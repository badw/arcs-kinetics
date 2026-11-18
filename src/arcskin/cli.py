"""
Command line interface for the full ARCS-Kinetics pipeline:

    nanoreactor -> nanoreactor analysis -> NEB endpoints -> NEB

Results are stored in a TinyDB database keyed on the (canonicalised, order
independent) reactant pair; running a pair that is already done is an error.
Every reactant and discovered product is also recorded in a `compounds` table.

    arcskin run "O=S(=O)=O" "O" --dir so3_h2o
    arcskin list
    arcskin show "O=S(=O)=O" "O"
    arcskin compounds
"""

import argparse
import json
import os
import re
import sys

from monty.json import jsanitize
from rdkit import Chem, RDLogger
from tinydb import Query, TinyDB

RDLogger.DisableLog("rdApp.warning")


def canon(smi, strict=False):
    """RDKit canonical SMILES (returns the input unchanged if unparsable)."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        if strict:
            raise ValueError(f"RDKit could not parse SMILES: {smi!r}")
        return smi
    return Chem.MolToSmiles(mol)


def reaction_key(smi1, smi2):
    """Order-independent key from RDKit canonical SMILES."""
    return ".".join(sorted(canon(s, strict=True) for s in (smi1, smi2)))


def get_entry(db, key):
    return db.get(Query().key == key)


def save(db, key, **fields):
    db.upsert(jsanitize({"key": key, **fields}, strict=False), Query().key == key)


def add_compounds(db, key, smiles, role):
    """Record compounds and the reaction they appeared in (role: reactant/product)."""
    table, C = db.table("compounds"), Query()
    for s in sorted({canon(x) for x in smiles}):
        doc = dict(table.get(C.smiles == s) or
                   {"smiles": s, "reactant_in": [], "product_in": []})
        if key not in doc[f"{role}_in"]:
            doc[f"{role}_in"].append(key)
        table.upsert(doc, C.smiles == s)


def best_pathways(ranking, ea_tol):
    """Lowest-Ea pathway plus any within ea_tol (eV) of it (sane ones preferred)."""
    sane = ranking[ranking["sane"].astype(bool)]
    pool = sane if len(sane) else ranking
    best = pool[pool["Ea"] <= pool["Ea"].min() + ea_tol]
    cols = ["pathway", "reactants", "products", "Ea", "dE", "sane"]
    return [{c: row[c] for c in cols} for _, row in best.iterrows()]


def run_pipeline(args, db, key):
    # heavy imports only when actually running
    from arcskin.make_neb_endpoints import make_endpoints
    from arcskin.nanoreactor_analysis import AnalyseNanoReactor
    from arcskin.nanoreactor_run import SMILESPack, XTBNanoReactor
    from arcskin.run_neb import run_all

    smiles = [args.smiles1, args.smiles2]
    directory = args.dir or re.sub(r"[^A-Za-z0-9]+", "_", key).strip("_")
    save(db, key, smiles=smiles, directory=directory, status="running",
         settings=vars(args) | {"func": None})
    add_compounds(db, key, smiles, "reactant")

    # 1. nanoreactor
    print(f"[1/4] nanoreactor: {smiles} -> {directory}")
    cells, charges = SMILESPack(smiles).run_multiple(
        args.num_runs,
        counts=[args.num_molecules, args.num_molecules],
        min_dist=args.min_dist,
    )
    xnr = XTBNanoReactor(
        combination=tuple(smiles), cells=cells, charges=charges,
        temperature=args.temperature, time=args.time,
    )
    xnr.run(location=directory)

    # 2. analysis
    print("[2/4] nanoreactor analysis")
    anr = AnalyseNanoReactor(skip_step=args.skip_step, reactants=smiles)
    anr.read_trajectory(xnr.trajectory_locations, format="xyz")
    anr.analyse_trajectories(n_cpus=args.n_cpus)
    possible_paths = anr.get_possible_paths_dict()
    save(db, key,
         species_counts=dict(anr.unique_smiles_counter),
         species=anr.unique_smiles,
         pathways=possible_paths)
    for v in (possible_paths or {}).values():
        add_compounds(db, key, v["products"], "product")
    if not possible_paths:
        print("no sensible pathways found")
        save(db, key, status="complete", best_pathways=[])
        return

    # 3. NEB endpoints
    print(f"[3/4] making endpoints for {len(possible_paths)} pathway(s)")
    endpoints = {}
    for k, v in possible_paths.items():
        try:
            reac, prod = make_endpoints(v["reactants"], v["products"])
            endpoints[k] = {"reactants": reac, "products": prod, "overall": v}
        except Exception as e:
            print(f"pathway {k} failed: {e}")
    if not endpoints:
        save(db, key, status="complete", best_pathways=[])
        return

    # 4. NEB
    print(f"[4/4] running NEB on {len(endpoints)} pathway(s)")
    ranking = run_all(endpoints, n_images=args.n_images, climb=True, fmax=args.fmax)
    ranking = ranking.reset_index(names="pathway")
    neb = ranking.to_dict("records")
    best = best_pathways(ranking, args.ea_tol)
    save(db, key, neb=neb, best_pathways=best, status="complete")
    print_best(best)


def print_best(best):
    if not best:
        print("no pathway found")
    for b in best:
        print(f"suggested pathway {b['pathway']}: {' + '.join(b['reactants'])} -> "
              f"{' + '.join(b['products'])}  Ea={b['Ea']:.3f} eV  dE={b['dE']:.3f} eV")


def cmd_run(args):
    db = TinyDB(args.db)
    key = reaction_key(args.smiles1, args.smiles2)
    entry = get_entry(db, key)
    if entry and entry.get("status") == "complete" and not args.force:
        print_best(entry.get("best_pathways", []))
        sys.exit(f"error: {key} has already been run (in {args.db}); use --force to rerun")
    try:
        run_pipeline(args, db, key)
    except Exception as e:
        save(db, key, status=f"failed: {e}")
        raise


def cmd_list(args):
    for e in TinyDB(args.db).all():
        best = e.get("best_pathways") or []
        sugg = "; ".join(f"{'+'.join(b['products'])} (Ea={b['Ea']:.2f})" for b in best)
        print(f"{e['key']:40s} {e.get('status', ''):10s} {sugg or '-'}")


def cmd_compounds(args):
    for c in TinyDB(args.db).table("compounds").all():
        print(f"{c['smiles']:30s} reactant in: {c['reactant_in'] or '-'}  "
              f"product of: {c['product_in'] or '-'}")


def cmd_show(args):
    entry = get_entry(TinyDB(args.db), reaction_key(args.smiles1, args.smiles2))
    if not entry:
        print("not in database")
        return
    if not args.full:  # drop the bulky NEB images / fits
        entry = dict(entry)
        entry["neb"] = [
            {k: v for k, v in r.items() if k not in ("images", "nebfit")}
            for r in entry.get("neb", [])
        ]
    print(json.dumps(entry, indent=2))


def main(argv=None):
    p = argparse.ArgumentParser(prog="arcskin", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", default=os.environ.get("ARCSKIN_DB", "arcskin_db.json"),
                   help="TinyDB file (default: arcskin_db.json or $ARCSKIN_DB)")
    sub = p.add_subparsers(required=True)

    r = sub.add_parser("run", help="run the full pipeline for two SMILES")
    r.add_argument("smiles1")
    r.add_argument("smiles2")
    r.add_argument("--dir", help="working directory (default: from SMILES)")
    r.add_argument("--num-molecules", type=int, default=4, help="copies of each molecule")
    r.add_argument("--num-runs", type=int, default=4, help="number of nanoreactor seeds")
    r.add_argument("--min-dist", type=float, default=2.2)
    r.add_argument("--temperature", type=int, default=1200)
    r.add_argument("--time", type=float, default=5.0, help="MD time (ps)")
    r.add_argument("--skip-step", type=int, default=10)
    r.add_argument("--n-cpus", type=int, default=4)
    r.add_argument("--n-images", type=int, default=16)
    r.add_argument("--fmax", type=float, default=0.05)
    r.add_argument("--ea-tol", type=float, default=0.1,
                   help="also suggest pathways with Ea within this of the best (eV)")
    r.add_argument("--force", action="store_true", help="rerun even if in database")
    r.set_defaults(func=cmd_run)

    l = sub.add_parser("list", help="list reactions in the database")
    l.set_defaults(func=cmd_list)

    s = sub.add_parser("show", help="show stored results for two SMILES")
    s.add_argument("smiles1")
    s.add_argument("smiles2")
    s.add_argument("--full", action="store_true", help="include NEB images")
    s.set_defaults(func=cmd_show)

    c = sub.add_parser("compounds", help="list all compounds in the database")
    c.set_defaults(func=cmd_compounds)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
