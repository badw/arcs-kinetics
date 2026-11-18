"""
Command line interface for the full ARCS-Kinetics pipeline:

    nanoreactor -> nanoreactor analysis -> NEB endpoints -> NEB

Results are stored in a TinyDB database keyed on the (canonicalised, order
independent) reactant pair, so a pair that has already been run is skipped.

    arcskin run "O=S(=O)=O" "O" --dir so3_h2o
    arcskin list
    arcskin show "O=S(=O)=O" "O"
"""

import argparse
import json
import os
import re

from monty.json import jsanitize
from rdkit import Chem
from tinydb import Query, TinyDB


def reaction_key(smi1, smi2):
    """Order-independent key from RDKit canonical SMILES."""
    can = []
    for s in (smi1, smi2):
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            raise ValueError(f"RDKit could not parse SMILES: {s!r}")
        can.append(Chem.MolToSmiles(mol))
    return ".".join(sorted(can))


def get_entry(db, key):
    return db.get(Query().key == key)


def save(db, key, **fields):
    db.upsert(jsanitize({"key": key, **fields}, strict=False), Query().key == key)


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
    if not possible_paths:
        print("no sensible pathways found")
        save(db, key, status="complete", best_pathway=None)
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
        save(db, key, status="complete", best_pathway=None)
        return

    # 4. NEB
    print(f"[4/4] running NEB on {len(endpoints)} pathway(s)")
    ranking = run_all(endpoints, n_images=args.n_images, climb=True, fmax=args.fmax)
    ranking = ranking.reset_index(names="pathway")
    neb = ranking.to_dict("records")
    sane = ranking[ranking["sane"].astype(bool)]
    best = (sane if len(sane) else ranking).iloc[0]
    best = {c: best[c] for c in ["pathway", "reactants", "products", "Ea", "dE", "sane"]}
    save(db, key, neb=neb, best_pathway=best, status="complete")
    print(f"best pathway: {best}")


def cmd_run(args):
    db = TinyDB(args.db)
    key = reaction_key(args.smiles1, args.smiles2)
    entry = get_entry(db, key)
    if entry and entry.get("status") == "complete" and not args.force:
        print(f"{key} already in {args.db} (use --force to rerun)")
        print(f"best pathway: {entry.get('best_pathway')}")
        return
    try:
        run_pipeline(args, db, key)
    except Exception as e:
        save(db, key, status=f"failed: {e}")
        raise


def cmd_list(args):
    for e in TinyDB(args.db).all():
        best = e.get("best_pathway") or {}
        print(f"{e['key']:40s} {e.get('status', ''):10s} "
              f"Ea={best.get('Ea', '-')}  products={best.get('products', '-')}")


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
    r.add_argument("--force", action="store_true", help="rerun even if in database")
    r.set_defaults(func=cmd_run)

    l = sub.add_parser("list", help="list reactions in the database")
    l.set_defaults(func=cmd_list)

    s = sub.add_parser("show", help="show stored results for two SMILES")
    s.add_argument("smiles1")
    s.add_argument("smiles2")
    s.add_argument("--full", action="store_true", help="include NEB images")
    s.set_defaults(func=cmd_show)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
