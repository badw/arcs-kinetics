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
    arcskin suggest -n 10

Each `run` writes a human-readable log to <dir>/arcskin.log.
"""

import argparse
import contextlib
import datetime
import itertools
import json
import os
import re
import sys
import traceback

from monty.json import jsanitize
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
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
    cols = ["pathway", "reactants", "products", "Ea", "dE", "sane", "kind", "reason"]
    return [{c: row.get(c) for c in cols} for _, row in best.iterrows()]


# --------------------------------------------------------------------------
# pretty logging: everything printed during a run is tee'd into <dir>/arcskin.log
# --------------------------------------------------------------------------
WIDTH = 78


class Tee:
    """Write to several streams at once (console + log file)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for st in self.streams:
            st.write(data)
            st.flush()
        return len(data)

    def flush(self):
        for st in self.streams:
            st.flush()

    def isatty(self):
        return False


def now():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg=""):
    print(f"[{now()[11:]}] {msg}" if msg else "")


def banner(title, char="="):
    print("\n" + char * WIDTH)
    print(f" {title}")
    print(char * WIDTH)


def section(title):
    print(f"\n--- {title} " + "-" * max(0, WIDTH - len(title) - 5))


def table(rows, headers):
    """Plain-text table; rows is a list of sequences."""
    rows = [["-" if v is None else str(v) for v in r] for r in rows]
    w = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h)
         for i, h in enumerate(headers)]
    fmt = "  ".join(f"{{:<{x}}}" for x in w)
    print("  " + fmt.format(*headers))
    print("  " + "  ".join("-" * x for x in w))
    for r in rows:
        print("  " + fmt.format(*r))


def fmt_rxn(reac, prod):
    return f"{' + '.join(reac)} -> {' + '.join(prod)}"


def fnum(x, f="{:.3f}"):
    try:
        return f.format(float(x))
    except (TypeError, ValueError):
        return "-"


@contextlib.contextmanager
def run_log(path):
    """Tee stdout into `path` for the duration of the block; log tracebacks."""
    with open(path, "a") as fh, contextlib.redirect_stdout(Tee(sys.stdout, fh)):
        try:
            yield
        except BaseException:
            section("ERROR")
            traceback.print_exc(file=sys.stdout)
            raise


def run_pipeline(args, db, key, directory):
    # heavy imports only when actually running
    from arcskin.make_neb_endpoints import make_endpoints
    from arcskin.nanoreactor_analysis import AnalyseNanoReactor
    from arcskin.nanoreactor_run import SMILESPack, XTBNanoReactor
    from arcskin.run_neb import run_all

    t0 = datetime.datetime.now()
    smiles = [args.smiles1, args.smiles2]
    known_before = {c["smiles"] for c in db.table("compounds").all()}
    settings = {k: v for k, v in vars(args).items() if k != "func"}

    banner(f"ARCS-Kinetics run: {' + '.join(smiles)}")
    print(f"  started   : {now()}")
    print(f"  key       : {key}")
    print(f"  directory : {os.path.abspath(directory)}")
    print(f"  database  : {os.path.abspath(args.db)}")
    section("settings")
    table([(k, v) for k, v in settings.items()
           if k not in ("smiles1", "smiles2", "db")], ["parameter", "value"])

    save(db, key, smiles=smiles, directory=directory, status="running",
         settings=settings, log=os.path.join(directory, "arcskin.log"))
    add_compounds(db, key, smiles, "reactant")

    # 1. nanoreactor
    section("[1/4] nanoreactor")
    log(f"{args.num_runs} run(s), {args.num_molecules} x each molecule, "
        f"T={args.temperature} K, t={args.time} ps")
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
    log("nanoreactor done")

    # 2. analysis
    section("[2/4] nanoreactor analysis")
    anr = AnalyseNanoReactor(skip_step=args.skip_step, reactants=smiles)
    anr.read_trajectory(xnr.trajectory_locations, format="xyz")
    anr.analyse_trajectories(n_cpus=args.n_cpus)
    possible_paths = anr.get_possible_paths_dict()
    save(db, key,
         species_counts=dict(anr.unique_smiles_counter),
         species=anr.unique_smiles,
         pathways=possible_paths)
    counts = dict(anr.unique_smiles_counter)
    log(f"{len(counts)} unique species observed")
    table(sorted(counts.items(), key=lambda kv: -kv[1]), ["species", "count"])
    for v in (possible_paths or {}).values():
        add_compounds(db, key, v["products"], "product")

    best = []
    if not possible_paths:
        log("no sensible pathways found")
    else:
        log(f"{len(possible_paths)} candidate pathway(s)")
        table([(k, fmt_rxn(v["reactants"], v["products"]))
               for k, v in possible_paths.items()], ["#", "pathway"])

        # 3. NEB endpoints
        section(f"[3/4] NEB endpoints ({len(possible_paths)} pathway(s))")
        endpoints = {}
        for k, v in possible_paths.items():
            try:
                reac, prod = make_endpoints(v["reactants"], v["products"])
                endpoints[k] = {"reactants": reac, "products": prod, "overall": v}
                log(f"pathway {k}: ok")
            except Exception as e:
                log(f"pathway {k}: FAILED ({e})")

        # 4. NEB
        if endpoints:
            section(f"[4/4] NEB ({len(endpoints)} pathway(s))")
            ranking = run_all(endpoints, n_images=args.n_images, climb=True,
                              fmax=args.fmax)
            ranking = ranking.reset_index(names="pathway")
            save(db, key, neb=ranking.to_dict("records"))
            best = best_pathways(ranking, args.ea_tol)
            log("NEB ranking:")
            table([(r["pathway"], fmt_rxn(r["reactants"], r["products"]),
                    fnum(r["Ea"]), fnum(r["dE"]), bool(r["sane"]),
                    r.get("kind"), r.get("reason") or "")
                   for _, r in ranking.sort_values("Ea").iterrows()],
                  ["#", "pathway", "Ea/eV", "dE/eV", "sane", "kind", "note"])
        else:
            log("no endpoints could be built; skipping NEB")

    save(db, key, best_pathways=best, status="complete")

    # summary
    all_compounds = {c["smiles"] for c in db.table("compounds").all()}
    new = sorted(all_compounds - known_before)
    banner("summary")
    print(f"  finished  : {now()}  (elapsed {str(datetime.datetime.now() - t0).split('.')[0]})")
    print(f"  compounds : {len(all_compounds)} in database, {len(new)} new from this run")
    for n in new:
        print(f"              + {n}")
    section("suggested pathway(s)")
    print_best(best)
    section("next combinations to try")
    print_suggestions(suggest_pairs(db, limit=5))
    print()


def print_best(best):
    if not best:
        print("  no pathway found")
        return
    table([(b["pathway"], fmt_rxn(b["reactants"], b["products"]),
            fnum(b["Ea"]), fnum(b["dE"]), b.get("kind") or "") for b in best],
          ["#", "pathway", "Ea/eV", "dE/eV", "kind"])


# --------------------------------------------------------------------------
# suggestions: compound pairs from the database that have not been run
# --------------------------------------------------------------------------
def is_stable(smi):
    """Neutral, closed-shell and parsable -> sensible nanoreactor reactant."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return False
    return (Chem.GetFormalCharge(mol) == 0
            and Descriptors.NumRadicalElectrons(mol) == 0)


def suggest_pairs(db, limit=20, include_self=False, allow_unstable=False,
                  containing=None):
    """Rank unprobed pairs of database compounds.

    A compound is "unexplored" if it has never been used as a reactant.
    Pairs score higher the more unexplored compounds they contain, then by how
    often the compounds have been produced (more robust products first).
    """
    compounds = db.table("compounds").all()
    done = {e["key"]: e.get("status", "") for e in db.all()}
    info = {c["smiles"]: c for c in compounds
            if allow_unstable or is_stable(c["smiles"])}
    pool = sorted(info)
    pairs = (itertools.combinations_with_replacement(pool, 2) if include_self
             else itertools.combinations(pool, 2))
    if containing:
        containing = canon(containing, strict=True)

    out = []
    for a, b in pairs:
        if containing and containing not in (a, b):
            continue
        key = reaction_key(a, b)
        status = done.get(key)
        if status is not None and not status.startswith("failed"):
            continue
        unexplored = sum(not info[x]["reactant_in"] for x in {a, b})
        produced = sum(len(info[x]["product_in"]) for x in {a, b})
        out.append({"smiles": (a, b), "key": key, "unexplored": unexplored,
                    "produced": produced,
                    "note": "previously failed" if status else ""})
    out.sort(key=lambda d: (-d["unexplored"], -d["produced"], d["key"]))
    return out[:limit] if limit else out


def print_suggestions(sugg):
    if not sugg:
        print("  no unprobed combinations left")
        return
    table([(i + 1, s["smiles"][0], s["smiles"][1], s["unexplored"],
            s["produced"], s["note"]) for i, s in enumerate(sugg)],
          ["rank", "smiles1", "smiles2", "unexplored", "produced", "note"])
    a, b = sugg[0]["smiles"]
    print(f"\n  e.g.  arcskin run '{a}' '{b}'")


def cmd_run(args):
    db = TinyDB(args.db)
    key = reaction_key(args.smiles1, args.smiles2)
    entry = get_entry(db, key)
    if entry and entry.get("status") == "complete" and not args.force:
        print_best(entry.get("best_pathways", []))
        sys.exit(f"error: {key} has already been run (in {args.db}); use --force to rerun")
    directory = args.dir or re.sub(r"[^A-Za-z0-9]+", "_", key).strip("_")
    os.makedirs(directory, exist_ok=True)
    logfile = os.path.join(directory, "arcskin.log")
    with run_log(logfile):
        try:
            run_pipeline(args, db, key, directory)
        except Exception as e:
            save(db, key, status=f"failed: {e}")
            raise
        finally:
            print(f"log written to {logfile}")


def cmd_list(args):
    for e in TinyDB(args.db).all():
        best = e.get("best_pathways") or []
        sugg = "; ".join(f"{'+'.join(b['products'])} (Ea={b['Ea']:.2f})" for b in best)
        print(f"{e['key']:40s} {e.get('status', ''):10s} {sugg or '-'}")


def cmd_compounds(args):
    for c in TinyDB(args.db).table("compounds").all():
        print(f"{c['smiles']:30s} reactant in: {c['reactant_in'] or '-'}  "
              f"product of: {c['product_in'] or '-'}")


def cmd_suggest(args):
    db = TinyDB(args.db)
    n = len(db.table("compounds"))
    print(f"{n} compound(s) in {args.db}; {len(db)} combination(s) probed\n")
    print_suggestions(suggest_pairs(db, limit=args.n, include_self=args.include_self,
                                    allow_unstable=args.allow_unstable,
                                    containing=args.containing))


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

    g = sub.add_parser("suggest",
                       help="suggest compound pairs from the database not yet run")
    g.add_argument("-n", type=int, default=20, help="number of suggestions (0 = all)")
    g.add_argument("--containing", metavar="SMILES",
                   help="only pairs that include this compound")
    g.add_argument("--include-self", action="store_true",
                   help="also suggest a compound paired with itself")
    g.add_argument("--allow-unstable", action="store_true",
                   help="include radicals and charged species")
    g.set_defaults(func=cmd_suggest)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
