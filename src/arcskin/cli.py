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
    arcskin network "2 [N]=O + O=O -> 2 [O][N]=O"

Each A + B pair runs in its own SMILES1_SMILES2/ directory with a log in
SMILES1_SMILES2/arcskin.log, whether started by `run` or `network`. `network`
appends its own summary to network.log; pairs already in the database are reused.
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
from collections import Counter

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
        except (Exception, KeyboardInterrupt):
            section("ERROR")
            traceback.print_exc(file=sys.stdout)
            raise


def run_pipeline(args, db, key, directory, suggest=True):
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
    if suggest:
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
    run_pair(args, db, key, args.dir or slug(key))


def slug(text):
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")


def run_pair(args, db, key, directory, suggest=True):
    """Run the pipeline for args.smiles1 + args.smiles2 with a log in `directory`."""
    os.makedirs(directory, exist_ok=True)
    logfile = os.path.join(directory, "arcskin.log")
    with run_log(logfile):
        try:
            run_pipeline(args, db, key, directory, suggest=suggest)
        except Exception as e:
            save(db, key, status=f"failed: {e}")
            raise
        finally:
            print(f"log written to {logfile}")


# --------------------------------------------------------------------------
# network: search A + B pairs until the overall reaction is connected
# --------------------------------------------------------------------------
def parse_side(text):
    """'2 [N]=O + O=O' -> ['[N]=O', '[N]=O', 'O=O'] (canonical SMILES)."""
    out = []
    for term in re.split(r"\s+\+\s+", text.strip()):
        term = term.strip()
        m = re.fullmatch(r"(\d+)\s*\*?\s*(\S+)", term)
        n, smi = (int(m.group(1)), m.group(2)) if m else (1, term)
        try:
            out += [canon(smi, strict=True)] * n
        except ValueError:
            raise ValueError(f"{smi!r} is not valid SMILES (formulas like NO2 are not "
                             f"supported; use e.g. [O][N]=O)") from None
    return out


def parse_reaction(text):
    """Split 'A + B -> C' (also '>>', '=>' or ' = ' with spaces) into two sides."""
    parts = re.split(r"\s*(?:->|>>|=>)\s*|\s+=\s+", text.strip())
    if len(parts) != 2:
        raise ValueError("write the reaction as 'A + B -> C' (or ' = ' with spaces)")
    return parse_side(parts[0]), parse_side(parts[1])


def atom_counts(smiles):
    c = Counter()
    for smi in smiles:
        mol = Chem.AddHs(Chem.MolFromSmiles(smi))
        c.update(a.GetSymbol() for a in mol.GetAtoms())
        c["charge"] += Chem.GetFormalCharge(mol)
    return +c


def skeleton(smi):
    """Connectivity-only key (all-H graph, single bonds, no charges/radicals), so
    e.g. [O]N=O and O=[N+][O-] both count as NO2."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return smi
    mol = Chem.RWMol(Chem.AddHs(mol))
    for b in mol.GetBonds():
        b.SetBondType(Chem.BondType.SINGLE)
        b.SetIsAromatic(False)
    for a in mol.GetAtoms():
        a.SetFormalCharge(0)
        a.SetNumRadicalElectrons(0)
        a.SetIsAromatic(False)
        a.SetNoImplicit(True)
    return Chem.MolToSmiles(mol)


def match_targets(targets, made):
    """{target: species in `made` with the same connectivity (or None)}."""
    by_skel = {}
    for sp in made:
        by_skel.setdefault(skeleton(sp), sp)
    return {t: (t if t in made else by_skel.get(skeleton(t))) for t in targets}


def reasonable_steps(db, max_ea):
    """Every sane NEB pathway in the database with Ea <= max_ea."""
    steps = []
    for e in db.all():
        if e.get("status") != "complete":
            continue
        for r in e.get("neb") or []:
            try:
                ea = float(r["Ea"])
            except (TypeError, ValueError, KeyError):
                continue
            if r.get("sane") and ea <= max_ea:
                steps.append({
                    "source": e["key"], "pathway": r.get("pathway"),
                    "reactants": [canon(x) for x in r["reactants"]],
                    "products": [canon(x) for x in r["products"]],
                    "Ea": ea, "dE": r.get("dE"), "kind": r.get("kind", ""),
                })
    return sorted(steps, key=lambda st: st["Ea"])


def reach(start, steps):
    """Species reachable from `start` through `steps`.
    Returns {species: producing step or None}, {species: generation}."""
    made = {s: None for s in start}
    gen = {s: 0 for s in start}
    changed = True
    while changed:
        changed = False
        for st in steps:  # lowest Ea first, so it wins when several steps make X
            if all(r in made for r in st["reactants"]):
                g = 1 + max(gen[r] for r in st["reactants"])
                for p in st["products"]:
                    if p not in made:
                        made[p], gen[p], changed = st, g, True
    return made, gen


def net(reactants, products, k=1):
    """Net stoichiometry {skeleton: coeff} (products +, reactants -)."""
    c = Counter()
    for x in products:
        c[skeleton(x)] += k
    for x in reactants:
        c[skeleton(x)] -= k
    return {sp: v for sp, v in c.items() if v}


def balance(reactants, products, steps, max_mult=4):
    """Find whole-number multiples of `steps` that add up exactly to the overall
    reaction (k x overall, k >= 1), so every intermediate cancels.

    Minimises the rate-limiting (largest) barrier first, then the number of step
    uses. Returns (k, [(multiplier, step), ...] in a runnable order) or None."""
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp

    target = net(reactants, products)
    uniq = {}  # one copy of each distinct, atom-balanced, non-trivial step
    for st in steps:
        v = net(st["reactants"], st["products"])
        if not v or atom_counts(st["reactants"]) != atom_counts(st["products"]):
            continue
        sig = tuple(sorted(v.items()))
        if sig not in uniq or st["Ea"] < uniq[sig][0]["Ea"]:
            uniq[sig] = (st, v)
    if not uniq:
        return None
    cand = sorted(uniq.values(), key=lambda sv: sv[0]["Ea"])
    species = sorted(set(target) | {sp for _, v in cand for sp in v})

    for cut in sorted({st["Ea"] for st, _ in cand}):  # minimax barrier
        use = [(st, v) for st, v in cand if st["Ea"] <= cut]
        n = len(use)
        A = np.array([[v.get(sp, 0) for _, v in use] + [-target.get(sp, 0)]
                      for sp in species], float)
        res = milp(c=np.r_[np.ones(n), 0.0],
                   constraints=LinearConstraint(A, 0, 0),
                   integrality=np.ones(n + 1),
                   bounds=Bounds(np.r_[np.zeros(n), 1], np.r_[np.full(n, max_mult), max_mult]))
        if res.success:
            x = np.rint(res.x).astype(int)
            chosen = [(int(m), st) for m, (st, _) in zip(x[:n], use) if m > 0]
            return int(x[n]), order_steps(chosen, reactants, int(x[n]))
    return None


def order_steps(chosen, reactants, k):
    """Order (mult, step) so each step's reactants exist when it runs."""
    have = Counter()
    for x in reactants:
        have[skeleton(x)] += k
    todo, out = [[m, st] for m, st in chosen], []
    while todo:
        for item in todo:
            need = Counter(skeleton(x) for x in item[1]["reactants"])
            if all(have[sp] >= c for sp, c in need.items()):
                have.subtract(need)
                have.update(skeleton(x) for x in item[1]["products"])
                item[0] -= 1
                if not out or out[-1][1] is not item[1]:
                    out.append([0, item[1]])
                out[-1][0] += 1
                if item[0] == 0:
                    todo.remove(item)
                break
        else:  # needs a species made later (cycle/catalyst): append the rest
            out += todo
            break
    return [(m, st) for m, st in out]


def candidate_pairs(db, pool, gen, tried, retry_failed=False):
    """Untried A + B pairs (A may equal B) from `pool`, earliest generation first."""
    status = {e["key"]: e.get("status", "") for e in db.all()}
    out = []
    for a, b in itertools.combinations_with_replacement(sorted(pool), 2):
        key = reaction_key(a, b)
        st = status.get(key)
        if key in tried or st == "complete" or st == "running":
            continue
        if st and st.startswith("failed") and not retry_failed:
            continue
        out.append(((max(gen[a], gen[b]), gen[a] + gen[b], a == b, key), a, b, key))
    return [(a, b, key) for _, a, b, key in sorted(out)]


def print_mechanism(steps):
    table([(i + 1, f"{m}x", fmt_rxn(st["reactants"], st["products"]), fnum(st["Ea"]),
            fnum(st["dE"]), st["kind"], f"{st['source']} #{st['pathway']}")
           for i, (m, st) in enumerate(steps)],
          ["step", "times", "reaction", "Ea/eV", "dE/eV", "kind", "from run"])


def cmd_network(args):
    db = TinyDB(args.db)
    reactants, products = parse_reaction(args.reaction)
    targets = sorted(set(products) - set(reactants))
    overall = f"{' + '.join(reactants)} -> {' + '.join(products)}"
    directory = args.dir or "."
    os.makedirs(directory, exist_ok=True)
    logfile = os.path.join(directory, "network.log")
    console = sys.stdout
    networks, N = db.table("networks"), Query()

    with run_log(logfile):
        banner(f"ARCS-Kinetics network search: {overall}")
        print(f"  started   : {now()}")
        print(f"  directory : {os.path.abspath(directory)}")
        print(f"  database  : {os.path.abspath(args.db)}")
        print(f"  targets   : {', '.join(targets)}")
        print(f"  criteria  : sane NEB pathway with Ea <= {args.max_ea} eV; "
              f"at most {args.max_runs} new pair run(s)")
        ra, pa = atom_counts(reactants), atom_counts(products)
        if ra != pa:
            print(f"  WARNING   : overall reaction is not balanced "
                  f"({dict(ra)} vs {dict(pa)})")
        if not targets:
            sys.exit("error: nothing to make (all products are also reactants)")

        runs, tried, mech = 0, set(), None
        while True:
            steps = reasonable_steps(db, args.max_ea)
            made, gen = reach(set(reactants), steps)
            found = match_targets(targets, made)
            have = [t for t in targets if found[t]]
            log(f"{len(steps)} reasonable step(s) in database; "
                f"{len(made)} species reachable; targets reached: "
                f"{len(have)}/{len(targets)}")
            if len(have) == len(targets):
                usable = [st for st in steps
                          if all(r in made for r in st["reactants"])]
                mech = balance(reactants, products, usable, args.max_mult)
                if mech:
                    rl = max(st["Ea"] for _, st in mech[1])
                    for t in targets:
                        if found[t] != t:
                            log(f"target {t} matched as {found[t]} (same connectivity)")
                    if not args.best:
                        break
                    log(f"balanced mechanism found (rate-limiting Ea = {rl:.3f} eV); "
                        f"--best: continuing to look for a lower one")
                else:
                    log("targets reachable, but no combination of steps balances "
                        "the overall reaction yet; continuing")
            cands = candidate_pairs(db, made, gen, tried, args.retry_failed)
            if not cands:
                log("no untried pairs left")
                break
            if runs >= args.max_runs:
                log(f"reached --max-runs ({args.max_runs}); "
                    f"{len(cands)} pair(s) still untried")
                break
            a, b, key = cands[0]
            runs += 1
            tried.add(key)
            section(f"run {runs}/{args.max_runs}: {a} + {b}  "
                    f"(generation {max(gen[a], gen[b])}, {len(cands) - 1} more queued)")
            pargs = argparse.Namespace(**vars(args))
            pargs.smiles1, pargs.smiles2, pargs.dir = a, b, None
            pair_dir = os.path.join(directory, slug(key))
            n_before = len(steps)
            try:
                with contextlib.redirect_stdout(console):  # details -> pair log only
                    run_pair(pargs, db, key, pair_dir, suggest=False)
            except Exception as e:
                log(f"FAILED: {e}  (see {pair_dir}/arcskin.log)")
                continue
            entry = get_entry(db, key)
            log(f"done -> {pair_dir}/arcskin.log")
            print_best(entry.get("best_pathways") or [])
            new = len(reasonable_steps(db, args.max_ea)) - n_before
            log(f"{new} new reasonable step(s)")

        banner("result")
        print(f"  finished  : {now()}  ({runs} new pair run(s))")
        if mech is None:
            missing = [t for t in targets if not found[t]]
            if missing:
                print(f"  NO mechanism found. Unreached target(s): {', '.join(missing)}")
            else:
                print("  NO mechanism found. All targets are reachable, but no "
                      "combination of steps balances the overall reaction")
            print(f"  reachable species: {', '.join(sorted(made))}")
            print("  try raising --max-ea / --max-runs, or --retry-failed")
            networks.upsert(jsanitize({"overall": overall, "status": "not found",
                                       "reachable": sorted(made), "updated": now()}),
                            N.overall == overall)
        else:
            k, steps_used = mech
            rls = max((st for _, st in steps_used), key=lambda st: st["Ea"])
            section("mechanism")
            print_mechanism(steps_used)
            total, name = Counter(), {}
            for m, st in steps_used:
                for x in st["reactants"] + st["products"]:
                    name.setdefault(skeleton(x), x)
                for sp, v in net(st["reactants"], st["products"], m).items():
                    total[name[sp]] += v
            lhs = [f"{-v} {sp}" for sp, v in sorted(total.items()) if v < 0]
            rhs = [f"{v} {sp}" for sp, v in sorted(total.items()) if v > 0]
            print(f"\n  net                : {' + '.join(lhs)} -> {' + '.join(rhs)}"
                  + (f"   (= {k} x overall)" if k > 1 else "   (= overall)"))
            print(f"  rate-limiting step : {fmt_rxn(rls['reactants'], rls['products'])}"
                  f"  (Ea = {rls['Ea']:.3f} eV)")
            networks.upsert(jsanitize({"overall": overall, "status": "found",
                                       "multiple": k,
                                       "mechanism": [dict(st, times=m)
                                                     for m, st in steps_used],
                                       "rate_limiting": rls, "updated": now()},
                                      strict=False),
                            N.overall == overall)
        print(f"\nlog written to {logfile}")


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

    def run_options(r):
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

    r = sub.add_parser("run", help="run the full pipeline for two SMILES")
    r.add_argument("smiles1")
    r.add_argument("smiles2")
    r.add_argument("--dir", help="working directory (default: from SMILES)")
    run_options(r)
    r.add_argument("--force", action="store_true", help="rerun even if in database")
    r.set_defaults(func=cmd_run)

    w = sub.add_parser("network", help="run A + B pairs until an overall reaction "
                       "is connected by reasonable steps")
    w.add_argument("reaction", help="SMILES reaction, e.g. "
                   "\"2 [N]=O + O=O -> 2 [O][N]=O\"")
    w.add_argument("--dir", help="where the SMILES1_SMILES2/ pair directories and "
                   "network.log go (default: current directory)")
    w.add_argument("--max-ea", type=float, default=1.0,
                   help="largest barrier (eV) a step may have (default 1.0)")
    w.add_argument("--max-runs", type=int, default=20,
                   help="maximum number of new pair runs (default 20)")
    w.add_argument("--max-mult", type=int, default=4,
                   help="most times one step may be used in the balanced "
                   "mechanism (default 4)")
    w.add_argument("--best", action="store_true",
                   help="don't stop at the first balanced mechanism: run pairs "
                   "until --max-runs / none left, then report the one with the "
                   "lowest rate-limiting barrier")
    w.add_argument("--retry-failed", action="store_true",
                   help="also rerun pairs whose earlier run failed")
    run_options(w)
    w.set_defaults(func=cmd_network)

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
