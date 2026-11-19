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
    arcskin batch "O=S=O" "[O]N=O" "[N]=O" "O=O" "O"

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


def loose_mol(smi):
    """Parse without sanitising, so OpenBabel-style SMILES such as O=N=O or
    N#N=O are read as written (no valence errors, no H silently added)."""
    mol = Chem.MolFromSmiles(smi, sanitize=False)
    if mol is not None:
        mol.UpdatePropertyCache(strict=False)
    return mol


def _composition(mol):
    c = Counter(a.GetSymbol() for a in mol.GetAtoms())
    c["H"] += sum(a.GetTotalNumHs() for a in mol.GetAtoms())
    return +c


def canon(smi, strict=False):
    """Canonical SMILES. Uses RDKit's sanitised form when that keeps the same
    atoms; otherwise (hypervalent OpenBabel output, or RDKit would add H) a
    canonical form of the molecule exactly as written."""
    mol = Chem.MolFromSmiles(smi)
    loose = loose_mol(smi)
    if loose is None:
        if strict:
            raise ValueError(f"RDKit could not parse SMILES: {smi!r}")
        return smi
    if mol is not None and _composition(mol) == _composition(loose):
        return Chem.MolToSmiles(mol)
    return Chem.MolToSmiles(loose)


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


def usable(rec, max_ea):
    """A NEB record that a network search could use as a step."""
    try:
        return bool(rec.get("sane")) and float(rec["Ea"]) <= max_ea
    except (TypeError, ValueError, KeyError):
        return False


def neb_pathways(args, db, key, paths):
    """Endpoints + NEB for `paths` ({id: pathway}); merges the results into the
    entry's existing NEB records and best pathways. Returns the new records."""
    import pandas as pd

    from arcskin.make_neb_endpoints import make_endpoints
    from arcskin.run_neb import run_all

    log(f"building NEB endpoints for {len(paths)} pathway(s)")
    endpoints = {}
    for k, v in paths.items():
        try:
            reac, prod = make_endpoints(v["reactants"], v["products"])
            endpoints[k] = {"reactants": reac, "products": prod, "overall": v}
            log(f"pathway {k}: ok")
        except Exception as e:
            log(f"pathway {k}: FAILED ({e})")
    if not endpoints:
        log("no endpoints could be built; skipping NEB")
        return []
    log(f"running NEB on {len(endpoints)} pathway(s)")
    ranking = run_all(endpoints, n_images=args.n_images, climb=True, fmax=args.fmax)
    ranking = ranking.reset_index(names="pathway")
    new = ranking.to_dict("records")
    table([(r["pathway"], fmt_rxn(r["reactants"], r["products"]),
            fnum(r["Ea"]), fnum(r["dE"]), bool(r["sane"]),
            r.get("kind"), r.get("reason") or "")
           for _, r in ranking.sort_values("Ea").iterrows()],
          ["#", "pathway", "Ea/eV", "dE/eV", "sane", "kind", "note"])

    done = {str(r["pathway"]) for r in new}
    old = [r for r in (get_entry(db, key) or {}).get("neb") or []
           if str(r.get("pathway")) not in done]
    merged = old + jsanitize(new, strict=False)
    best = best_pathways(pd.DataFrame(merged), args.ea_tol)
    save(db, key, neb=merged, best_pathways=best)
    return new


def neb_in_tiers(args, db, key, paths, tier):
    """NEB `paths` tier by tier (tier(pathway) -> int, lowest first), stopping
    after the first tier that gives a usable step. Returns the ids not run."""
    tiers = {k: tier(v) for k, v in paths.items()}
    left = dict(paths)
    for t in sorted(set(tiers.values())):
        group = {k: v for k, v in left.items() if tiers[k] == t}
        section(f"NEB tier {t} ({TIER_NAMES.get(t, '')}): {len(group)} pathway(s)")
        new = neb_pathways(args, db, key, group)
        for k in group:
            left.pop(k)
        if any(usable(r, args.max_ea) for r in new):
            if left:
                log(f"usable step found; {len(left)} lower-priority pathway(s) "
                    f"deferred (run later only if needed)")
            break
    return sorted(left, key=str)


TIER_NAMES = {0: "completes the overall reaction", 1: "makes a target",
              2: "other", 3: "not possible in this flask"}


def run_pipeline(args, db, key, directory, suggest=True):
    # heavy imports only when actually running
    from arcskin.nanoreactor_analysis import AnalyseNanoReactor
    from arcskin.nanoreactor_run import SMILESPack, XTBNanoReactor

    t0 = datetime.datetime.now()
    smiles = [args.smiles1, args.smiles2]
    known_before = {c["smiles"] for c in db.table("compounds").all()}
    settings = {k: v for k, v in vars(args).items() if k not in ("func", "focus")}

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

    tier = getattr(args, "focus", None)
    if not possible_paths:
        log("no sensible pathways found")
        save(db, key, neb=[], best_pathways=[], deferred=[])
    else:
        log(f"{len(possible_paths)} candidate pathway(s)")
        if tier:
            table([(k, fmt_rxn(v["reactants"], v["products"]), tier(v))
                   for k, v in sorted(possible_paths.items(),
                                      key=lambda kv: (tier(kv[1]), kv[0]))],
                  ["#", "pathway", "tier"])
            print("  tiers: " + "; ".join(f"{t} = {n}" for t, n in TIER_NAMES.items()))
        else:
            table([(k, fmt_rxn(v["reactants"], v["products"]))
                   for k, v in possible_paths.items()], ["#", "pathway"])
        save(db, key, neb=[], best_pathways=[])
        if tier:  # network search: NEB the pathways that matter first
            deferred = neb_in_tiers(args, db, key, possible_paths, tier)
        else:
            section(f"[3/4 + 4/4] NEB endpoints and NEB ({len(possible_paths)} pathway(s))")
            neb_pathways(args, db, key, possible_paths)
            deferred = []
        save(db, key, deferred=deferred)

    best = (get_entry(db, key) or {}).get("best_pathways") or []
    save(db, key, status="complete")

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
    if not entry:  # same molecules, different SMILES spelling?
        key2, st = run_status(db).get(pair_skeleton([args.smiles1, args.smiles2]),
                                      (None, None))
        if key2 and st == "complete":
            entry = get_entry(db, key2)
            print(f"note: same pair already run as {key2}")
    if entry and entry.get("status") == "complete" and not args.force:
        print_best(entry.get("best_pathways", []))
        sys.exit(f"error: {entry['key']} has already been run (in {args.db}); "
                 f"use --force to rerun")
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
    """Element counts (H included) plus total charge, as written."""
    c = Counter()
    for smi in smiles:
        mol = loose_mol(smi)
        if mol is None:
            raise ValueError(f"RDKit could not parse SMILES: {smi!r}")
        c.update(_composition(mol))
        c["charge"] += sum(a.GetFormalCharge() for a in mol.GetAtoms())
    return +c


def skeleton(smi):
    """Connectivity-only key (all-H graph, single bonds, no charges/radicals), so
    e.g. [O]N=O and O=[N+][O-] both count as NO2."""
    mol = loose_mol(smi)
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


def all_steps(db, max_ea):
    """Every NEB pathway in the database, each with `problem` set to why it
    can't be used ('' if it can)."""
    steps = []
    for e in db.all():
        if e.get("status") != "complete":
            continue
        for r in e.get("neb") or []:
            st = {"source": e["key"], "pathway": r.get("pathway"),
                  "reactants": [canon(x) for x in r.get("reactants") or []],
                  "products": [canon(x) for x in r.get("products") or []],
                  "Ea": None, "dE": r.get("dE"), "kind": r.get("kind", ""),
                  "problem": ""}
            try:
                st["Ea"] = float(r["Ea"])
            except (TypeError, ValueError, KeyError):
                st["problem"] = "no Ea"
            if not st["problem"] and not r.get("sane"):
                st["problem"] = f"not sane ({r.get('kind') or 'old check'})"
            elif not st["problem"] and st["Ea"] > max_ea:
                st["problem"] = f"Ea > {max_ea}"
            if not st["problem"]:
                try:
                    ra, pa = atom_counts(st["reactants"]), atom_counts(st["products"])
                    if ra != pa:
                        st["problem"] = "atoms don't balance"
                except ValueError as err:
                    st["problem"] = str(err)
            steps.append(st)
    return sorted(steps, key=lambda st: (st["Ea"] is None, st["Ea"] or 0))


def reasonable_steps(db, max_ea):
    """Every sane, atom-balanced NEB pathway in the database with Ea <= max_ea."""
    return [st for st in all_steps(db, max_ea) if not st["problem"]]


def print_step_report(db, max_ea, states, name):
    """Why each step in the database was or wasn't usable."""
    reached = {k for sk in states for k, _ in sk}
    rows = []
    for st in all_steps(db, max_ea):
        why = st["problem"]
        if not why:
            need = Counter(skeleton(x) for x in st["reactants"])
            if not any(all(dict(sk).get(k, 0) >= v for k, v in need.items())
                       for sk in states):
                missing = [name.get(k, k) for k in need if k not in reached]
                why = ("needs " + ", ".join(missing) if missing
                       else "reactants never together in the flask")
        rows.append((fmt_rxn(st["reactants"], st["products"]), fnum(st["Ea"]),
                     st["source"], why or "usable"))
    if rows:
        table(rows, ["step", "Ea/eV", "from run", "status"])
    else:
        print("  (no NEB results in the database)")


# A "flask" is a multiset of molecules ({skeleton: count}), starting from the
# overall reactants. A step can only fire if its reactants are all in the flask,
# and it consumes them, so e.g. after 2 NO -> N2O2 in 2 NO + O2 there is no NO
# left for NO + N2O2.
def _fkey(c):
    return tuple(sorted((k, v) for k, v in c.items() if v > 0))


def flask_steps(steps, name):
    """(reactant Counter, product Counter, step) per distinct, non-trivial step
    (lowest Ea kept). Fills `name` {skeleton: SMILES}."""
    best = {}
    for st in steps:
        r = Counter(skeleton(x) for x in st["reactants"])
        p = Counter(skeleton(x) for x in st["products"])
        if r == p:
            continue
        for x in st["reactants"] + st["products"]:
            name.setdefault(skeleton(x), x)
        k = (_fkey(r), _fkey(p))
        if k not in best or st["Ea"] < best[k][2]["Ea"]:
            best[k] = (r, p, st)
    return list(best.values())


def explore(start, steps, max_states=20000):
    """Every flask state reachable from `start`, each with its cheapest route.

    Cost of a route = (largest Ea along it, number of steps), so the first
    route to any state is the one with the lowest rate-limiting barrier.
    Returns ({state: (cost, parent state, step)}, truncated)."""
    import heapq

    s0 = _fkey(start)
    best = {s0: ((0.0, 0), None, None)}
    heap, done, truncated = [(0.0, 0, s0)], set(), False
    while heap:
        ea, n, sk = heapq.heappop(heap)
        if sk in done:
            continue
        done.add(sk)
        flask = Counter(dict(sk))
        for r, p, st in steps:
            if any(flask[x] < v for x, v in r.items()):
                continue
            nk = _fkey(flask - r + p)
            cost = (max(ea, st["Ea"]), n + 1)
            if nk not in best or cost < best[nk][0]:
                if nk not in best and len(best) >= max_states:
                    truncated = True
                    continue
                best[nk] = (cost, sk, st)
                heapq.heappush(heap, (*cost, nk))
    return best, truncated


def route_to(states, sk):
    """Steps leading to state `sk`, first step first."""
    out = []
    while states[sk][1] is not None:
        _, sk, st = states[sk]
        out.append(st)
    return out[::-1]


def flask_depths(start, states, steps):
    """Fewest steps from `start` to each explored flask state (BFS)."""
    from collections import deque

    s0 = _fkey(start)
    depth, queue = {s0: 0}, deque([s0])
    while queue:
        sk = queue.popleft()
        flask = Counter(dict(sk))
        for r, p, _ in steps:
            if any(flask[x] < v for x, v in r.items()):
                continue
            nk = _fkey(flask - r + p)
            if nk in states and nk not in depth:
                depth[nk] = depth[sk] + 1
                queue.append(nk)
    return depth


def flask_pairs(states, depth, goal=None):
    """{(a, b): (generation, state)} for every pair that can meet in some
    reachable flask (a == b needs two of them). generation = 1 + fewest steps
    needed before they can meet, so generation 1 = pairs of the starting flask.
    Flasks that already hold `goal` are skipped (nothing left to make)."""
    pairs = {}
    for sk, d in depth.items():
        flask = dict(sk)
        if goal and all(flask.get(k, 0) >= v for k, v in goal.items()):
            continue
        for a, b in itertools.combinations_with_replacement(sorted(flask), 2):
            if a == b and flask[a] < 2:
                continue
            g = (d + 1, states[sk][0])
            if (a, b) not in pairs or g < pairs[(a, b)][:2]:
                pairs[(a, b)] = (d + 1, states[sk][0], sk)
    return {k: (g, sk) for k, (g, _, sk) in pairs.items()}


def pair_skeleton(smiles):
    return tuple(sorted(skeleton(x) for x in smiles))


def run_status(db):
    """{skeleton pair: (key, status)}: one entry per pair, however it was spelt."""
    out = {}
    for e in db.all():
        smiles = e.get("smiles") or e["key"].split(".")
        sk = pair_skeleton(smiles)
        st = e.get("status", "")
        if sk not in out or st == "complete":
            out[sk] = (e["key"], st)
    return out


def candidate_pairs(db, pairs, name, tried, retry_failed=False):
    """Untried pairs as (generation, smiles1, smiles2, key, state), lowest
    generation first, A+B before A+A. A pair counts as run whatever SMILES it
    was run with (O=N=O + X is the same pair as O=[N+][O-] + X)."""
    status = run_status(db)
    out = []
    for (a, b), (gen, sk) in pairs.items():
        sa, sb = name[a], name[b]
        key = reaction_key(sa, sb)
        _, st = status.get((a, b) if a <= b else (b, a), (None, None))
        if key in tried or (a, b) in tried or st in ("complete", "running"):
            continue
        if st and st.startswith("failed") and not retry_failed:
            continue
        out.append(((gen, a == b, key), gen, sa, sb, key, sk))
    return [c[1:] for c in sorted(out)]


def make_tier(pair, depth, steps, goal, targets, max_states=2000):
    """tier(pathway) for a pair run in a network search, judged against every
    flask in which the pair can meet: 0 = completes the overall reaction (or
    leads to it with steps already known), 1 = makes a target, 2 = other,
    3 = can't happen in any of those flasks."""
    pair = Counter(skeleton(x) for x in pair)
    flasks = [Counter(dict(sk)) for sk in depth
              if all(dict(sk).get(k, 0) >= v for k, v in pair.items())]
    target_sk = {skeleton(t) for t in targets}
    holds = lambda f: all(f.get(k, 0) >= v for k, v in goal.items())  # noqa: E731
    cache = {}

    def tier(v):
        r = Counter(skeleton(x) for x in v["reactants"])
        p = Counter(skeleton(x) for x in v["products"])
        k = (_fkey(r), _fkey(p))
        if k in cache:
            return cache[k]
        best = 3
        for f in flasks:
            if any(f[x] < c for x, c in r.items()):
                continue
            new = f - r + p
            if holds(new) or any(holds(Counter(dict(sk)))
                                 for sk in explore(new, steps, max_states)[0]):
                best = 0
                break
            best = min(best, 1 if target_sk & set(p) else 2)
        cache[k] = best
        return best

    return tier


def deferred_candidates(db, pairs):
    """[(generation, key, [ids])] for entries with deferred pathways whose pair
    can meet in a reachable flask, lowest generation first."""
    out = []
    for e in db.all():
        ids = e.get("deferred") or []
        if not ids or e.get("status") != "complete":
            continue
        a, b = sorted(skeleton(x) for x in (e.get("smiles") or e["key"].split(".")))
        if (a, b) in pairs:
            out.append((pairs[(a, b)][0], e["key"], ids))
    return sorted(out)


def fmt_flask(sk, name):
    return " + ".join(f"{v} {name.get(k, k)}" if v > 1 else name.get(k, k)
                      for k, v in sk) or "(empty)"


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

    # skeleton -> SMILES used to run/show a species: your reactants as written,
    # otherwise the nanoreactor's own spelling (so the analysis recognises it),
    # with your product spelling only as a fallback
    base_name = {}
    for x in reactants:
        base_name.setdefault(skeleton(x), x)
    name = dict(base_name)
    start = Counter({k: v * args.scale for k, v in
                     Counter(skeleton(x) for x in reactants).items()})
    goal = Counter({k: v * args.scale for k, v in
                    Counter(skeleton(x) for x in products).items()})

    with run_log(logfile):
        banner(f"ARCS-Kinetics network search: {overall}")
        print(f"  started   : {now()}")
        print(f"  directory : {os.path.abspath(directory)}")
        print(f"  database  : {os.path.abspath(args.db)}")
        print(f"  targets   : {', '.join(targets)}")
        print(f"  flask     : {fmt_flask(_fkey(start), name)}"
              + (f"  (overall x {args.scale})" if args.scale > 1 else ""))
        print(f"  criteria  : sane NEB pathway with Ea <= {args.max_ea} eV; "
              f"at most {args.max_runs} new pair run(s)")
        ra, pa = atom_counts(reactants), atom_counts(products)
        if ra != pa:
            print(f"  WARNING   : overall reaction is not balanced "
                  f"({dict(ra)} vs {dict(pa)})")
        if not targets:
            sys.exit("error: nothing to make (all products are also reactants)")

        runs, tried, route, said = 0, set(), None, set()
        while True:
            name = dict(base_name)
            steps = flask_steps(reasonable_steps(db, args.max_ea), name)
            for x in products:
                name.setdefault(skeleton(x), x)
            states, truncated = explore(start, steps, args.max_states)
            if truncated:
                log(f"WARNING: stopped exploring at --max-states ({args.max_states})")
            depth = flask_depths(start, states, steps)
            done = [sk for sk in states
                    if all(dict(sk).get(k, 0) >= v for k, v in goal.items())]
            reached = {k for sk in states for k, _ in sk}
            have = [t for t in targets if skeleton(t) in reached]
            for t in have:
                if name[skeleton(t)] != t and t not in said:
                    said.add(t)
                    log(f"target {t} found as {name[skeleton(t)]} (same molecule)")
            log(f"{len(steps)} usable step(s); {len(states)} flask state(s), "
                f"{len(reached)} species reachable; targets reached: "
                f"{len(have)}/{len(targets)}")
            if done:
                sk = min(done, key=lambda s: states[s][0])
                route = (sk, route_to(states, sk))
                rl = states[sk][0][0]
                if not args.best:
                    break
                log(f"mechanism found (rate-limiting Ea = {rl:.3f} eV); "
                    f"--best: continuing to look for a lower one")
            elif len(have) == len(targets):
                log("targets can be made, but not all from the available "
                    "amounts at once; continuing")
            fp = flask_pairs(states, depth, goal)
            cands = candidate_pairs(db, fp, name, tried, args.retry_failed)
            gen = cands[0][0] if cands else None
            if not cands or gen > args.max_generations:
                defer = [d for d in deferred_candidates(db, fp)
                         if d[0] <= args.max_generations]
                if defer and runs < args.max_runs:
                    g, key, ids = defer[0]
                    entry = get_entry(db, key)
                    a, b = entry.get("smiles") or key.split(".")
                    runs += 1
                    section(f"deferred pathways, run {runs}/{args.max_runs}: "
                            f"{a} + {b} ({len(ids)} pathway(s))")
                    pair_dir = entry.get("directory") or os.path.join(directory, slug(key))
                    paths = {i: entry["pathways"][str(i)] for i in ids
                             if str(i) in (entry.get("pathways") or {})}
                    pargs = argparse.Namespace(**vars(args))
                    tier = make_tier([a, b], depth, steps, goal, targets)
                    try:
                        os.makedirs(pair_dir, exist_ok=True)
                        with contextlib.redirect_stdout(console), \
                                run_log(os.path.join(pair_dir, "arcskin.log")):
                            banner(f"deferred NEB for {a} + {b}")
                            left = neb_in_tiers(pargs, db, key, paths, tier)
                        save(db, key, deferred=left)
                        log(f"{len(paths) - len(left)} deferred pathway(s) done, "
                            f"{len(left)} still deferred")
                    except Exception as e:
                        save(db, key, deferred=[])
                        log(f"FAILED: {e}  (see {pair_dir}/arcskin.log)")
                    continue
                if not cands:
                    log("no untried pairs left")
                else:
                    log(f"next pairs are generation {gen} > --max-generations "
                        f"({args.max_generations}); stopping")
                break
            batch = [c for c in cands if c[0] == gen]
            banner(f"generation {gen}: {len(batch)} pair(s)", "-")
            for _, a, b, _, sk in batch:
                print(f"  {a} + {b}" + (f"    (from flask {fmt_flask(sk, name)})"
                                       if gen > 1 else ""))
            stop = False
            for i, (_, a, b, key, sk) in enumerate(batch):
                if runs >= args.max_runs:
                    log(f"reached --max-runs ({args.max_runs}); "
                        f"{len(batch) - i} pair(s) of generation {gen} not run")
                    stop = True
                    break
                runs += 1
                tried.update({key, pair_skeleton([a, b])})
                section(f"gen {gen}, run {runs}/{args.max_runs}: {a} + {b}")
                pargs = argparse.Namespace(**vars(args))
                pargs.smiles1, pargs.smiles2, pargs.dir = a, b, None
                if args.focus:
                    pargs.focus = make_tier([a, b], depth, steps, goal, targets)
                pair_dir = os.path.join(directory, slug(key))
                n_before = len(reasonable_steps(db, args.max_ea))
                try:
                    with contextlib.redirect_stdout(console):  # details -> pair log
                        run_pair(pargs, db, key, pair_dir, suggest=False)
                except Exception as e:
                    log(f"FAILED: {e}  (see {pair_dir}/arcskin.log)")
                    continue
                entry = get_entry(db, key)
                log(f"done -> {pair_dir}/arcskin.log")
                print_best(entry.get("best_pathways") or [])
                new = len(reasonable_steps(db, args.max_ea)) - n_before
                log(f"{new} new usable step(s)")
            if stop:
                break

        banner("result")
        print(f"  finished  : {now()}  ({runs} new pair run(s))")
        if route is None:
            missing = [t for t in targets if skeleton(t) not in reached]
            if missing:
                print(f"  NO mechanism found. Unreached target(s): {', '.join(missing)}")
            else:
                print("  NO mechanism found. Every target can be made, but not all "
                      "of them from the available amounts")
            print(f"  reachable species: {', '.join(sorted(name.get(k, k) for k in reached))}")
            print("  try raising --max-ea / --max-runs / --scale, or --retry-failed")
            section("steps in the database")
            print_step_report(db, args.max_ea, states, name)
            networks.upsert(jsanitize({"overall": overall, "status": "not found",
                                       "reachable": sorted(name.get(k, k) for k in reached),
                                       "updated": now()}),
                            N.overall == overall)
        else:
            sk, seq = route
            grouped = []  # collapse repeated consecutive steps
            for st in seq:
                if grouped and grouped[-1][1] is st:
                    grouped[-1][0] += 1
                else:
                    grouped.append([1, st])
            rls = max(seq, key=lambda st: st["Ea"])
            section("mechanism")
            print_mechanism(grouped)
            flask = start
            print("\n  flask:")
            print(f"    start     : {fmt_flask(_fkey(flask), name)}")
            for i, (m, st) in enumerate(grouped):
                for _ in range(m):
                    flask = (flask - Counter(skeleton(x) for x in st["reactants"])
                             + Counter(skeleton(x) for x in st["products"]))
                print(f"    after {i + 1:<3d} : {fmt_flask(_fkey(flask), name)}")
            print(f"\n  rate-limiting step : {fmt_rxn(rls['reactants'], rls['products'])}"
                  f"  (Ea = {rls['Ea']:.3f} eV)")
            networks.upsert(jsanitize({"overall": overall, "status": "found",
                                       "scale": args.scale,
                                       "mechanism": [dict(st, times=m)
                                                     for m, st in grouped],
                                       "rate_limiting": rls, "updated": now()},
                                      strict=False),
                            N.overall == overall)
        print(f"\nlog written to {logfile}")


def read_smiles_list(args):
    """SMILES from the command line and/or --file (one per line, '#' comments,
    anything after the first whitespace ignored, so 'O=S=O  SO2' works)."""
    items = list(args.smiles)
    if args.file:
        with open(args.file) as fh:
            for line in fh:
                line = line.split("#")[0].strip()
                if line:
                    items.append(line.split()[0])
    out, seen = [], set()
    for smi in items:
        canon(smi, strict=True)  # fail early on a typo
        sk = skeleton(smi)
        if sk in seen:
            print(f"note: {smi} is the same molecule as one already listed; skipped")
            continue
        seen.add(sk)
        out.append(smi)
    return out


def cmd_batch(args):
    db = TinyDB(args.db)
    species = read_smiles_list(args)
    if len(species) < (1 if args.self_pairs else 2):
        sys.exit("error: give at least two SMILES (or one with self pairs)")
    combos = (itertools.combinations_with_replacement(species, 2) if args.self_pairs
              else itertools.combinations(species, 2))
    pairs = sorted(combos, key=lambda ab: ab[0] == ab[1])  # A+B first, then A+A
    directory = args.dir or "."
    os.makedirs(directory, exist_ok=True)
    logfile = os.path.join(directory, "batch.log")
    console = sys.stdout
    results, runs = [], 0

    with run_log(logfile):
        banner(f"ARCS-Kinetics batch: {len(species)} species, {len(pairs)} pair(s)")
        print(f"  started   : {now()}")
        print(f"  directory : {os.path.abspath(directory)}")
        print(f"  database  : {os.path.abspath(args.db)}")
        print(f"  species   : {', '.join(species)}")
        status = run_status(db)
        todo = []
        for a, b in pairs:
            key2, st = status.get(pair_skeleton([a, b]), (None, None))
            if st == "complete" and not args.force:
                results.append((a, b, "in database", get_entry(db, key2)))
            elif st and st.startswith("failed") and not (args.retry_failed or args.force):
                results.append((a, b, "failed before", get_entry(db, key2)))
            else:
                todo.append((a, b))
        print(f"  to run    : {len(todo)} pair(s); "
              f"{len(pairs) - len(todo)} already in the database")
        if args.max_runs is not None and len(todo) > args.max_runs:
            print(f"  --max-runs: only the first {args.max_runs} will be run")

        for i, (a, b) in enumerate(todo):
            if args.max_runs is not None and runs >= args.max_runs:
                results.append((a, b, "not run (--max-runs)", None))
                continue
            key = reaction_key(a, b)
            pair_dir = os.path.join(directory, slug(key))
            section(f"pair {i + 1}/{len(todo)}: {a} + {b}")
            pargs = argparse.Namespace(**vars(args))
            pargs.smiles1, pargs.smiles2, pargs.dir = a, b, None
            runs += 1
            try:
                with contextlib.redirect_stdout(console):  # details -> pair log
                    run_pair(pargs, db, key, pair_dir, suggest=False)
                entry = get_entry(db, key)
                results.append((a, b, "run", entry))
                log(f"done -> {pair_dir}/arcskin.log")
                print_best(entry.get("best_pathways") or [])
            except Exception as e:
                results.append((a, b, "FAILED", get_entry(db, key)))
                log(f"FAILED: {e}  (see {pair_dir}/arcskin.log)")

        banner("summary")
        print(f"  finished  : {now()}  ({runs} pair(s) run)")
        rows = []
        for a, b, what, entry in results:
            best = (entry or {}).get("best_pathways") or []
            if not best:
                rows.append((f"{a} + {b}", what, "-", "-", "-"))
            for j, bp in enumerate(best):
                rows.append((f"{a} + {b}" if j == 0 else "", what if j == 0 else "",
                             " + ".join(bp["products"]), fnum(bp["Ea"]),
                             bp.get("kind") or ""))
        table(rows, ["pair", "status", "best products", "Ea/eV", "kind"])
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
    w.add_argument("--scale", type=int, default=1,
                   help="amount of material: N x the overall reactants "
                   "(default 1, i.e. exactly what the equation says)")
    w.add_argument("--max-states", type=int, default=20000,
                   help="cap on flask states explored (default 20000)")
    w.add_argument("--max-generations", type=int, default=4,
                   help="deepest generation to run (default 4); generation 1 = "
                   "pairs of the starting flask, 2 = pairs possible after one "
                   "step, ...")
    w.add_argument("--no-focus", dest="focus", action="store_false",
                   help="NEB every candidate pathway of each pair, instead of "
                   "the ones that complete the reaction first")
    w.add_argument("--best", action="store_true",
                   help="don't stop at the first mechanism found: run pairs "
                   "until --max-runs / none left, then report the one with the "
                   "lowest rate-limiting barrier")
    w.add_argument("--retry-failed", action="store_true",
                   help="also rerun pairs whose earlier run failed")
    run_options(w)
    w.set_defaults(func=cmd_network)

    bt = sub.add_parser("batch", help="run every pair from a list of SMILES "
                        "(A+B for all A, B, plus A+A)")
    bt.add_argument("smiles", nargs="*", help="SMILES to combine")
    bt.add_argument("--file", "-f", help="text file with one SMILES per line")
    bt.add_argument("--no-self", dest="self_pairs", action="store_false",
                    help="skip A + A pairs")
    bt.add_argument("--dir", help="where pair directories and batch.log go "
                    "(default: current directory)")
    bt.add_argument("--max-runs", type=int, default=None,
                    help="stop after this many new pair runs")
    bt.add_argument("--retry-failed", action="store_true",
                    help="rerun pairs whose earlier run failed")
    bt.add_argument("--force", action="store_true",
                    help="rerun pairs even if already in the database")
    run_options(bt)
    bt.set_defaults(func=cmd_batch)

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
