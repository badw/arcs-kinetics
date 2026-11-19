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
    arcskin decompose "O=N(=O)ON(=O)=O"
    arcskin traj O_O-N_O                 # what's in O_O-N_O/trajectories.json
    arcskin traj O_O-N_O --neb 0         # -> neb_0.traj for `ase gui`

Each pair directory keeps one trajectories.json with the xtb MD frames and every
NEB band (images with energies and forces, reaction coordinate, fitted energy
profile, Ea / Ea_rev / dE); the xtb seed_*/ directories are deleted once the
frames are saved (--keep-xtb keeps them). See arcskin.trajectories for reading
it in Python.

Each A + B pair runs in its own SMILES1-SMILES2/ directory with a log in
SMILES1-SMILES2/arcskin.log, whether started by `run` or `network`. `network`
writes its summary to network_<reaction>.log (e.g. network_2N_O+O_O_to_2O_N_O.log),
so several reactions can share a directory; pairs already in the database are
reused. Single-NEB steps go in proposed/<reaction>.log (A + B -> missing
products) and decomposed/<species>.log (X -> A + B).
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

import numpy as np
from collections import Counter

from monty.json import jsanitize
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from tinydb import Query, TinyDB

from arcskin.spin import DEFAULT_MODEL

RDLogger.DisableLog("rdApp.warning")
RDLogger.DisableLog("rdApp.error")  # hypervalent spellings (O=N=O) are handled; no noise


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
    """Record compounds and the reaction they appeared in (role: reactant/product).
    One record per molecule (matched by connectivity, so O=N=O and [O]N=O are
    the same), keeping the spelling it was first seen with."""
    table = db.table("compounds")
    by_skel = {skeleton(c["smiles"]): c for c in table.all()}
    for s in sorted(set(smiles)):
        sk = skeleton(s)
        doc = dict(by_skel.get(sk) or {"smiles": s, "reactant_in": [], "product_in": []})
        if key not in doc[f"{role}_in"]:
            doc[f"{role}_in"].append(key)
        table.upsert(doc, Query().smiles == doc["smiles"])
        by_skel[sk] = doc


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

    new = neb_records(args, paths)
    if not new:
        return []
    directory = (get_entry(db, key) or {}).get("directory")
    if directory and os.path.isdir(directory):
        save_neb_bands(os.path.join(directory, "trajectories.json"),
                       {r["pathway"]: r for r in new})
    done = {str(r["pathway"]) for r in new}
    old = [r for r in (get_entry(db, key) or {}).get("neb") or []
           if str(r.get("pathway")) not in done]
    merged = old + jsanitize(new, strict=False)
    best = best_pathways(pd.DataFrame(merged), args.ea_tol)
    save(db, key, neb=merged, best_pathways=best)
    return new


def save_neb_bands(path, records):
    """Keep NEB images/energies/reaction coordinate in trajectories.json."""
    from arcskin.trajectories import save_neb

    try:
        save_neb(path, records)
        log(f"NEB band(s) saved to {path}")
    except Exception as e:  # never lose a run over this
        log(f"could not save NEB bands to {path}: {e}")


def remove_xtb_files(directories, trajectories):
    """Delete the xtb seed_*/ directories once their frames are safely in
    trajectories.json."""
    import shutil

    try:
        from monty.serialization import loadfn

        saved = loadfn(trajectories)
    except Exception as e:
        log(f"keeping xtb files: could not read {trajectories} ({e})")
        return
    for d in directories:
        seed = os.path.basename(os.path.normpath(d))
        if not seed.startswith("seed_") or not saved.get(seed):
            log(f"keeping {d}: its frames are not in {trajectories}")
            continue
        shutil.rmtree(d, ignore_errors=True)
    log(f"xtb files removed (MD frames kept in {trajectories})")


def neb_records(args, paths):
    """Endpoints + NEB for `paths` ({id: pathway}); returns the NEB records."""
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
    ranking = run_all(endpoints, n_images=args.n_images, climb=True, fmax=args.fmax,
                      model=args.model, separate_products=args.separate_products,
                      spin_envelope=getattr(args, "spin_envelope", True))
    ranking = ranking.reset_index(names="pathway")
    new = ranking.to_dict("records")
    table([(r["pathway"], fmt_rxn(r["reactants"], r["products"]),
            fnum(r["Ea"]), fnum(r["dE"]), r.get("mult"), bool(r["sane"]),
            r.get("kind"), r.get("reason") or "")
           for _, r in ranking.sort_values("Ea").iterrows()],
          ["#", "pathway", "Ea/eV", "dE/eV", "mult", "sane", "kind", "note"])
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
         settings=settings, model=args.model, log=os.path.join(directory, "arcskin.log"))
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
    anr = AnalyseNanoReactor(skip_step=args.skip_step, reactants=smiles,
                             model=args.model)
    anr.read_trajectory(xnr.trajectory_locations, format="xyz")
    anr.analyse_trajectories(n_cpus=args.n_cpus)
    possible_paths = anr.get_possible_paths_dict()
    save(db, key,
         species_counts=dict(anr.unique_smiles_counter),
         species=anr.unique_smiles,
         pathways=possible_paths)
    if not getattr(args, "keep_xtb", False):
        remove_xtb_files(xnr.directories, os.path.join(directory, "trajectories.json"))
    counts = dict(anr.unique_smiles_counter)
    log(f"{len(counts)} unique species observed")
    table(sorted(counts.items(), key=lambda kv: -kv[1]), ["species", "count"])
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
    for b in best:  # only what NEB says is the outcome, not every candidate
        if b.get("sane"):
            add_compounds(db, key, b["products"], "product")
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
    if entry and entry_model(entry) != args.model:
        print(f"note: {key} was computed with {entry_model(entry)}; "
              f"rerunning with {args.model}")
        entry = None
    if not entry:  # same molecules, different SMILES spelling?
        key2, st = run_status(db, args.model).get(pair_skeleton([args.smiles1, args.smiles2]),
                                      (None, None))
        if key2 and st == "complete":
            entry = get_entry(db, key2)
            print(f"note: same pair already run as {key2}")
    if entry and entry.get("status") == "complete" and not args.force:
        print_best(entry.get("best_pathways", []))
        sys.exit(f"error: {entry['key']} has already been run (in {args.db}); "
                 f"use --force to rerun")
    run_pair(args, db, key, args.dir or pair_dirname(key))


def pair_dirname(key):
    """Directory for a pair: each SMILES made filesystem-safe, joined by '-'
    (e.g. O=O.[N]=O -> O_O-N_O)."""
    return "-".join(slug(x) or "X" for x in key.split("."))


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


def suggest_balance(reactants, products, max_coef=6):
    """Smallest whole-number coefficients that balance the species of an
    unbalanced reaction, as text (e.g. '2 O=[N+][O-] + O -> ...'), or None."""
    left, right = sorted(set(reactants)), sorted(set(products))
    species = left + right
    try:
        counts = [atom_counts([x]) for x in species]
    except ValueError:
        return None
    elements = sorted({e for c in counts for e in c})
    for total in range(len(species), max_coef * len(species) + 1):
        for coefs in itertools.product(range(1, max_coef + 1), repeat=len(species)):
            if sum(coefs) != total:
                continue
            if all(sum(n * c[e] for n, c in zip(coefs[:len(left)], counts))
                   == sum(n * c[e] for n, c in zip(coefs[len(left):],
                                                    counts[len(left):]))
                   for e in elements):
                side = lambda xs, ns: " + ".join(  # noqa: E731
                    f"{n} {x}" if n > 1 else x for x, n in zip(xs, ns))
                return f"{side(left, coefs[:len(left)])} -> " \
                       f"{side(right, coefs[len(left):])}"
    return None


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


def weak_complex(r, weak_max):
    """Binding energy (eV, from the isolated species) if NEB record `r` is a
    weakly bound association A + B -> C, else None.

    C must be a single molecule that the NEB found unbound (it fell apart into
    A + B when relaxed) and must lie no more than weak_max eV above A + B.
    Such complexes (the NO dimer, ONOO, ...) aren't minima at this level but
    are the pre-equilibria that termolecular reactions like 2 NO + O2 run
    through, so a search may treat them as fast, reversible steps."""
    if weak_max is None or r.get("kind") != "unbound":
        return None
    if len(r.get("reactants") or []) != 2 or len(r.get("products") or []) != 1:
        return None
    try:
        diff = float(r.get("diff"))
    except (TypeError, ValueError):
        return None
    return diff if np.isfinite(diff) and diff <= weak_max else None


def all_steps(db, max_ea, keep="best", ea_tol=0.1, model=None, weak_max=None,
              reverse=False):
    """Every NEB pathway in the database, each with `problem` set to why it
    can't be used ('' if it can).

    weak_max: if set, unbound associations A + B -> C lying <= weak_max eV
      above A + B become usable "weak complex" steps (Ea 0, dE = binding
      energy); see weak_complex(). They are kept in addition to each run's
      best step, not in competition with it.

    keep: which usable pathways of each pair run go into the search
      "best" - only the lowest-Ea one (the dominant channel)
      "tol"  - the lowest-Ea one plus any within ea_tol eV of it
      "all"  - every sane pathway with Ea <= max_ea

    reverse: also add the reverse of every sane pathway (products ->
      reactants) with Ea_rev = Ea - dE (the band's highest point measured from
      its product end) and -dE, so a run of the overall *products* (e.g.
      HONO + HNO3) can supply steps for the forward reaction. Weak complexes
      get their dissociation (Ea = max(0, -binding energy)). The --steps choice
      is applied to each direction separately."""
    steps = []
    proposals = [{"key": f"{d.get('origin', 'proposed')}:{d['key']}",
                  "status": "complete",
                  "model": d.get("model"), "neb": [d["neb"]] if d.get("neb") else []}
                 for d in db.table("proposals").all()]
    for e in db.all() + proposals:
        if e.get("status") != "complete":
            continue
        other = model and entry_model(e) != model
        for r in e.get("neb") or []:
            st = {"source": e["key"], "pathway": r.get("pathway"),
                  "reactants": list(r.get("reactants") or []),
                  "products": list(r.get("products") or []),
                  "Ea": None, "dE": r.get("dE"), "kind": r.get("kind", ""),
                  "problem": ""}
            try:
                st["Ea"] = float(r["Ea"])
            except (TypeError, ValueError, KeyError):
                st["problem"] = "no Ea"
            bind = None if other else weak_complex(r, weak_max)
            if other:
                st["problem"] = f"computed with {entry_model(e)}, not {model}"
            elif bind is not None:
                st.update(Ea=0.0, dE=bind, kind="weak complex", problem="")
            elif not st["problem"] and not r.get("sane"):
                st["problem"] = f"not sane ({r.get('kind') or 'old check'})"
            if not st["problem"]:
                try:
                    ra, pa = atom_counts(st["reactants"]), atom_counts(st["products"])
                    if ra != pa:
                        st["problem"] = "atoms don't balance"
                except ValueError as err:
                    st["problem"] = str(err)
            new = [st]
            if reverse and not st["problem"]:
                rv = dict(st, reactants=st["products"], products=st["reactants"],
                          direction="reverse")
                try:
                    dE = float(st["dE"])
                    if bind is not None:
                        rv.update(Ea=max(0.0, -bind), dE=-bind,
                                  kind="weak dissociation")
                    else:
                        rv.update(Ea=max(0.0, st["Ea"] - dE), dE=-dE,
                                  kind=f"reverse {st['kind']}".strip())
                    new.append(rv)
                except (TypeError, ValueError):
                    pass
            for x in new:
                x.setdefault("direction", "forward")
                if (not x["problem"] and not x["kind"].startswith("weak")
                        and x["Ea"] > max_ea):
                    x["problem"] = f"Ea > {max_ea}"
                steps.append(x)
    if keep != "all":
        groups = {}
        for st in steps:
            if not st["problem"] and not st["kind"].startswith("weak"):
                groups.setdefault((st["source"], st["direction"],
                                   pair_skeleton(st["reactants"])), []).append(st)
        for grp in groups.values():
            low = min(st["Ea"] for st in grp)
            first = True
            for st in sorted(grp, key=lambda st: st["Ea"]):
                if keep == "best" and not first:
                    st["problem"] = f"not the best of its run (best Ea {low:.3f})"
                elif keep == "tol" and st["Ea"] > low + ea_tol:
                    st["problem"] = (f"more than {ea_tol} eV above the best of "
                                     f"its run ({low:.3f})")
                first = False
    return sorted(steps, key=lambda st: (st["Ea"] is None, st["Ea"] or 0))


def reasonable_steps(db, max_ea, keep="best", ea_tol=0.1, model=None, weak_max=None,
                     reverse=False):
    """The NEB pathways a network search may use (see all_steps)."""
    return [st for st in all_steps(db, max_ea, keep, ea_tol, model, weak_max, reverse)
            if not st["problem"]]


def print_step_report(db, max_ea, states, name, keep="best", ea_tol=0.1, model=None,
                      weak_max=None, reverse=False):
    """Why each step in the database was or wasn't usable."""
    reached = {k for sk in states for k, _ in sk}
    rows = []
    for st in all_steps(db, max_ea, keep, ea_tol, model, weak_max, reverse):
        why = st["problem"]
        if not why:
            need = Counter(skeleton(x) for x in st["reactants"])
            if not any(all(dict(sk).get(k, 0) >= v for k, v in need.items())
                       for sk in states):
                missing = [name.get(k, k) for k in need if k not in reached]
                why = ("needs " + ", ".join(missing) if missing
                       else "reactants never together in the flask")
        if not why and st["kind"] == "weak complex":
            why = f"usable (weak complex, {st['dE']:+.3f} eV)"
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
    (lowest Ea kept). Fills `name` {skeleton: SMILES} for species not already
    named, using the most closed-shell spelling seen (see spelling_score)."""
    best, spellings = {}, {}
    for st in steps:
        r = Counter(skeleton(x) for x in st["reactants"])
        p = Counter(skeleton(x) for x in st["products"])
        if r == p:
            continue
        for x in st["reactants"] + st["products"]:
            spellings.setdefault(skeleton(x), set()).add(x)
        k = (_fkey(r), _fkey(p))
        if k not in best or st["Ea"] < best[k][2]["Ea"]:
            best[k] = (r, p, st)
    for k, options in spellings.items():
        if k not in name:
            name[k] = min(options, key=lambda x: (spelling_score(x), x))
    return list(best.values())


def spelling_score(smi):
    """Lower = better SMILES to start a nanoreactor run from: valid for RDKit,
    then fewest radical electrons, then fewest formal charges. OpenBabel often
    writes e.g. H2SO4 as the 'diradical' [O]S(O)(O)[O]; O=S(=O)(O)O wins."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        mol, bad = loose_mol(smi), 1
    else:
        bad = 0
    if mol is None:
        return (2, 0, 0)
    return (bad, sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms()),
            sum(abs(a.GetFormalCharge()) for a in mol.GetAtoms()))


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


def entry_model(e):
    """Model an entry was computed with (entries from before models were
    recorded used aimnet2)."""
    return e.get("model") or (e.get("settings") or {}).get("model") or "aimnet2"


def pair_skeleton(smiles):
    return tuple(sorted(skeleton(x) for x in smiles))


def run_status(db, model=None):
    """{skeleton pair: (key, status)}: one entry per pair, however it was spelt.
    With `model`, entries computed with another model are ignored (they count
    as not run)."""
    out = {}
    for e in db.all():
        if model and entry_model(e) != model:
            continue
        smiles = e.get("smiles") or e["key"].split(".")
        sk = pair_skeleton(smiles)
        st = e.get("status", "")
        if sk not in out or st == "complete":
            out[sk] = (e["key"], st)
    return out


def candidate_pairs(db, pairs, name, tried, retry_failed=False, model=None):
    """Untried pairs as (generation, smiles1, smiles2, key, state), lowest
    generation first, A+B before A+A. A pair counts as run whatever SMILES it
    was run with (O=N=O + X is the same pair as O=[N+][O-] + X)."""
    status = run_status(db, model)
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


def proposal_sig(reactants, products):
    return f"{'.'.join(sorted(map(skeleton, reactants)))}>>" \
           f"{'.'.join(sorted(map(skeleton, products)))}"


def proposals(db, depth, goal, name, model):
    """Steps suggested by the overall equation: A + B -> (exactly what is still
    missing), for every pair A + B that can meet in a reachable flask where
    turning A + B into the missing products completes the reaction. Each is
    one NEB, much cheaper than a nanoreactor run, and catches steps the short
    MD never sampled (e.g. N2O2 + O2 -> 2 NO2). Already-tried ones are skipped.
    Returns [(depth, reactants, products, sig)] shallowest first."""
    done = {(d["sig"], d.get("model")) for d in db.table("proposals").all()}
    out, seen = [], set()
    for sk, d in sorted(depth.items(), key=lambda kv: kv[1]):
        flask = Counter(dict(sk))
        if all(flask.get(k, 0) >= v for k, v in goal.items()):
            continue
        for a, b in itertools.combinations_with_replacement(sorted(flask), 2):
            if a == b and flask[a] < 2:
                continue
            rest = flask - Counter([a, b])
            if any(v > goal.get(k, 0) for k, v in rest.items()):
                continue  # leftovers that aren't products: wouldn't finish
            need = goal - rest
            reac = [name[a], name[b]]
            prod = [name[k] for k, v in sorted(need.items()) for _ in range(v)]
            if not prod or Counter([a, b]) == need:
                continue
            try:
                if atom_counts(reac) != atom_counts(prod):
                    continue
            except ValueError:
                continue
            sig = proposal_sig(reac, prod)
            if sig in seen or (sig, model) in done:
                continue
            seen.add(sig)
            out.append((d, reac, prod, sig))
    return out


def _fragment_smiles(frag):
    """SMILES of a fragment cut out of a molecule. A charged fragment (from a
    zwitterionic spelling like O=[N+][O-]) is rewritten neutral, with radicals
    on atoms short of their usual valence."""
    frag = Chem.RWMol(frag)
    if sum(a.GetFormalCharge() for a in frag.GetAtoms()) != 0:
        pt = Chem.GetPeriodicTable()
        for a in frag.GetAtoms():
            a.SetFormalCharge(0)
            order = sum(b.GetBondTypeAsDouble() for b in a.GetBonds())
            val = pt.GetDefaultValence(a.GetAtomicNum())
            a.SetNumRadicalElectrons(max(0, int(val - order - a.GetTotalNumHs())))
    frag.UpdatePropertyCache(strict=False)
    raw = Chem.MolToSmiles(frag, allHsExplicit=True)  # every atom bracketed
    nice = canon(raw)
    try:
        return nice if atom_counts([nice]) == atom_counts([raw]) else raw
    except ValueError:
        return raw


def fragmentations(smi):
    """Every way to split `smi` into two pieces by breaking one bond between
    heavy atoms (not in a ring), as [[frag1, frag2], ...], one per distinct pair
    of fragments. E.g. O2N-O-NO2 -> [NO2 + NO3], [O2N-O-NO + O]."""
    mol = loose_mol(smi)
    if mol is None:
        return []
    mol = Chem.RWMol(mol)
    for a in mol.GetAtoms():  # pin H counts so nothing is added on re-reading
        a.SetNumExplicitHs(a.GetTotalNumHs())
        a.SetNoImplicit(True)
    out, seen = [], set()
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if b.IsInRing() or 1 in (mol.GetAtomWithIdx(i).GetAtomicNum(),
                                 mol.GetAtomWithIdx(j).GetAtomicNum()):
            continue
        rw = Chem.RWMol(mol)
        n = int(round(b.GetBondTypeAsDouble()))
        rw.RemoveBond(i, j)
        for k in (i, j):  # each end keeps the electrons of the broken bond
            at = rw.GetAtomWithIdx(k)
            at.SetNumRadicalElectrons(at.GetNumRadicalElectrons() + n)
        rw.UpdatePropertyCache(strict=False)
        frags = Chem.GetMolFrags(rw, asMols=True, sanitizeFrags=False)
        if len(frags) != 2:
            continue
        prods = sorted(_fragment_smiles(f) for f in frags)
        k = tuple(sorted(skeleton(p) for p in prods))
        if k in seen:
            continue
        seen.add(k)
        try:
            if atom_counts([smi]) != atom_counts(prods):
                continue
        except ValueError:
            continue
        out.append(prods)
    return out


def decompositions(db, states, start, goal, name, model, steps, which="intermediates"):
    """Unimolecular steps X -> A + B to try: every one-bond split of each
    species X that a network search can make (see fragmentations), skipping
    ones already in the database. Catches decompositions such as
    N2O5 -> NO2 + NO3 that a bimolecular nanoreactor run never tests.

    which: "intermediates" - species reached that are neither starting
             materials nor overall products (default)
           "all" - every species in a reachable flask
    Species only formed as weak complexes are skipped (their falling apart is
    already the reverse of the pre-equilibrium). Fragments that are known
    species get their usual spelling. Returns [(X, [A, B], sig)]."""
    done = {(d["sig"], d.get("model")) for d in db.table("proposals").all()}
    weak = {k for r, p, st in steps if st["kind"] == "weak complex" for k in p}
    made = {k for r, p, st in steps if st["kind"] != "weak complex" for k in p}
    species = {k for sk in states for k, _ in sk}
    if which != "all":
        species = {k for k in species if k not in start and k not in goal}
    out = []
    for k in sorted(species, key=lambda k: name.get(k, k)):
        if k in weak and k not in made:
            continue
        x = name.get(k, k)
        if loose_mol(x) is None or loose_mol(x).GetNumAtoms() < 2:
            continue
        for prods in fragmentations(x):
            prods = sorted(name.get(skeleton(p), p) for p in prods)
            sig = proposal_sig([x], prods)
            if (sig, model) not in done:
                out.append((x, prods, sig))
    return out


def reaction_slug(reactants, products):
    """Filename-safe name for an overall reaction, e.g.
    2 [N]=O + O=O -> 2 O=[N+][O-]  ->  2N_O+O_O_to_2O_N_O"""
    def side(smiles):
        return "+".join(f"{n if n > 1 else ''}{slug(x) or 'X'}"
                        for x, n in Counter(smiles).items())
    return f"{side(reactants)}_to_{side(products)}"


def run_proposal(args, db, reac, prod, sig, directory, tag="proposed",
                 origin="proposed"):
    """NEB one proposed step (origin "proposed", log in proposed/<tag>.log) or
    decomposition (origin "decomposed", log in decomposed/<tag>.log) and store
    it in the 'proposals' table."""
    from arcskin.nanoreactor_analysis import AnalyseNanoReactor

    anr = AnalyseNanoReactor(model=args.model)
    path = {"reactants": reac, "products": prod, "origin": "proposed"}
    path["reactants_energies"] = [anr.species_energy(x) for x in reac]
    path["products_energies"] = [anr.species_energy(x) for x in prod]
    path["diff"] = float(np.sum(path["products_energies"])
                         - np.sum(path["reactants_energies"]))
    path["origin"] = origin
    pdir = os.path.join(directory, origin)
    os.makedirs(pdir, exist_ok=True)
    rec = None
    with run_log(os.path.join(pdir, f"{tag}.log")):
        banner(f"{origin} step: {fmt_rxn(reac, prod)}")
        recs = neb_records(args, {0: path})
        rec = jsanitize(recs[0], strict=False) if recs else None
        if recs:
            save_neb_bands(os.path.join(pdir, "trajectories.json"),
                           {fmt_rxn(reac, prod): recs[0]})
    key = (f"{'.'.join(sorted(canon(x) for x in reac))}>>"
           f"{'.'.join(sorted(canon(x) for x in prod))}")
    db.table("proposals").upsert(
        {"sig": sig, "key": key, "model": args.model, "reactants": reac,
         "products": prod, "origin": origin, "neb": rec, "updated": now()},
        (Query().sig == sig) & (Query().model == args.model))
    return rec


def run_decomposition(args, db, x, prod, sig, directory, console=None):
    """NEB X -> A + B (log in decomposed/<X>.log); a failure is recorded so it
    isn't retried. Returns the NEB record or None."""
    try:
        with contextlib.redirect_stdout(console or sys.stdout):
            return run_proposal(args, db, [x], prod, sig, directory,
                                slug(x) or "X", origin="decomposed")
    except Exception as e:
        log(f"FAILED: {e}")
        db.table("proposals").upsert(
            {"sig": sig, "key": sig, "model": args.model, "reactants": [x],
             "products": prod, "origin": "decomposed", "neb": None,
             "error": str(e), "updated": now()},
            (Query().sig == sig) & (Query().model == args.model))
        return None


def cmd_decompose(args):
    """NEB every one-bond split of each given species."""
    db = TinyDB(args.db)
    directory = args.dir or "."
    os.makedirs(directory, exist_ok=True)
    done = {(d["sig"], d.get("model")) for d in db.table("proposals").all()}
    known = {skeleton(c["smiles"]): c["smiles"] for c in db.table("compounds").all()}
    rows = []
    for x in args.smiles:
        x = canon(x, strict=True)
        splits = fragmentations(x)
        if not splits:
            print(f"{x}: no single-bond splits into two pieces")
        for prod in splits:
            prod = sorted(known.get(skeleton(p), p) for p in prod)
            sig = proposal_sig([x], prod)
            if (sig, args.model) in done and not args.force:
                old = db.table("proposals").get((Query().sig == sig)
                                                & (Query().model == args.model))
                rec = (old or {}).get("neb")
                note = "already in database"
            else:
                print(f"NEB {fmt_rxn([x], prod)}")
                rec = run_decomposition(args, db, x, prod, sig, directory)
                note = ""
            rec = rec or {}
            rows.append((fmt_rxn([x], prod), fnum(rec.get("Ea")), fnum(rec.get("dE")),
                         bool(rec.get("sane")), rec.get("kind") or "failed",
                         note or rec.get("reason") or ""))
    section("decompositions")
    table(rows, ["reaction", "Ea/eV", "dE/eV", "sane", "kind", "note"])
    print(f"\nlogs in {os.path.join(os.path.abspath(directory), 'decomposed')}/")


def deferred_candidates(db, pairs, model=None):
    """[(generation, key, [ids])] for entries with deferred pathways whose pair
    can meet in a reachable flask, lowest generation first."""
    out = []
    for e in db.all():
        ids = e.get("deferred") or []
        if not ids or e.get("status") != "complete":
            continue
        if model and entry_model(e) != model:
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
    logfile = os.path.join(directory,
                           f"network_{reaction_slug(reactants, products)}.log")
    console = sys.stdout
    networks, N = db.table("networks"), Query()
    use_model = None if args.any_model else args.model
    weak_max = None if args.weak_max < 0 else args.weak_max

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
        print(f"  model     : {args.model}"
              + ("  (also using results from other models)" if args.any_model else ""))
        print("  complexes : " + (f"weakly bound associations within {weak_max} eV "
                                  "of their parts used as fast pre-equilibria"
                                  if weak_max is not None else "not used"))
        print(f"  criteria  : sane NEB pathway with Ea <= {args.max_ea} eV; "
              f"steps kept per pair run: {args.steps}"
              + (f" (within {args.ea_tol} eV)" if args.steps == "tol" else "") + "; "
              f"at most {args.max_runs} new pair run(s)")
        print("  direction : " + ("forward, plus pair runs from the product side "
                                  "(steps used in both directions)" if args.reverse
                                  else "forward only"))
        ra, pa = atom_counts(reactants), atom_counts(products)
        if ra != pa:
            fix = suggest_balance(reactants, products)
            sys.exit(f"error: overall reaction is not balanced ({dict(ra)} vs "
                     f"{dict(pa)}), so no mechanism can complete it"
                     + (f"; did you mean: {fix}" if fix else ""))
        if not targets:
            sys.exit("error: nothing to make (all products are also reactants)")

        rtargets = sorted(set(reactants) - set(products))
        runs, tried, route, said = 0, set(), None, set()
        while True:
            name = dict(base_name)
            steps = flask_steps(reasonable_steps(db, args.max_ea, args.steps, args.ea_tol,
                                                 use_model, weak_max, args.reverse), name)
            for x in products:
                name.setdefault(skeleton(x), x)
            states, truncated = explore(start, steps, args.max_states)
            if truncated:
                log(f"WARNING: stopped exploring at --max-states ({args.max_states})")
            depth = flask_depths(start, states, steps)
            # product side: flasks reachable from the overall products with the
            # same (two-way) steps, i.e. ones that lead *to* the products
            if args.reverse:
                states_b, _ = explore(goal, steps, args.max_states)
                depth_b = flask_depths(goal, states_b, steps)
            else:
                states_b, depth_b = {}, {}
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
            if args.propose:
                props = proposals(db, depth, goal, name, args.model)
                if args.reverse:  # product-side pair -> exactly the missing reactants
                    sigs = {x[3] for x in props}
                    props += [x for x in proposals(db, depth_b, start, name, args.model)
                              if x[3] not in sigs]
                if props:
                    banner(f"proposed steps from the overall equation: {len(props)}", "-")
                    for _, reac, prod, sig in props:
                        log(f"NEB {fmt_rxn(reac, prod)}")
                        try:
                            with contextlib.redirect_stdout(console):
                                rec = run_proposal(args, db, reac, prod, sig, directory,
                                                   reaction_slug(reactants, products))
                        except Exception as e:
                            log(f"FAILED: {e}")
                            db.table("proposals").upsert(
                                {"sig": sig, "key": sig, "model": args.model,
                                 "reactants": reac, "products": prod, "neb": None,
                                 "error": str(e), "updated": now()},
                                (Query().sig == sig) & (Query().model == args.model))
                            continue
                        if rec:
                            log(f"  Ea={fnum(rec.get('Ea'))} dE={fnum(rec.get('dE'))} "
                                f"{rec.get('kind')}" + ("" if rec.get("sane")
                                                        else f" ({rec.get('reason')})"))
                    continue  # re-explore with whatever the proposals gave
            if args.decompose != "none":
                decs = decompositions(db, {**states, **states_b}, start, goal, name,
                                      args.model, steps, args.decompose)
                if decs:
                    banner(f"decompositions of reachable species: {len(decs)}", "-")
                    for x, prod, sig in decs:
                        log(f"NEB {fmt_rxn([x], prod)}")
                        rec = run_decomposition(args, db, x, prod, sig, directory,
                                                console)
                        if rec:
                            log(f"  Ea={fnum(rec.get('Ea'))} dE={fnum(rec.get('dE'))} "
                                f"{rec.get('kind')}" + ("" if rec.get("sane")
                                                        else f" ({rec.get('reason')})"))
                    continue  # re-explore with the new unimolecular steps
            fp = flask_pairs(states, depth, goal)
            side = dict.fromkeys(fp, "reactant")
            if args.reverse:
                for k, v in flask_pairs(states_b, depth_b, start).items():
                    if k not in fp or v[0] < fp[k][0]:
                        fp[k], side[k] = v, "product"

            def tier_for(a, b):
                k = tuple(sorted((skeleton(a), skeleton(b))))
                if side.get(k) == "product":
                    return make_tier([a, b], depth_b, steps, start, rtargets)
                return make_tier([a, b], depth, steps, goal, targets)

            cands = candidate_pairs(db, fp, name, tried, args.retry_failed,
                                    None if args.any_model else args.model)
            gen = cands[0][0] if cands else None
            if not cands or gen > args.max_generations:
                defer = [d for d in deferred_candidates(db, fp, use_model)
                         if d[0] <= args.max_generations]
                if defer and runs < args.max_runs:
                    g, key, ids = defer[0]
                    entry = get_entry(db, key)
                    a, b = entry.get("smiles") or key.split(".")
                    runs += 1
                    section(f"deferred pathways, run {runs}/{args.max_runs}: "
                            f"{a} + {b} ({len(ids)} pathway(s))")
                    pair_dir = entry.get("directory") or os.path.join(directory, pair_dirname(key))
                    paths = {i: entry["pathways"][str(i)] for i in ids
                             if str(i) in (entry.get("pathways") or {})}
                    pargs = argparse.Namespace(**vars(args))
                    tier = tier_for(a, b)
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
                k = tuple(sorted((skeleton(a), skeleton(b))))
                where = "product side" if side.get(k) == "product" else ""
                if gen > 1:
                    where = (where + ", " if where else "") + \
                        f"from flask {fmt_flask(sk, name)}"
                print(f"  {a} + {b}" + (f"    ({where})" if where else ""))
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
                    pargs.focus = tier_for(a, b)
                pair_dir = os.path.join(directory, pair_dirname(key))
                n_before = len(reasonable_steps(db, args.max_ea, args.steps, args.ea_tol, use_model, weak_max,
                                            args.reverse))
                try:
                    with contextlib.redirect_stdout(console):  # details -> pair log
                        run_pair(pargs, db, key, pair_dir, suggest=False)
                except Exception as e:
                    log(f"FAILED: {e}  (see {pair_dir}/arcskin.log)")
                    continue
                entry = get_entry(db, key)
                log(f"done -> {pair_dir}/arcskin.log")
                print_best(entry.get("best_pathways") or [])
                new = len(reasonable_steps(db, args.max_ea, args.steps, args.ea_tol, use_model, weak_max,
                                            args.reverse)) - n_before
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
            print_step_report(db, args.max_ea, states, name, args.steps, args.ea_tol,
                              use_model, weak_max, args.reverse)
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
            # energy profile: where each step starts (sum of earlier dE) and the
            # highest point it reaches, relative to the starting flask
            e_now, peak, peak_step = 0.0, 0.0, None
            for i, (m, st) in enumerate(grouped):
                for _ in range(m):
                    top = e_now + st["Ea"]
                    if top > peak or peak_step is None:
                        peak, peak_step = max(peak, top), i + 1
                    try:
                        e_now += float(st["dE"])
                    except (TypeError, ValueError):
                        pass
            if any(st["kind"] == "weak complex" for _, st in grouped):
                print("\n  energy profile (relative to the start, sum of dE):")
                e_now = 0.0
                for i, (m, st) in enumerate(grouped):
                    for _ in range(m):
                        top = e_now + st["Ea"]
                        try:
                            e_now += float(st["dE"])
                        except (TypeError, ValueError):
                            pass
                    print(f"    step {i + 1:<3d}: highest point {top:+.3f} eV, "
                          f"then {e_now:+.3f} eV")
                print(f"  apparent barrier   : {peak:+.3f} eV (step {peak_step}; "
                      "includes uphill pre-equilibria)")
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


def batch_products(entries, known, max_ea=None, stable_only=False):
    """New species from a generation's pair results: products of each pair's
    best (sane) NEB pathways that aren't already in `known` (skeletons),
    spelt the most closed-shell way seen. Returns ({skeleton: SMILES}, rows)
    where rows = (product, from pair, Ea) for the log."""
    spellings, origin = {}, {}
    for entry in entries:
        for bp in (entry or {}).get("best_pathways") or []:
            if not bp.get("sane"):
                continue
            try:
                ea = float(bp["Ea"])
            except (TypeError, ValueError, KeyError):
                continue
            if max_ea is not None and ea > max_ea:
                continue
            for x in bp.get("products") or []:
                sk = skeleton(x)
                if sk in known or (stable_only and not is_stable(x)):
                    continue
                spellings.setdefault(sk, set()).add(x)
                if sk not in origin or ea < origin[sk][1]:
                    origin[sk] = (" + ".join(entry.get("smiles") or []), ea)
    new = {sk: min(v, key=lambda x: (spelling_score(x), x))
           for sk, v in spellings.items()}
    rows = [(new[sk], origin[sk][0], fnum(origin[sk][1])) for sk in sorted(new, key=new.get)]
    return new, rows


def cmd_batch(args):
    db = TinyDB(args.db)
    species = read_smiles_list(args)
    if len(species) < (1 if args.self_pairs else 2):
        sys.exit("error: give at least two SMILES (or one with self pairs)")
    directory = args.dir or "."
    os.makedirs(directory, exist_ok=True)
    logfile = os.path.join(directory, "batch.log")
    console = sys.stdout
    results, runs = [], 0
    known = {skeleton(x): x for x in species}  # every species so far
    new = dict(known)                          # this generation's new species
    stop_reason = None

    with run_log(logfile):
        banner(f"ARCS-Kinetics batch: {len(species)} species"
               + (f", up to {args.generations} generation(s)" if args.generations > 1
                  else ""))
        print(f"  started   : {now()}")
        print(f"  directory : {os.path.abspath(directory)}")
        print(f"  database  : {os.path.abspath(args.db)}")
        print(f"  species   : {', '.join(species)}")
        if args.generations > 1:
            print("  new species: products of each pair's best sane pathway"
                  + (f" with Ea <= {args.product_max_ea} eV" if args.product_max_ea
                     is not None else "")
                  + ("; closed-shell neutral only" if args.stable_only else ""))

        for gen in range(1, args.generations + 1):
            # pairs with at least one species new in this generation
            old = [x for k, x in known.items() if k not in new]
            fresh = list(new.values())
            pairs = list(itertools.combinations(fresh, 2))
            pairs += [(a, b) for a in fresh for b in old]
            if args.self_pairs:
                pairs += [(a, a) for a in fresh]
            pairs.sort(key=lambda ab: ab[0] == ab[1])  # A+B first, then A+A
            banner(f"generation {gen}: {len(fresh)} new species, {len(pairs)} pair(s)", "-")
            if gen > 1:
                print(f"  new species: {', '.join(fresh)}")

            status = run_status(db, args.model)
            todo, gen_entries = [], []
            for a, b in pairs:
                key2, st = status.get(pair_skeleton([a, b]), (None, None))
                if st == "complete" and not args.force:
                    entry = get_entry(db, key2)
                    results.append((gen, a, b, "in database", entry))
                    gen_entries.append(entry)
                elif st and st.startswith("failed") and not (args.retry_failed or args.force):
                    results.append((gen, a, b, "failed before", get_entry(db, key2)))
                else:
                    todo.append((a, b))
            print(f"  to run    : {len(todo)} pair(s); "
                  f"{len(pairs) - len(todo)} already in the database")

            for i, (a, b) in enumerate(todo):
                if args.max_runs is not None and runs >= args.max_runs:
                    results.append((gen, a, b, "not run (--max-runs)", None))
                    stop_reason = f"reached --max-runs ({args.max_runs})"
                    continue
                key = reaction_key(a, b)
                pair_dir = os.path.join(directory, pair_dirname(key))
                section(f"gen {gen}, pair {i + 1}/{len(todo)}: {a} + {b}")
                pargs = argparse.Namespace(**vars(args))
                pargs.smiles1, pargs.smiles2, pargs.dir = a, b, None
                runs += 1
                try:
                    with contextlib.redirect_stdout(console):  # details -> pair log
                        run_pair(pargs, db, key, pair_dir, suggest=False)
                except Exception as e:
                    results.append((gen, a, b, "FAILED", get_entry(db, key)))
                    log(f"FAILED: {e}  (see {pair_dir}/arcskin.log)")
                    continue
                entry = get_entry(db, key)
                results.append((gen, a, b, "run", entry))
                gen_entries.append(entry)
                log(f"done -> {pair_dir}/arcskin.log")
                print_best(entry.get("best_pathways") or [])

            if stop_reason:
                break
            if gen == args.generations:
                if args.generations > 1:
                    stop_reason = f"reached --generations ({args.generations})"
                break
            new, rows = batch_products(gen_entries, known, args.product_max_ea,
                                       args.stable_only)
            if not new:
                stop_reason = f"no new species from generation {gen}"
                log(stop_reason)
                break
            if args.max_species and len(known) + len(new) > args.max_species:
                keep = sorted(new, key=lambda k: float(
                    next(r[2] for r in rows if r[0] == new[k])))
                keep = keep[:max(0, args.max_species - len(known))]
                if not keep:
                    stop_reason = f"reached --max-species ({args.max_species})"
                    log(stop_reason)
                    break
                log(f"--max-species ({args.max_species}): keeping the "
                    f"{len(keep)} new species made with the lowest Ea")
                new = {k: new[k] for k in keep}
                rows = [r for r in rows if r[0] in new.values()]
            section(f"new species from generation {gen}: {len(new)}")
            table(rows, ["species", "made by", "Ea/eV"])
            known.update(new)

        banner("summary")
        print(f"  finished  : {now()}  ({runs} pair(s) run)")
        if stop_reason:
            print(f"  stopped   : {stop_reason}")
        print(f"  species   : {len(known)} ({', '.join(known.values())})")
        rows = []
        for gen, a, b, what, entry in results:
            best = (entry or {}).get("best_pathways") or []
            if not best:
                rows.append((gen, f"{a} + {b}", what, "-", "-", "-"))
            for j, bp in enumerate(best):
                rows.append((gen if j == 0 else "", f"{a} + {b}" if j == 0 else "",
                             what if j == 0 else "",
                             " + ".join(bp["products"]), fnum(bp["Ea"]),
                             bp.get("kind") or ""))
        table(rows, ["gen", "pair", "status", "best products", "Ea/eV", "kind"])
        print(f"\nlog written to {logfile}")


def cmd_traj(args):
    """List what's in a trajectories.json, or write NEB / MD trajectories as
    ASE files (energies and forces included, so `ase gui neb_0.traj` ->
    Tools -> NEB shows the energy profile)."""
    from ase.io import write

    from arcskin.trajectories import load, md_frames, neb_images

    data = load(args.location)
    seeds = sorted((k for k in data if k.startswith("seed_")),
                   key=lambda k: int(k.split("_")[1]))
    neb = data.get("neb") or {}
    if not args.neb and not args.md:
        if data.get("combinations"):
            print(f"pair: {' + '.join(data['combinations'])}")
        print(f"MD: {', '.join(f'{s} ({len(data[s])} frames)' for s in seeds) or 'none'}")
        rows = [(k, fmt_rxn(v["reactants"], v["products"]), fnum(v.get("Ea")),
                 fnum(v.get("Ea_rev")), fnum(v.get("dE")), v.get("kind"),
                 len(v.get("images") or []))
                for k, v in sorted(neb.items(), key=lambda kv: str(kv[0]))]
        if rows:
            table(rows, ["id", "reaction", "Ea/eV", "Ea_rev/eV", "dE/eV", "kind",
                         "images"])
        else:
            print("NEB: none")
        return
    safe = lambda k: slug(str(k)) or "X"  # noqa: E731
    if args.neb:
        ids = sorted(neb, key=str) if args.neb == "all" else [args.neb]
        for k in ids:
            if args.neb == "all" or not args.output:
                out = os.path.join(args.output or ".", f"neb_{safe(k)}.traj")
            else:
                out = args.output
            os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
            write(out, neb_images(args.location, k))
            print(f"NEB {k}: {fmt_rxn(neb[k]['reactants'], neb[k]['products'])} -> {out}")
    if args.md:
        out = args.output if args.output and not args.neb else f"{args.md}.traj"
        write(out, md_frames(args.location, args.md))
        print(f"MD {args.md} -> {out}")


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
        r.add_argument("--model", default=DEFAULT_MODEL,
                       help=f"AIMNet2 model (default {DEFAULT_MODEL}; aimnet2 is "
                       "closed-shell and overbinds radicals). Results from other "
                       "models are not reused")
        r.add_argument("--no-separate-products", dest="separate_products",
                       action="store_false",
                       help="relax the NEB product endpoint as one complex instead "
                       "of one molecule at a time (products may then bond to each "
                       "other)")
        r.add_argument("--no-spin-envelope", dest="spin_envelope",
                       action="store_false",
                       help="run every NEB on one spin state; by default bands "
                       "involving radicals follow the lower of the two lowest "
                       "spin states at each image, so bonds can break into "
                       "radicals (aimnet2nse's singlet is closed-shell)")
        r.add_argument("--keep-xtb", action="store_true",
                       help="keep the xtb seed_*/ directories (by default they are "
                       "deleted once the MD frames are in trajectories.json)")
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
    w.add_argument("--dir", help="where the SMILES1-SMILES2/ pair directories and "
                   "network_<reaction>.log go (default: current directory)")
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
    w.add_argument("--no-propose", dest="propose", action="store_false",
                   help="don't NEB steps suggested by the overall equation "
                   "(A + B -> exactly the missing products) before nanoreactor runs")
    w.add_argument("--no-reverse", dest="reverse", action="store_false",
                   help="search forward only: don't use NEB steps backwards "
                   "(Ea_rev = Ea - dE) or run pairs from the product side "
                   "(e.g. HONO + HNO3 for 2 NO2 + H2O -> HONO + HNO3)")
    w.add_argument("--decompose", choices=["intermediates", "all", "none"],
                   default="intermediates",
                   help="NEB unimolecular decompositions X -> A + B (every "
                   "one-bond split) of reachable species, e.g. N2O5 -> NO2 + NO3: "
                   "intermediates = species that are neither starting materials "
                   "nor overall products (default), all = every reachable "
                   "species, none = don't")
    w.add_argument("--weak-max", type=float, default=0.15,
                   help="use unbound/weakly bound complexes A + B -> C (NO dimer, "
                   "ONOO, ...) as fast pre-equilibrium steps if C is at most this "
                   "many eV above A + B (default 0.15; negative = don't)")
    w.add_argument("--steps", choices=["best", "tol", "all"], default="best",
                   help="which usable NEB pathways of each pair run the search "
                   "may use: best = only the lowest-Ea one (default), tol = also "
                   "those within --ea-tol of it, all = every sane one with "
                   "Ea <= --max-ea")
    w.add_argument("--any-model", action="store_true",
                   help="also use results computed with other models")
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
    bt.add_argument("--generations", type=int, default=1,
                    help="keep going: after each generation, the products of its "
                    "pairs' best sane pathways become new species, and the next "
                    "generation runs every pair involving a new species. Stops "
                    "after this many generations or when no new species appear "
                    "(default 1 = just the given list)")
    bt.add_argument("--product-max-ea", type=float, default=None,
                    help="only products made with Ea <= this (eV) become new "
                    "species (default: any sane pathway)")
    bt.add_argument("--stable-only", action="store_true",
                    help="only closed-shell, neutral products become new species "
                    "(no radicals)")
    bt.add_argument("--max-species", type=int, default=None,
                    help="cap on the total number of species; extra new ones are "
                    "dropped, highest Ea first")
    run_options(bt)
    bt.set_defaults(func=cmd_batch)

    d = sub.add_parser("decompose", help="NEB every one-bond decomposition "
                       "X -> A + B of the given species")
    d.add_argument("smiles", nargs="+", help="species to decompose")
    d.add_argument("--dir", help="where decomposed/<species>.log goes "
                   "(default: current directory)")
    d.add_argument("--force", action="store_true",
                   help="rerun decompositions already in the database")
    run_options(d)
    d.set_defaults(func=cmd_decompose)

    t = sub.add_parser("traj", help="list or export the MD / NEB trajectories "
                       "in a trajectories.json (for ASE)")
    t.add_argument("location", help="pair directory (or proposed/, decomposed/), "
                   "or a trajectories.json file")
    t.add_argument("--neb", metavar="ID", help="NEB pathway to export "
                   "(id as listed; 'all' = every one)")
    t.add_argument("--md", metavar="SEED", help="MD seed to export, e.g. seed_0")
    t.add_argument("-o", "--output", help="output file or, with --neb all, "
                   "directory (default: neb_<id>.traj / <seed>.traj)")
    t.set_defaults(func=cmd_traj)

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
