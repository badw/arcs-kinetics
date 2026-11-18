# from tblite.ase import TBLite
from aimnet.calculators import AIMNet2ASE
from ase.build import minimize_rotation_and_translation
from ase.mep import NEB
import warnings
from ase.optimize import BFGS, FIRE
import numpy as np
import pandas as pd
from ase.utils import forcecurve
import tqdm

from arcskin.check_endpoints import expected_bonds, geometry_bonds

MIN_PATH = 0.5  # Å; shorter bands mean the endpoints are (nearly) the same structure


def relax(atoms, calc, fmax=0.05, steps=200, logfile=None):
    """Optional endpoint relaxation. Note: loose complexes may drift apart."""
    atoms.calc = calc
    BFGS(atoms, logfile=logfile).run(fmax=fmax, steps=steps)
    return atoms


def run_neb(
    reac,
    prod,
    calc,
    n_images=8,
    fmax=0.05,
    steps=500,
    climb=True,
    method="idpp",
    trajectory=None,
    logfile="-",
):
    """Interpolate (IDPP), relax the band, then switch on climbing image.
    `calc` is either one calculator shared by all images (e.g. MACE) or a factory
    `atoms -> Calculator` (e.g. lambda a: TBLite(charge=a.info['charge'],
    multiplicity=a.info['mult'])). Returns (images, energies, barrier_fwd, dE)."""
    prod = prod.copy()
    minimize_rotation_and_translation(reac, prod)
    images = [reac.copy()] + [reac.copy() for _ in range(n_images)] + [prod]
    shared = not (callable(calc) and not hasattr(calc, "get_potential_energy"))
    for im in images:
        im.calc = calc
    neb = NEB(
        images, climb=False, allow_shared_calculator=shared, method="improvedtangent"
    )
    neb.interpolate(method=method)
    opt = FIRE(neb, trajectory=trajectory, logfile=logfile)
    opt.run(fmax=max(fmax, 0.5) if climb else fmax, steps=steps)
    converged = True
    if climb:
        neb.climb = True
        converged = opt.run(fmax=fmax, steps=steps)
    if not converged:
        warnings.warn(f"NEB not converged to fmax={fmax} in {steps} steps")
    E = np.array([im.get_potential_energy() for im in images])
    E -= E[0]
    return images, E, E.max(), E[-1]


def _bond_labels(atoms, bonds):
    sym = atoms.get_chemical_symbols()
    return ", ".join(f"{sym[i]}{i}-{sym[j]}{j}" for i, j in sorted(bonds))


def diagnose(reac, prod, reactant_smiles, path_length, scale=1.25):
    """Classify a finished NEB by comparing the bonding in the relaxed endpoints
    with the bonds the reactant/product SMILES say should be there.

    Returns (sane, kind, reason). kind is one of
      "barrier"      - both endpoints kept their bonding and the band has length
      "barrierless"  - the reactant complex relaxed straight into the products
      "unbound"      - the product complex fell apart back into the reactants
      "rearranged"   - an endpoint relaxed into some other species
      "collapsed"    - endpoints are the same structure for some other reason
      "unchecked"    - bonding could not be checked (missing endpoint info)
    """
    try:
        exp_r, exp_p = expected_bonds(reactant_smiles, reac)
    except Exception as e:  # e.g. endpoints built without formed/broken info
        sane = path_length > MIN_PATH
        return sane, "unchecked", f"bond check skipped ({e}); path {path_length:.2f} Å"
    got_r, got_p = geometry_bonds(reac, scale), geometry_bonds(prod, scale)

    if got_r == exp_p and got_r != exp_r:
        return (True, "barrierless",
                f"reactant complex relaxed directly into the products "
                f"(formed {_bond_labels(reac, exp_p - exp_r)}); no barrier")
    if got_p == exp_r and got_p != exp_p:
        return (False, "unbound",
                f"product complex fell apart into the reactants "
                f"(lost {_bond_labels(prod, exp_p - exp_r)}); product not a minimum")
    problems = []
    for name, atoms, got, exp in (("reactant", reac, got_r, exp_r),
                                  ("product", prod, got_p, exp_p)):
        if name == "reactant" and got > exp:
            problems.append(
                f"reactant complex reacted on relaxation "
                f"(+[{_bond_labels(atoms, got - exp)}]); barrier is from that "
                f"intermediate, not the separated reactants")
        elif got != exp:
            msg = f"{name} rearranged"
            if got - exp:
                msg += f" +[{_bond_labels(atoms, got - exp)}]"
            if exp - got:
                msg += f" -[{_bond_labels(atoms, exp - got)}]"
            problems.append(msg)
    if problems:
        return False, "rearranged", "; ".join(problems)
    if path_length <= MIN_PATH:
        return (False, "collapsed",
                f"endpoints almost identical (path {path_length:.2f} Å)")
    return True, "barrier", ""


def determine_best(results):
    new_results = {}
    for i in results:
        new_results[i] = {
            "Ea": results[i]["Ea"],
            "dE": results[i]["dE"],
            "reactants": results[i]["overall"]["reactants"],
            "products": results[i]["overall"]["products"],
            "reactants_energies": results[i]["overall"]["reactants_energies"],
            "products_energies": results[i]["overall"]["products_energies"],
            "diff": results[i]["overall"]["diff"],
            "images": results[i]["images"],
            "nebfit": results[i]["nebfit"],
            "sane": results[i]["sane"],
            "kind": results[i]["kind"],
            "reason": results[i]["reason"],
            "path_length": results[i]["path_length"],
        }
    ranking = pd.DataFrame(new_results).T.sort_values(by=["Ea", "dE"], ascending=True)
    return ranking


def run_all(
    endpoints_dict: dict,
    calculator=None,
    n_images=5,
    fmax=0.1,
    climb=False,
    steps=500,
    trajectory=None,
    logfile=None,
):
    if not calculator:
        print("running with aimnet")
        calc = AIMNet2ASE("aimnet2")
    else:
        calc = calculator
    results = {}

    for k, v in tqdm.tqdm(endpoints_dict.items()):
        reactants = relax(v["reactants"], calc)
        products = relax(v["products"], calc)
        images, E, Ea, dE = run_neb(
            reactants,
            products,
            calc,
            n_images=n_images,
            fmax=fmax,
            climb=climb,
            steps=steps,
            trajectory=trajectory,
            logfile=logfile,
        )
        fit = forcecurve.fit_images(images)
        path_length = float(np.max(fit.path))
        sane, kind, reason = diagnose(
            reactants, products, v["overall"]["reactants"], path_length
        )
        if kind == "barrierless" and "diff" in v["overall"]:
            # both NEB endpoints are the product, so the band's dE is 0 by
            # construction; use the isolated-species reaction energy instead
            dE = float(v["overall"]["diff"])
            reason += f"; dE from isolated species ({dE:.3f} eV)"

        results[k] = {
            "nebfit": fit._asdict(),
            "images": [img.todict() for img in images],
            "E": E,
            "Ea": Ea,
            "dE": dE,
            "overall": v["overall"],
            "sane": sane,
            "kind": kind,
            "reason": reason,
            "path_length": path_length,
        }

    results = determine_best(results)

    return results
