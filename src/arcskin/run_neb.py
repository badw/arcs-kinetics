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
        if np.max(fit.path) <= 0.5:
            # need more checks
            sane = False
        else:
            sane = True

        results[k] = {
            "nebfit": fit._asdict(),
            "images": [img.todict() for img in images],
            "E": E,
            "Ea": Ea,
            "dE": dE,
            "overall": v["overall"],
            "sane": sane,
        }

    results = determine_best(results)

    return results
