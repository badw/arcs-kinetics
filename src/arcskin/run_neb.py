# from tblite.ase import TBLite
import warp as wp

wp.config.log_level = wp.LOG_WARNING  # no 'Warp initialized' banner
from aimnet.calculators import AIMNet2ASE  # noqa: F401  (kept for callers)
from ase.build import minimize_rotation_and_translation
from ase.mep import NEB
import warnings
from ase.optimize import BFGS, FIRE
import numpy as np
import pandas as pd
from ase.utils import forcecurve
import tqdm

from arcskin.check_endpoints import expected_bonds, geometry_bonds
from arcskin.freq import (check as freq_check, fmt as fmt_freq, label_fragments,
                          make_minimum)
from arcskin.make_neb_endpoints import _kabsch
from arcskin.spin import (DEFAULT_MODEL, SpinEnvelope, envelope_mults,
                          fragment_mults, fragments, get_calculator, is_nse,
                          set_state, system_mult)

def freq_report(images, E, fr_r, fr_p, m_r, m_p, charge, mult, calc, band_calc,
                overall, threshold=50.0):
    """Frequencies of each reactant/product molecule (own spin state) and of
    the band's highest interior image (band's own calculator)."""
    out = {}
    for side, atoms, frags, mults, smiles in (
            ("reactants", images[0], fr_r, m_r, overall.get("reactants") or []),
            ("products", images[-1], fr_p, m_p, overall.get("products") or [])):
        rows = []
        for i, idx in enumerate(frags):
            sub = atoms[idx]
            if len(frags) == 1:
                set_state(sub, charge, mult)
            else:
                set_state(sub, 0, mults[i] if mults else None)
            r = freq_check(sub, calc, threshold)
            r["smiles"] = label_fragments(atoms, frags, smiles)[i]
            r["mult"] = sub.info["mult"]
            rows.append(r)
        out[side] = rows
    top = int(np.argmax(E))
    if 0 < top < len(images) - 1:
        ts = images[top].copy()
        ts.info.update(images[top].info)
        r = freq_check(ts, band_calc, threshold)
        r["image"] = top
        r["verdict"] = ("transition state" if r["n_imag"] == 1 else
                        "no imaginary mode (not a saddle point)" if r["n_imag"] == 0
                        else f"{r['n_imag']} imaginary modes (higher-order saddle "
                             f"or unconverged band)")
        out["ts"] = r
    else:
        out["ts"] = {"verdict": "no interior maximum (barrierless / monotonic)",
                     "image": top, "n_imag": None, "imag": [], "freqs": []}
    return out


MIN_PATH = 0.5  # Å; shorter bands mean the endpoints are (nearly) the same structure


def relax(atoms, calc, fmax=0.05, steps=200, logfile=None):
    """Optional endpoint relaxation. Note: loose complexes may drift apart."""
    atoms.calc = calc
    BFGS(atoms, logfile=logfile).run(fmax=fmax, steps=steps)
    return atoms


def relax_fragments(atoms, frags, mults, calc, fmax=0.05, steps=200):
    """Relax each molecule on its own (in its own spin state) and put it back
    where it was (Kabsch), so molecules placed next to each other can't bond
    during the relaxation. A single fragment is just relaxed."""
    if len(frags) <= 1:
        return relax(atoms, calc, fmax, steps)
    for idx, m in zip(frags, mults):
        sub = atoms[idx]
        set_state(sub, 0, m)
        relax(sub, calc, fmax, steps)
        atoms.positions[idx] = _kabsch(sub.positions, atoms.positions[idx])[0]
    return atoms


def endpoint_fragments(reac, reactant_smiles):
    """Molecules of each endpoint from the intended bonding (None if unknown)."""
    try:
        exp_r, exp_p = expected_bonds(reactant_smiles, reac)
    except Exception:
        return None, None
    return fragments(len(reac), exp_r), fragments(len(reac), exp_p)


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
    for im in images:  # keep energy/forces with each image (for trajectories.json)
        im.info["energy"] = float(im.get_potential_energy())
        im.set_array("forces", np.array(im.get_forces()))
    E = np.array([im.info["energy"] for im in images])
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
            "mult": results[i].get("mult"),
            "spin_allowed": results[i].get("spin_allowed", True),
            "spin_envelope": results[i].get("spin_envelope"),
            "image_mults": results[i].get("image_mults"),
            "freq": results[i].get("freq"),
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
    model=None,
    separate_products=True,
    spin_envelope=True,
    freq=False,
    freq_threshold=50.0,
):
    """NEB every pathway in endpoints_dict.

    model: AIMNet2 model when no calculator is given (default aimnet2nse).
    Spin: each molecule gets its ground multiplicity; the NEB runs at the lowest
    total spin both sides can reach (e.g. N2O2 + O2(triplet) -> 2 NO2 is a
    triplet NEB). separate_products: relax the product molecules one at a time,
    so products placed side by side can't bond to each other.
    spin_envelope: when the two lowest total spins both sides can reach differ
    (radicals involved), run the band on the lower of the two at each image
    (see spin.SpinEnvelope); otherwise a bond breaking into two radicals is
    forced onto the closed-shell singlet and its energy is far too high.
    freq: also compute harmonic frequencies (arcskin.freq) of every reactant
    and product molecule and of the highest image. A molecule with an
    imaginary mode (|nu| > freq_threshold cm^-1) isn't a minimum, so the
    pathway is marked not sane; the highest image should have exactly one
    (a transition state), which is reported but doesn't change `sane`."""
    if not calculator:
        model = model or DEFAULT_MODEL
        print(f"running with {model}")
        calc = get_calculator(model)
    else:
        calc = calculator
    results = {}

    for k, v in tqdm.tqdm(endpoints_dict.items()):
        reactants, products = v["reactants"], v["products"]
        charge = int(reactants.info.get("charge", 0))
        fr_r, fr_p = endpoint_fragments(reactants, v["overall"]["reactants"])
        if fr_r is None:
            fr_r = fr_p = [list(range(len(reactants)))]
        if charge == 0:
            m_r = fragment_mults(reactants, fr_r, calc)
            m_p = fragment_mults(products, fr_p, calc)
            mult, spin_ok = system_mult(m_r, m_p)
        else:  # charged: no per-fragment charges known, use lowest spin
            m_r = m_p = None
            mult, spin_ok = None, True
        set_state(reactants, charge, mult)
        set_state(products, charge, mult)
        mult = reactants.info["mult"]

        reactants = relax(reactants, calc)
        if separate_products and m_p is not None:
            products = relax_fragments(products, fr_p, m_p, calc)
            set_state(products, charge, mult)  # back to the system's state
        else:
            products = relax(products, calc)
        pushed = []
        if freq and m_r is not None:
            # make every end molecule a true minimum before the band is run
            for side, atoms, frags, mults in (("reactants", reactants, fr_r, m_r),
                                              ("products", products, fr_p, m_p)):
                names = label_fragments(atoms, frags, v["overall"][side])
                for i, idx in enumerate(frags):
                    sub = atoms[idx]
                    set_state(sub, charge if len(frags) == 1 else 0,
                              mult if len(frags) == 1 else mults[i])
                    _, n = make_minimum(sub, calc, freq_threshold)
                    if n:
                        atoms.positions[idx] = _kabsch(sub.positions,
                                                       atoms.positions[idx])[0]
                        pushed.append(f"{names[i]} ({side[:-1]})")
            if pushed and len(fr_r) > 1:
                reactants = relax(reactants, calc)  # complex around the new shapes
            set_state(reactants, charge, mult)
            set_state(products, charge, mult)
        band_calc, env = calc, None
        if spin_envelope and m_r is not None and is_nse(calc):
            env = envelope_mults(m_r, m_p)
            if len(env) > 1:
                band_calc = SpinEnvelope(calc, env)
            else:
                env = None
        images, E, Ea, dE = run_neb(
            reactants,
            products,
            band_calc,
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
        image_mults = None
        if env:
            image_mults = []
            for im in images:  # which state each image ended up on
                band_calc.get_potential_energy(im)
                image_mults.append(band_calc.mult)
                im.info["band_mult"] = band_calc.mult
        if not spin_ok:
            reason = (reason + "; " if reason else "") + (
                f"spin-forbidden: reactant spins {m_r} can't couple to the "
                f"same total as product spins {m_p} (run at mult {mult})")
        freq_res = None
        if freq:
            freq_res = freq_report(images, E, fr_r, fr_p, m_r, m_p, charge, mult,
                                   calc, band_calc, v["overall"], freq_threshold)
            freq_res["pushed_off_saddle"] = pushed
            bad = [f"{w} {x['smiles']} ({fmt_freq(x)})"
                   for w in ("reactants", "products") for x in freq_res[w]
                   if x["n_imag"]]
            if bad:
                sane = False
                reason = (reason + "; " if reason else "") + (
                    "not a minimum: " + ", ".join(bad))
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
            "mult": mult,
            "spin_allowed": spin_ok,
            "spin_envelope": env,
            "image_mults": image_mults,
            "freq": freq_res,
        }

    results = determine_best(results)

    return results
