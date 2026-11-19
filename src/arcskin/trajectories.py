"""
Reading and writing trajectories.json: the MD frames of a nanoreactor run and
the NEB bands computed from it, in one file per pair directory.

Layout::

    {"combinations": [smiles1, smiles2],
     "seed_0": [atoms dict, ...], "seed_1": [...], ...    # xtb MD frames
     "neb": {"<pathway id>": {
         "reactants": [...], "products": [...],
         "Ea": .., "dE": .., "Ea_rev": ..,                 # eV
         "kind": .., "sane": .., "reason": .., "mult": .., "image_mults": [...],
         "images": [atoms dict, ...],                     # info["energy"], forces
         "energies": [...],                               # eV, relative to image 0
         "reaction_coordinate": [...],                    # Å along the band
         "fit_coordinate": [...], "fit_energies": [...]}}}  # smoothed profile

Reading it back in ASE::

    from arcskin.trajectories import neb_images, md_frames
    from ase.visualize import view
    images = neb_images("O_O-N_O/trajectories.json", 0)   # energies + forces
    view(images)                      # or ase.io.write("neb.traj", images)
    md = md_frames("O_O-N_O/trajectories.json", "seed_0")

`arcskin traj` does the same from the command line (and writes .traj files that
`ase gui` can show, including Tools -> NEB for the energy profile).
"""

from __future__ import annotations

import os

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from monty.serialization import dumpfn, loadfn


def _path(location):
    return (location if location.endswith(".json")
            else os.path.join(location, "trajectories.json"))


def load(location) -> dict:
    """Raw contents of trajectories.json (a file, or the directory holding it)."""
    return loadfn(_path(location))


def to_atoms(d) -> Atoms:
    """Atoms from a stored dict, with a SinglePointCalculator holding the
    stored energy (info['energy']) and forces, if any."""
    if isinstance(d, Atoms):
        return d
    d = dict(d)
    forces = d.pop("forces", None)
    atoms = Atoms.fromdict(d)
    energy = atoms.info.get("energy")
    if energy is not None or forces is not None:
        atoms.calc = SinglePointCalculator(
            atoms, energy=energy,
            forces=None if forces is None else np.asarray(forces))
    return atoms


def neb_images(location, pathway) -> list[Atoms]:
    """The NEB band of one pathway, image 0 = reactant end."""
    neb = load(location).get("neb") or {}
    if str(pathway) not in neb:
        raise KeyError(f"no NEB for pathway {pathway!r}; have {sorted(neb)}")
    return [to_atoms(d) for d in neb[str(pathway)]["images"]]


def md_frames(location, seed="seed_0") -> list[Atoms]:
    """The MD frames of one nanoreactor seed."""
    data = load(location)
    if seed not in data:
        raise KeyError(f"no {seed}; have {sorted(k for k in data if k.startswith('seed_'))}")
    return [to_atoms(d) for d in data[seed]]


def _str_keys(x):
    """JSON needs string keys; endpoint info has e.g. {(i, j): distance}."""
    if isinstance(x, dict):
        return {(k if isinstance(k, str) else str(k)): _str_keys(v)
                for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_str_keys(v) for v in x)
    return x


def neb_entry(rec) -> dict:
    """What trajectories.json keeps of one NEB record (from run_neb.run_all)."""
    fit = rec.get("nebfit") or {}
    images = [_str_keys(im) for im in rec.get("images") or []]
    e = [(im.get("info") or {}).get("energy") for im in images]
    energies = ([x - e[0] for x in e] if e and all(x is not None for x in e)
                else (list(fit.get("energies")) if fit.get("energies") is not None
                      else None))
    try:
        ea_rev = float(rec["Ea"]) - float(rec["dE"])
    except (TypeError, ValueError, KeyError):
        ea_rev = None
    return {
        "reactants": list(rec.get("reactants") or []),
        "products": list(rec.get("products") or []),
        "Ea": rec.get("Ea"), "dE": rec.get("dE"), "Ea_rev": ea_rev,
        "kind": rec.get("kind"), "sane": rec.get("sane"),
        "reason": rec.get("reason"), "mult": rec.get("mult"),
        "image_mults": rec.get("image_mults"),
        "images": images,
        "energies": energies,
        "reaction_coordinate": fit.get("path"),
        "fit_coordinate": fit.get("fit_path"),
        "fit_energies": fit.get("fit_energies"),
    }


def save_neb(location, records: dict):
    """Add/replace NEB bands ({id: NEB record}) in trajectories.json, keeping
    whatever else is in it."""
    path = _path(location)
    data = loadfn(path) if os.path.exists(path) else {}
    neb = data.setdefault("neb", {})
    for k, rec in records.items():
        neb[str(k)] = neb_entry(rec)
    # write a temporary file and swap it in, so a failed write can never
    # truncate the existing file (it holds the only copy of the MD frames)
    tmp = f"{path}.tmp"
    try:
        dumpfn(data, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return path
