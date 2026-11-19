"""
Calculator and spin-state handling for AIMNet2.

The open-shell model ``aimnet2nse`` (the default) needs the charge and spin
multiplicity of every structure; the closed-shell ``aimnet2`` ignores the spin
and badly overbinds radicals (e.g. it puts N2O4 ~4.4 eV below 2 NO2; the
experimental value is ~0.6 eV, which aimnet2nse reproduces).

AIMNet2ASE reads ``atoms.info["charge"]`` and ``atoms.info["mult"]`` on every
call, so one shared calculator works for many species as long as each Atoms
carries its own state (use ``set_state``; a missing charge silently reuses the
previous one).

Set ARCSKIN_MODEL to change the default model.
"""

from __future__ import annotations

import os
from collections import Counter

import networkx as nx
import numpy as np
from ase.data import covalent_radii
from ase.optimize import BFGS

DEFAULT_MODEL = os.environ.get("ARCSKIN_MODEL", "aimnet2nse")
SMALL = 4  # molecules with <= this many atoms get their ground spin computed

_calcs = {}
_ground = {}

# Hund's-rule ground states of neutral free atoms (Z: multiplicity)
ATOM_MULT = {1: 2, 5: 2, 6: 3, 7: 4, 8: 3, 9: 2, 14: 3, 15: 4, 16: 3, 17: 2,
             33: 4, 34: 3, 35: 2, 53: 2}


def get_calculator(model=None):
    """One AIMNet2ASE per model name per process."""
    import warp as wp

    wp.config.log_level = wp.LOG_WARNING
    from aimnet.calculators import AIMNet2ASE

    model = model or DEFAULT_MODEL
    if model not in _calcs:
        _calcs[model] = AIMNet2ASE(model)
    return _calcs[model]


def is_nse(calc):
    return bool(getattr(getattr(calc, "base_calc", None), "is_nse", False))


def model_name(calc):
    """Best-effort model name of a calculator (for the database)."""
    for name, c in _calcs.items():
        if c is calc:
            return name
    return "aimnet2nse" if is_nse(calc) else "aimnet2"


def set_state(atoms, charge=0, mult=None):
    """Store charge and multiplicity where AIMNet2ASE reads them."""
    if mult is None:
        mult = lowest_mult(atoms, charge)
    atoms.info["charge"] = int(charge)
    atoms.info["mult"] = int(mult)
    return atoms


def lowest_mult(atoms, charge=0):
    """Lowest multiplicity allowed by the electron count."""
    return (int(atoms.numbers.sum()) - int(charge)) % 2 + 1


def _signature(atoms, scale=1.25):
    """Formula + element-pair bond counts: cache key for small molecules."""
    rc = covalent_radii[atoms.numbers]
    d = atoms.get_all_distances()
    sym = atoms.get_chemical_symbols()
    i, j = np.where(np.triu(d < scale * (rc[:, None] + rc[None, :]), 1))
    bonds = Counter(tuple(sorted((sym[a], sym[b]))) for a, b in zip(i, j))
    return tuple(sorted(Counter(sym).items())), tuple(sorted(bonds.items()))


def relaxed_energy(atoms, calc, charge, mult, fmax=0.05, steps=300):
    a = atoms.copy()
    set_state(a, charge, mult)
    a.calc = calc
    try:
        BFGS(a, logfile=None).run(fmax=fmax, steps=steps)
        return float(a.get_potential_energy()), a
    except Exception:
        return np.inf, a


def ground_mult(atoms, calc, charge=0):
    """Ground-state multiplicity of ONE molecule.

    Closed-shell model or large molecule: lowest spin from the electron count.
    Open-shell model and <= SMALL atoms: the lower in energy of the lowest spin
    and the next one up (catches triplet O2, O, S, SO, NH, CH2, quartet N...).
    """
    low = lowest_mult(atoms, charge)
    if not is_nse(calc) or len(atoms) > SMALL:
        return low
    if len(atoms) == 1 and charge == 0 and atoms.numbers[0] in ATOM_MULT:
        return ATOM_MULT[int(atoms.numbers[0])]  # models are unreliable for atoms
    key = (_signature(atoms), int(charge), model_name(calc))
    if key not in _ground:
        e = {m: relaxed_energy(atoms, calc, charge, m)[0] for m in (low, low + 2)}
        _ground[key] = min(e, key=e.get)
    return _ground[key]


def couple(mults):
    """All total multiplicities two or more spins can couple to."""
    total = {1}
    for m in mults:
        total = {t for a in total for t in range(abs(a - m) + 1, a + m, 2)}
    return total


def system_mult(reac_mults, prod_mults):
    """(multiplicity, spin_allowed) for a reaction: the lowest total spin that
    both sides can reach; if there is none, the reactants' lowest one."""
    r, p = couple(reac_mults), couple(prod_mults)
    common = r & p
    if common:
        return min(common), True
    return min(r), False


def fragments(n, bonds):
    """Connected components (sorted index lists) of a bond list."""
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(bonds)
    return sorted(sorted(c) for c in nx.connected_components(G))


def fragment_mults(atoms, frags, calc):
    """Ground multiplicity of each fragment (neutral fragments assumed, unless
    the whole structure is one fragment)."""
    charge = int(atoms.info.get("charge", 0))
    if len(frags) == 1:
        return [ground_mult(atoms, calc, charge)]
    return [ground_mult(atoms[idx], calc, 0) for idx in frags]
