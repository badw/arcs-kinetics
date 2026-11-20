"""
Harmonic frequencies: is a structure a true minimum, or a transition state?

The bonding check in run_neb.diagnose only asks whether a relaxed structure
kept its bonds. This module looks at the curvature instead: a minimum has no
imaginary frequencies, a first-order saddle point (transition state) exactly
one. The Hessian is built by central finite differences of the forces, so it
works with any ASE calculator (including spin.SpinEnvelope), at 6N force calls
per structure: a few seconds for small molecules with AIMNet2.

    from arcskin.freq import check
    res = check(atoms, calc)      # atoms carries charge/mult in atoms.info
    res["n_imag"], res["imag"], res["freqs"]

Frequencies are in cm^-1; imaginary ones are reported as negative numbers.
Modes softer than `threshold` (default 50 cm^-1) are ignored when counting:
residual forces from a relaxation to fmax ~0.05 eV/Å, and the floppy modes of
loose complexes, give small spurious imaginary values.
"""

from __future__ import annotations

import numpy as np
from ase import units

# sqrt(eV / (Å^2 amu)) in cm^-1
_CM = np.sqrt(units._e / (1e-20 * units._amu)) / (2 * np.pi * units._c * 100)
THRESHOLD = 50.0


def hessian(atoms, calc, delta=0.01):
    """Cartesian Hessian (eV/Å^2) by central differences of the forces."""
    a = atoms.copy()
    a.info.update(atoms.info)
    a.calc = calc
    x0 = a.get_positions()
    n = len(a)
    H = np.zeros((3 * n, 3 * n))
    for i in range(n):
        for k in range(3):
            f = []
            for sgn in (1, -1):
                x = x0.copy()
                x[i, k] += sgn * delta
                a.set_positions(x)
                f.append(a.get_forces().ravel())
            H[3 * i + k] = -(f[0] - f[1]) / (2 * delta)
    a.set_positions(x0)
    return 0.5 * (H + H.T)


def _external_modes(atoms):
    """Orthonormal mass-weighted translation + rotation vectors (3N x k)."""
    m = atoms.get_masses()
    sm = np.repeat(np.sqrt(m), 3)
    r = atoms.get_positions() - atoms.get_center_of_mass()
    vecs = []
    for k in range(3):
        t = np.zeros((len(atoms), 3))
        t[:, k] = 1
        vecs.append(t.ravel() * sm)
    for k in range(3):
        e = np.zeros(3)
        e[k] = 1
        vecs.append(np.cross(r, e).ravel() * sm)
    u, s, _ = np.linalg.svd(np.array(vecs).T, full_matrices=False)
    return u[:, s > 1e-6 * s.max()]  # 5 for linear molecules, 6 otherwise


def normal_modes(atoms, H):
    """(frequencies in cm^-1 with imaginary < 0, Cartesian mode vectors
    (n_modes, N, 3), unit length), translations and rotations projected out."""
    if len(atoms) < 2:
        return np.array([]), np.zeros((0, len(atoms), 3))
    m = np.repeat(atoms.get_masses(), 3)
    Hm = H / np.sqrt(np.outer(m, m))
    ext = _external_modes(atoms)
    # basis of the internal (vibrational) space: complement of ext
    q, _ = np.linalg.qr(np.hstack([ext, np.eye(len(m))]))
    D = q[:, ext.shape[1]:len(m)]
    w, v = np.linalg.eigh(D.T @ Hm @ D)
    modes = (D @ v).T / np.sqrt(m)  # back to Cartesian displacements
    modes /= np.linalg.norm(modes, axis=1, keepdims=True)
    return np.sign(w) * np.sqrt(np.abs(w)) * _CM, modes.reshape(-1, len(atoms), 3)


def frequencies(atoms, H):
    """Vibrational frequencies (cm^-1, imaginary < 0)."""
    return normal_modes(atoms, H)[0]


def check(atoms, calc, threshold=THRESHOLD, delta=0.01):
    """{'freqs', 'n_imag', 'imag', 'lowest'} for one structure."""
    if len(atoms) < 2:
        return {"freqs": [], "n_imag": 0, "imag": [], "lowest": None}
    nu = frequencies(atoms, hessian(atoms, calc, delta))
    imag = [float(x) for x in nu if x < -threshold]
    return {"freqs": [float(x) for x in nu], "n_imag": len(imag), "imag": imag,
            "lowest": float(nu.min()) if len(nu) else None}


def make_minimum(atoms, calc, threshold=THRESHOLD, step=0.15, fmax=0.01,
                 max_tries=3):
    """Relax `atoms` (in place, its own charge/mult in info) to a true minimum:
    while it has imaginary modes, push it along the most negative one and
    re-relax. Gets molecules off symmetric saddle points that BFGS can't
    leave by itself (e.g. NO3 stuck at D3h). Returns (check result, number of
    pushes)."""
    from ase.optimize import BFGS

    atoms.calc = calc
    BFGS(atoms, logfile=None).run(fmax=fmax, steps=500)
    for n in range(max_tries + 1):
        nu, modes = normal_modes(atoms, hessian(atoms, calc))
        imag = [float(x) for x in nu if x < -threshold]
        res = {"freqs": [float(x) for x in nu], "n_imag": len(imag), "imag": imag,
               "lowest": float(nu.min()) if len(nu) else None}
        if not imag or n == max_tries:
            return res, n
        e0 = atoms.get_potential_energy()
        mode = modes[int(np.argmin(nu))]
        best = None
        for sgn in (1, -1):  # try both directions, keep the lower minimum
            a = atoms.copy()
            a.info.update(atoms.info)
            a.positions += sgn * step * mode / np.abs(mode).max()
            a.calc = calc
            BFGS(a, logfile=None).run(fmax=fmax, steps=500)
            if best is None or a.get_potential_energy() < best.get_potential_energy():
                best = a
        if best.get_potential_energy() > e0 - 1e-5:
            return res, n  # no lower structure found: leave it
        atoms.positions = best.positions
    return res, max_tries


def smiles_formula(smi):
    """Element counts of a SMILES (H included), e.g. for matching fragments."""
    from collections import Counter

    from rdkit import Chem

    mol = Chem.MolFromSmiles(smi, sanitize=False)
    if mol is None:
        return None
    mol.UpdatePropertyCache(strict=False)
    c = Counter(a.GetSymbol() for a in mol.GetAtoms())
    c["H"] += sum(a.GetTotalNumHs() for a in mol.GetAtoms())
    return +c


def label_fragments(atoms, frags, smiles):
    """SMILES for each fragment (index lists), matched by composition."""
    from collections import Counter

    left = list(smiles)
    out = []
    for idx in frags:
        comp = Counter(atoms[idx].get_chemical_symbols())
        hit = next((s for s in left if smiles_formula(s) == comp), None)
        if hit is not None:
            left.remove(hit)
        out.append(hit or atoms[idx].get_chemical_formula())
    return out


def fmt(res):
    """'min' / 'TS (-523i)' / '2 imag (-610i, -80i)' style summary."""
    if res is None:
        return "-"
    n = res["n_imag"]
    if n == 0:
        return "min"
    vals = ", ".join(f"{abs(x):.0f}i" for x in res["imag"][:3])
    return f"{n} imag ({vals})"
