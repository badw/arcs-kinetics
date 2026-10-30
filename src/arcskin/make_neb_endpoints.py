"""
SMILES reaction -> aligned NEB endpoints -> ASE NEB with any calculator.

Approach (borrowed from autodE, reimplemented so autodE isn't needed):
  1. Atom mapping: find the smallest set of broken/formed bonds that makes the
     reactant graph isomorphic to the product graph, only trying bond types
     whose counts change (autode.bond_rearrangement.get_bond_rearrangs).
  2. Template: the side with fewer molecules (e.g. the single product of A+B->C)
     is embedded once. Each molecule on the other side is Kabsch-aligned onto its
     mapped atoms in the template, so its orientation already matches the end
     state and the interpolation doesn't drag atoms through each other.
  3. Rigid-body fit (cf. autode's translate_rotate_reactant / attack_cost):
     molecules are rotated/translated so bonds that change sit at
     shift_factor * (r_cov_i + r_cov_j) and no other inter-molecular pair is
     closer than that.
"""

import itertools
import warnings
from collections import Counter

import networkx as nx
import numpy as np
from networkx.algorithms import isomorphism as iso
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from ase import Atoms
from ase.build import minimize_rotation_and_translation
from ase.data import covalent_radii
from ase.optimize import BFGS, FIRE

RDLogger.DisableLog("rdApp.warning")

try:
    from ase.mep import NEB, NEBTools
except ImportError:  # ASE < 3.23
    from ase.neb import NEB, NEBTools


# --------------------------------------------------------------------------- #
# Embedding
# --------------------------------------------------------------------------- #
def embed(smi, seed=42):
    """SMILES -> (Atoms, bonds). ETKDG + MMFF (UFF fallback); handles single atoms.
    atoms.info holds 'charge' and 'n_radicals'."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smi!r}")
    mol = Chem.AddHs(mol)
    if mol.GetNumAtoms() == 1:
        pos = np.zeros((1, 3))
    else:
        params = AllChem.ETKDGv3()
        params.randomSeed = seed
        if AllChem.EmbedMolecule(mol, params) != 0:
            raise RuntimeError(f"Embedding failed for {smi!r}")
        try:
            if AllChem.MMFFHasAllMoleculeParams(mol):
                AllChem.MMFFOptimizeMolecule(mol)
            else:
                AllChem.UFFOptimizeMolecule(mol)
        except Exception:
            pass  # keep the ETKDG geometry
        pos = mol.GetConformer().GetPositions()
    atoms = Atoms([a.GetSymbol() for a in mol.GetAtoms()], positions=pos)
    atoms.positions -= atoms.get_center_of_mass()
    atoms.info.update(
        charge=Chem.GetFormalCharge(mol),
        n_radicals=sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms()),
    )
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
    return atoms, bonds


def conformers(smi, n=10, seed=42):
    """List of position arrays (same atom order as `embed`) for n conformers."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smi))
    if mol.GetNumAtoms() == 1 or n <= 1:
        return [embed(smi, seed)[0].positions]
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    ids = list(AllChem.EmbedMultipleConfs(mol, n, params))
    if not ids:
        return [embed(smi, seed)[0].positions]
    try:
        if AllChem.MMFFHasAllMoleculeParams(mol):
            AllChem.MMFFOptimizeMoleculeConfs(mol)
        else:
            AllChem.UFFOptimizeMoleculeConfs(mol)
    except Exception:
        pass
    return [mol.GetConformer(i).GetPositions() for i in ids]


def smiles_to_atoms(smi, seed=42, return_charge=True):
    """Your original helper, kept for compatibility."""
    atoms, _ = embed(smi, seed)
    return (atoms, atoms.info["charge"]) if return_charge else atoms


def _side(smiles, seed=42):
    """Embed and concatenate molecules -> (Atoms, nx.Graph, list of index arrays)."""
    atoms, frags, G = Atoms(), [], nx.Graph()
    charge = n_rad = 0
    for smi in smiles:
        a, bonds = embed(smi, seed)
        n0 = len(atoms)
        frags.append(np.arange(n0, n0 + len(a)))
        G.add_nodes_from((n0 + i, {"Z": z}) for i, z in enumerate(a.numbers))
        G.add_edges_from((n0 + i, n0 + j) for i, j in bonds)
        charge += a.info["charge"]
        n_rad += a.info["n_radicals"]
        atoms += a
    # lowest spin consistent with the electron count; override if needed
    mult = (int(atoms.numbers.sum()) - charge) % 2 + 1
    atoms.info.update(charge=charge, mult=mult, n_radicals=n_rad, smiles=list(smiles))
    return atoms, G, frags


# --------------------------------------------------------------------------- #
# Atom mapping (autodE-style bond rearrangement search)
# --------------------------------------------------------------------------- #
def _btype(G, i, j):
    return tuple(sorted((G.nodes[i]["Z"], G.nodes[j]["Z"])))


def _signature(G):
    return Counter((G.nodes[n]["Z"], d) for n, d in G.degree())


def find_rearrangements(G_r, G_p, max_extra=1, skip_identity=False):
    """Yield (fbonds, bbonds, GraphMatcher), fewest bond changes first.
    Only bond *types* whose counts change are tried; `max_extra` additionally
    allows that many break+form pairs of an unchanged type (e.g. H transfers).
    skip_identity drops the 'no bonds change' answer (e.g. H + H2 -> H2 + H)."""
    if sorted(nx.get_node_attributes(G_r, "Z").values()) != sorted(
        nx.get_node_attributes(G_p, "Z").values()
    ):
        raise ValueError("Reactants and products don't have the same atoms")
    cr = Counter(_btype(G_r, *e) for e in G_r.edges)
    cp = Counter(_btype(G_p, *e) for e in G_p.edges)
    form0, break0 = cp - cr, cr - cp

    bonds_by_type, pairs_by_type = {}, {}
    for e in G_r.edges:
        bonds_by_type.setdefault(_btype(G_r, *e), []).append(e)
    for i, j in itertools.combinations(G_r.nodes, 2):
        if not G_r.has_edge(i, j):
            pairs_by_type.setdefault(_btype(G_r, i, j), []).append((i, j))

    def choose(pool, need):
        per = [
            list(itertools.combinations(pool.get(t, []), k)) for t, k in need.items()
        ]
        for combo in itertools.product(*per):
            yield [b for c in combo for b in c]

    sig_p, nm = _signature(G_p), iso.categorical_node_match("Z", None)
    for n_extra in range(max_extra + 1):
        extras = (
            [Counter()]
            if n_extra == 0
            else [
                Counter(c) for c in itertools.combinations_with_replacement(cr, n_extra)
            ]
        )
        for ex in extras:
            for bb in choose(bonds_by_type, break0 + ex):
                for fb in choose(pairs_by_type, form0 + ex):
                    if skip_identity and not (bb or fb):
                        continue
                    G = G_r.copy()
                    G.remove_edges_from(bb)
                    G.add_edges_from(fb)
                    if _signature(G) != sig_p:
                        continue
                    gm = iso.GraphMatcher(G_p, G, node_match=nm)
                    if gm.is_isomorphic():
                        yield fb, bb, gm


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def _kabsch(P, Q):
    """Rigidly move P onto Q. Returns (moved P, rmsd)."""
    pc, qc = P.mean(0), Q.mean(0)
    if len(P) == 1:
        return Q.copy(), 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        R, _ = Rotation.align_vectors(Q - qc, P - pc)
    moved = R.apply(P - pc) + qc
    return moved, float(np.sqrt(((moved - Q) ** 2).sum(1).mean()))


def _apply(x, pos, frags):
    out = pos.copy()
    for k, idx in enumerate(frags):
        c = pos[idx].mean(0)
        out[idx] = (
            Rotation.from_rotvec(x[6 * k : 6 * k + 3]).apply(pos[idx] - c)
            + c
            + x[6 * k + 3 : 6 * k + 6]
        )
    return out


def rigid_fit(
    atoms, frags, targets, ref=None, shift=1.5, w_ref=0.01, n_starts=10, seed=0
):
    """Rotate/translate each molecule (rigidly) to minimise
        Σ_targets (r_ij - d_ij)^2 + 10 Σ_inter max(0, r_min - r_ij)^2 + w_ref Σ |x - ref|^2
    with d_ij = r_min = shift * (r_cov_i + r_cov_j) (r_min >= 2 Å for non-target pairs)."""
    pos, Z = atoms.positions, atoms.numbers
    rc = covalent_radii[Z]
    fid = np.empty(len(atoms), int)
    for k, idx in enumerate(frags):
        fid[idx] = k
    tset = {tuple(sorted(t)) for t in targets}
    ti = np.array([t[0] for t in tset], int)
    tj = np.array([t[1] for t in tset], int)
    td = shift * (rc[ti] + rc[tj])
    pr = [
        (i, j)
        for i, j in itertools.combinations(range(len(atoms)), 2)
        if fid[i] != fid[j] and (i, j) not in tset
    ]
    ri, rj = (np.array(a, int) for a in zip(*pr)) if pr else (np.array([], int),) * 2
    rmin = np.maximum(shift * (rc[ri] + rc[rj]), 2.0)

    def cost(x):
        X = _apply(x, pos, frags)
        c = ((np.linalg.norm(X[ti] - X[tj], axis=1) - td) ** 2).sum()
        c += (
            10
            * (
                np.clip(rmin - np.linalg.norm(X[ri] - X[rj], axis=1), 0, None) ** 2
            ).sum()
        )
        if ref is not None:
            c += w_ref * ((X - ref) ** 2).sum()
        return c

    rng = np.random.default_rng(seed)
    starts = [np.zeros(6 * len(frags))] if ref is not None else []
    if ref is None:  # random orientations, molecules spread out on a sphere
        size = [
            np.ptp(pos[idx], axis=0).max() if len(idx) > 1 else 1.0 for idx in frags
        ]
        for _ in range(n_starts):
            x = np.zeros(6 * len(frags))
            for k in range(1, len(frags)):
                v = rng.normal(size=3)
                x[6 * k : 6 * k + 3] = Rotation.random(random_state=rng).as_rotvec()
                x[6 * k + 3 : 6 * k + 6] = (
                    v / np.linalg.norm(v) * (size[0] + size[k]) / 2 + 2
                )
            starts.append(x)
    best = min(
        (minimize(cost, x0, method="L-BFGS-B") for x0 in starts), key=lambda r: r.fun
    )
    out = atoms.copy()
    out.positions = _apply(best.x, pos, frags)
    return out, best.fun


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
def make_endpoints(
    reactants,
    products,
    shift=1.5,
    seed=42,
    n_confs=10,
    max_isomorphisms=500,
    verbose=True,
):
    """Build (reactant_complex, product_complex) Atoms with identical atom order,
    positioned/rotated consistently for NEB. Info dicts hold charge, mult and the
    formed/broken bonds (reactant indices)."""
    R, G_r, fr_r = _side(reactants, seed)
    P, G_p, fr_p = _side(products, seed)
    try:
        fb, bb, gm = next(
            find_rearrangements(G_r, G_p, skip_identity=len(fr_r) + len(fr_p) > 2)
        )
    except StopIteration:
        raise RuntimeError("No bond rearrangement found (try max_extra=2)") from None

    # template = side with fewer molecules (products on a tie)
    prod_is_tmpl = len(fr_p) <= len(fr_r)
    T, fr_t, O, fr_o = (P, fr_p, R, fr_r) if prod_is_tmpl else (R, fr_r, P, fr_p)
    G_o = G_r if prod_is_tmpl else G_p
    G_t = G_p if prod_is_tmpl else G_r

    # changed bonds, in template indices, for the template's own rigid fit
    def to_tmap(m):  # GraphMatcher mapping is product -> rearranged reactant
        inv = {r: p for p, r in m.items()}
        n = len(R)
        return (
            np.array([inv[i] for i in range(n)])
            if prod_is_tmpl
            else np.array([m[i] for i in range(n)])
        )

    # template geometry (only needs fitting if it has >1 molecule)
    m0 = to_tmap(gm.mapping)
    if len(fr_t) > 1:
        t_targets = [
            (m0[i], m0[j]) for i, j in G_o.edges if not G_t.has_edge(m0[i], m0[j])
        ]
        T, _ = rigid_fit(T, fr_t, t_targets, shift=shift)

    # pick the symmetry-equivalent mapping + conformer of each molecule on the
    # other side that best overlays the template (e.g. s-cis butadiene for D-A)
    confs = [conformers(s, n_confs, seed) for s in O.info["smiles"]]
    maps = [
        to_tmap(m) for m in itertools.islice(gm.isomorphisms_iter(), max_isomorphisms)
    ]

    def best_confs(tm):
        out = []
        for idx, cs in zip(fr_o, confs):
            fits = [_kabsch(c, T.positions[tm[idx]]) for c in cs]
            out.append(min(fits, key=lambda f: f[1]))
        return out

    tmap = min(maps, key=lambda tm: sum(r for _, r in best_confs(tm)))
    ref = T.positions[tmap]
    O = O.copy()
    for idx, (moved, _) in zip(fr_o, best_confs(tmap)):
        O.positions[idx] = moved

    # then separate the molecules rigidly
    o_targets = [
        (i, j)
        for i, j in itertools.combinations(range(len(O)), 2)
        if G_t.has_edge(tmap[i], tmap[j]) and not G_o.has_edge(i, j)
    ]
    O, _ = rigid_fit(O, fr_o, o_targets, ref=ref, shift=shift)

    # put both in reactant atom order
    if prod_is_tmpl:
        reac, prod = O, T[tmap]
    else:
        inv = np.argsort(tmap)
        reac, prod = T, O[inv]
    prod.info = dict(P.info)
    reac.positions -= reac.get_center_of_mass()
    minimize_rotation_and_translation(reac, prod)
    for a in (reac, prod):
        a.info.update(
            formed=[tuple(map(int, b)) for b in fb],
            broken=[tuple(map(int, b)) for b in bb],
        )
    if verbose:
        sym = reac.get_chemical_symbols()
        lab = lambda b: f"{sym[b[0]]}{b[0]}-{sym[b[1]]}{b[1]}"
        print(
            f"{' + '.join(reactants)} -> {' + '.join(products)} | "
            f"form: {[lab(b) for b in fb]}  break: {[lab(b) for b in bb]}"
        )
    return reac, prod
