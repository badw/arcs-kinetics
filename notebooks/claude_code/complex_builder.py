"""
complex_builder.py
------------------
Builds geometrically sensible reactant and product complexes for NEB,
solving the two core problems in automated NEB setup:

  Problem 1 - Atom mapping
    Reactant and product Atoms must have atoms in the SAME ORDER so that
    atom i in the reactant maps to atom i in the product. Solved via
    element-constrained Hungarian assignment on a pairwise distance matrix,
    after Kabsch alignment of each molecular fragment.

  Problem 2 - Reactive geometry
    Molecules must be oriented so forming/breaking bonds are at chemically
    reasonable distances. We identify reactive atom pairs from the mapping,
    then translate/rotate fragments so those atoms face each other at
    ~1.8x their equilibrium bond length and minimise steric clash.

Entry point
-----------
    from complex_builder import ComplexBuilder, ElementPair

    builder = ComplexBuilder()

    # NO + O2 -> NO3
    reactant_complex, product_complex = builder.build(
        reactant_fragments = [no_atoms, o2_atoms],
        product_fragments  = [no3_atoms],
        forming_bond       = ElementPair("N", "O"),
    )
"""

from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation

from ase import Atoms
from ase.data import covalent_radii, atomic_numbers


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class ElementPair:
    """Identifies a bond by the two element symbols it connects."""
    elem1: str
    elem2: str

    def matches(self, s1: str, s2: str) -> bool:
        return (self.elem1 == s1 and self.elem2 == s2) or \
               (self.elem1 == s2 and self.elem2 == s1)


def _rcov(sym: str) -> float:
    return covalent_radii[atomic_numbers[sym]]


def _eq_bond(s1: str, s2: str) -> float:
    return _rcov(s1) + _rcov(s2)


def _center_at_origin(atoms: Atoms) -> Atoms:
    a = atoms.copy()
    a.positions -= a.positions.mean(axis=0)
    return a


def _kabsch_rotate(mobile: np.ndarray, ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Kabsch algorithm.
    Returns (rotated_mobile, rotation_matrix R) such that mobile @ R.T ~ ref.
    Both inputs must be pre-centred (N,3).
    """
    H = mobile.T @ ref
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1.0, 1.0, d])          # fix improper rotation
    R = Vt.T @ D @ U.T
    return mobile @ R.T, R


def _bonded_pairs(atoms: Atoms, scale: float = 1.3) -> list[tuple[int, int]]:
    """
    Return all bonded pairs (i<j) based on covalent radii * scale.
    """
    syms = atoms.get_chemical_symbols()
    pos = atoms.get_positions()
    n = len(atoms)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            thresh = (_rcov(syms[i]) + _rcov(syms[j])) * scale
            if np.linalg.norm(pos[i] - pos[j]) < thresh:
                pairs.append((i, j))
    return pairs


# ---------------------------------------------------------------------------
# Problem 1: Atom mapping
# ---------------------------------------------------------------------------

def map_atoms(
    reactant: Atoms,
    product: Atoms,
    verbose: bool = False,
) -> list[int]:
    """
    Find the permutation `perm` such that reactant[i] corresponds to
    product[perm[i]], preserving element identity and minimising total
    displacement after Kabsch alignment.

    Returns
    -------
    perm : list of length len(reactant)
        product index for each reactant atom
    """
    syms_r = reactant.get_chemical_symbols()
    syms_p = product.get_chemical_symbols()

    if sorted(syms_r) != sorted(syms_p):
        raise ValueError(
            f"Reactant and product have different compositions: "
            f"{sorted(syms_r)} vs {sorted(syms_p)}"
        )

    pos_r = reactant.get_positions()
    pos_p = product.get_positions()
    n = len(reactant)

    # Centre both
    cr = pos_r.mean(axis=0)
    cp = pos_p.mean(axis=0)
    pr = pos_r - cr
    pp = pos_p - cp

    # Kabsch align product onto reactant (initial orientation)
    pp_rot, _ = _kabsch_rotate(pp, pr)

    # Build element-masked cost matrix (inf where elements differ)
    INF = 1e9
    cost = np.full((n, n), INF)
    for i in range(n):
        for j in range(n):
            if syms_r[i] == syms_p[j]:
                cost[i, j] = np.linalg.norm(pr[i] - pp_rot[j])

    row_ind, col_ind = linear_sum_assignment(cost)

    # Check assignment is feasible
    for i, j in zip(row_ind, col_ind):
        if cost[i, j] >= INF:
            raise ValueError(
                f"Cannot find atom mapping: reactant atom {i} ({syms_r[i]}) "
                f"has no matching element in the product."
            )

    perm = list(col_ind)
    if verbose:
        total_disp = sum(
            np.linalg.norm(pr[i] - pp_rot[perm[i]]) for i in range(n)
        )
        print(
            f"[map_atoms] Total displacement after mapping: {total_disp:.4f} Å")
        for i in range(n):
            j = perm[i]
            d = np.linalg.norm(pr[i] - pp_rot[j])
            print(
                f"  R[{i}]({syms_r[i]}) -> P[{j}]({syms_p[j]})  disp={d:.3f} Å")

    return perm


def reorder_product(product: Atoms, perm: list[int]) -> Atoms:
    """
    Return a copy of `product` with atoms reordered so that atom i
    corresponds to atom i in the reactant (using the mapping from map_atoms).
    """
    new_pos = product.get_positions()[perm]
    new_syms = [product.get_chemical_symbols()[j] for j in perm]
    reordered = product.copy()
    reordered.set_positions(new_pos)
    reordered.set_chemical_symbols(new_syms)
    return reordered


# ---------------------------------------------------------------------------
# Identify which bonds form and break
# ---------------------------------------------------------------------------

def _find_bond_changes(
    reactant: Atoms,
    product: Atoms,
    bond_scale: float = 1.3,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """
    Given reactant and product with atoms in the SAME order,
    return (bonds_formed, bonds_broken) as lists of (i, j) index pairs.

    A bond is 'formed'  if it exists in product  but not in reactant.
    A bond is 'broken'  if it exists in reactant but not in product.
    """
    r_bonds = set(_bonded_pairs(reactant, bond_scale))
    p_bonds = set(_bonded_pairs(product,  bond_scale))
    formed = sorted(p_bonds - r_bonds)
    broken = sorted(r_bonds - p_bonds)
    return formed, broken


# ---------------------------------------------------------------------------
# Problem 2: Reactive geometry
# ---------------------------------------------------------------------------

def _clash_energy(positions: np.ndarray, symbols: list[str]) -> float:
    """
    Soft repulsion energy to measure steric clash.
    E = sum_{i<j} max(0, r0_ij - d_ij)^2 where r0_ij = 0.7*(rcov_i + rcov_j).
    """
    n = len(positions)
    E = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(positions[i] - positions[j])
            r0 = 0.7 * (_rcov(symbols[i]) + _rcov(symbols[j]))
            delta = r0 - d
            if delta > 0:
                E += delta ** 2
    return E


def _random_rotation() -> np.ndarray:
    return Rotation.random().as_matrix()


def _place_two_fragments(
    frag_a: Atoms,
    frag_b: Atoms,
    reactive_idx_a: int,
    reactive_idx_b: int,
    approach_distance: float,
    n_orientations: int = 72,
    rng: Optional[np.random.Generator] = None,
) -> tuple[Atoms, Atoms]:
    """
    Orient frag_a and frag_b so that reactive_idx_a (in frag_a) and
    reactive_idx_b (in frag_b) are `approach_distance` apart along the z-axis,
    choosing the rotation of frag_b that minimises steric clash.

    Returns (placed_a, placed_b) both centred so reactive atoms are separated.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    syms_a = frag_a.get_chemical_symbols()
    syms_b = frag_b.get_chemical_symbols()

    # Centre each fragment on its reactive atom
    a = frag_a.copy()
    b = frag_b.copy()
    a.positions -= a.positions[reactive_idx_a]
    b.positions -= b.positions[reactive_idx_b]

    # Place A reactive atom at origin, B reactive atom along +z
    z_vec = np.array([0.0, 0.0, 1.0])
    a_reactive_pos = np.array([0.0, 0.0, 0.0])
    b_reactive_pos = np.array([0.0, 0.0, approach_distance])

    # Shift B so its reactive atom sits at b_reactive_pos
    b.positions += b_reactive_pos

    # Try many orientations of B around z-axis, keep lowest clash
    best_clash = np.inf
    best_b_pos = b.get_positions().copy()

    # Systematic: rotate B around z-axis in n_orientations steps
    for angle in np.linspace(0, 2 * np.pi, n_orientations, endpoint=False):
        R_z = Rotation.from_rotvec(angle * z_vec).as_matrix()
        trial = (b.get_positions() - b_reactive_pos) @ R_z.T + b_reactive_pos
        combined_pos = np.vstack([a.get_positions(), trial])
        combined_syms = syms_a + syms_b
        clash = _clash_energy(combined_pos, combined_syms)
        if clash < best_clash:
            best_clash = clash
            best_b_pos = trial.copy()

    # Also try random orientations in case the systematic scan missed a minimum
    for _ in range(32):
        R_rand = _random_rotation()
        trial = (b.get_positions() - b_reactive_pos) @ R_rand.T + \
            b_reactive_pos
        combined_pos = np.vstack([a.get_positions(), trial])
        combined_syms = syms_a + syms_b
        clash = _clash_energy(combined_pos, combined_syms)
        if clash < best_clash:
            best_clash = clash
            best_b_pos = trial.copy()

    b.set_positions(best_b_pos)
    return a, b


def _place_single_product(frag: Atoms) -> Atoms:
    """Centre a single-fragment product at origin."""
    a = frag.copy()
    a.positions -= a.positions.mean(axis=0)
    return a


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

class ComplexBuilder:
    """
    Builds matched reactant and product Atoms objects for NEB.

    Parameters
    ----------
    cell_size        : cubic cell edge length in Angstrom
    approach_scale   : reactive atom separation = approach_scale * eq_bond_length
    bond_scale       : multiplier on covalent radii to detect bonds
    n_orientations   : number of rotations tried when minimising steric clash
    vacuum           : vacuum padding around the complex (A), added to cell
    verbose          : print progress
    """

    def __init__(
        self,
        cell_size: float = 15.0,
        approach_scale: float = 1.8,
        bond_scale: float = 1.3,
        n_orientations: int = 72,
        vacuum: float = 4.0,
        verbose: bool = True,
    ):
        self.cell_size = cell_size
        self.approach_scale = approach_scale
        self.bond_scale = bond_scale
        self.n_orientations = n_orientations
        self.vacuum = vacuum
        self.verbose = verbose

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[ComplexBuilder] {msg}")

    def _find_reactive_atom(
        self,
        frag: Atoms,
        elem: str,
        prefer_undercoordinated: bool = True,
    ) -> int:
        """
        Find the index of the best reactive atom of a given element in a fragment.
        Prefers atoms with fewer bonds (more undercoordinated = more reactive).
        """
        syms = frag.get_chemical_symbols()
        bonds = _bonded_pairs(frag, self.bond_scale)

        # Count bonds per atom
        bond_count = {i: 0 for i in range(len(frag))}
        for i, j in bonds:
            bond_count[i] += 1
            bond_count[j] += 1

        candidates = [i for i, s in enumerate(syms) if s == elem]
        if not candidates:
            raise ValueError(
                f"Element '{elem}' not found in fragment {frag.get_chemical_formula()}")

        if prefer_undercoordinated:
            return min(candidates, key=lambda i: bond_count[i])
        return candidates[0]

    def build(
        self,
        reactant_fragments: list[Atoms],
        product_fragments: list[Atoms],
        forming_bond: Optional[ElementPair] = None,
        breaking_bond: Optional[ElementPair] = None,
        atom_map_verbose: bool = False,
    ) -> tuple[Atoms, Atoms]:
        """
        Build a geometrically sensible (reactant_complex, product_complex) pair
        with atoms in matched order, ready for NEB interpolation.

        Parameters
        ----------
        reactant_fragments : list of Atoms, one per reactant molecule
            e.g. [no_atoms, o2_atoms]   for  NO + O2
        product_fragments  : list of Atoms, one per product molecule
            e.g. [no3_atoms]            for  NO3
        forming_bond       : ElementPair describing the bond that FORMS
            (used to choose which atoms to orient toward each other)
            e.g. ElementPair("N", "O")
        breaking_bond      : ElementPair describing the bond that BREAKS
            (alternative hint if you have a dissociation)
        atom_map_verbose   : print atom-by-atom mapping details

        Returns
        -------
        (reactant_complex, product_complex) : two Atoms objects
            - same number of atoms, same atom ordering
            - placed in a cubic cell of size self.cell_size
            - reactive atoms within approach_scale * eq_bond_length of each other
        """
        # --- 1. Combine all fragments into one Atoms each side ---
        raw_reactant = self._combine_fragments(reactant_fragments, "reactant")
        raw_product = self._combine_fragments(product_fragments,  "product")

        self._log(f"Reactant: {raw_reactant.get_chemical_formula()}  "
                  f"({len(raw_reactant)} atoms)")
        self._log(f"Product:  {raw_product.get_chemical_formula()}  "
                  f"({len(raw_product)} atoms)")

        # --- 2. Atom mapping (element-constrained Hungarian) ---
        self._log("Solving atom mapping...")
        perm = map_atoms(raw_reactant, raw_product, verbose=atom_map_verbose)
        matched_product = reorder_product(raw_product, perm)

        # --- 3. Identify bond changes ---
        formed, broken = _find_bond_changes(
            raw_reactant, matched_product, self.bond_scale
        )
        syms_r = raw_reactant.get_chemical_symbols()
        self._log(
            f"Bonds formed: {[(syms_r[i], syms_r[j]) for i, j in formed]}")
        self._log(
            f"Bonds broken: {[(syms_r[i], syms_r[j]) for i, j in broken]}")

        # --- 4. Build reactive geometry ---
        reactant_complex = self._build_reactant_complex(
            reactant_fragments, forming_bond, breaking_bond, formed, broken, syms_r
        )
        product_complex = self._build_product_complex(
            product_fragments, raw_reactant, matched_product
        )

        # --- 5. Align product onto reactant (Kabsch) so interpolation is smooth ---
        product_complex = self._align_product_to_reactant(
            reactant_complex, product_complex
        )

        # --- 6. Put both in a cell centred at box midpoint ---
        reactant_complex = self._place_in_cell(reactant_complex)
        product_complex = self._place_in_cell(product_complex)

        self._log("Done.")
        return reactant_complex, product_complex

    # ------------------------------------------------------------------
    # Internal steps
    # ------------------------------------------------------------------

    def _combine_fragments(self, fragments: list[Atoms], label: str) -> Atoms:
        """Concatenate fragments, spacing them 5 Å apart along x."""
        if len(fragments) == 1:
            return _center_at_origin(fragments[0])
        result = None
        x_offset = 0.0
        for frag in fragments:
            f = _center_at_origin(frag)
            span = np.ptp(f.positions[:, 0]) if len(f) > 1 else 1.0
            f.positions[:, 0] += x_offset + span / 2 + 1.5
            x_offset += span + 4.0
            result = f if result is None else result + f
        result.positions -= result.positions.mean(axis=0)
        return result

    def _build_reactant_complex(
        self,
        fragments: list[Atoms],
        forming_bond: Optional[ElementPair],
        breaking_bond: Optional[ElementPair],
        formed: list[tuple[int, int]],
        broken: list[tuple[int, int]],
        syms_r: list[str],
    ) -> Atoms:
        """
        Place reactant fragments with reactive atoms facing each other.
        """
        if len(fragments) == 1:
            # Unimolecular — just centre it
            return _center_at_origin(fragments[0])

        if len(fragments) == 2:
            frag_a, frag_b = [_center_at_origin(f) for f in fragments]
            syms_a = frag_a.get_chemical_symbols()
            syms_b = frag_b.get_chemical_symbols()
            n_a = len(frag_a)

            # Determine which atoms in each fragment are reactive
            # Priority: (1) user-specified forming_bond hint,
            #           (2) atoms involved in formed/broken bonds,
            #           (3) most undercoordinated atom of the right element
            reactive_a, reactive_b = self._pick_reactive_pair(
                frag_a, frag_b, syms_a, syms_b, n_a,
                forming_bond, breaking_bond, formed, broken, syms_r
            )

            eq_d = _eq_bond(syms_a[reactive_a], syms_b[reactive_b])
            app_d = eq_d * self.approach_scale
            self._log(f"  Reactive pair: {syms_a[reactive_a]}(frag_a[{reactive_a}]) "
                      f"-- {syms_b[reactive_b]}(frag_b[{reactive_b}])  "
                      f"target separation = {app_d:.2f} Å")

            placed_a, placed_b = _place_two_fragments(
                frag_a, frag_b, reactive_a, reactive_b,
                approach_distance=app_d,
                n_orientations=self.n_orientations,
            )
            combined = placed_a + placed_b
            combined.positions -= combined.positions.mean(axis=0)
            return combined

        # 3+ fragments: place sequentially, each approaching the growing complex
        self._log("3+ fragments: sequential placement")
        result = _center_at_origin(fragments[0])
        for frag in fragments[1:]:
            f = _center_at_origin(frag)
            syms_f = f.get_chemical_symbols()
            # Heuristic: pick most undercoordinated atom on each side
            elem_hint = forming_bond.elem1 if forming_bond else None
            ri = self._most_exposed(result, elem_hint)
            fi = self._most_exposed(f, elem_hint)
            eq_d = _eq_bond(
                result.get_chemical_symbols()[ri], syms_f[fi]
            )
            placed_r, placed_f = _place_two_fragments(
                result, f, ri, fi,
                approach_distance=eq_d * self.approach_scale,
                n_orientations=self.n_orientations,
            )
            result = placed_r + placed_f
            result.positions -= result.positions.mean(axis=0)
        return result

    def _pick_reactive_pair(
        self, frag_a, frag_b, syms_a, syms_b, n_a,
        forming_bond, breaking_bond, formed, broken, syms_r
    ) -> tuple[int, int]:
        """
        Choose the reactive atom index in frag_a and frag_b.
        Returns (idx_in_frag_a, idx_in_frag_b).
        """
        # Option 1: use the forming-bond hint
        if forming_bond is not None:
            e1, e2 = forming_bond.elem1, forming_bond.elem2
            # Try e1 in A, e2 in B
            cands_a1 = [i for i, s in enumerate(syms_a) if s == e1]
            cands_b2 = [i for i, s in enumerate(syms_b) if s == e2]
            if cands_a1 and cands_b2:
                return self._most_exposed(frag_a, e1), self._most_exposed(frag_b, e2)
            # Try e2 in A, e1 in B
            cands_a2 = [i for i, s in enumerate(syms_a) if s == e2]
            cands_b1 = [i for i, s in enumerate(syms_b) if s == e1]
            if cands_a2 and cands_b1:
                return self._most_exposed(frag_a, e2), self._most_exposed(frag_b, e1)

        # Option 2: use the bond-change list (atoms in formed/broken bonds
        # that come from different fragments: index < n_a is in frag_a)
        bond_changes = formed + broken
        cross_bonds = [(i, j - n_a) for i, j in bond_changes
                       if i < n_a and j >= n_a]
        cross_bonds += [(j - n_a, i) for i, j in bond_changes    # reversed
                        if j < n_a and i >= n_a]
        if cross_bonds:
            ia, ib = cross_bonds[0]
            return ia, ib

        # Option 3: fallback — most undercoordinated heavy atom on each side
        self._log(
            "  No reactive pair hint found; using most undercoordinated atoms")
        heavy_a = [i for i, s in enumerate(syms_a) if s != "H"]
        heavy_b = [i for i, s in enumerate(syms_b) if s != "H"]
        ra = self._most_exposed(frag_a, None)
        rb = self._most_exposed(frag_b, None)
        return ra, rb

    def _most_exposed(self, atoms: Atoms, elem: Optional[str]) -> int:
        """
        Return index of atom with fewest bonds (most undercoordinated).
        If elem is given, restricted to that element; else all heavy atoms.
        """
        syms = atoms.get_chemical_symbols()
        bonds = _bonded_pairs(atoms, self.bond_scale)
        bcount = {i: 0 for i in range(len(atoms))}
        for i, j in bonds:
            bcount[i] += 1
            bcount[j] += 1
        if elem is not None:
            cands = [i for i, s in enumerate(syms) if s == elem]
        else:
            cands = [i for i, s in enumerate(syms) if s != "H"]
        if not cands:
            cands = list(range(len(atoms)))
        return min(cands, key=lambda i: bcount[i])

    def _build_product_complex(
        self,
        product_fragments: list[Atoms],
        reactant_combined: Atoms,
        matched_product: Atoms,
    ) -> Atoms:
        """
        For the product side, use the matched_product geometry directly
        (it already has atoms in the correct order and Kabsch-aligned to reactant).
        """
        return _center_at_origin(matched_product)

    def _align_product_to_reactant(
        self,
        reactant: Atoms,
        product: Atoms,
    ) -> Atoms:
        """
        Kabsch-align product onto reactant so interpolation stays smooth.
        This minimises unnecessary rotation in the NEB path.
        """
        pr = reactant.get_positions()
        pp = product.get_positions()
        cr = pr.mean(axis=0)
        cp = pp.mean(axis=0)
        pp_rot, R = _kabsch_rotate(pp - cp, pr - cr)
        aligned = product.copy()
        aligned.set_positions(pp_rot + cr)
        return aligned

    def _place_in_cell(self, atoms: Atoms) -> Atoms:
        """Put atoms in a cubic cell, centred at the box midpoint."""
        a = atoms.copy()
        a.set_cell([self.cell_size] * 3)
        a.set_pbc(True)
        a.positions += np.array([self.cell_size / 2] * 3)
        return a

    # ------------------------------------------------------------------
    # Convenience: build from a whole reaction route
    # ------------------------------------------------------------------

    def build_all_steps(
        self,
        route: list[dict],
        species_geometries: dict[str, Atoms],
        forming_bonds: Optional[dict[str, ElementPair]] = None,
    ) -> list[dict]:
        """
        Build (reactant_complex, product_complex) for every elementary step
        in a reaction route.

        Parameters
        ----------
        route : list of dicts with keys 'reactants' and 'products',
                each a list of formula strings.
                e.g. [
                  {'reactants': ['NO', 'O2'], 'products': ['NO3']},
                  {'reactants': ['NO3', 'NO'], 'products': ['NO2', 'NO2']},
                ]
        species_geometries : dict mapping formula -> Atoms object
                e.g. {'NO': no_atoms, 'O2': o2_atoms, 'NO3': no3_atoms, ...}
        forming_bonds : optional dict mapping step label -> ElementPair
                e.g. {'NO+O2->NO3': ElementPair('N','O')}

        Returns
        -------
        List of dicts, each with:
          'label'    : str
          'reactant' : Atoms (reactant complex, atoms matched to product)
          'product'  : Atoms (product complex)
        """
        forming_bonds = forming_bonds or {}
        steps_out = []

        for step in route:
            r_frags = [species_geometries[f] for f in step["reactants"]]
            p_frags = [species_geometries[f] for f in step["products"]]
            r_str = "+".join(step["reactants"])
            p_str = "+".join(step["products"])
            label = f"{r_str}->{p_str}"

            fb = forming_bonds.get(label)
            self._log(f"\nBuilding: {label}")

            r_complex, p_complex = self.build(
                reactant_fragments=r_frags,
                product_fragments=p_frags,
                forming_bond=fb,
            )
            steps_out.append({
                "label":    label,
                "reactant": r_complex,
                "product":  p_complex,
            })

        return steps_out


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

def check_complex_pair(
    reactant: Atoms,
    product: Atoms,
    verbose: bool = True,
) -> dict:
    """
    Run sanity checks on a (reactant, product) complex pair before NEB.

    Checks:
      1. Same number of atoms
      2. Same element composition
      3. Atom ordering consistent (element[i] matches)
      4. No severe steric clashes (min interatomic distance)
      5. Atom displacement reasonable (max displacement < 8 Å warns)

    Returns dict of check results.
    """
    syms_r = reactant.get_chemical_symbols()
    syms_p = product.get_chemical_symbols()
    pos_r = reactant.get_positions()
    pos_p = product.get_positions()

    results = {}

    # 1. Same count
    results["same_count"] = len(reactant) == len(product)

    # 2. Same composition
    results["same_composition"] = sorted(syms_r) == sorted(syms_p)

    # 3. Element ordering
    mismatches = [i for i in range(len(syms_r)) if syms_r[i] != syms_p[i]]
    results["ordering_ok"] = len(mismatches) == 0
    results["mismatches"] = mismatches

    # 4. Min distance (clash check)
    n = len(reactant)
    min_d_r = np.inf
    min_d_p = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            min_d_r = min(min_d_r, np.linalg.norm(pos_r[i] - pos_r[j]))
            min_d_p = min(min_d_p, np.linalg.norm(pos_p[i] - pos_p[j]))
    results["min_dist_reactant"] = float(min_d_r)
    results["min_dist_product"] = float(min_d_p)
    results["no_clash"] = min_d_r > 0.5 and min_d_p > 0.5

    # 5. Max atom displacement
    disps = np.linalg.norm(pos_r - pos_p, axis=1)
    results["max_displacement"] = float(disps.max())
    results["mean_displacement"] = float(disps.mean())
    results["displacement_ok"] = float(disps.max()) < 8.0

    if verbose:
        def ok(b): return "OK" if b else "FAIL"
        print(f"\n[ComplexBuilder] Sanity check:")
        print(f"  Same atom count:       {ok(results['same_count'])}")
        print(f"  Same composition:      {ok(results['same_composition'])}")
        print(f"  Atom ordering:         {ok(results['ordering_ok'])}"
              + (f"  (mismatches at {mismatches})" if mismatches else ""))
        print(f"  No steric clash:       {ok(results['no_clash'])}  "
              f"(min dist R={min_d_r:.2f}A  P={min_d_p:.2f}A)")
        print(f"  Max displacement:      {results['max_displacement']:.2f} A  "
              f"{'OK' if results['displacement_ok'] else 'WARN: large'}")

    return results
