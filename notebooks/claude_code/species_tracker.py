"""
species_tracker.py
------------------
Bond-order-based molecular species detection and persistence tracking
for ReaxFF / MACE / NequIP trajectories run through ASE.

Works on any ASE Atoms trajectory — the bond orders are computed via
a simple distance + covalent-radii heuristic when no ReaxFF bond file
is present, or you can supply explicit bond-order arrays.

Key classes
-----------
BondOrderAnalyser   — convert an Atoms frame → molecular graph
SpeciesTracker      — accumulate frames and apply persistence filters
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from ase import Atoms
from ase.data import covalent_radii, atomic_numbers


# ---------------------------------------------------------------------------
# Default bond-order cutoffs by element pair
# ---------------------------------------------------------------------------

# Symmetric: key = frozenset({sym1, sym2})
_DEFAULT_BO_CUTOFFS: dict[frozenset, float] = {
    frozenset({"H", "H"}): 0.3,
    frozenset({"H", "C"}): 0.3,
    frozenset({"H", "N"}): 0.3,
    frozenset({"H", "O"}): 0.3,
    frozenset({"H", "S"}): 0.3,
    frozenset({"C", "C"}): 0.3,
    frozenset({"C", "N"}): 0.3,
    frozenset({"C", "O"}): 0.3,
    frozenset({"N", "N"}): 0.3,
    frozenset({"N", "O"}): 0.3,
    frozenset({"O", "O"}): 0.3,  # catches peroxides, ONOO·
}
_GLOBAL_BO_CUTOFF_DEFAULT = 0.3


def _bo_cutoff(sym1: str, sym2: str, overrides: dict[frozenset, float]) -> float:
    key = frozenset({sym1, sym2})
    return overrides.get(key, _DEFAULT_BO_CUTOFFS.get(key, _GLOBAL_BO_CUTOFF_DEFAULT))


# ---------------------------------------------------------------------------
# Distance-based pseudo bond order
# ---------------------------------------------------------------------------

def _distance_bond_orders(atoms: Atoms, skin: float = 0.2) -> list[tuple[int, int, float]]:
    """
    Estimate bond orders from interatomic distances using covalent radii.

    Returns a list of (i, j, bo) for all pairs within
    (r_cov_i + r_cov_j + skin) Å.  The 'bond order' is a simple
    exponential decay so that it is ~1.0 at the sum of covalent radii
    and falls to ~0.1 at 1.4× that distance.  This is not a ReaxFF BO
    but is sufficient for connectivity analysis.
    """
    from ase.neighborlist import neighbor_list

    symbols = atoms.get_chemical_symbols()
    rcov = np.array([covalent_radii[atomic_numbers[s]] for s in symbols])

    # Build per-pair cutoff dict keyed by (atomic_number_i, atomic_number_j)
    unique_symbols = set(symbols)
    pair_cutoffs = {}
    for s1 in unique_symbols:
        for s2 in unique_symbols:
            z1, z2 = atomic_numbers[s1], atomic_numbers[s2]
            # 1.3× sum of covalent radii catches all normal bonds incl. short O-H
            r_cut = (covalent_radii[z1] + covalent_radii[z2]) * 1.3 + skin
            pair_cutoffs[(z1, z2)] = r_cut

    i_idx, j_idx, d = neighbor_list("ijd", atoms, cutoff=pair_cutoffs)

    bonds = []
    for i, j, dist in zip(i_idx, j_idx, d):
        if i >= j:
            continue
        r_sum = rcov[i] + rcov[j]
        if r_sum < 1e-3:
            continue
        bo = np.exp(-3.0 * (dist / r_sum - 1.0))
        bonds.append((int(i), int(j), float(bo)))
    return bonds


# ---------------------------------------------------------------------------
# Molecular graph from a single frame
# ---------------------------------------------------------------------------

def _union_find_components(n: int, edges: list[tuple[int, int]]) -> list[int]:
    """Return component label per atom using union-find."""
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for a, b in edges:
        union(a, b)
    return [find(i) for i in range(n)]


def _canonical_formula(symbols: list[str]) -> str:
    """Hill-order formula string for a list of element symbols."""
    counts: dict[str, int] = defaultdict(int)
    for s in symbols:
        counts[s] += 1
    ordered = []
    for elem in (["C", "H"] + sorted(s for s in counts if s not in ("C", "H"))):
        if elem in counts:
            n = counts[elem]
            ordered.append(elem if n == 1 else f"{elem}{n}")
    return "".join(ordered)


@dataclass
class MolecularSpecies:
    """One detected molecular species in a single frame."""
    atom_indices: tuple[int, ...]    # indices into the Atoms object
    formula: str                     # Hill-order formula
    heavy_count: int                 # number of non-H atoms
    symbols: tuple[str, ...]


def frame_to_species(
    atoms: Atoms,
    bo_cutoffs: Optional[dict[frozenset, float]] = None,
    explicit_bonds: Optional[list[tuple[int, int, float]]] = None,
    skin: float = 0.2,
) -> list[MolecularSpecies]:
    """
    Convert a single ASE Atoms frame into a list of MolecularSpecies.

    Parameters
    ----------
    atoms          : ASE Atoms object (single frame)
    bo_cutoffs     : optional per-bond-type overrides, e.g.
                     {frozenset({'O','O'}): 0.25}
    explicit_bonds : pre-computed (i, j, bo) list (e.g. from ReaxFF bonds.reax).
                     If None, distances + covalent radii are used.
    skin           : extra distance allowance for distance-based BO (Å)
    """
    overrides = bo_cutoffs or {}
    symbols = atoms.get_chemical_symbols()
    n = len(atoms)

    raw_bonds = explicit_bonds if explicit_bonds is not None else _distance_bond_orders(
        atoms, skin)

    # --- H disambiguation: for each H, keep only bond to highest-BO heavy atom
    h_indices = {i for i, s in enumerate(symbols) if s == "H"}
    h_best: dict[int, tuple[int, float]] = {}  # h_idx -> (heavy_idx, bo)
    non_h_bonds: list[tuple[int, int, float]] = []

    for i, j, bo in raw_bonds:
        si, sj = symbols[i], symbols[j]
        pair_cutoff = _bo_cutoff(si, sj, overrides)
        if bo < pair_cutoff:
            continue
        is_h_i = i in h_indices
        is_h_j = j in h_indices
        if is_h_i and not is_h_j:
            if i not in h_best or bo > h_best[i][1]:
                h_best[i] = (j, bo)
        elif is_h_j and not is_h_i:
            if j not in h_best or bo > h_best[j][1]:
                h_best[j] = (i, bo)
        else:
            non_h_bonds.append((i, j, bo))

    # Build final edge list
    edges: list[tuple[int, int]] = [(i, j) for i, j, _ in non_h_bonds]
    for h_idx, (heavy_idx, _) in h_best.items():
        edges.append((h_idx, heavy_idx))

    # Connected components
    comp_labels = _union_find_components(n, edges)
    components: dict[int, list[int]] = defaultdict(list)
    for atom_idx, comp_id in enumerate(comp_labels):
        components[comp_id].append(atom_idx)

    species_list = []
    for indices in components.values():
        syms = [symbols[i] for i in indices]
        formula = _canonical_formula(syms)
        heavy = sum(1 for s in syms if s != "H")
        species_list.append(MolecularSpecies(
            atom_indices=tuple(sorted(indices)),
            formula=formula,
            heavy_count=heavy,
            symbols=tuple(syms),
        ))
    return species_list


# ---------------------------------------------------------------------------
# Species occurrence across a trajectory
# ---------------------------------------------------------------------------

@dataclass
class SpeciesRecord:
    """Accumulates frame-level observations of one formula."""
    formula: str
    frame_indices: list[int] = field(
        default_factory=list)   # every frame it appeared
    # List of contiguous runs: each run is (start_frame, end_frame, n_frames)
    runs: list[tuple[int, int, int]] = field(default_factory=list)

    @property
    def max_run_length(self) -> int:
        return max((r[2] for r in self.runs), default=0)

    @property
    def total_count(self) -> int:
        return len(self.frame_indices)


def _compute_runs(
    frame_indices: list[int],
    gap_tolerance: int = 2,
) -> list[tuple[int, int, int]]:
    """
    Find contiguous runs in a sorted list of frame indices.

    gap_tolerance: merge runs separated by ≤ this many missing frames.
    Returns list of (start_frame, end_frame, n_frames_in_run).
    """
    if not frame_indices:
        return []
    fi = sorted(frame_indices)
    runs = []
    run_start = fi[0]
    run_end = fi[0]
    for idx in fi[1:]:
        if idx - run_end <= gap_tolerance + 1:
            run_end = idx
        else:
            runs.append((run_start, run_end, run_end - run_start + 1))
            run_start = idx
            run_end = idx
    runs.append((run_start, run_end, run_end - run_start + 1))
    return runs


# ---------------------------------------------------------------------------
# Main tracker class
# ---------------------------------------------------------------------------

class SpeciesTracker:
    """
    Accumulates per-frame species detections across a trajectory and
    applies persistence + frequency filters to identify stable intermediates.

    Usage
    -----
    tracker = SpeciesTracker(
        timestep_fs=0.25,
        tau_min_fs=100.0,
        min_heavy_atoms=2,
        bo_cutoffs={frozenset({'O','O'}): 0.25},
        gap_tolerance_frames=2,
    )
    for frame_atoms in trajectory:
        tracker.add_frame(frame_atoms)

    result = tracker.get_intermediates(reactants={'CO', 'H2O'}, products={'CO2', 'H2'})
    """

    def __init__(
        self,
        timestep_fs: float = 0.25,
        tau_min_fs: float = 100.0,
        min_heavy_atoms: int = 2,
        bo_cutoffs: Optional[dict[frozenset, float]] = None,
        gap_tolerance_frames: int = 2,
        skin: float = 0.2,
        verbose: bool = True,
    ):
        """
        Parameters
        ----------
        timestep_fs          : MD timestep in femtoseconds
        tau_min_fs           : minimum continuous lifetime to count as stable (fs)
        min_heavy_atoms      : minimum number of non-H atoms (filters bare H, O etc.)
        bo_cutoffs           : per-bond-type BO overrides, e.g. {frozenset({'O','O'}): 0.25}
        gap_tolerance_frames : merge runs split by ≤ this many missing frames
        skin                 : extra Å tolerance for distance-based bond detection
        verbose              : print progress to stdout
        """
        self.timestep_fs = timestep_fs
        self.tau_min_fs = tau_min_fs
        self.min_heavy_atoms = min_heavy_atoms
        self.bo_cutoffs = bo_cutoffs or {}
        self.gap_tolerance_frames = gap_tolerance_frames
        self.skin = skin
        self.verbose = verbose

        self._tau_min_frames = max(1, int(np.ceil(tau_min_fs / timestep_fs)))
        self._records: dict[str, SpeciesRecord] = defaultdict(
            lambda: SpeciesRecord(formula="")
        )
        self._frame_count = 0
        self._explicit_bonds_cache: Optional[list[tuple[int, int, float]]] = None

    # ------------------------------------------------------------------
    # Frame ingestion
    # ------------------------------------------------------------------

    def add_frame(
        self,
        atoms: Atoms,
        explicit_bonds: Optional[list[tuple[int, int, float]]] = None,
        frame_index: Optional[int] = None,
    ) -> list[str]:
        """
        Analyse one MD frame and record detected species.

        Parameters
        ----------
        atoms          : ASE Atoms for this timestep
        explicit_bonds : (i, j, bo) list from ReaxFF bonds.reax; if None,
                         distance-based estimation is used
        frame_index    : override frame counter (useful when subsampling)

        Returns
        -------
        List of formula strings detected in this frame.
        """
        fidx = frame_index if frame_index is not None else self._frame_count
        species = frame_to_species(
            atoms,
            bo_cutoffs=self.bo_cutoffs,
            explicit_bonds=explicit_bonds,
            skin=self.skin,
        )
        detected = []
        for sp in species:
            formula = sp.formula
            rec = self._records[formula]
            rec.formula = formula
            rec.frame_indices.append(fidx)
            detected.append(formula)

        self._frame_count += 1
        if self.verbose and self._frame_count % 500 == 0:
            print(f"  [SpeciesTracker] processed {self._frame_count} frames, "
                  f"{len(self._records)} unique formulae seen")
        return detected

    def add_trajectory(
        self,
        trajectory,
        explicit_bonds_per_frame: Optional[list] = None,
        stride: int = 1,
    ) -> None:
        """
        Convenience method: iterate over an ASE trajectory or list of Atoms.

        Parameters
        ----------
        trajectory             : iterable of ASE Atoms objects
        explicit_bonds_per_frame : optional list of bond lists, one per frame
        stride                 : only analyse every N-th frame
        """
        for raw_idx, atoms in enumerate(trajectory):
            if raw_idx % stride != 0:
                continue
            bonds = None
            if explicit_bonds_per_frame is not None:
                bonds = explicit_bonds_per_frame[raw_idx]
            self.add_frame(atoms, explicit_bonds=bonds, frame_index=raw_idx)

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    def _build_runs(self) -> None:
        """Compute contiguous runs for every record (call once after ingestion)."""
        for rec in self._records.values():
            rec.runs = _compute_runs(
                rec.frame_indices, self.gap_tolerance_frames)

    def get_intermediates(
        self,
        reactants: Optional[set[str]] = None,
        products: Optional[set[str]] = None,
        min_trajectory_fraction: float = 0.0,
    ) -> dict[str, SpeciesRecord]:
        """
        Return filtered intermediates after applying all criteria.

        Parameters
        ----------
        reactants                : set of formula strings to exclude (e.g. {'CO','H2O'})
        products                 : set of formula strings to exclude (e.g. {'CO2','H2'})
        min_trajectory_fraction  : minimum fraction of total frames the species
                                   must appear in (0.0 = no frequency filter)

        Returns
        -------
        dict mapping formula -> SpeciesRecord for all species passing the filters.
        """
        self._build_runs()
        exclude = (reactants or set()) | (products or set())
        min_frames_total = int(min_trajectory_fraction * self._frame_count)

        kept = {}
        for formula, rec in self._records.items():
            if formula in exclude:
                continue
            if rec.heavy_count_for_formula() < self.min_heavy_atoms:
                continue
            if rec.max_run_length < self._tau_min_frames:
                continue
            if rec.total_count < max(1, min_frames_total):
                continue
            kept[formula] = rec

        if self.verbose:
            print(f"\n[SpeciesTracker] Intermediates passing all filters "
                  f"(τ_min={self.tau_min_fs} fs, heavy≥{self.min_heavy_atoms}, "
                  f"freq≥{min_trajectory_fraction:.1%}):")
            for f, r in sorted(kept.items(), key=lambda x: -x[1].max_run_length):
                t_max = r.max_run_length * self.timestep_fs
                print(f"  {f:12s}  max_lifetime={t_max:.1f} fs  "
                      f"appearances={r.total_count}")
        return kept

    def summary_dataframe(
        self,
        reactants: Optional[set[str]] = None,
        products: Optional[set[str]] = None,
        min_trajectory_fraction: float = 0.0,
    ):
        """
        Return a pandas DataFrame summary (requires pandas).
        Columns: formula, max_lifetime_fs, total_appearances,
                 n_runs, trajectory_fraction
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for summary_dataframe()")

        intermediates = self.get_intermediates(
            reactants=reactants,
            products=products,
            min_trajectory_fraction=min_trajectory_fraction,
        )
        rows = []
        for formula, rec in intermediates.items():
            rows.append({
                "formula": formula,
                "max_lifetime_fs": rec.max_run_length * self.timestep_fs,
                "total_appearances": rec.total_count,
                "n_runs": len(rec.runs),
                "trajectory_fraction": rec.total_count / max(1, self._frame_count),
            })
        df = pd.DataFrame(rows).sort_values("max_lifetime_fs", ascending=False)
        return df.reset_index(drop=True)

    def plot_timeline(
        self,
        reactants: Optional[set[str]] = None,
        products: Optional[set[str]] = None,
        min_trajectory_fraction: float = 0.0,
        max_species: int = 20,
        figsize: tuple = (12, 6),
    ):
        """
        Plot a presence/absence timeline for all filtered intermediates.
        Requires matplotlib.
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
        except ImportError:
            raise ImportError("matplotlib is required for plot_timeline()")

        intermediates = self.get_intermediates(
            reactants=reactants,
            products=products,
            min_trajectory_fraction=min_trajectory_fraction,
        )
        # Sort by first appearance
        sorted_items = sorted(
            intermediates.items(),
            key=lambda x: min(x[1].frame_indices),
        )[:max_species]

        total_time_fs = self._frame_count * self.timestep_fs
        fig, ax = plt.subplots(figsize=figsize)

        cmap = plt.cm.get_cmap("tab20", len(sorted_items))
        for row_idx, (formula, rec) in enumerate(sorted_items):
            color = cmap(row_idx)
            for start_f, end_f, _ in rec.runs:
                t_start = start_f * self.timestep_fs
                t_end = end_f * self.timestep_fs
                ax.barh(
                    row_idx, t_end - t_start,
                    left=t_start, height=0.7,
                    color=color, alpha=0.85,
                )

        ax.set_yticks(range(len(sorted_items)))
        ax.set_yticklabels([f for f, _ in sorted_items], fontsize=10)
        ax.set_xlabel("Time (fs)")
        ax.set_xlim(0, total_time_fs)
        ax.set_title(f"Intermediate species timeline  "
                     f"(τ_min={self.tau_min_fs} fs, BO cutoff={_GLOBAL_BO_CUTOFF_DEFAULT})")
        ax.axvline(0, color="k", lw=0.5)
        plt.tight_layout()
        return fig, ax


# ---------------------------------------------------------------------------
# Monkey-patch SpeciesRecord to have heavy_count from formula
# ---------------------------------------------------------------------------

def _heavy_count_for_formula(self) -> int:
    """Count non-H heavy atoms from a Hill formula string."""
    import re
    count = 0
    for elem, n in re.findall(r"([A-Z][a-z]?)(\d*)", self.formula):
        if elem and elem != "H":
            count += int(n) if n else 1
    return count


SpeciesRecord.heavy_count_for_formula = _heavy_count_for_formula
