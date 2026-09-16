"""
examples.py
-----------
Ready-to-run examples for:
  1. CO + H2O  (water-gas shift)
  2. 2NO + O2  (NO oxidation — targets N2O2, NO3, ONOO·)

Each example shows two usage modes:
  A) Full pipeline: run MD + analyse in one call
  B) Post-hoc analysis: analyse an existing trajectory file

Replace the calculator with your real one (MACE, ReaxFF, etc.).
The ASE EMT calculator is used below only so these examples import
without requiring a GPU or ReaxFF installation.
"""

import numpy as np
from ase import Atoms, units
from ase.build import molecule

# ============================================================
# Helper: build a simulation cell packed with reactant copies
# ============================================================


def build_cell(molecules_dict: dict[str, int], cell_size: float = 12.0) -> Atoms:
    """
    Pack N copies of each molecule into a cubic cell.

    Parameters
    ----------
    molecules_dict : {ase_molecule_name: count}, e.g. {'CO': 8, 'H2O': 8}
    cell_size      : cubic cell edge in Å

    Returns
    -------
    ASE Atoms object with PBC
    """
    from ase.build import molecule as ase_mol
    rng = np.random.default_rng(42)
    all_atoms = None

    for mol_name, count in molecules_dict.items():
        template = ase_mol(mol_name)
        for _ in range(count):
            m = template.copy()
            # Random placement inside cell with 2 Å buffer
            buf = 2.0
            pos = rng.uniform(buf, cell_size - buf, size=3)
            m.translate(pos - m.get_center_of_mass())

            if all_atoms is None:
                all_atoms = m
            else:
                all_atoms += m

    all_atoms.set_cell([cell_size, cell_size, cell_size])
    all_atoms.set_pbc(True)
    return all_atoms


# ============================================================
# Example 1: CO + H2O  (water-gas shift)
# ============================================================

def example_co_h2o_full_run():
    """
    Full MD + analysis pipeline for CO + H2O → CO2 + H2.
    Targets: HOCO·, COOH (cis/trans), HCO, H2CO3
    """
    from nano_reactor import NanoReactor

    # --- Build your atoms object (or load from file) ---
    atoms = build_cell({"CO": 8, "H2O": 8}, cell_size=12.0)
    print(f"Cell: {len(atoms)} atoms, {atoms.get_chemical_formula()}")

    # --- Attach your calculator ---
    # Replace this with your real calculator, e.g.:
    #   from mace.calculators import MACECalculator
    #   calc = MACECalculator(model_paths=["my_model.model"], device="cuda")
    from ase.calculators.emt import EMT   # placeholder only
    calc = EMT()

    # --- Create and run the reactor ---
    reactor = NanoReactor(
        atoms=atoms,
        calculator=calc,
        temperature=3000,           # K — elevated for timescale compression
        pressure_GPa=0.0,           # NVT (Langevin)
        total_time_fs=5000,       # 50 ps
        timestep_fs=0.25,
        friction=0.01,

        # Intermediate detection
        tau_min_fs=100.0,           # 100 fs minimum lifetime
        min_heavy_atoms=2,          # exclude bare H·, O· etc.
        gap_tolerance=2,            # merge runs split by ≤2 frames
        bo_cutoffs={},              # use defaults (0.3 for all bonds)

        # Trajectory & analysis
        analysis_stride=1,          # analyse every MD step
        traj_stride=20,             # write traj every 20 steps (5 fs)
        transition_window=10,       # frames window for edge detection
        min_edge_count=3,           # prune rarely-observed transitions

        # Chemistry
        reactants={"CO", "H2O"},
        products={"CO2", "H2"},
        min_traj_fraction=0.01,     # must appear in ≥1% of frames

        output_dir="output_co_h2o",
        verbose=True,
    )

    reactor.run()
    reactor.report()
    reactor.plot(show=False)

    # Save CSV
    reactor.save_summary_csv()

    # Access results programmatically
    intermediates = reactor.get_intermediates()
    neb_pairs = reactor.get_neb_pairs()

    print(f"\nFound {len(intermediates)} intermediates")
    print(f"Generated {len(neb_pairs)} NEB pairs")
    return reactor


def example_co_h2o_posthoc(trajectory_path: str):
    """
    Post-hoc analysis: you already have a trajectory file.
    No calculator needed.
    """
    from nano_reactor import NanoReactor
    from ase.calculators.emt import EMT

    # Atoms object just for metadata — not used for MD here
    atoms = build_cell({"CO": 8, "H2O": 8}, cell_size=12.0)

    reactor = NanoReactor(
        atoms=atoms,
        calculator=EMT(),          # not used in post-hoc mode
        tau_min_fs=100.0,
        min_heavy_atoms=2,
        reactants={"CO", "H2O"},
        products={"CO2", "H2"},
        output_dir="output_co_h2o_posthoc",
        verbose=True,
    )

    # Analyse the existing trajectory
    reactor.analyse_trajectory(trajectory_path, stride=1)
    reactor.report()
    reactor.plot(show=False)
    return reactor


# ============================================================
# Example 2: 2NO + O2  (NO oxidation)
# ============================================================

def example_no_oxidation_full_run():
    """
    Full MD + analysis pipeline for 2NO + O2 → 2NO2.
    Targets: N2O2 (dimer), NO3, N2O3, ONOO· (peroxynitrite).

    Key difference from CO/H2O: we lower the O-O bond-order cutoff
    to 0.25 to catch the weak O-O bond in ONOO·.
    """
    from nano_reactor import NanoReactor

    atoms = build_cell({"NO": 12, "O2": 6}, cell_size=14.0)
    print(f"Cell: {len(atoms)} atoms, {atoms.get_chemical_formula()}")

    from ase.calculators.emt import EMT   # replace with your real calculator
    calc = EMT()

    # Bond-order cutoff override: O-O bonds need lower threshold
    # to catch ONOO· (peroxynitrite) and NO3
    bo_overrides = {
        frozenset({"O", "O"}): 0.25,   # weaker than default 0.3
        frozenset({"N", "O"}): 0.28,   # N-O in NO3, ONOO· can be stretched
        frozenset({"N", "N"}): 0.30,   # N-N in N2O2 dimer
    }

    reactor = NanoReactor(
        atoms=atoms,
        calculator=calc,
        temperature=2500,            # slightly lower — NOx less stable at 3000 K
        pressure_GPa=0.0,
        total_time_fs=5000,       # 100 ps to capture N2O2 dimer events
        timestep_fs=0.25,
        friction=0.01,

        # Intermediate detection — slightly looser to catch short-lived N2O2
        tau_min_fs=50.0,             # N2O2 dimer can be short-lived
        min_heavy_atoms=2,
        gap_tolerance=3,             # N2O2 can flicker — be generous
        bo_cutoffs=bo_overrides,

        analysis_stride=1,
        traj_stride=20,
        transition_window=15,        # wider window for slower NOx chemistry
        min_edge_count=2,

        reactants={"NO", "O2"},
        products={"NO2"},
        min_traj_fraction=0.005,     # 0.5% — NOx intermediates can be rare

        output_dir="output_no_oxidation",
        verbose=True,
    )

    reactor.run()
    reactor.report()
    reactor.plot(show=False)
    reactor.save_summary_csv()

    return reactor


# ============================================================
# Example 3: Lightweight — just the tracker, no MD
# ============================================================

def example_tracker_only(frames):
    """
    Use SpeciesTracker standalone if you manage MD yourself.

    Parameters
    ----------
    frames : list of ASE Atoms objects (your trajectory)
    """
    from species_tracker import SpeciesTracker
    from reaction_network import ReactionNetwork

    tracker = SpeciesTracker(
        timestep_fs=0.25,
        tau_min_fs=100.0,
        min_heavy_atoms=2,
        bo_cutoffs={frozenset({"O", "O"}): 0.25},
        gap_tolerance_frames=2,
        verbose=True,
    )

    network = ReactionNetwork(
        transition_window_frames=10,
        min_edge_count=2,
        verbose=True,
    )

    for frame_idx, atoms in enumerate(frames):
        formulae = tracker.add_frame(atoms, frame_index=frame_idx)
        network.add_frame_species(frame_idx, formulae)

    # Build network edges
    network._derive_edges()

    # Get results
    intermediates = tracker.get_intermediates(
        reactants={"CO", "H2O"},
        products={"CO2", "H2"},
    )

    neb_pairs = network.get_neb_pairs()
    network.print_network()

    return intermediates, neb_pairs


# ============================================================
# Example 4: Using explicit ReaxFF bond orders
# ============================================================

def parse_reaxff_bonds(bonds_file: str, n_atoms: int):
    """
    Parse a ReaxFF bonds.reax file and return per-frame bond lists.

    Each entry in the output is a list of (i, j, bo) tuples (0-indexed).
    Pass these as explicit_bonds to tracker.add_frame().

    File format (LAMMPS ReaxFF output):
        # Frame N
        atom_id  nbonds  partner1 bo1 partner2 bo2 ...
    """
    frames = []
    current = []
    with open(bonds_file) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                if current:
                    frames.append(current)
                    current = []
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            atom_i = int(parts[0]) - 1   # 0-index
            nbonds = int(parts[1])
            for k in range(nbonds):
                atom_j = int(parts[2 + 2*k]) - 1
                bo = float(parts[3 + 2*k])
                if atom_i < atom_j:   # avoid double-counting
                    current.append((atom_i, atom_j, bo))
    if current:
        frames.append(current)
    return frames


if __name__ == "__main__":
    print("Running CO + H2O minimal test (EMT placeholder)...")
    # This won't produce chemically meaningful results with EMT,
    # but exercises the full code path.
    #reactor = example_co_h2o_full_run()
    reactor =example_no_oxidation_full_run() 
    print("\nDone.")
