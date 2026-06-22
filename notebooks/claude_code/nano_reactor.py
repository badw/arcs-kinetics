"""
nano_reactor.py
---------------
Top-level NanoReactor class.  Give it an ASE Atoms object (your seeded
simulation cell) and it handles:

  1. Running reactive MD with a supplied ASE Calculator
     (MACE, NequIP, ReaxFF via ASE interface, or any other)
  2. On-the-fly intermediate detection via SpeciesTracker
  3. Reaction network construction via ReactionNetwork
  4. Summary reporting and plot generation

Minimal example
---------------
    from ase.io import read
    from mace.calculators import MACECalculator

    atoms = read("my_cell.xyz")
    calc  = MACECalculator(model_paths=["mace_model.model"], device="cuda")

    reactor = NanoReactor(
        atoms        = atoms,
        calculator   = calc,
        temperature  = 3000,      # K — elevated to compress timescales
        pressure_GPa = 0.0,       # NVT by default; set > 0 for NPT
        total_time_fs= 100_000,   # 100 ps
        timestep_fs  = 0.25,
        tau_min_fs   = 100.0,
        reactants    = {"CO", "H2O"},
        products     = {"CO2", "H2"},
    )
    reactor.run()
    reactor.report()
    reactor.plot()
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
from ase import Atoms, units
from ase.md.langevin import Langevin
from ase.md.nptberendsen import NPTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.io import Trajectory, write

from claude_code.species_tracker import SpeciesTracker
from claude_code.reaction_network import ReactionNetwork


class NanoReactor:
    """
    Reactive nano-reactor: run MD → detect intermediates → build network.

    Parameters
    ----------
    atoms              : ASE Atoms — your pre-built simulation cell with
                         reactant molecules already placed.
    calculator         : any ASE-compatible calculator (MACE, ReaxFF, etc.)
    temperature        : MD temperature in K (elevated for timescale compression)
    pressure_GPa       : target pressure in GPa; 0 = NVT (Langevin thermostat)
    total_time_fs      : total simulation time in fs
    timestep_fs        : MD integration timestep in fs
    friction           : Langevin friction coefficient (fs⁻¹)
    tau_min_fs         : minimum species lifetime to call it an intermediate
    bo_cutoffs         : per-bond-type bond-order overrides (see SpeciesTracker)
    min_heavy_atoms    : minimum heavy-atom count for an intermediate
    gap_tolerance      : merge runs split by ≤ this many frames (see SpeciesTracker)
    analysis_stride    : analyse bond topology every N MD steps (default 1)
    traj_stride        : write trajectory every N MD steps
    transition_window  : frames within which a disappearance+appearance = transition
    min_edge_count     : minimum observations to keep a network edge
    reactants          : set of formula strings to exclude from intermediates
    products           : set of formula strings to exclude from intermediates
    min_traj_fraction  : minimum fraction of frames a species must appear in
    output_dir         : directory for trajectory, logs, and plots
    skin               : extra Å tolerance for distance-based bond detection
    verbose            : print progress
    """

    def __init__(
        self,
        atoms: Atoms,
        calculator,
        temperature: float = 3000.0,
        pressure_GPa: float = 0.0,
        total_time_fs: float = 100_000.0,
        timestep_fs: float = 0.25,
        friction: float = 0.01,
        tau_min_fs: float = 100.0,
        bo_cutoffs: Optional[dict] = None,
        min_heavy_atoms: int = 2,
        gap_tolerance: int = 2,
        analysis_stride: int = 1,
        traj_stride: int = 10,
        transition_window: int = 10,
        min_edge_count: int = 2,
        reactants: Optional[set[str]] = None,
        products: Optional[set[str]] = None,
        min_traj_fraction: float = 0.0,
        output_dir: str = "nano_reactor_output",
        skin: float = 0.2,
        verbose: bool = True,
    ):
        self.atoms = atoms.copy()
        self.calculator = calculator
        self.temperature = temperature
        self.pressure_GPa = pressure_GPa
        self.total_time_fs = total_time_fs
        self.timestep_fs = timestep_fs
        self.friction = friction
        self.tau_min_fs = tau_min_fs
        self.bo_cutoffs = bo_cutoffs or {}
        self.min_heavy_atoms = min_heavy_atoms
        self.gap_tolerance = gap_tolerance
        self.analysis_stride = analysis_stride
        self.traj_stride = traj_stride
        self.transition_window = transition_window
        self.min_edge_count = min_edge_count
        self.reactants = reactants or set()
        self.products = products or set()
        self.min_traj_fraction = min_traj_fraction
        self.output_dir = Path(output_dir)
        self.skin = skin
        self.verbose = verbose

        self._n_steps = int(total_time_fs / timestep_fs)
        self._analysis_steps = 0

        self.tracker = SpeciesTracker(
            timestep_fs=timestep_fs * analysis_stride,
            tau_min_fs=tau_min_fs,
            min_heavy_atoms=min_heavy_atoms,
            bo_cutoffs=bo_cutoffs,
            gap_tolerance_frames=gap_tolerance,
            skin=skin,
            verbose=verbose,
        )
        self.network = ReactionNetwork(
            transition_window_frames=transition_window,
            min_edge_count=min_edge_count,
            verbose=verbose,
        )

        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # MD setup
    # ------------------------------------------------------------------

    def _setup_md(self):
        self.atoms.calc = self.calculator
        MaxwellBoltzmannDistribution(
            self.atoms, temperature_K=self.temperature)

        dt = self.timestep_fs * units.fs
        if self.pressure_GPa > 0:
            self._md = NPTBerendsen(
                self.atoms,
                timestep=dt,
                temperature_K=self.temperature,
                pressure_au=self.pressure_GPa * units.GPa,
                taut=100 * units.fs,
                taup=500 * units.fs,
                compressibility_au=4.57e-5 / units.bar,
            )
        else:
            self._md = Langevin(
                self.atoms,
                timestep=dt,
                temperature_K=self.temperature,
                friction=self.friction / units.fs,
            )

    # ------------------------------------------------------------------
    # Per-step callback
    # ------------------------------------------------------------------

    def _analysis_callback(self):
        """Called every analysis_stride MD steps."""
        frame_idx = self._analysis_steps
        formulae = self.tracker.add_frame(self.atoms, frame_index=frame_idx)
        self.network.add_frame_species(frame_idx, formulae)
        self._analysis_steps += 1

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Run the full MD + analysis pipeline."""
        t0 = time.time()
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"NanoReactor starting")
            print(f"  {len(self.atoms)} atoms | T={self.temperature} K | "
                  f"{'NVT' if self.pressure_GPa == 0 else f'NPT {self.pressure_GPa} GPa'}")
            print(
                f"  {self._n_steps} steps × {self.timestep_fs} fs = {self.total_time_fs/1000:.1f} ps")
            print(f"  τ_min = {self.tau_min_fs} fs | "
                  f"BO cutoffs = {self.bo_cutoffs or 'defaults (0.3)'}")
            print(f"  Output → {self.output_dir}")
            print(f"{'='*60}\n")

        self._setup_md()

        # Trajectory writer
        traj_path = str(self.output_dir / "trajectory.traj")
        traj_writer = Trajectory(traj_path, "w", self.atoms)

        step_counter = [0]

        def combined_callback():
            step_counter[0] += 1
            if step_counter[0] % self.traj_stride == 0:
                traj_writer.write(self.atoms)
            if step_counter[0] % self.analysis_stride == 0:
                self._analysis_callback()

        self._md.attach(combined_callback, interval=1)

        if self.verbose:
            print("Running MD...")
        self._md.run(self._n_steps)
        traj_writer.close()

        elapsed = time.time() - t0
        if self.verbose:
            print(f"\nMD complete in {elapsed:.1f} s  "
                  f"({elapsed/self._n_steps*1000:.2f} ms/step)")

        # Build network edges from accumulated frame data
        if self.verbose:
            print("\nBuilding reaction network...")
        self.network._derive_edges()

    # ------------------------------------------------------------------
    # Analysis without running MD (use pre-existing trajectory)
    # ------------------------------------------------------------------

    def analyse_trajectory(self, trajectory_path: str, stride: int = 1) -> None:
        """
        Run the intermediate detection on an already-computed trajectory
        (ASE .traj file or any format readable by ase.io.read).

        Use this if you already have MD output and just want the analysis.

        Parameters
        ----------
        trajectory_path : path to ASE trajectory or XYZ/extxyz file
        stride          : analyse every N-th frame
        """
        from ase.io import read as ase_read

        if self.verbose:
            print(
                f"\nAnalysing trajectory: {trajectory_path} (stride={stride})")

        frames = ase_read(trajectory_path, index=":")
        for raw_idx, atoms in enumerate(frames):
            if raw_idx % stride != 0:
                continue
            formulae = self.tracker.add_frame(atoms, frame_index=raw_idx)
            self.network.add_frame_species(raw_idx, formulae)

        if self.verbose:
            print(f"  Processed {self.tracker._frame_count} analysis frames")

        self.network._derive_edges()

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def get_intermediates(self) -> dict:
        """Return filtered intermediates dict (formula → SpeciesRecord)."""
        return self.tracker.get_intermediates(
            reactants=self.reactants,
            products=self.products,
            min_trajectory_fraction=self.min_traj_fraction,
        )

    def get_neb_pairs(self) -> list[tuple[str, str, int]]:
        """Return list of (source, target, count) NEB pairs."""
        return self.network.get_neb_pairs(
            reactants=self.reactants,
            products=self.products,
        )

    def report(self) -> None:
        """Print a full human-readable summary to stdout."""
        print(f"\n{'='*60}")
        print("NANO-REACTOR ANALYSIS REPORT")
        print(f"{'='*60}")

        print(f"\nSimulation: {self.total_time_fs/1000:.1f} ps | "
              f"T = {self.temperature} K | "
              f"τ_min = {self.tau_min_fs} fs")

        print(f"\nReactants (excluded): {sorted(self.reactants)}")
        print(f"Products  (excluded): {sorted(self.products)}")

        intermediates = self.get_intermediates()
        if not intermediates:
            print(
                "\n[!] No intermediates found — consider lowering τ_min or BO cutoff")
        else:
            print(f"\nIntermediate species ({len(intermediates)} found):")
            print(f"  {'Formula':>14}  {'Max lifetime (fs)':>18}  "
                  f"{'Appearances':>12}  {'Traj. fraction':>14}")
            print("  " + "-"*64)
            for formula, rec in sorted(
                intermediates.items(),
                key=lambda x: -x[1].max_run_length,
            ):
                frac = rec.total_count / max(1, self.tracker._frame_count)
                print(f"  {formula:>14}  "
                      f"{rec.max_run_length * self.tracker.timestep_fs:>18.1f}  "
                      f"{rec.total_count:>12}  "
                      f"{frac:>14.2%}")

        pairs = self.get_neb_pairs()
        print(f"\nNEB job list ({len(pairs)} pairs):")
        print(f"  {'Source':>14}  →  {'Target':<14}  {'Observed':>8}")
        print("  " + "-"*46)
        for src, tgt, count in pairs[:30]:
            print(f"  {src:>14}  →  {tgt:<14}  {count:>8}")
        if len(pairs) > 30:
            print(f"  ... and {len(pairs)-30} more (see neb_jobs.txt)")

        neb_path = str(self.output_dir / "neb_jobs.txt")
        self.network.export_neb_input(
            neb_path,
            reactants=self.reactants,
            products=self.products,
        )
        print(f"\nFull NEB list written to: {neb_path}")
        print(f"{'='*60}\n")

    def save_summary_csv(self, path: Optional[str] = None) -> str:
        """Save intermediate summary as CSV (requires pandas)."""
        df = self.tracker.summary_dataframe(
            reactants=self.reactants,
            products=self.products,
            min_trajectory_fraction=self.min_traj_fraction,
        )
        out = path or str(self.output_dir / "intermediates.csv")
        df.to_csv(out, index=False)
        if self.verbose:
            print(f"Saved intermediates CSV → {out}")
        return out

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def plot(self, show: bool = True) -> None:
        """Generate and save timeline + network plots."""
        try:
            import matplotlib
            matplotlib.use("Agg") if not show else None
            import matplotlib.pyplot as plt
        except ImportError:
            print("[NanoReactor] matplotlib not available — skipping plots")
            return

        # Timeline
        fig1, _ = self.tracker.plot_timeline(
            reactants=self.reactants,
            products=self.products,
            min_trajectory_fraction=self.min_traj_fraction,
        )
        timeline_path = str(self.output_dir / "timeline.png")
        fig1.savefig(timeline_path, dpi=150, bbox_inches="tight")
        print(f"Timeline plot → {timeline_path}")

        # Network
        try:
            fig2, _ = self.network.plot_network(
                reactants=self.reactants,
                products=self.products,
            )
            net_path = str(self.output_dir / "reaction_network.png")
            fig2.savefig(net_path, dpi=150, bbox_inches="tight")
            print(f"Network plot → {net_path}")
        except Exception as e:
            print(f"[NanoReactor] Network plot failed: {e}")

        if show:
            plt.show()
        plt.close("all")
