"""
neb_runner.py
-------------
Runs CI-NEB for each elementary step using any ASE-compatible GNN calculator
(CHGNet, MACE, NequIP, etc.) and returns forward/reverse barriers.

For each elementary step A + B -> C + D you provide:
  - reactant_complex : ASE Atoms  (A and B placed near each other in a cell)
  - product_complex  : ASE Atoms  (C and D placed near each other)

The runner:
  1. Relaxes both endpoints with the GNN calculator
  2. Interpolates N images along the path (IDPP by default)
  3. Runs CI-NEB to find the saddle point
  4. Returns a NEBResult with Ea_forward, Ea_reverse, deltaE, and the TS geometry
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from ase import Atoms, units
from ase.optimize import FIRE, BFGS
from ase.mep import NEB
from ase.io import write


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class NEBResult:
    """Results from one CI-NEB calculation. Energies in eV."""
    label: str
    Ea_forward: float       # activation energy forward  (eV)
    Ea_reverse: float       # activation energy reverse  (eV)
    delta_E: float          # reaction energy E_prod - E_react (eV)
    E_reactant: float       # absolute energy of relaxed reactant (eV)
    E_product: float        # absolute energy of relaxed product  (eV)
    E_ts: float             # absolute energy of transition state (eV)
    ts_image_index: int
    n_images: int
    image_energies: list[float] = field(default_factory=list)  # eV, relative
    converged: bool = True

    def __str__(self) -> str:
        conv = "converged" if self.converged else "NOT CONVERGED"
        return (
            f"\nNEB result [{conv}]: {self.label}\n"
            f"  delta_E     = {self.delta_E:+.4f} eV\n"
            f"  Ea(forward) = {self.Ea_forward:.4f} eV\n"
            f"  Ea(reverse) = {self.Ea_reverse:.4f} eV\n"
            f"  TS image    = {self.ts_image_index} / {self.n_images - 1}\n"
        )

    def to_dict(self) -> dict:
        return {
            "label":          self.label,
            "Ea_forward_eV":  self.Ea_forward,
            "Ea_reverse_eV":  self.Ea_reverse,
            "delta_E_eV":     self.delta_E,
            "E_reactant_eV":  self.E_reactant,
            "E_product_eV":   self.E_product,
            "E_ts_eV":        self.E_ts,
            "converged":      self.converged,
        }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class NEBRunner:
    """
    Wraps CI-NEB for GNN calculators.

    Parameters
    ----------
    calc_factory  : callable() -> fresh ASE calculator. Called once per NEB
                    image so each image has an independent instance.
                    e.g.  lambda: CHGNetCalculator()
    n_images      : number of images *between* endpoints (total = n + 2)
    fmax_relax    : force convergence for endpoint relaxation (eV/A)
    fmax_neb      : force convergence for NEB (eV/A)
    max_steps_relax / max_steps_neb : iteration caps
    spring_k      : NEB spring constant (eV/A^2)
    climb         : use CI-NEB (recommended True)
    optimizer     : 'FIRE' (default, robust) or 'BFGS'
    interpolation : 'idpp' (better) or 'linear'
    verbose       : print progress
    """

    def __init__(
        self,
        calc_factory: Callable,
        n_images: int = 7,
        fmax_relax: float = 0.05,
        fmax_neb: float = 0.05,
        max_steps_relax: int = 500,
        max_steps_neb: int = 1000,
        spring_k: float = 0.1,
        climb: bool = True,
        optimizer: str = "FIRE",
        interpolation: str = "idpp",
        verbose: bool = True,
    ):
        self.calc_factory    = calc_factory
        self.n_images        = n_images
        self.fmax_relax      = fmax_relax
        self.fmax_neb        = fmax_neb
        self.max_steps_relax = max_steps_relax
        self.max_steps_neb   = max_steps_neb
        self.spring_k        = spring_k
        self.climb           = climb
        self.optimizer       = optimizer
        self.interpolation   = interpolation
        self.verbose         = verbose

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[NEBRunner] {msg}")

    def _make_opt(self, obj, logfile="-", trajectory=None):
        kwargs = dict(logfile=logfile)
        if trajectory:
            kwargs["trajectory"] = trajectory
        return FIRE(obj, **kwargs) if self.optimizer == "FIRE" else BFGS(obj, **kwargs)

    def _relax(self, atoms: Atoms, label: str, out: Path) -> Atoms:
        a = atoms.copy()
        a.calc = self.calc_factory()
        opt = self._make_opt(
            a,
            logfile=str(out / f"{label}_relax.log"),
            trajectory=str(out / f"{label}_relax.traj"),
        )
        conv = opt.run(fmax=self.fmax_relax, steps=self.max_steps_relax)
        e = a.get_potential_energy()
        self._log(f"  {label}: E = {e:.6f} eV  ({'OK' if conv else 'not converged'})")
        return a

    def run(
        self,
        label: str,
        reactant: Atoms,
        product: Atoms,
        output_dir: str = "neb_output",
        relax_endpoints: bool = True,
    ) -> NEBResult:
        """
        Run CI-NEB for one elementary step.

        Parameters
        ----------
        label          : e.g. "NO+O2->NO3"
        reactant       : Atoms object for the reactant complex
        product        : Atoms object for the product complex
        output_dir     : directory for logs / trajectories
        relax_endpoints: relax endpoint geometries first (recommended)
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        self._log(f"{'='*55}")
        self._log(f"NEB: {label}")
        self._log(f"{'='*55}")

        # 1. Relax endpoints
        if relax_endpoints:
            self._log("Relaxing reactant...")
            r = self._relax(reactant, f"{label}_R", out)
            self._log("Relaxing product...")
            p = self._relax(product,  f"{label}_P", out)
        else:
            r = reactant.copy(); r.calc = self.calc_factory()
            p = product.copy();  p.calc = self.calc_factory()

        E_r = r.get_potential_energy()
        E_p = p.get_potential_energy()
        self._log(f"  E_R = {E_r:.6f}  E_P = {E_p:.6f}  dE = {E_p-E_r:+.6f} eV")

        write(str(out / f"{label}_R_relaxed.xyz"), r)
        write(str(out / f"{label}_P_relaxed.xyz"), p)

        # 2. Build images
        n_total = self.n_images + 2
        images  = [r.copy() for _ in range(n_total)]
        images[-1] = p.copy()
        for img in images[1:-1]:
            img.calc = self.calc_factory()
        images[0].calc  = r.calc
        images[-1].calc = p.calc

        neb = NEB(images, k=self.spring_k, climb=False, method="aseneb")

        # 3. Interpolate
        self._log(f"Interpolating ({self.interpolation}, {n_total} images)...")
        neb.interpolate(method=self.interpolation, mic=True)
        write(str(out / f"{label}_initial_path.xyz"), images)

        # 4. Pre-relax (no CI) for a smoother path before turning on climbing
        self._log("Pre-relaxing path...")
        self._make_opt(
            neb, logfile=str(out / f"{label}_pre.log")
        ).run(fmax=max(self.fmax_neb * 3, 0.15), steps=200)

        # 5. CI-NEB
        if self.climb:
            neb.climb = True
            self._log("Running CI-NEB...")
        opt_ci = self._make_opt(
            neb,
            logfile=str(out / f"{label}_ci.log"),
            trajectory=str(out / f"{label}_neb.traj"),
        )
        converged = opt_ci.run(fmax=self.fmax_neb, steps=self.max_steps_neb)
        self._log(f"CI-NEB {'converged' if converged else 'did NOT converge'}")

        # 6. Barriers
        E_abs = [img.get_potential_energy() for img in images]
        E_rel = [e - E_r for e in E_abs]
        ts_idx = int(np.argmax(E_abs))
        E_ts   = E_abs[ts_idx]
        Ea_fwd = max(0.0, E_ts - E_r)
        Ea_rev = max(0.0, E_ts - E_p)

        write(str(out / f"{label}_TS.xyz"),         images[ts_idx])
        write(str(out / f"{label}_final_path.xyz"), images)

        # Energy profile text
        with open(str(out / f"{label}_profile.txt"), "w") as fh:
            fh.write(f"# {label}\n# image  E_rel(eV)\n")
            for i, e in enumerate(E_rel):
                fh.write(f"  {i:3d}   {e:+.6f}\n")
            fh.write(f"# Ea_fwd = {Ea_fwd:.4f}  Ea_rev = {Ea_rev:.4f}  dE = {E_p-E_r:+.4f}\n")

        result = NEBResult(
            label=label, Ea_forward=Ea_fwd, Ea_reverse=Ea_rev,
            delta_E=E_p - E_r, E_reactant=E_r, E_product=E_p,
            E_ts=E_ts, ts_image_index=ts_idx, n_images=n_total,
            image_energies=E_rel, converged=converged,
        )
        self._log(str(result))
        return result

    def run_batch(
        self,
        steps: list[dict],
        output_base: str = "neb_batch",
    ) -> list[NEBResult]:
        """
        Run NEB for a list of elementary steps.

        Each dict in `steps` needs:
          'label'    : str
          'reactant' : Atoms
          'product'  : Atoms
          'relax'    : bool  (optional, default True)
        """
        results = []
        for i, step in enumerate(steps):
            self._log(f"\nStep {i+1}/{len(steps)}: {step['label']}")
            results.append(self.run(
                label=step["label"],
                reactant=step["reactant"],
                product=step["product"],
                output_dir=str(Path(output_base) / step["label"].replace(" ", "_")),
                relax_endpoints=step.get("relax", True),
            ))
        return results


# ---------------------------------------------------------------------------
# Convenience plotter
# ---------------------------------------------------------------------------

def plot_energy_profiles(
    results: list[NEBResult],
    temperature: float = 298.15,
    figsize: tuple = (10, 5),
    save_path: Optional[str] = None,
):
    """Plot NEB energy profiles for all elementary steps side by side."""
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    n   = len(results)
    fig = plt.figure(figsize=figsize)
    gs  = gridspec.GridSpec(1, n, figure=fig, wspace=0.4)
    clr = plt.cm.tab10(np.linspace(0, 1, n))

    for idx, res in enumerate(results):
        ax = fig.add_subplot(gs[idx])
        x  = np.arange(len(res.image_energies))
        ax.plot(x, res.image_energies, "o-", color=clr[idx], linewidth=2, markersize=5)
        ax.axhline(0,          color="grey", lw=0.7, ls="--")
        ax.axhline(res.delta_E, color="grey", lw=0.7, ls=":")
        ts_e = res.image_energies[res.ts_image_index]
        ax.annotate(
            f"Ea={res.Ea_forward:.3f} eV",
            xy=(res.ts_image_index, ts_e),
            xytext=(res.ts_image_index - 0.8, ts_e + abs(ts_e) * 0.1 + 0.02),
            fontsize=8, color=clr[idx],
            arrowprops=dict(arrowstyle="-|>", color=clr[idx], lw=0.8),
        )
        ax.set_title(res.label, fontsize=9)
        ax.set_xlabel("Image")
        if idx == 0:
            ax.set_ylabel("Energy (eV, rel. reactant)")
        conv = "" if res.converged else "\n[not converged]"
        ax.text(0.97, 0.05, f"dE={res.delta_E:+.3f} eV{conv}",
                transform=ax.transAxes, ha="right", fontsize=8, color="grey")

    fig.suptitle(f"NEB profiles  (T={temperature:.0f} K)", fontsize=11)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
