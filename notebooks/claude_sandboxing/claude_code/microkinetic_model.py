"""
microkinetic_model.py
---------------------
Builds and solves a microkinetic model from NEB barriers.

Takes NEBResult objects (from neb_runner.py) and a list of elementary
steps, then:

  1. Converts barriers -> Eyring rate constants k(T)
  2. Writes the ODE system for all species concentrations
  3. Integrates to steady state with scipy solve_ivp
  4. Extracts overall rate, rate-determining step, and species profiles

Designed to slot directly into the nano_reactor pipeline:

    from neb_runner import NEBResult
    from microkinetic_model import MicroKineticModel, ElementaryStep

    steps = [
        ElementaryStep(
            label      = "NO + O2 -> NO3",
            reactants  = {"NO": 1, "O2": 1},
            products   = {"NO3": 1},
            neb_result = neb_results[0],       # NEBResult from neb_runner
        ),
        ElementaryStep(
            label      = "NO3 + NO -> 2NO2",
            reactants  = {"NO3": 1, "NO": 1},
            products   = {"NO2": 2},
            neb_result = neb_results[1],
        ),
    ]

    model = MicroKineticModel(
        steps            = steps,
        temperature      = 500.0,          # K
        pressure         = 1.0,            # atm
        initial_conc     = {"NO": 1.0, "O2": 0.5},   # mol/L
        target_reaction  = "2NO + O2 -> 2NO2",
    )

    result = model.run()
    result.report()
    result.plot()
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve, root

# Physical constants
_KB = 8.617333e-5   # eV / K     (Boltzmann)
_H = 4.135668e-15  # eV · s     (Planck)
_R = 8.314462e-3   # kJ / mol K (gas constant, for unit notes)
_EV_TO_KJ = 96.4853  # eV -> kJ/mol


# ---------------------------------------------------------------------------
# Concentration units
# ---------------------------------------------------------------------------

class ConcentrationUnit:
    """
    Describes what "1.0" means in your initial_conc dict.

    The key quantity is `alpha`: how many mol/L equals one user unit.
    All internal ODE arithmetic stays in user units; rate constants are
    rescaled by alpha^(molecularity-1) so the ODE is dimensionally consistent
    regardless of what unit you choose.

    Built-in constructors
    ---------------------
    ConcentrationUnit.mol_per_L()           1.0 = 1 mol/L   (default)
    ConcentrationUnit.mol_per_m3()          1.0 = 1 mol/m³
    ConcentrationUnit.ppm(T, P_atm)         1.0 = 1 ppm (ideal gas at T/P)
    ConcentrationUnit.ppb(T, P_atm)         1.0 = 1 ppb
    ConcentrationUnit.ppt(T, P_atm)         1.0 = 1 ppt
    ConcentrationUnit.molecules_per_cm3()   1.0 = 1 molecule/cm³
    ConcentrationUnit.custom(alpha, label)  1.0 = alpha mol/L

    Parameters
    ----------
    alpha : float
        Number of mol/L equal to one user unit.
        e.g. for ppm at STP: alpha ≈ 4.46e-8 mol/L
    label : str
        Human-readable unit string used in reports and plots.
    """

    _NA = 6.02214076e23   # mol^-1
    _R = 8.314462        # J/mol/K
    _ATM = 101325.0        # Pa per atm

    def __init__(self, alpha: float, label: str = "user units"):
        if alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha}")
        self.alpha = float(alpha)   # mol/L per 1 user unit
        self.label = label

    # ------------------------------------------------------------------
    # Built-in constructors
    # ------------------------------------------------------------------

    @classmethod
    def mol_per_L(cls) -> "ConcentrationUnit":
        """1 user unit = 1 mol/L  (SI chemistry default)."""
        return cls(1.0, "mol/L")

    @classmethod
    def mol_per_m3(cls) -> "ConcentrationUnit":
        """1 user unit = 1 mol/m³."""
        return cls(1e-3, "mol/m³")

    @classmethod
    def _ideal_gas_conc_mol_per_L(cls, T: float, P_atm: float) -> float:
        """Total molar concentration of ideal gas at T (K), P (atm) in mol/L."""
        P_Pa = P_atm * cls._ATM
        return (P_Pa / (cls._R * T)) / 1000.0   # mol/m³ -> mol/L

    @classmethod
    def ppm(cls, T: float = 298.15, P_atm: float = 1.0) -> "ConcentrationUnit":
        """1 user unit = 1 ppm (mole fraction 1e-6) of ideal gas at T, P."""
        c_total = cls._ideal_gas_conc_mol_per_L(T, P_atm)
        return cls(c_total * 1e-6, f"ppm @ {T:.0f}K/{P_atm:.2f}atm")

    @classmethod
    def ppb(cls, T: float = 298.15, P_atm: float = 1.0) -> "ConcentrationUnit":
        """1 user unit = 1 ppb (mole fraction 1e-9) of ideal gas at T, P."""
        c_total = cls._ideal_gas_conc_mol_per_L(T, P_atm)
        return cls(c_total * 1e-9, f"ppb @ {T:.0f}K/{P_atm:.2f}atm")

    @classmethod
    def ppt(cls, T: float = 298.15, P_atm: float = 1.0) -> "ConcentrationUnit":
        """1 user unit = 1 ppt (mole fraction 1e-12) of ideal gas at T, P."""
        c_total = cls._ideal_gas_conc_mol_per_L(T, P_atm)
        return cls(c_total * 1e-12, f"ppt @ {T:.0f}K/{P_atm:.2f}atm")

    @classmethod
    def molecules_per_cm3(cls) -> "ConcentrationUnit":
        """1 user unit = 1 molecule/cm³  (atmospheric chemistry convention)."""
        # 1 molecule/cm3 = 1e6 molecule/m3 = 1e6/NA mol/m3 = 1e3/NA mol/L
        alpha = 1e3 / cls._NA
        return cls(alpha, "molecule/cm³")

    @classmethod
    def custom(cls, alpha: float, label: str = "custom") -> "ConcentrationUnit":
        """1 user unit = alpha mol/L."""
        return cls(alpha, label)

    # ------------------------------------------------------------------
    # Conversions
    # ------------------------------------------------------------------

    def to_mol_per_L(self, value: float) -> float:
        """Convert a value in user units -> mol/L."""
        return value * self.alpha

    def from_mol_per_L(self, value: float) -> float:
        """Convert mol/L -> user units."""
        return value / self.alpha

    def k_scale(self, molecularity: int) -> float:
        """
        Factor to multiply a SI rate constant (mol/L base) by so that
        rate = k_scaled * [A]_user * [B]_user ...  gives d[X]_user/dt.

        For a step with `molecularity` reactant molecules:
            k_user = k_SI * alpha^(molecularity - 1)
        """
        return self.alpha ** (molecularity - 1)

    def rate_scale(self) -> float:
        """
        Factor to convert a rate in mol/L/s to user-units/s.
        rate_user = rate_SI / alpha
        """
        return 1.0 / self.alpha

    def __repr__(self) -> str:
        return f"ConcentrationUnit(alpha={self.alpha:.4e}, label='{self.label}')"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ElementaryStep:
    """
    One elementary step in the mechanism.

    Parameters
    ----------
    label     : human-readable name, e.g. "NO + O2 -> NO3"
    reactants : stoichiometry dict, e.g. {"NO": 1, "O2": 1}
    products  : stoichiometry dict, e.g. {"NO3": 1}
    neb_result: NEBResult from neb_runner (provides Ea_forward, Ea_reverse)
    Ea_forward_eV / Ea_reverse_eV : manual barriers if neb_result is None
    A_factor  : pre-exponential factor override (s^-1 or L/mol/s);
                if None, Eyring theory is used: A = kT/h
    """
    label:          str
    reactants:      dict[str, int]
    products:       dict[str, int]
    neb_result:     object = None          # NEBResult | None
    Ea_forward_eV:  Optional[float] = None
    Ea_reverse_eV:  Optional[float] = None
    A_factor:       Optional[float] = None

    def get_barriers(self) -> tuple[float, float]:
        """Return (Ea_fwd, Ea_rev) in eV, from neb_result or manual values."""
        if self.neb_result is not None:
            return self.neb_result.Ea_forward, self.neb_result.Ea_reverse
        if self.Ea_forward_eV is not None and self.Ea_reverse_eV is not None:
            return self.Ea_forward_eV, self.Ea_reverse_eV
        raise ValueError(
            f"Step '{self.label}' has no barrier data. "
            "Provide a neb_result or set Ea_forward_eV / Ea_reverse_eV."
        )

    def get_delta_E(self) -> float:
        """Return ΔE in eV (negative = exothermic)."""
        if self.neb_result is not None:
            return self.neb_result.delta_E
        ea_f, ea_r = self.get_barriers()
        return ea_f - ea_r

    def molecularity(self) -> int:
        """Total number of reactant molecules (1 = unimolecular, 2 = bimolecular)."""
        return sum(self.reactants.values())


@dataclass
class MKMResult:
    """
    Holds all output from a MicroKineticModel run.
    """
    # Setup
    species:        list[str]
    step_labels:    list[str]
    temperature:    float
    pressure:       float
    t_span:         tuple[float, float]

    # Eyring rate constants
    k_forward:      np.ndarray    # shape (n_steps,)
    k_reverse:      np.ndarray

    # Time evolution
    t_points:       np.ndarray
    concentrations: np.ndarray    # shape (n_species, n_timepoints)

    # Steady state
    ss_concentrations:  dict[str, float]
    # net rate of each elementary step (mol/L/s)
    ss_rates:           np.ndarray
    overall_rate:       float         # mol/L/s, rate of overall reaction
    target_species:     Optional[str]  # species used to define overall rate

    # Analysis
    rate_determining_step: str
    degree_of_rate_control: np.ndarray   # DRC per step
    equilibrium_constants:  np.ndarray   # Keq = kf/kr per step
    ode_success:            bool
    conc_unit:              "ConcentrationUnit"

    def report(self) -> None:
        """Print a full human-readable summary."""
        print(f"\n{'='*65}")
        print(f"MICROKINETIC MODEL REPORT")
        print(f"{'='*65}")
        print(f"  T = {self.temperature:.1f} K  |  P = {self.pressure:.3f} atm")
        print(
            f"  Integration: {'converged' if self.ode_success else 'DID NOT CONVERGE'}")

        print(f"\n{'─'*65}")
        print(f"  Rate constants (Eyring, T = {self.temperature:.1f} K)")
        print(f"  {'Step':<28}  {'kf (s⁻¹ or L/mol/s)':>20}  {'kr':>20}")
        print(f"  {'─'*64}")
        for i, lbl in enumerate(self.step_labels):
            print(
                f"  {lbl:<28}  {self.k_forward[i]:>20.4e}  {self.k_reverse[i]:>20.4e}")

        print(f"\n{'─'*65}")
        print(f"  Equilibrium constants  Keq = kf / kr")
        for i, lbl in enumerate(self.step_labels):
            print(f"  {lbl:<28}  Keq = {self.equilibrium_constants[i]:.4e}")

        print(f"\n{'─'*65}")
        print(f"  Steady-state concentrations ({self.conc_unit.label})")
        for sp, c in sorted(self.ss_concentrations.items()):
            print(f"  {sp:<16}  {c:.6e}")

        print(f"\n{'─'*65}")
        print(
            f"  Net elementary step rates at steady state ({self.conc_unit.label} / s)")
        for i, lbl in enumerate(self.step_labels):
            print(f"  {lbl:<28}  {self.ss_rates[i]:>+.4e}")

        print(f"\n{'─'*65}")
        if self.target_species:
            print(f"  Overall rate (d[{self.target_species}]/dt at SS) = "
                  f"{self.overall_rate:.4e} {self.conc_unit.label} / s")
        print(f"  Rate-determining step: {self.rate_determining_step}")

        print(f"\n{'─'*65}")
        print(f"  Degree of Rate Control (Campbell DRC)")
        for i, lbl in enumerate(self.step_labels):
            print(f"  {lbl:<28}  X_RC = {self.degree_of_rate_control[i]:+.4f}")
        print(f"{'='*65}\n")

    def plot(
        self,
        figsize: tuple = (14, 9),
        save_path: Optional[str] = None,
        show: bool = True,
    ):
        """
        Four-panel plot:
          1. Concentration vs time (all species)
          2. Net step rates at steady state
          3. Energy diagram (from Keq and Ea)
          4. Degree of rate control bar chart
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
        except ImportError:
            raise ImportError("matplotlib required for plot()")

        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.38)

        colors = plt.cm.tab10(np.linspace(0, 1, len(self.species)))
        sp_color = {sp: colors[i] for i, sp in enumerate(self.species)}

        # ── Panel 1: concentration vs time ──────────────────────────────
        ax1 = fig.add_subplot(gs[0, 0])
        for i, sp in enumerate(self.species):
            ax1.semilogy(self.t_points, self.concentrations[i],
                         label=sp, color=sp_color[sp], linewidth=1.8)
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Concentration (mol / L)")
        ax1.set_title("Species concentration vs time")
        ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax1.legend(fontsize=8, loc="best")
        ax1.axvline(self.t_points[-1] * 0.9, color="grey",
                    lw=0.7, ls="--", label="SS region")

        # ── Panel 2: net step rates at SS ───────────────────────────────
        ax2 = fig.add_subplot(gs[0, 1])
        colors2 = ["#2196F3" if r >= 0 else "#F44336" for r in self.ss_rates]
        ax2.barh(self.step_labels, self.ss_rates, color=colors2, alpha=0.85)
        ax2.axvline(0, color="black", lw=0.8)
        ax2.set_xlabel("Net rate (mol / L / s)")
        ax2.set_title("Elementary step rates at steady state")
        ax2.tick_params(axis="y", labelsize=8)

        # ── Panel 3: potential energy diagram ───────────────────────────
        ax3 = fig.add_subplot(gs[1, 0])
        _plot_energy_diagram(ax3, self.step_labels,
                             [s.get_barriers()[0] for s in self._steps],
                             [s.get_barriers()[1] for s in self._steps],
                             [s.get_delta_E() for s in self._steps])
        ax3.set_title(
            f"Potential energy diagram  (T = {self.temperature:.0f} K)")

        # ── Panel 4: degree of rate control ─────────────────────────────
        ax4 = fig.add_subplot(gs[1, 1])
        drc_colors = ["#4CAF50" if d > 0 else "#9E9E9E"
                      for d in self.degree_of_rate_control]
        ax4.barh(self.step_labels, self.degree_of_rate_control,
                 color=drc_colors, alpha=0.85)
        ax4.axvline(0, color="black", lw=0.8)
        ax4.axvline(1, color="grey",  lw=0.7, ls="--")
        ax4.set_xlabel("Degree of Rate Control  $X_{RC}$")
        ax4.set_title("Campbell degree of rate control")
        ax4.tick_params(axis="y", labelsize=8)

        fig.suptitle(
            f"Microkinetic model  |  T = {self.temperature:.0f} K  "
            f"|  P = {self.pressure:.2f} atm",
            fontsize=12, fontweight="bold",
        )

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Plot saved → {save_path}")
        if show:
            plt.show()
        return fig

    # Internal reference to steps for plotting
    _steps: list = field(default_factory=list)


def _plot_energy_diagram(ax, labels, ea_fwd, ea_rev, delta_e):
    """Draw a reaction coordinate energy diagram across all steps."""
    energies = [0.0]
    ts_energies = []
    for i in range(len(labels)):
        ts_energies.append(energies[-1] + ea_fwd[i])
        energies.append(energies[-1] + delta_e[i])

    # x positions: each step spans 2 units, TS at midpoint
    x_min = []
    x_ts = []
    x_max = []
    for i in range(len(labels)):
        x_min.append(i * 2.0)
        x_ts.append(i * 2.0 + 1.0)
        x_max.append(i * 2.0 + 2.0)

    x_levels = [0.0] + [x * 2.0 for x in range(1, len(labels) + 1)]
    y_levels = energies

    # Draw levels as horizontal bars
    for i, (x, y) in enumerate(zip(x_levels, y_levels)):
        ax.hlines(y, x - 0.4, x + 0.4, colors="#333", linewidths=2.5)

    # Draw connecting paths through TS
    for i in range(len(labels)):
        xpts = [x_levels[i], x_ts[i], x_levels[i + 1]]
        ypts = [energies[i], ts_energies[i], energies[i + 1]]
        ax.plot(xpts, ypts, "b--", linewidth=1.2, alpha=0.7)
        # Annotate TS barrier
        ax.annotate(
            f"{ea_fwd[i]:.2f} eV",
            xy=(x_ts[i], ts_energies[i]),
            xytext=(x_ts[i] + 0.15, ts_energies[i] + 0.02),
            fontsize=7, color="#1565C0",
        )

    ax.set_xticks(x_levels)
    ax.set_xticklabels(
        ["R"] + [f"I{i+1}" if i < len(labels) - 1 else "P"
                 for i in range(len(labels))],
        fontsize=8,
    )
    ax.set_ylabel("Energy (eV)")
    ax.axhline(0, color="grey", lw=0.5, ls=":")


# ---------------------------------------------------------------------------
# Main model class
# ---------------------------------------------------------------------------

class MicroKineticModel:
    """
    Microkinetic model: ODE integration from Eyring rate constants.

    Parameters
    ----------
    steps            : list of ElementaryStep
    temperature      : temperature in K
    pressure         : pressure in atm (used for ideal-gas concentration
                       conversion if needed; doesn't affect rate constants
                       directly unless you set pressure-dependent A_factor)
    conc_unit        : ConcentrationUnit describing what "1.0" means in
                       initial_conc.  Default: mol/L.
                       e.g. ConcentrationUnit.ppm(T=500, P_atm=1.0)
                            ConcentrationUnit.molecules_per_cm3()
                            ConcentrationUnit.custom(alpha=4e-8, label="ppm")
    initial_conc     : dict mapping species -> initial concentration in the
                       units described by conc_unit.  Species not listed
                       default to 0.
    target_reaction  : overall reaction string e.g. "2NO + O2 -> 2NO2",
                       used to define which species tracks the overall rate.
    target_species   : formula of the species whose d[X]/dt gives the overall
                       rate (e.g. "NO2"). Inferred from target_reaction if None.
    t_end            : integration end time in seconds (default: auto)
    n_points         : number of time points to record
    ss_fraction      : fraction of t_end at which to read steady-state values
    drc_delta        : fractional barrier perturbation for DRC calculation
    verbose          : print progress
    """

    def __init__(
        self,
        steps:            list[ElementaryStep],
        temperature:      float = 500.0,
        pressure:         float = 1.0,
        conc_unit:        "ConcentrationUnit | None" = None,
        initial_conc:     Optional[dict[str, float]] = None,
        target_reaction:  Optional[str] = None,
        target_species:   Optional[str] = None,
        t_end:            Optional[float] = None,
        n_points:         int = 2000,
        ss_fraction:      float = 0.85,
        drc_delta:        float = 0.001,
        verbose:          bool = True,
    ):
        self.steps = steps
        self.temperature = temperature
        self.pressure = pressure
        self.conc_unit = conc_unit if conc_unit is not None else ConcentrationUnit.mol_per_L()
        self.initial_conc = initial_conc or {}
        self.target_reaction = target_reaction
        self.target_species = target_species or self._infer_target()
        self.t_end = t_end
        self.n_points = n_points
        self.ss_fraction = ss_fraction
        self.drc_delta = drc_delta
        self.verbose = verbose

        # Collect all species (preserving order: reactants first, then products)
        seen = {}
        for step in steps:
            for sp in list(step.reactants) + list(step.products):
                if sp not in seen:
                    seen[sp] = len(seen)
        self.species = list(seen)
        self.species_idx = {sp: i for i, sp in enumerate(self.species)}
        self.n_species = len(self.species)
        self.n_steps = len(steps)

    # ------------------------------------------------------------------
    # Eyring rate constants
    # ------------------------------------------------------------------

    def _eyring_k(self, Ea_eV: float) -> float:
        """
        k(T) = (kT/h) * exp(-Ea / kT)

        Units: s^-1 for unimolecular, L/mol/s for bimolecular (the kT/h
        prefactor has units s^-1; for bimolecular steps we don't add a
        concentration factor here — that's handled by the rate expression
        in the ODE).  If you have a surface model you'd replace this with
        the surface Eyring expression.
        """
        kT = _KB * self.temperature      # eV
        prefactor = kT / _H               # s^-1  (~6e12 at 300K)
        return prefactor * np.exp(-Ea_eV / kT)

    def compute_rate_constants(
        self,
        barrier_overrides: Optional[dict[int, tuple[float, float]]] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute forward and reverse rate constants for all steps.

        Parameters
        ----------
        barrier_overrides : optional dict {step_index: (Ea_fwd, Ea_rev)}
            used internally for DRC perturbation.

        Returns
        -------
        k_fwd, k_rev : arrays of shape (n_steps,)
        """
        k_fwd = np.zeros(self.n_steps)
        k_rev = np.zeros(self.n_steps)
        overrides = barrier_overrides or {}

        for i, step in enumerate(self.steps):
            if i in overrides:
                ea_f, ea_r = overrides[i]
            else:
                ea_f, ea_r = step.get_barriers()

            # Rescale so rate constants are consistent with user concentration units.
            # For a bimolecular step in mol/L: k has units L/mol/s.
            # If the user works in units U where 1 U = alpha mol/L:
            #   k_user = k_SI * alpha^(molecularity - 1)
            # This keeps  rate_user = k_user * [A]_user * [B]_user  correct.
            mol = step.molecularity()
            scale_fwd = self.conc_unit.k_scale(mol)
            # Reverse molecularity = number of product molecules
            mol_rev = sum(step.products.values())
            scale_rev = self.conc_unit.k_scale(mol_rev)

            if step.A_factor is not None:
                k_fwd[i] = step.A_factor * \
                    np.exp(-ea_f / (_KB * self.temperature)) * scale_fwd
                k_rev[i] = step.A_factor * \
                    np.exp(-ea_r / (_KB * self.temperature)) * scale_rev
            else:
                k_fwd[i] = self._eyring_k(ea_f) * scale_fwd
                k_rev[i] = self._eyring_k(ea_r) * scale_rev

        return k_fwd, k_rev

    # ------------------------------------------------------------------
    # Stoichiometric matrix
    # ------------------------------------------------------------------

    def _stoich_matrix(self) -> np.ndarray:
        """
        nu[i, j] = net stoichiometry of species j in step i.
        Positive = produced, negative = consumed.
        Shape: (n_steps, n_species).
        """
        nu = np.zeros((self.n_steps, self.n_species))
        for i, step in enumerate(self.steps):
            for sp, coeff in step.reactants.items():
                nu[i, self.species_idx[sp]] -= coeff
            for sp, coeff in step.products.items():
                nu[i, self.species_idx[sp]] += coeff
        return nu

    # ------------------------------------------------------------------
    # ODE system
    # ------------------------------------------------------------------

    def _make_ode(
        self,
        k_fwd: np.ndarray,
        k_rev: np.ndarray,
        nu:    np.ndarray,
    ):
        """
        Return f(t, c) for solve_ivp.

        Rate of step i:
            r_i = k_fwd[i] * prod(c[j]^nu_ij for reactants)
                - k_rev[i] * prod(c[j]^nu_ij for products)

        dc[j]/dt = sum_i  nu[i,j] * r_i
        """
        n_steps = self.n_steps
        n_species = self.n_species

        stoich_r = np.where(nu < 0, -nu, 0.0)   # (n_steps, n_species)
        stoich_p = np.where(nu > 0,  nu, 0.0)

        def f(t, c):
            c = np.maximum(c, 0.0)
            c_row = c[np.newaxis, :]              # (1, n_species)
            rf = k_fwd * np.prod(
                np.where(stoich_r > 0, c_row ** stoich_r, 1.0), axis=1)
            rr = k_rev * np.prod(
                np.where(stoich_p > 0, c_row ** stoich_p, 1.0), axis=1)
            return nu.T @ (rf - rr)

        return f

    # ------------------------------------------------------------------
    # Integration
    # ------------------------------------------------------------------

    def _auto_t_end(self, k_fwd: np.ndarray, k_rev: np.ndarray) -> float:
        """
        Estimate a sensible integration time from the slowest rate constant.
        t_end ~ 100 / k_min  (give slowest step ~100 turnovers to reach SS).
        """
        k_all = np.concatenate([k_fwd, k_rev])
        k_all = k_all[k_all > 0]
        k_slow = k_all.min() if len(k_all) > 0 else 1.0
        t_end = min(max(100.0 / k_slow, 1e-10), 1e12)
        return t_end

    def _integrate(
        self,
        k_fwd:   np.ndarray,
        k_rev:   np.ndarray,
        nu:      np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        """
        Integrate ODEs. Returns (t, c_matrix, success).
        c_matrix shape: (n_species, n_timepoints).
        """
        t_end = self.t_end or self._auto_t_end(k_fwd, k_rev)

        c0 = np.array([
            self.initial_conc.get(sp, 0.0) for sp in self.species
        ])

        ode_f = self._make_ode(k_fwd, k_rev, nu)

        t_start_eval = max(t_end * 1e-6, 1e-20)
        t_end_eval = t_end * (1.0 - 1e-10)
        if t_start_eval >= t_end_eval:
            t_start_eval = t_end_eval * 1e-6
        t_eval = np.logspace(
            np.log10(t_start_eval),
            np.log10(t_end_eval),
            self.n_points,
        )

        sol = solve_ivp(
            ode_f,
            t_span=(0.0, float(t_end)),
            y0=c0,
            method="LSODA",          # auto-stiff; faster than Radau for plotting
            t_eval=t_eval,
            rtol=1e-6,
            atol=1e-10,
            dense_output=False,
        )

        return sol.t, sol.y, sol.success

    # ------------------------------------------------------------------
    # Steady-state analysis
    # ------------------------------------------------------------------

    def _solve_ss(
        self,
        k_fwd: np.ndarray,
        k_rev:  np.ndarray,
        nu:     np.ndarray,
    ) -> tuple[dict[str, float], bool]:
        """
        Solve dc/dt = 0 algebraically for steady-state concentrations.
        Much faster than integrating to SS with Radau.

        Strategy
        --------
        1. Short LSODA pre-run to get into the basin of attraction.
        2. scipy.optimize.root in log-space (hybr, then krylov).
        3. Fallback: full Radau integration (slow, last resort only).
        """
        ode_f = self._make_ode(k_fwd, k_rev, nu)
        c0 = np.maximum(
            np.array([self.initial_conc.get(sp, 0.0) for sp in self.species]),
            1e-30,
        )

        # Pre-integrate to near-SS. Use the SLOWEST rate constant to set
        # the timescale — the fastest would be orders of magnitude too short.
        k_all_ss = np.concatenate([k_fwd, k_rev])
        k_all_ss = k_all_ss[k_all_ss > 0]
        k_slow_ss = k_all_ss.min() if len(k_all_ss) else 1.0
        t_pre = min(max(100.0 / k_slow_ss, 1e-20), 1e10)
        try:
            pre = solve_ivp(ode_f, t_span=(0.0, t_pre), y0=c0,
                            method="LSODA", rtol=1e-6, atol=1e-10)
            c_guess = np.maximum(pre.y[:, -1], 1e-30) if pre.success else c0
        except Exception:
            c_guess = c0

        # Residual in log-space: dc/dt / c = 0
        def residual(log_c):
            c = np.exp(np.clip(log_c, -500, 500))
            return ode_f(0.0, c) / np.maximum(c, 1e-30)

        log_g = np.log(c_guess)

        # Attempt 1: hybr (MINPACK, fast)
        sol = root(residual, log_g, method="hybr",
                   options={"maxfev": 2000, "xtol": 1e-10})
        if sol.success and np.all(np.isfinite(sol.x)):
            c_ss = np.exp(np.clip(sol.x, -500, 500))
            return {sp: float(c_ss[i]) for i, sp in enumerate(self.species)}, True

        # Attempt 2: krylov (more robust)
        sol2 = root(residual, log_g, method="krylov",
                    options={"maxiter": 500, "fatol": 1e-8})
        if sol2.success and np.all(np.isfinite(sol2.x)):
            c_ss = np.exp(np.clip(sol2.x, -500, 500))
            return {sp: float(c_ss[i]) for i, sp in enumerate(self.species)}, True

        # Attempt 3: full Radau fallback
        warnings.warn(
            "Algebraic SS solve failed; falling back to Radau integration.",
            RuntimeWarning,
        )
        t_end = self._auto_t_end(k_fwd, k_rev)
        fb = solve_ivp(ode_f, t_span=(0.0, float(t_end)), y0=c0,
                       method="Radau", rtol=1e-8, atol=1e-12)
        c_end = np.maximum(fb.y[:, -1], 0.0) if fb.success else c0
        return {sp: float(c_end[i]) for i, sp in enumerate(self.species)}, fb.success

    def _steady_state_rates(
        self,
        c_ss: dict[str, float],
        k_fwd: np.ndarray,
        k_rev: np.ndarray,
        nu:    np.ndarray,
    ) -> np.ndarray:
        """Net rate of each elementary step at steady state."""
        c = np.array([c_ss.get(sp, 0.0) for sp in self.species])
        c = np.maximum(c, 0.0)
        rates = np.zeros(self.n_steps)
        for i in range(self.n_steps):
            rf = k_fwd[i]
            for j in range(self.n_species):
                if nu[i, j] < 0:
                    rf *= c[j] ** (-nu[i, j])
            rr = k_rev[i]
            for j in range(self.n_species):
                if nu[i, j] > 0:
                    rr *= c[j] ** nu[i, j]
            rates[i] = rf - rr
        return rates

    def _overall_rate_from_target(
        self,
        c_ss:  dict[str, float],
        k_fwd: np.ndarray,
        k_rev: np.ndarray,
        nu:    np.ndarray,
    ) -> float:
        """
        d[target_species]/dt at steady state.
        For the overall reaction rate, pick a product (positive contribution).
        """
        if self.target_species is None:
            return 0.0
        step_rates = self._steady_state_rates(c_ss, k_fwd, k_rev, nu)
        j = self.species_idx.get(self.target_species)
        if j is None:
            return 0.0
        return float(nu[:, j] @ step_rates)

    # ------------------------------------------------------------------
    # Campbell Degree of Rate Control
    # ------------------------------------------------------------------

    def _compute_drc(
        self,
        c_ss_ref:   dict[str, float],
        k_fwd_ref:  np.ndarray,
        k_rev_ref:  np.ndarray,
        nu:         np.ndarray,
        r_ref:      float,
    ) -> np.ndarray:
        """
        Campbell DRC: X_RC,i = (k_i / r) * (dr / dk_i) at constant Keq.

        Implemented by perturbing Ea_forward by +delta*kT for each step
        while holding Ea_reverse fixed (so Keq changes negligibly),
        re-integrating, and computing the numerical derivative.
        """
        drc = np.zeros(self.n_steps)
        kT = _KB * self.temperature
        delta = self.drc_delta

        if abs(r_ref) < 1e-30:
            return drc

        for i in range(self.n_steps):
            ea_f_orig, ea_r_orig = self.steps[i].get_barriers()
            overrides = {i: (ea_f_orig + delta * kT, ea_r_orig)}
            k_fwd_p, k_rev_p = self.compute_rate_constants(overrides)

            # Algebraic SS — no ODE integration, fast
            c_ss_p, ok_p = self._solve_ss(k_fwd_p, k_rev_p, nu)
            if not ok_p:
                warnings.warn(f"DRC solve failed for step {i}")
                continue

            r_pert = self._overall_rate_from_target(
                c_ss_p, k_fwd_p, k_rev_p, nu)
            d_ln_k = -delta
            if abs(d_ln_k) > 0 and abs(r_ref) > 0:
                drc[i] = ((r_pert - r_ref) / r_ref) / d_ln_k

        return drc

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------

    def run(self) -> MKMResult:
        """
        Run the full microkinetic model and return a MKMResult.
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Microkinetic Model")
            print(
                f"  T = {self.temperature:.1f} K  |  P = {self.pressure:.3f} atm")
            print(f"  Concentration unit: {self.conc_unit.label}  "
                  f"(1 unit = {self.conc_unit.alpha:.4e} mol/L)")
            print(f"  {self.n_steps} elementary steps  |  {self.n_species} species")
            print(f"  Species: {self.species}")
            print(f"{'='*60}")

        nu = self._stoich_matrix()

        # Rate constants
        k_fwd, k_rev = self.compute_rate_constants()

        if self.verbose:
            kT = _KB * self.temperature
            print(
                f"\n  Prefactor kT/h = {kT/_H:.3e} s⁻¹  (T = {self.temperature:.0f} K)")
            for i, step in enumerate(self.steps):
                ea_f, ea_r = step.get_barriers()
                print(f"  {step.label}")
                print(f"    Ea_fwd = {ea_f:.4f} eV  "
                      f"kf = {k_fwd[i]:.4e}  "
                      f"kr = {k_rev[i]:.4e}  "
                      f"Keq = {k_fwd[i]/max(k_rev[i], 1e-300):.4e}")

        # ── Steady-state: fast algebraic solve ────────────────────────
        if self.verbose:
            print("\n  Solving steady state (algebraic)...")
        c_ss, ss_ok = self._solve_ss(k_fwd, k_rev, nu)
        if not ss_ok:
            warnings.warn("Steady-state solve did not fully converge.")
        elif self.verbose:
            print("  Steady state converged.")

        ss_rates = self._steady_state_rates(c_ss, k_fwd, k_rev, nu)
        overall_rate = self._overall_rate_from_target(c_ss, k_fwd, k_rev, nu)

        # ── DRC: N algebraic solves, not N Radau integrations ─────────
        if self.verbose:
            print("  Computing degree of rate control (DRC)...")
        drc = self._compute_drc(c_ss, k_fwd, k_rev, nu, overall_rate)

        # ── Time evolution: LSODA, only for the plot ───────────────────
        if self.verbose:
            t_end_used = self.t_end or self._auto_t_end(k_fwd, k_rev)
            print(
                f"  Building time-evolution profile (LSODA, t={t_end_used:.2e} s)...")
        t_pts, c_mat, success = self._integrate(k_fwd, k_rev, nu)
        if not success:
            warnings.warn(
                "Time-evolution integration did not converge (plot may be rough).")
        rds_idx = int(np.argmax(np.abs(drc)))
        rds = self.steps[rds_idx].label

        keq = np.where(k_rev > 0, k_fwd / k_rev, np.inf)

        result = MKMResult(
            species=self.species,
            step_labels=[s.label for s in self.steps],
            temperature=self.temperature,
            pressure=self.pressure,
            t_span=(float(t_pts[0]), float(t_pts[-1])),
            k_forward=k_fwd,
            k_reverse=k_rev,
            t_points=t_pts,
            concentrations=c_mat,
            ss_concentrations=c_ss,
            ss_rates=ss_rates,
            overall_rate=overall_rate,
            target_species=self.target_species,
            rate_determining_step=rds,
            degree_of_rate_control=drc,
            equilibrium_constants=keq,
            ode_success=success,
            conc_unit=self.conc_unit,
        )
        # Attach steps reference for plotting
        object.__setattr__(result, '_steps', self.steps)

        if self.verbose:
            result.report()

        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _infer_target(self) -> Optional[str]:
        """
        Guess the target product species from target_reaction string.
        e.g. "2NO + O2 -> 2NO2"  ->  "NO2"
        """
        if not self.target_reaction:
            return None
        try:
            _, rhs = self.target_reaction.replace("=", "->").split("->")
            # Take the last product listed
            parts = [p.strip() for p in rhs.split("+")]
            last = parts[-1]
            # Strip leading stoichiometry number
            import re
            m = re.match(r"^\d*\s*([A-Za-z].*)", last)
            return m.group(1).strip() if m else last
        except Exception:
            return None
