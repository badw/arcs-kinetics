"""
microkinetic_model.py
---------------------
Microkinetic rate modelling from NEB barriers.

Given a sequence of elementary steps with forward/reverse barriers,
this module:

  1. Computes Eyring rate constants k(T) for each step
  2. Builds and integrates the ODE system for species concentrations
  3. Extracts the overall rate, turnover frequency, and rate-determining step
  4. Optionally derives the quasi-steady-state (QSS) analytical rate law
     for simple linear mechanisms

Physical assumptions
--------------------
- Gas-phase, ideal mixture: rate = k × [A] × [B] (collision frequency
  already folded into the pre-exponential via Eyring TST)
- Reversible elementary steps (both k_f and k_r computed from barriers)
- Concentration units: molecules per unit volume (normalised to 1 here;
  pass molar fractions or number densities as initial conditions)
- Temperature-independent barriers (no quantum tunnelling correction by
  default; enable Wigner below)

Eyring equation
---------------
    k(T) = (κ kB T / h) × exp(-Ea / kB T)

where κ is the transmission coefficient (default 1.0).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

# Physical constants (SI)
kB_SI  = 1.380649e-23   # J/K
h_SI   = 6.62607015e-34 # J·s
kB_eV  = 8.617333e-5    # eV/K
NA     = 6.02214076e23  # mol⁻¹

# eV → J
eV_to_J = 1.602176634e-19


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ElementaryStep:
    """
    One reversible elementary step in the mechanism.

    reactants  : list of species names on the left  e.g. ['NO', 'O2']
    products   : list of species names on the right e.g. ['NO3']
    Ea_f_eV    : forward barrier from NEB (eV)
    Ea_r_eV    : reverse barrier from NEB (eV)
    label      : human-readable name
    kappa      : Eyring transmission coefficient (1 = classical, no tunnelling)

    The reaction energy is implied: ΔE = Ea_f - Ea_r
    """
    reactants: list[str]
    products:  list[str]
    Ea_f_eV:   float
    Ea_r_eV:   float
    label:     str = ""
    kappa:     float = 1.0

    def k_forward(self, T: float) -> float:
        """Forward rate constant (s⁻¹ or s⁻¹·[conc]^(n-1)) at temperature T (K)."""
        return _eyring(self.Ea_f_eV, T, self.kappa)

    def k_reverse(self, T: float) -> float:
        return _eyring(self.Ea_r_eV, T, self.kappa)

    def delta_E_eV(self) -> float:
        return self.Ea_f_eV - self.Ea_r_eV

    def equilibrium_constant(self, T: float) -> float:
        """Dimensionless Keq = kf / kr."""
        kr = self.k_reverse(T)
        return self.k_forward(T) / kr if kr > 0 else np.inf

    def summary(self, T: float) -> str:
        kf = self.k_forward(T)
        kr = self.k_reverse(T)
        return (
            f"  {self.label or '+'.join(self.reactants)+'→'+'+'.join(self.products)}\n"
            f"    Ea_f = {self.Ea_f_eV:.3f} eV  Ea_r = {self.Ea_r_eV:.3f} eV  "
            f"ΔE = {self.delta_E_eV():+.3f} eV\n"
            f"    k_f({T:.0f}K) = {kf:.3e} s⁻¹   k_r = {kr:.3e} s⁻¹   "
            f"Keq = {kf/kr if kr else np.inf:.3e}"
        )


@dataclass
class MechanismStep:
    """Associates an ElementaryStep with its stoichiometric coefficients."""
    step: ElementaryStep
    reactant_stoich: dict[str, int]   # e.g. {'NO': 2}
    product_stoich:  dict[str, int]   # e.g. {'N2O2': 1}


@dataclass
class MicrokineticResult:
    """All outputs from a microkinetic integration."""
    T_K: float
    t_span: tuple[float, float]
    species: list[str]
    t: np.ndarray                          # time axis
    concentrations: np.ndarray            # shape (n_species, n_time)
    rates: np.ndarray                     # shape (n_steps, n_time)
    overall_rate: float                   # d[product]/dt at quasi-steady state
    rate_determining_step: int            # index of rate-limiting step
    degree_of_rate_control: np.ndarray   # DRC per step
    converged: bool

    def summary(self, mechanism: list[MechanismStep]) -> str:
        lines = [f"\nMicrokinetic result at T = {self.T_K:.0f} K"]
        lines.append(f"  Overall rate = {self.overall_rate:.3e} (conc/s)")
        rds = mechanism[self.rate_determining_step].step.label or \
              f"Step {self.rate_determining_step+1}"
        lines.append(f"  Rate-determining step: {rds} "
                     f"(DRC = {self.degree_of_rate_control[self.rate_determining_step]:.3f})")
        lines.append(f"  Degree of rate control per step:")
        for i, ms in enumerate(mechanism):
            lbl = ms.step.label or f"Step {i+1}"
            lines.append(f"    [{i+1}] {lbl}: {self.degree_of_rate_control[i]:.3f}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core physics
# ---------------------------------------------------------------------------

def _eyring(Ea_eV: float, T: float, kappa: float = 1.0) -> float:
    """Eyring TST rate constant in s⁻¹ (per-collision units)."""
    if T <= 0:
        return 0.0
    Ea_J = Ea_eV * eV_to_J
    prefactor = kappa * kB_SI * T / h_SI
    exponent  = -Ea_J / (kB_SI * T)
    # Guard against underflow
    if exponent < -700:
        return 0.0
    return prefactor * np.exp(exponent)


def _wigner_tunnelling(Ea_eV: float, T: float, nu_imag_cm1: float = 1000.0) -> float:
    """
    Wigner tunnelling correction factor (multiplicative on Eyring k).
    nu_imag_cm1: imaginary frequency of TS in cm⁻¹ (estimate from TS Hessian).
    Only significant for H-transfer steps at low T.
    """
    nu_Hz = nu_imag_cm1 * 2.998e10     # cm⁻¹ → Hz
    u = h_SI * nu_Hz / (kB_SI * T)
    return 1.0 + u**2 / 24.0


# ---------------------------------------------------------------------------
# ODE system builder
# ---------------------------------------------------------------------------

class MicrokineticModel:
    """
    Build and integrate the microkinetic ODE for a gas-phase mechanism.

    Parameters
    ----------
    mechanism    : list of MechanismStep objects
    species      : list of all species names (determines ODE dimension)
    T_K          : temperature in Kelvin
    wigner       : apply Wigner tunnelling correction
    nu_imag_cm1  : imaginary TS frequencies for Wigner (one per step)

    Example
    -------
    steps = [
        MechanismStep(
            step=ElementaryStep(['NO','O2'], ['NO3'],
                                Ea_f_eV=0.5, Ea_r_eV=1.2, label='NO+O2→NO3'),
            reactant_stoich={'NO':1, 'O2':1},
            product_stoich={'NO3':1},
        ),
        MechanismStep(
            step=ElementaryStep(['NO3','NO'], ['NO2'],
                                Ea_f_eV=0.2, Ea_r_eV=0.8, label='NO3+NO→2NO2'),
            reactant_stoich={'NO3':1, 'NO':1},
            product_stoich={'NO2':2},
        ),
    ]
    model = MicrokineticModel(steps, species=['NO','O2','NO3','NO2'], T_K=600)
    result = model.integrate({'NO':1.0, 'O2':0.5}, t_span=(0, 1e-3))
    print(result.summary(steps))
    """

    def __init__(
        self,
        mechanism: list[MechanismStep],
        species: list[str],
        T_K: float,
        wigner: bool = False,
        nu_imag_cm1: Optional[list[float]] = None,
    ):
        self.mechanism   = mechanism
        self.species     = species
        self.T_K         = T_K
        self.wigner      = wigner
        self.nu_imag_cm1 = nu_imag_cm1 or [1000.0] * len(mechanism)

        self._idx = {s: i for i, s in enumerate(species)}
        self._kf, self._kr = self._compute_rate_constants()

    def _compute_rate_constants(self) -> tuple[list[float], list[float]]:
        kf, kr = [], []
        for i, ms in enumerate(self.mechanism):
            kfi = ms.step.k_forward(self.T_K)
            kri = ms.step.k_reverse(self.T_K)
            if self.wigner:
                w = _wigner_tunnelling(ms.step.Ea_f_eV, self.T_K,
                                       self.nu_imag_cm1[i])
                kfi *= w
                kri *= w
            kf.append(kfi)
            kr.append(kri)
        return kf, kr

    def _net_rate(self, conc: np.ndarray, step_idx: int) -> float:
        """Net rate of step i = kf·∏[reactants]^stoich - kr·∏[products]^stoich."""
        ms = self.mechanism[step_idx]
        rf = self._kf[step_idx]
        rr = self._kr[step_idx]
        for sp, n in ms.reactant_stoich.items():
            c = max(conc[self._idx[sp]], 0.0)
            rf *= c ** n
        for sp, n in ms.product_stoich.items():
            c = max(conc[self._idx[sp]], 0.0)
            rr *= c ** n
        return rf - rr

    def _ode_rhs(self, t: float, conc: np.ndarray) -> np.ndarray:
        """d[species]/dt from all elementary steps."""
        dcdt = np.zeros(len(self.species))
        for i, ms in enumerate(self.mechanism):
            r = self._net_rate(conc, i)
            for sp, n in ms.reactant_stoich.items():
                dcdt[self._idx[sp]] -= n * r
            for sp, n in ms.product_stoich.items():
                dcdt[self._idx[sp]] += n * r
        return dcdt

    def integrate(
        self,
        initial_concentrations: dict[str, float],
        t_span: tuple[float, float] = (0, 1.0),
        n_points: int = 1000,
        method: str = "BDF",        # stiff-safe; use 'RK45' for non-stiff
        rtol: float = 1e-8,
        atol: float = 1e-12,
    ) -> MicrokineticResult:
        """
        Integrate the ODE system.

        Parameters
        ----------
        initial_concentrations : {species: concentration} — species not listed
                                 default to 0.
        t_span  : (t_start, t_end) in seconds
        n_points: number of output timepoints
        method  : scipy ODE solver ('BDF' for stiff systems, recommended)
        """
        c0 = np.zeros(len(self.species))
        for sp, val in initial_concentrations.items():
            if sp in self._idx:
                c0[self._idx[sp]] = val

        t_eval = np.linspace(t_span[0], t_span[1], n_points)

        sol = solve_ivp(
            self._ode_rhs, t_span, c0,
            method=method, t_eval=t_eval,
            rtol=rtol, atol=atol, dense_output=False,
        )

        converged = sol.success
        if not converged:
            warnings.warn(f"ODE integration did not converge: {sol.message}")

        # Per-step net rates at each timepoint
        rates = np.zeros((len(self.mechanism), len(sol.t)))
        for ti, conc in enumerate(sol.y.T):
            for i in range(len(self.mechanism)):
                rates[i, ti] = self._net_rate(conc, i)

        # Overall rate: production rate of the first listed product of last step
        # (use the quasi-steady-state tail — last 10% of trajectory)
        last_step  = self.mechanism[-1]
        first_prod = list(last_step.product_stoich.keys())[0]
        prod_idx   = self._idx[first_prod]
        tail_start = int(0.9 * len(sol.t))
        prod_conc  = sol.y[prod_idx, tail_start:]
        t_tail     = sol.t[tail_start:]

        if len(t_tail) > 1:
            overall_rate = float(np.polyfit(t_tail, prod_conc, 1)[0])
        else:
            overall_rate = 0.0

        # Rate-determining step: step with lowest average net rate in QSS tail
        mean_rates = np.mean(np.abs(rates[:, tail_start:]), axis=1)
        rds = int(np.argmin(mean_rates))

        # Degree of rate control (Campbell DRC): numerical perturbation
        drc = self._compute_drc(c0, t_span, overall_rate,
                                 method, rtol, atol)

        return MicrokineticResult(
            T_K=self.T_K,
            t_span=t_span,
            species=self.species,
            t=sol.t,
            concentrations=sol.y,
            rates=rates,
            overall_rate=overall_rate,
            rate_determining_step=rds,
            degree_of_rate_control=drc,
            converged=converged,
        )

    def _compute_drc(
        self,
        c0: np.ndarray,
        t_span: tuple,
        r_ref: float,
        method: str,
        rtol: float,
        atol: float,
        delta: float = 0.01,       # 1% perturbation on kf and kr
    ) -> np.ndarray:
        """
        Campbell degree of rate control:
          DRC_i = (∂ ln r / ∂ ln k_i) at constant Keq

        Numerical approximation: perturb kf_i and kr_i by +δ simultaneously
        (keeping Keq = kf/kr constant), reintegrate, measure change in rate.
        """
        drc = np.zeros(len(self.mechanism))
        if abs(r_ref) < 1e-30:
            return drc

        for i in range(len(self.mechanism)):
            # Save original
            kf_orig = self._kf[i]
            kr_orig = self._kr[i]

            # Perturb upward (keep Keq constant)
            self._kf[i] = kf_orig * (1 + delta)
            self._kr[i] = kr_orig * (1 + delta)

            t_eval = np.linspace(*t_span, 500)
            sol_p = solve_ivp(self._ode_rhs, t_span, c0,
                              method=method, t_eval=t_eval,
                              rtol=rtol, atol=atol)

            # Restore
            self._kf[i] = kf_orig
            self._kr[i] = kr_orig

            if not sol_p.success:
                drc[i] = 0.0
                continue

            last_step  = self.mechanism[-1]
            first_prod = list(last_step.product_stoich.keys())[0]
            prod_idx   = self._idx[first_prod]
            tail = int(0.9 * len(t_eval))
            prod_p = sol_p.y[prod_idx, tail:]
            t_p    = t_eval[tail:]
            r_p    = float(np.polyfit(t_p, prod_p, 1)[0]) if len(t_p) > 1 else 0.0

            # DRC ≈ (Δr/r) / (Δk/k) = (r_p - r_ref)/r_ref / delta
            drc[i] = ((r_p - r_ref) / r_ref) / delta if abs(r_ref) > 0 else 0.0

        return drc

    # ------------------------------------------------------------------
    # Analytical QSS rate law (linear mechanisms only)
    # ------------------------------------------------------------------

    def qss_rate_law(
        self,
        intermediate_species: list[str],
        initial_concentrations: dict[str, float],
    ) -> str:
        """
        Derive a symbolic quasi-steady-state rate expression for simple
        linear chain mechanisms.  Prints the analytical form and returns
        it as a string.

        Works for mechanisms where intermediates are consumed as fast as
        they are made (d[I]/dt ≈ 0).

        For the 2NO+O2 example:
            QSS on [NO3]: k1f[NO][O2] - k1r[NO3] - k2f[NO3][NO] = 0
            → [NO3]_ss = k1f[NO][O2] / (k1r + k2f[NO])
            → r_overall = k2f[NO3]_ss[NO] = k1f·k2f[NO]²[O2] / (k1r + k2f[NO])
        """
        kf = self._kf
        kr = self._kr

        lines = ["Quasi-steady-state rate law (analytical approximation):"]
        lines.append("")

        # For the specific 2-step case we can give the exact QSS expression
        if len(self.mechanism) == 2:
            ms1 = self.mechanism[0]
            ms2 = self.mechanism[1]

            # Step 1: A + B ⇌ I
            # Step 2: I + C ⇌ products
            r1_specs = list(ms1.reactant_stoich.keys())
            r2_specs = list(ms2.reactant_stoich.keys())

            k1f_val = kf[0]
            k1r_val = kr[0]
            k2f_val = kf[1]
            k2r_val = kr[1]

            # Concentrations at t=0
            c = {sp: initial_concentrations.get(sp, 0.0) for sp in self.species}

            # Symbolic expression
            lines.append(f"Step 1: {' + '.join(ms1.reactants)} ⇌ {' + '.join(ms1.products)}")
            lines.append(f"  k1f = {k1f_val:.3e} s⁻¹   k1r = {k1r_val:.3e} s⁻¹")
            lines.append(f"Step 2: {' + '.join(ms2.reactants)} ⇌ {' + '.join(ms2.products)}")
            lines.append(f"  k2f = {k2f_val:.3e} s⁻¹   k2r = {k2r_val:.3e} s⁻¹")
            lines.append("")

            # Find the intermediate (product of step 1, reactant of step 2)
            intermediates = set(ms1.product_stoich) & set(ms2.reactant_stoich)
            if intermediates:
                I = list(intermediates)[0]

                # QSS: k1f·∏[R1] - k1r·[I] - k2f·[I]·∏[R2_excl_I] = 0
                r2_excl = [s for s in ms2.reactant_stoich if s != I]
                r2_prod = np.prod([c[s] ** ms2.reactant_stoich[s]
                                   for s in r2_excl]) if r2_excl else 1.0
                r1_prod = np.prod([c[s] ** ms1.reactant_stoich[s]
                                   for s in ms1.reactants])

                I_ss_num = k1f_val * r1_prod
                I_ss_den = k1r_val + k2f_val * r2_prod
                I_ss     = I_ss_num / I_ss_den if I_ss_den > 0 else 0.0

                r_qss = k2f_val * I_ss * r2_prod

                lines.append(f"QSS on [{I}]:")
                lines.append(f"  [{I}]_ss = k1f·∏[R1] / (k1r + k2f·∏[R2])")
                lines.append(f"          = {I_ss:.3e}")
                lines.append("")
                lines.append(f"Overall rate (QSS):")
                lines.append(f"  r = k2f · [{I}]_ss · ∏[R2]")
                lines.append(f"    = k1f·k2f·∏[R1]·∏[R2] / (k1r + k2f·∏[R2])")
                lines.append(f"    = {r_qss:.3e}  (at initial concentrations)")

                # Rate-limiting regimes
                if k1r_val > k2f_val * r2_prod:
                    lines.append("")
                    lines.append("  Regime: k1r >> k2f·[R2]  →  step 1 pre-equilibrium")
                    lines.append(f"  r ≈ (k1f/k1r)·k2f·∏[R1]·∏[R2] = Keq1·k2f·∏[all]")
                else:
                    lines.append("")
                    lines.append("  Regime: k2f·[R2] >> k1r  →  step 1 rate-determining")
                    lines.append(f"  r ≈ k1f·∏[R1]  (step 2 fast, step 1 limits)")

        result = "\n".join(lines)
        print(result)
        return result

    # ------------------------------------------------------------------
    # Arrhenius fit across temperatures
    # ------------------------------------------------------------------

    def arrhenius_fit(
        self,
        T_range: tuple[float, float] = (300, 1000),
        n_T: int = 20,
        initial_concentrations: dict[str, float] = None,
        t_span: tuple[float, float] = (0, 1e-3),
    ) -> dict:
        """
        Integrate the microkinetic model at multiple temperatures and fit
        an effective Arrhenius expression to the overall rate:
            r(T) = A · exp(-Ea_eff / kB T)

        Returns dict with Ea_eff_eV, A, T_values, rates.
        """
        T_values = np.linspace(T_range[0], T_range[1], n_T)
        rates    = []
        c0 = initial_concentrations or {}

        for T in T_values:
            model_T = MicrokineticModel(
                self.mechanism, self.species, T_K=float(T),
                wigner=self.wigner,
            )
            try:
                res = model_T.integrate(c0, t_span=t_span, n_points=300)
                rates.append(max(res.overall_rate, 1e-100))
            except Exception:
                rates.append(1e-100)

        rates = np.array(rates)
        inv_T = 1.0 / T_values
        ln_r  = np.log(rates)
        # Linear fit: ln(r) = ln(A) - Ea_eff/(kB·T)
        coeffs = np.polyfit(inv_T, ln_r, 1)
        Ea_eff_eV = -coeffs[0] * kB_eV
        A_prefactor = np.exp(coeffs[1])

        return {
            "Ea_eff_eV":   Ea_eff_eV,
            "A_prefactor": A_prefactor,
            "T_values":    T_values,
            "rates":       rates,
        }


# ---------------------------------------------------------------------------
# Convenience builder from NEBResults
# ---------------------------------------------------------------------------

def mechanism_from_neb_results(
    neb_results: list,         # list of NEBResult from neb_runner.py
    step_definitions: list[dict],
    species: list[str],
    T_K: float,
) -> tuple[MicrokineticModel, list[MechanismStep]]:
    """
    Build a MicrokineticModel directly from a list of NEBResult objects.

    Parameters
    ----------
    neb_results      : list of NEBResult (one per elementary step, in order)
    step_definitions : list of dicts, each with keys:
                       'reactants'       : list[str]  — left-hand species
                       'products'        : list[str]  — right-hand species
                       'reactant_stoich' : dict[str,int]
                       'product_stoich'  : dict[str,int]
                       'label'           : str
    species          : full list of species names (for ODE dimension)
    T_K              : temperature in Kelvin

    Example
    -------
    neb_results = [result_step1, result_step2]
    step_defs = [
        dict(
            reactants=['NO','O2'], products=['NO3'],
            reactant_stoich={'NO':1,'O2':1},
            product_stoich={'NO3':1},
            label='NO+O2→NO3',
        ),
        dict(
            reactants=['NO3','NO'], products=['NO2'],
            reactant_stoich={'NO3':1,'NO':1},
            product_stoich={'NO2':2},
            label='NO3+NO→2NO2',
        ),
    ]
    model, mechanism = mechanism_from_neb_results(
        neb_results, step_defs,
        species=['NO','O2','NO3','NO2'],
        T_K=600,
    )
    """
    assert len(neb_results) == len(step_definitions), \
        "Must supply one NEBResult per step definition"

    mechanism = []
    for neb, defn in zip(neb_results, step_definitions):
        step = ElementaryStep(
            reactants = defn["reactants"],
            products  = defn["products"],
            Ea_f_eV   = neb.Ea_forward_eV,
            Ea_r_eV   = neb.Ea_reverse_eV,
            label     = defn.get("label", ""),
        )
        mechanism.append(MechanismStep(
            step             = step,
            reactant_stoich  = defn["reactant_stoich"],
            product_stoich   = defn["product_stoich"],
        ))

    model = MicrokineticModel(mechanism, species, T_K)
    return model, mechanism
