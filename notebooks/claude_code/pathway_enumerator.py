"""
pathway_enumerator.py
---------------------
Enumerates chemically valid reaction pathways from a pool of species
(reactants + MD-discovered intermediates + products), constrained so
that each pathway's net stoichiometry exactly matches the overall reaction.

Algorithm
---------
1. Parse the overall reaction into a target stoichiometry vector.
2. Generate all candidate elementary steps from the species pool:
     - unimolecular:  A -> B + C
     - bimolecular:   A + B -> C,  A + B -> C + D
   keeping only atom-balanced steps.
3. Optionally filter to steps that were actually observed in MD
   (via the ReactionNetwork edge list).
4. Find all combinations of steps (with integer multipliers up to
   max_step_count) whose net stoichiometry vector equals the target,
   using integer linear programming (PuLP/CBC).
5. Return ranked Pathway objects — ranked by total number of steps
   (shorter = simpler mechanism).

Usage
-----
    from pathway_enumerator import PathwayEnumerator, parse_reaction

    overall = "2NO + O2 = 2NO2"

    # Species pool: reactants + MD intermediates + products
    pool = ["NO", "O2", "NO2", "NO3", "N2O2", "N2O3", "ONOO"]

    # Optionally: edges actually observed in MD (formula pairs)
    observed_edges = {("NO", "NO3"), ("NO3", "NO2"), ("N2O2", "NO2")}

    enumerator = PathwayEnumerator(
        overall_reaction = overall,
        species_pool     = pool,
        observed_edges   = observed_edges,   # None = don't filter
        max_step_mult    = 3,   # max times any one step can repeat
        max_path_steps   = 4,   # max elementary steps per pathway
        max_paths        = 20,  # stop after this many found
    )

    pathways = enumerator.enumerate()
    enumerator.print_pathways(pathways)
"""

from __future__ import annotations

import re
import itertools
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Formula parsing
# ---------------------------------------------------------------------------

def parse_formula(formula: str) -> dict[str, int]:
    """
    Parse a molecular formula string into element counts.
    e.g. "NO3" -> {"N":1, "O":3},  "N2O2" -> {"N":2, "O":2}
    """
    pattern = r"([A-Z][a-z]?)(\d*)"
    counts: dict[str, int] = {}
    for elem, n in re.findall(pattern, formula):
        if elem:
            counts[elem] = counts.get(elem, 0) + int(n or 1)
    return counts


def formula_to_vector(
    formula: str,
    elements: list[str],
) -> np.ndarray:
    """
    Convert a formula to a stoichiometry vector over `elements`.
    """
    counts = parse_formula(formula)
    return np.array([counts.get(e, 0) for e in elements], dtype=int)


def parse_reaction(reaction_str: str) -> tuple[dict[str, int], dict[str, int]]:
    """
    Parse "2NO + O2 = 2NO2" into
        reactants = {"NO": 2, "O2": 1}
        products  = {"NO2": 2}

    Supports "=", "->", "=>".
    """
    for sep in [" = ", " -> ", " => ", "=", "->", "=>"]:
        if sep in reaction_str:
            lhs, rhs = reaction_str.split(sep, 1)
            break
    else:
        raise ValueError(f"Cannot parse reaction: {reaction_str!r}")

    def parse_side(s: str) -> dict[str, int]:
        result = {}
        for term in re.split(r"\s*\+\s*", s.strip()):
            term = term.strip()
            m = re.match(r"^(\d+)?\s*([A-Za-z]\S*)$", term)
            if not m:
                raise ValueError(f"Cannot parse term: {term!r}")
            coeff = int(m.group(1)) if m.group(1) else 1
            species = m.group(2)
            result[species] = result.get(species, 0) + coeff
        return result

    return parse_side(lhs), parse_side(rhs)


# ---------------------------------------------------------------------------
# Elementary step generation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ElementaryStep:
    """
    One candidate elementary step, atom-balanced.
    Stored as sorted tuples for hashability.
    """
    reactants: tuple[tuple[str, int], ...]   # sorted ((species, coeff), ...)
    products:  tuple[tuple[str, int], ...]

    @classmethod
    def make(
        cls,
        reactants: dict[str, int],
        products:  dict[str, int],
    ) -> "ElementaryStep":
        return cls(
            reactants=tuple(sorted(reactants.items())),
            products =tuple(sorted(products.items())),
        )

    def reactant_dict(self) -> dict[str, int]:
        return dict(self.reactants)

    def product_dict(self) -> dict[str, int]:
        return dict(self.products)

    def label(self) -> str:
        def fmt(items):
            parts = []
            for sp, n in sorted(items):
                parts.append(sp if n == 1 else f"{n}{sp}")
            return " + ".join(parts)
        return f"{fmt(self.reactants)} -> {fmt(self.products)}"

    def net_stoich(self, all_species: list[str]) -> np.ndarray:
        """
        Net stoichiometry vector: positive = produced, negative = consumed.
        """
        vec = np.zeros(len(all_species), dtype=int)
        idx = {sp: i for i, sp in enumerate(all_species)}
        for sp, n in self.reactants:
            if sp in idx:
                vec[idx[sp]] -= n
        for sp, n in self.products:
            if sp in idx:
                vec[idx[sp]] += n
        return vec

    def all_species(self) -> set[str]:
        return {sp for sp, _ in self.reactants} | {sp for sp, _ in self.products}

    def is_atom_balanced(self, elements: list[str]) -> bool:
        lhs = defaultdict(int)
        rhs = defaultdict(int)
        for sp, n in self.reactants:
            for e, c in parse_formula(sp).items():
                lhs[e] += n * c
        for sp, n in self.products:
            for e, c in parse_formula(sp).items():
                rhs[e] += n * c
        return all(lhs[e] == rhs[e] for e in elements)


def generate_candidate_steps(
    species_pool:   list[str],
    elements:       list[str],
    max_reactants:  int = 2,
    max_products:   int = 2,
    max_coeff:      int = 2,
    observed_edges: Optional[set[tuple[str, str]]] = None,
) -> list[ElementaryStep]:
    """
    Generate all atom-balanced elementary steps from the species pool.

    Parameters
    ----------
    species_pool   : list of formula strings available as reactants/products
    elements       : list of element symbols present in the system
    max_reactants  : max distinct reactant species (1 or 2 typical)
    max_products   : max distinct product species (1 or 2 typical)
    max_coeff      : maximum stoichiometric coefficient for any species
    observed_edges : if given, only keep steps where at least one
                     (reactant, product) pair appears in this set

    Returns
    -------
    List of unique atom-balanced ElementaryStep objects.
    """
    steps: set[ElementaryStep] = set()
    coeffs = range(1, max_coeff + 1)

    # Generate reactant multisets
    reactant_combos = []
    for n_r in range(1, max_reactants + 1):
        for combo in itertools.combinations_with_replacement(species_pool, n_r):
            # Convert to dict of counts
            rd: dict[str, int] = defaultdict(int)
            for sp in combo:
                rd[sp] += 1
            reactant_combos.append(dict(rd))

    # Generate product multisets
    product_combos = []
    for n_p in range(1, max_products + 1):
        for combo in itertools.combinations_with_replacement(species_pool, n_p):
            pd: dict[str, int] = defaultdict(int)
            for sp in combo:
                pd[sp] += 1
            product_combos.append(dict(pd))

    for rd in reactant_combos:
        for pd in product_combos:
            # Skip trivial identity
            if rd == pd:
                continue
            # Skip if reactants == products as sets (just different coeffs)
            if set(rd.keys()) == set(pd.keys()) and rd == pd:
                continue

            step = ElementaryStep.make(rd, pd)

            # Atom balance check
            if not step.is_atom_balanced(elements):
                continue

            # Observed-edge filter: at least one (r,p) pair in observed_edges
            if observed_edges is not None:
                r_species = set(rd.keys())
                p_species = set(pd.keys())
                has_observed = any(
                    (r, p) in observed_edges or (p, r) in observed_edges
                    for r in r_species for p in p_species
                )
                if not has_observed:
                    continue

            steps.add(step)

    return sorted(steps, key=lambda s: s.label())


# ---------------------------------------------------------------------------
# Pathway: a combination of steps with multiplicities
# ---------------------------------------------------------------------------

@dataclass
class Pathway:
    """
    A sequence of elementary steps (with integer multiplicities)
    whose net stoichiometry equals the overall reaction.
    """
    steps:         list[ElementaryStep]
    multiplicities: list[int]            # how many times each step runs
    net_stoich:    np.ndarray            # should equal overall target

    def total_step_count(self) -> int:
        return sum(self.multiplicities)

    def label(self) -> str:
        lines = []
        step_num = 1
        for step, mult in zip(self.steps, self.multiplicities):
            prefix = f"  Step {step_num}:" if mult == 1 else f"  Step {step_num} (×{mult}):"
            lines.append(f"{prefix}  {step.label()}")
            step_num += 1
        return "\n".join(lines)

    def intermediates(
        self,
        reactants: set[str],
        products:  set[str],
    ) -> set[str]:
        """Species that appear but are neither overall reactants nor products."""
        all_sp: set[str] = set()
        for step in self.steps:
            all_sp |= step.all_species()
        return all_sp - reactants - products

    def is_duplicate(self, other: "Pathway") -> bool:
        """Two pathways are duplicates if they have the same steps+multiplicities."""
        if len(self.steps) != len(other.steps):
            return False
        return (
            set(zip([s.label() for s in self.steps], self.multiplicities)) ==
            set(zip([s.label() for s in other.steps], other.multiplicities))
        )


# ---------------------------------------------------------------------------
# ILP-based pathway finder
# ---------------------------------------------------------------------------

def _find_pathways_ilp(
    steps:          list[ElementaryStep],
    all_species:    list[str],
    target:         np.ndarray,
    max_step_mult:  int,
    max_path_steps: int,
    max_paths:      int,
) -> list[Pathway]:
    """
    Use integer linear programming to find combinations of steps
    (with multiplicities) whose net stoichiometry equals `target`.

    For each solution found, we add a constraint excluding it so
    the solver finds the next one (solution enumeration).
    """
    try:
        import pulp
    except ImportError:
        raise ImportError("pip install pulp  (required for ILP pathway search)")

    n_steps   = len(steps)
    n_species = len(all_species)

    # Net stoichiometry matrix: S[i, j] = net stoich of species j in step i
    S = np.array([step.net_stoich(all_species) for step in steps], dtype=int)

    pathways: list[Pathway] = []

    # We'll iteratively solve, excluding previous solutions
    excluded: list[tuple[int, ...]] = []   # list of previous multiplicity tuples

    for _ in range(max_paths * 10):   # generous iteration cap
        if len(pathways) >= max_paths:
            break

        prob = pulp.LpProblem("pathway", pulp.LpMinimize)

        # Decision variables: multiplicity of each step (0..max_step_mult)
        m = [
            pulp.LpVariable(f"m{i}", lowBound=0, upBound=max_step_mult, cat="Integer")
            for i in range(n_steps)
        ]

        # Objective: minimise total steps (prefer simpler pathways)
        prob += pulp.lpSum(m)

        # Stoichiometric constraints: S^T m = target
        for j in range(n_species):
            prob += pulp.lpSum(int(S[i, j]) * m[i] for i in range(n_steps)) == int(target[j])

        # At least one step must be used
        prob += pulp.lpSum(m) >= 1

        # Cap total steps
        prob += pulp.lpSum(m) <= max_path_steps

        # Exclude previously found solutions
        for prev in excluded:
            # Add constraint: can't use exactly this combination
            # At least one variable must differ from the previous solution
            active = [i for i in range(n_steps) if prev[i] > 0]
            if active:
                prob += pulp.lpSum(m[i] for i in active) <= sum(prev[i] for i in active) - 1

        # Solve silently
        prob.solve(pulp.PULP_CBC_CMD(msg=0))

        if pulp.LpStatus[prob.status] != "Optimal":
            break

        mults = tuple(int(round(pulp.value(m[i]))) for i in range(n_steps))

        # Only keep steps actually used
        used_steps = [steps[i] for i in range(n_steps) if mults[i] > 0]
        used_mults = [mults[i] for i in range(n_steps) if mults[i] > 0]

        if not used_steps:
            excluded.append(mults)
            continue

        # Verify net stoich
        net = sum(
            used_mults[k] * used_steps[k].net_stoich(all_species)
            for k in range(len(used_steps))
        )
        if not np.array_equal(net, target):
            excluded.append(mults)
            continue

        pw = Pathway(
            steps=used_steps,
            multiplicities=used_mults,
            net_stoich=net,
        )

        # Deduplication
        if not any(pw.is_duplicate(p) for p in pathways):
            pathways.append(pw)

        excluded.append(mults)

    return sorted(pathways, key=lambda p: p.total_step_count())


# ---------------------------------------------------------------------------
# Main enumerator class
# ---------------------------------------------------------------------------

class PathwayEnumerator:
    """
    Enumerate all elementary step sequences consistent with an overall reaction.

    Parameters
    ----------
    overall_reaction : str, e.g. "2NO + O2 = 2NO2"
    species_pool     : list of all species to consider as intermediates
                       (should include reactants and products)
    observed_edges   : set of (species_a, species_b) pairs actually seen
                       transitioning in your MD/ReactionNetwork output.
                       If None, all atom-balanced steps are considered.
    max_step_mult    : maximum times any single elementary step can repeat
                       within one pathway (default 3)
    max_path_steps   : maximum total elementary steps in a pathway (default 4)
    max_reactants    : max reactant species per elementary step (default 2)
    max_products     : max product species per elementary step (default 2)
    max_coeff        : max stoichiometric coefficient per species (default 2)
    max_paths        : maximum number of pathways to return (default 20)
    verbose          : print progress
    """

    def __init__(
        self,
        overall_reaction: str,
        species_pool:     list[str],
        observed_edges:   Optional[set[tuple[str, str]]] = None,
        max_step_mult:    int = 3,
        max_path_steps:   int = 4,
        max_reactants:    int = 2,
        max_products:     int = 2,
        max_coeff:        int = 2,
        max_paths:        int = 20,
        verbose:          bool = True,
    ):
        self.overall_reaction = overall_reaction
        self.species_pool     = list(dict.fromkeys(species_pool))  # dedup, preserve order
        self.observed_edges   = observed_edges
        self.max_step_mult    = max_step_mult
        self.max_path_steps   = max_path_steps
        self.max_reactants    = max_reactants
        self.max_products     = max_products
        self.max_coeff        = max_coeff
        self.max_paths        = max_paths
        self.verbose          = verbose

        # Parse overall reaction
        self.overall_reactants, self.overall_products = parse_reaction(overall_reaction)

        # Collect elements present
        all_formulae = list(self.overall_reactants) + list(self.overall_products) + self.species_pool
        elem_set: set[str] = set()
        for f in all_formulae:
            elem_set |= set(parse_formula(f).keys())
        self.elements = sorted(elem_set)

        # All species (pool + any mentioned in overall reaction not already there)
        sp_set = set(self.species_pool)
        for sp in list(self.overall_reactants) + list(self.overall_products):
            if sp not in sp_set:
                self.species_pool.append(sp)
                sp_set.add(sp)
        self.all_species = self.species_pool

        # Build target stoichiometry vector
        self.target = self._build_target()

    def _build_target(self) -> np.ndarray:
        """
        Net stoichiometry of the overall reaction as a vector over all_species.
        Positive = produced, negative = consumed.
        """
        target = np.zeros(len(self.all_species), dtype=int)
        idx = {sp: i for i, sp in enumerate(self.all_species)}
        for sp, n in self.overall_products.items():
            if sp in idx:
                target[idx[sp]] += n
        for sp, n in self.overall_reactants.items():
            if sp in idx:
                target[idx[sp]] -= n
        return target

    def enumerate(self) -> list[Pathway]:
        """
        Run the full enumeration and return ranked Pathway objects.
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Pathway Enumerator")
            print(f"  Overall: {self.overall_reaction}")
            print(f"  Pool:    {self.species_pool}")
            print(f"  Elements: {self.elements}")
            if self.observed_edges:
                print(f"  MD edges: {len(self.observed_edges)} observed transitions")
            print(f"{'='*60}")

        # Generate candidate elementary steps
        if self.verbose:
            print(f"\nGenerating candidate elementary steps...")

        candidates = generate_candidate_steps(
            species_pool   = self.all_species,
            elements       = self.elements,
            max_reactants  = self.max_reactants,
            max_products   = self.max_products,
            max_coeff      = self.max_coeff,
            observed_edges = self.observed_edges,
        )

        if self.verbose:
            print(f"  {len(candidates)} atom-balanced candidate steps")
            if len(candidates) < 30:
                for s in candidates:
                    print(f"    {s.label()}")

        if not candidates:
            print("  No candidate steps found. Check species pool and observed_edges.")
            return []

        # ILP search
        if self.verbose:
            print(f"\nSearching for pathways (ILP)...")
            print(f"  max_step_mult={self.max_step_mult}  "
                  f"max_path_steps={self.max_path_steps}  "
                  f"max_paths={self.max_paths}")

        pathways = _find_pathways_ilp(
            steps          = candidates,
            all_species    = self.all_species,
            target         = self.target,
            max_step_mult  = self.max_step_mult,
            max_path_steps = self.max_path_steps,
            max_paths      = self.max_paths,
        )

        if self.verbose:
            print(f"\n  Found {len(pathways)} valid pathways.")

        return pathways

    def print_pathways(self, pathways: list[Pathway]) -> None:
        """Pretty-print all pathways with intermediates labelled."""
        overall_r = set(self.overall_reactants)
        overall_p = set(self.overall_products)

        print(f"\n{'='*60}")
        print(f"PATHWAYS for:  {self.overall_reaction}")
        print(f"{'='*60}")

        if not pathways:
            print("  No valid pathways found.")
            return

        for i, pw in enumerate(pathways, 1):
            intermediates = pw.intermediates(overall_r, overall_p)
            print(f"\nPathway {i}  ({pw.total_step_count()} elementary steps"
                  + (f", intermediates: {', '.join(sorted(intermediates))}" if intermediates else "")
                  + ")")
            print(pw.label())

        print(f"\n{'='*60}")

    def to_neb_steps(self, pathways: list[Pathway]) -> list[dict]:
        """
        Convert pathways to the format expected by NEBRunner.run_batch():
        list of dicts with 'label', 'reactants', 'products'.

        Only unique elementary steps across all pathways are returned.
        """
        seen: set[str] = set()
        neb_steps = []
        for pw in pathways:
            for step in pw.steps:
                lbl = step.label()
                if lbl not in seen:
                    seen.add(lbl)
                    neb_steps.append({
                        "label":     lbl,
                        "reactants": step.reactant_dict(),
                        "products":  step.product_dict(),
                    })
        return neb_steps
