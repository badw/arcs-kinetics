"""
Sanity checks for NEB endpoints from make_neb_endpoints.make_endpoints.

Idea: relax each endpoint with the calculator you'll use for the NEB, rebuild the
bond graph from the relaxed geometry and compare it with the bonds the SMILES say
should be there. If the reactant complex relaxes into a new bond (or the product
complex falls apart / re-forms a broken bond), the endpoint is not a minimum of the
intended species and the NEB barrier is meaningless.

    from arcskin.check_endpoints import check_endpoints, make_checked_endpoints

    calc_factory = lambda atoms: TBLite(method="GFN2-xTB",
                                        charge=atoms.info["charge"],
                                        multiplicity=atoms.info["mult"], verbosity=0)
    reac, prod, report = make_checked_endpoints(["[SH]", "N(=O)[O]"], ["N=O", "O=S"],
                                                calc_factory)
"""

import numpy as np
import networkx as nx
from ase.constraints import FixBondLengths
from ase.data import covalent_radii
from ase.optimize import BFGS

from arkskin.make_neb_endpoints import _side, make_endpoints


# --------------------------------------------------------------------------- #
# Connectivity
# --------------------------------------------------------------------------- #
def geometry_bonds(atoms, scale=1.25):
    """Bonds from distances: r_ij < scale * (r_cov_i + r_cov_j).
    scale must sit below make_endpoints' shift (1.5 by default) so the unrelaxed
    pre-complex is read as separate molecules."""
    rc = covalent_radii[atoms.numbers]
    d = atoms.get_all_distances()
    i, j = np.where(np.triu(d < scale * (rc[:, None] + rc[None, :]), 1))
    return {(int(a), int(b)) for a, b in zip(i, j)}


def expected_bonds(reactant_smiles, reac):
    """(reactant bonds, product bonds), both in the endpoints' (reactant) atom order.
    Uses the SMILES graph, not the geometry, plus the formed/broken bonds that
    make_endpoints stored in atoms.info."""
    _, G_r, _ = _side(reactant_smiles)
    rb = {tuple(sorted(map(int, e))) for e in G_r.edges}
    formed = {tuple(sorted(b)) for b in reac.info["formed"]}
    broken = {tuple(sorted(b)) for b in reac.info["broken"]}
    return rb, (rb - broken) | formed


def _fragments(n, bonds):
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(bonds)
    return [sorted(c) for c in nx.connected_components(G)]


def closest_contact(atoms, bonds):
    """Smallest r_ij / (r_cov_i + r_cov_j) between atoms in different molecules
    (molecules defined by `bonds`). inf if there is only one molecule."""
    frags = _fragments(len(atoms), bonds)
    if len(frags) < 2:
        return np.inf, None
    fid = np.empty(len(atoms), int)
    for k, f in enumerate(frags):
        fid[f] = k
    rc = covalent_radii[atoms.numbers]
    ratio = atoms.get_all_distances() / (rc[:, None] + rc[None, :])
    ratio[fid[:, None] == fid[None, :]] = np.inf
    i, j = np.unravel_index(np.argmin(ratio), ratio.shape)
    return float(ratio[i, j]), (int(min(i, j)), int(max(i, j)))


# --------------------------------------------------------------------------- #
# Relaxation + checks
# --------------------------------------------------------------------------- #
def relax(atoms, calc_factory, fmax=0.05, steps=300, fix_pairs=(), logfile=None):
    """Relax a copy. fix_pairs: atom pairs whose distance is frozen (e.g. the
    reacting bonds, to keep a pre-complex apart while everything else relaxes)."""
    a = atoms.copy()
    a.info = dict(atoms.info)
    a.calc = calc_factory(a)
    if fix_pairs:
        a.set_constraint(FixBondLengths([list(p) for p in fix_pairs]))
    e0 = a.get_potential_energy()
    opt = BFGS(a, logfile=logfile)
    converged = opt.run(fmax=fmax, steps=steps)
    a.set_constraint()
    return a, e0, a.get_potential_energy(), bool(converged)


def check_endpoint(
    atoms,
    expected,
    calc_factory,
    scale=1.25,
    fix_pairs=(),
    min_contact=None,
    **relax_kw,
):
    """Relax `atoms` and compare its connectivity with `expected` (set of bonds).

    Returns a dict:
      ok          - True if the relaxed graph equals the expected graph
      extra       - bonds present after relaxation but not expected (e.g. the
                    reactants have already reacted -> shift too small)
      missing     - expected bonds that are gone after relaxation
      start_extra - extra bonds already present *before* relaxation
      contact     - closest inter-molecular r / (r_cov_i + r_cov_j) after relaxing
      dE          - E(relaxed) - E(unrelaxed)
      rmsd        - displacement of the relaxation (Å)
      relaxed     - the relaxed Atoms
    If min_contact is given, ok also requires contact >= min_contact."""
    expected = {tuple(sorted(b)) for b in expected}
    start = geometry_bonds(atoms, scale)
    try:
        relaxed, e0, e1, conv = relax(
            atoms, calc_factory, fix_pairs=fix_pairs, **relax_kw
        )
    except Exception as exc:  # SCF failure etc. -> count as a failed endpoint
        return dict(
            ok=False,
            error=f"{type(exc).__name__}: {exc}",
            extra=[],
            missing=[],
            start_extra=sorted(start - expected),
            contact=np.nan,
            contact_pair=None,
            dE=np.nan,
            rmsd=np.nan,
            converged=False,
            energy=np.nan,
            relaxed=None,
        )
    got = geometry_bonds(relaxed, scale)
    contact, pair = closest_contact(relaxed, expected)
    ok = got == expected and (min_contact is None or contact >= min_contact)
    return dict(
        ok=ok,
        extra=sorted(got - expected),
        missing=sorted(expected - got),
        start_extra=sorted(start - expected),
        contact=contact,
        contact_pair=pair,
        dE=e1 - e0,
        rmsd=float(np.sqrt(((relaxed.positions - atoms.positions) ** 2).sum(1).mean())),
        converged=conv,
        energy=e1,
        relaxed=relaxed,
    )


def check_endpoints(
    reac,
    prod,
    reactant_smiles,
    calc_factory,
    scale=1.25,
    freeze_reacting=False,
    verbose=True,
    **kw,
):
    """Check both endpoints. freeze_reacting=True holds the forming bonds (reactant
    side) and breaking bonds (product side) at their starting length, so you only
    test that each molecule is sensible, not whether the complex is a minimum.
    Use that for barrierless associations (radical + radical), which will always
    collapse in a free relaxation."""
    rb, pb = expected_bonds(reactant_smiles, reac)
    formed, broken = reac.info["formed"], reac.info["broken"]
    out = {}
    for name, atoms, exp, fix in (
        ("reactants", reac, rb, formed),
        ("products", prod, pb, broken),
    ):
        out[name] = check_endpoint(
            atoms,
            exp,
            calc_factory,
            scale,
            fix_pairs=fix if freeze_reacting else (),
            **kw,
        )
    out["ok"] = out["reactants"]["ok"] and out["products"]["ok"]
    if verbose:
        sym = reac.get_chemical_symbols()
        lab = lambda bs: [f"{sym[i]}{i}-{sym[j]}{j}" for i, j in bs]
        for name in ("reactants", "products"):
            r = out[name]
            if "error" in r:
                print(f"  FAIL {name:9s} calculator error: {r['error']}")
                continue
            flag = "OK  " if r["ok"] else "FAIL"
            msg = (
                f"  {flag} {name:9s} dE={r['dE']:+7.3f} eV  rmsd={r['rmsd']:.2f} Å  "
                f"closest contact={r['contact']:.2f}×Σr_cov"
            )
            if r["extra"]:
                msg += f"  new bonds: {lab(r['extra'])}"
            if r["missing"]:
                msg += f"  lost bonds: {lab(r['missing'])}"
            if r["start_extra"]:
                msg += f"  (already bonded before relaxing: {lab(r['start_extra'])})"
            if not r["converged"]:
                msg += "  [not converged]"
            print(msg)
    return out


def make_checked_endpoints(
    reactants,
    products,
    calc_factory,
    shifts=(1.5, 2.0, 2.5, 3.0),
    scale=1.25,
    freeze_reacting=False,
    relax_endpoints=True,
    verbose=True,
    check_kw=None,
    **endpoint_kw,
):
    """Try make_endpoints with increasing shift until both relaxed endpoints keep
    the intended connectivity. Returns (reac, prod, report). If relax_endpoints,
    the relaxed geometries are returned (re-aligned), which is what you want for
    the NEB anyway. freeze_reacting: False (free relaxation), True (reacting bonds
    held fixed) or "fallback" (try free first, then frozen; report["frozen"] says
    which one passed). Raises RuntimeError if nothing works (err.report has all
    attempts)."""
    from ase.build import minimize_rotation_and_translation

    tried = {}
    modes = [freeze_reacting] if freeze_reacting != "fallback" else [False, True]
    for s, frz in ((s, f) for f in modes for s in shifts):
        reac, prod = make_endpoints(
            reactants, products, shift=s, verbose=False, **endpoint_kw
        )
        if verbose:
            print(f"shift={s}" + ("  (reacting bonds frozen)" if frz else ""))
        rep = check_endpoints(
            reac,
            prod,
            reactants,
            calc_factory,
            scale=scale,
            freeze_reacting=frz,
            verbose=verbose,
            **(check_kw or {}),
        )
        rep["frozen"] = frz
        tried[(s, frz)] = rep
        if rep["ok"]:
            if relax_endpoints:
                r, p = rep["reactants"]["relaxed"], rep["products"]["relaxed"]
                for a, ref in ((r, reac), (p, prod)):
                    a.calc = None
                    a.info = dict(ref.info)
                minimize_rotation_and_translation(r, p)
                reac, prod = r, p
            rep["shift"] = s
            return reac, prod, rep
    err = RuntimeError(
        f"No shift in {shifts} gave stable endpoints for {reactants} -> {products}"
    )
    err.report = tried
    raise err
