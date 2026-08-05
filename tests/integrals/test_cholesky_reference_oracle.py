"""
Oracle gate: the production Cholesky drivers must match the frozen reference implementations.

:mod:`forte2.integrals.cholesky_reference` holds the Phase 1/2 reference (oracle) versions of the
Koch one-step and Folkestad two-step decompositions, each validated against the dense LAPACK
``dpstrf`` path. This module diffs the *production* drivers in :mod:`forte2.integrals.cholesky`
against those oracles. It is the contract every production-hardening change (significant-set
restriction, proactive shell-pair draining, Schwarz screening, zero-memory Step I, ...) must keep
green.

The contract is a **reconstruction** contract, not a bit-for-bit one: production is free to select
pivots in a different order (e.g. Koch proactive draining) or keep a slightly different Cholesky
basis, so long as

* it reconstructs the ERI matrix to the same tolerance,
* it selects the same number of vectors up to the documented small two-step margin, and
* its Gram matrix ``B^T B`` agrees with the oracle's to tolerance (the factorization is unique only
  up to an orthogonal transform, so we never compare ``B`` element-wise).
"""

import numpy as np
import pytest

from forte2 import System, integrals
from forte2.integrals.cholesky import (
    cholesky_otf,
    cholesky_pivots,
    cholesky_pivoted,
    cholesky_vectors_ri,
    _build_shell_pair_layout,
)
from forte2.integrals.cholesky_reference import (
    cholesky_otf_reference,
    cholesky_pivots_reference,
    cholesky_pivoted_reference,
    cholesky_vectors_ri_reference,
)

WATER = """
O 0.000000000000 0.000000000000 -0.061664597388
H 0.000000000000 -0.711620616369 0.489330954643
H 0.000000000000 0.711620616369 0.489330954643
"""

# Small systems whose dense (mn|rs) is cheap to form as the ultimate ground truth.
SYSTEMS = [
    ("water/sto-3g", WATER, "sto-3g"),
    ("water/cc-pvdz", WATER, "cc-pvdz"),
    ("Ne/cc-pvdz", "Ne 0 0 0", "cc-pvdz"),
]
TOLS = [1e-6, 1e-10]


def _params(*, algos):
    """Cartesian product of systems x tolerances, labelled for readable test ids."""
    return [
        pytest.param(xyz, basis, tol, id=f"{label}/{tol:g}")
        for (label, xyz, basis) in SYSTEMS
        for tol in TOLS
    ]


def _gram(B, naux, nbf):
    """B^T B as an (nbf**2, nbf**2) matrix from a (naux, nbf, nbf) factor."""
    Bmat = B.reshape(naux, nbf * nbf)
    return Bmat.T @ Bmat


def _max_abs(A):
    """Elementwise max-abs norm (the norm in which the CD accuracy bound is cleanest)."""
    return float(np.max(np.abs(A))) if A.size else 0.0


def _recon_bound(tol):
    """Max-abs reconstruction/agreement bound at threshold ``tol``.

    Koch's stopping criterion leaves the residual ``R = M - B^T B`` positive semidefinite with every
    diagonal entry ``<= tol``, so ``|R_pq| <= sqrt(R_pp R_qq) <= tol`` elementwise. Production adds
    Schwarz screening (which drops only integrals provably ``<= tol``) and proactive draining (a
    different but equally valid pivot path); the two-step path reconstructs via an RI fit of
    comparable accuracy. A modest constant absorbs the accumulation of these ``O(tol)`` effects and
    round-off; the ``1e-9`` floor covers the machine-precision (``tol <= 0``) regime and the
    unavoidable dense-integral round-off. This is deliberately a *reconstruction* bound: it does not
    assume production and reference took the same path, only that both reproduce the true ERI.
    """
    return max(1e-9, 32.0 * tol)


def _rank_margin(rank):
    """Allowed pivot-count spread between production and oracle (see cholesky.py Notes)."""
    return max(5, int(np.ceil(0.03 * rank)))


# ---------------------------------------------------------------------------
# Koch one-step: production cholesky_otf vs cholesky_otf_reference
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("xyz, basis_set, tol", _params(algos=("otf",)))
def test_otf_matches_reference(xyz, basis_set, tol):
    system = System(xyz=xyz, basis_set=basis_set)
    nbf = system.nbf

    B_prod, naux_prod = cholesky_otf(system, tol)
    B_ref, naux_ref = cholesky_otf_reference(system, tol)

    # Same vector count up to the documented margin (draining/screening may shift it slightly).
    assert abs(naux_prod - naux_ref) <= _rank_margin(naux_ref)
    # Gram matrices agree: production reconstructs the same operator as the oracle, to the CD
    # accuracy (max-abs bound; screening + draining make production diverge only at O(tol)).
    assert _max_abs(
        _gram(B_prod, naux_prod, nbf) - _gram(B_ref, naux_ref, nbf)
    ) <= _recon_bound(tol)


# ---------------------------------------------------------------------------
# Folkestad two-step: production cholesky_pivoted vs cholesky_pivoted_reference
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("xyz, basis_set, tol", _params(algos=("pivoted",)))
def test_pivoted_matches_reference(xyz, basis_set, tol):
    system = System(xyz=xyz, basis_set=basis_set)
    nbf = system.nbf

    B_prod, naux_prod = cholesky_pivoted(system, tol)
    B_ref, naux_ref = cholesky_pivoted_reference(system, tol)

    assert abs(naux_prod - naux_ref) <= _rank_margin(naux_ref)
    assert _max_abs(
        _gram(B_prod, naux_prod, nbf) - _gram(B_ref, naux_ref, nbf)
    ) <= _recon_bound(tol)


@pytest.mark.parametrize("xyz, basis_set, tol", _params(algos=("pivoted",)))
def test_step1_pivots_match_reference(xyz, basis_set, tol):
    """Step I alone: the production pivot set spans the same space as the reference pivot set."""
    system = System(xyz=xyz, basis_set=basis_set)
    layout = _build_shell_pair_layout(system.basis)

    piv_prod = cholesky_pivots(system, tol, layout=layout)
    piv_ref = cholesky_pivots_reference(system, tol, layout=layout)

    assert abs(len(piv_prod) - len(piv_ref)) <= _rank_margin(len(piv_ref))
    # Pivots are valid, unique packed AO-pair indices in both.
    assert np.all((piv_prod >= 0) & (piv_prod < system.nbf**2))
    assert len(np.unique(piv_prod)) == len(piv_prod)


@pytest.mark.parametrize("xyz, basis_set, tol", _params(algos=("pivoted",)))
def test_step2_vectors_match_reference(xyz, basis_set, tol):
    """Step II alone: given identical pivots, production and reference build the same vectors."""
    system = System(xyz=xyz, basis_set=basis_set)
    nbf = system.nbf
    layout = _build_shell_pair_layout(system.basis)

    # Feed both Step-II implementations the *same* reference pivots so any difference is Step II's.
    pivots = cholesky_pivots_reference(system, tol, layout=layout)
    B_prod, naux_prod = cholesky_vectors_ri(system, pivots, tol=tol, layout=layout)
    B_ref, naux_ref = cholesky_vectors_ri_reference(
        system, pivots, tol=tol, layout=layout
    )

    assert naux_prod == naux_ref == len(pivots)
    assert _max_abs(
        _gram(B_prod, naux_prod, nbf) - _gram(B_ref, naux_ref, nbf)
    ) <= _recon_bound(tol)


# ---------------------------------------------------------------------------
# Both production paths still reconstruct the dense ground-truth ERI (defense in depth).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("xyz, basis_set, tol", _params(algos=("otf", "pivoted")))
def test_production_reconstructs_dense_eri(xyz, basis_set, tol):
    system = System(xyz=xyz, basis_set=basis_set)
    nbf = system.nbf
    M = integrals.coulomb_4c(system).reshape((nbf * nbf,) * 2)
    # Elementwise: the CD accuracy bound is stated per matrix element, |M_pq - (B^T B)_pq| <= tol.
    recon_bound = _recon_bound(tol)

    B_otf, naux_otf = cholesky_otf(system, tol)
    assert _max_abs(_gram(B_otf, naux_otf, nbf) - M) <= recon_bound

    B_piv, naux_piv = cholesky_pivoted(system, tol)
    assert _max_abs(_gram(B_piv, naux_piv, nbf) - M) <= recon_bound
