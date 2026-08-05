"""
Tests for the two-step (Folkestad 2019) pivoted Cholesky decomposition of the ERIs.

The two-step factor is validated against the dense reference oracle
(``FockBuilder._build_B_cholesky_dense`` / ``cholesky_wrapper``) and against the one-step on-the-fly
path (``cholesky_otf``): it must reconstruct the four-center ERI tensor to the requested tolerance,
recover the same numerical rank, and -- routed through ``cholesky_tei="pivoted"`` -- give an SCF
energy identical to the dense and on-the-fly paths.
"""

import numpy as np
import pytest

from forte2 import System, RHF, integrals
from forte2.integrals.cholesky import (
    cholesky_otf,
    cholesky_pivots,
    cholesky_pivoted,
    cholesky_vectors_ri,
    _build_shell_pair_layout,
)
from forte2.helpers.matrix_functions import cholesky_wrapper

WATER = """
O 0.000000000000 0.000000000000 -0.061664597388
H 0.000000000000 -0.711620616369 0.489330954643
H 0.000000000000 0.711620616369 0.489330954643
"""


def _dense_eri_matrix(system):
    nbf = system.nbf
    return integrals.coulomb_4c(system).reshape((nbf**2,) * 2)


def test_pivoted_reconstructs_eri():
    """B B^T == the full ERI matrix to the pivot tolerance."""
    tol = 1e-10
    system = System(
        xyz=WATER, basis_set="cc-pvdz", cholesky_tei="pivoted", cholesky_tol=tol
    )
    nbf = system.nbf
    M = _dense_eri_matrix(system)

    B, naux = cholesky_pivoted(system, tol)
    assert B.shape == (naux, nbf, nbf)
    Bmat = B.reshape(naux, nbf * nbf)
    assert np.linalg.norm(Bmat.T @ Bmat - M) < 1e-8


def test_pivoted_diagonal_accurate():
    """The reconstructed diagonal matches the (mn|mn) primitive to the tolerance."""
    tol = 1e-12
    system = System(
        xyz=WATER, basis_set="cc-pvdz", cholesky_tei="pivoted", cholesky_tol=tol
    )
    nbf = system.nbf
    B, naux = cholesky_pivoted(system, tol)
    Bmat = B.reshape(naux, nbf * nbf)
    diag_reco = np.einsum("Jp,Jp->p", Bmat, Bmat)
    diag_ref = np.asarray(integrals.coulomb_4c_diagonal(system))
    assert np.linalg.norm(diag_reco - diag_ref) < 1e-8


def _rank_margin(rank):
    """Allowed pivot overshoot of two-step vs strict-greedy CD (see module docstring)."""
    return max(5, int(np.ceil(0.03 * rank)))


def test_pivoted_matches_dense_rank_and_span():
    """Two-step and dense pivoted Cholesky recover (nearly) the same rank and the same row space."""
    tol = 1e-10
    system = System(
        xyz=WATER, basis_set="cc-pvdz", cholesky_tei="pivoted", cholesky_tol=tol
    )
    nbf = system.nbf
    M = _dense_eri_matrix(system)

    B_piv, naux_piv = cholesky_pivoted(system, tol)
    B_dense = cholesky_wrapper(M, tol=tol)  # (naux, nbf**2), B.T @ B == M
    rank = B_dense.shape[0]

    # The two-step "Cholesky basis" is not guaranteed minimal: qualified-batch greedy pivoting can
    # keep a few near-dependent columns that strict global greedy (dense/OTF) would have screened.
    # It never selects *fewer* than the numerical rank, and the overshoot is small.
    assert naux_piv >= rank
    assert naux_piv <= rank + _rank_margin(rank)
    # Regardless of the extra pivots, the Gram reconstruction matches the dense factor to tolerance
    # (the factorization is unique only up to an orthogonal transform, so compare B^T B, not B).
    Bmat = B_piv.reshape(naux_piv, nbf * nbf)
    assert np.linalg.norm(Bmat.T @ Bmat - B_dense.T @ B_dense) < 1e-8


def test_pivoted_rank_matches_otf():
    """The two-step and one-step (Koch) paths select nearly the same number of pivots."""
    tol = 1e-10
    system = System(
        xyz=WATER, basis_set="cc-pvdz", cholesky_tei="pivoted", cholesky_tol=tol
    )
    _, naux_otf = cholesky_otf(system, tol)
    _, naux_piv = cholesky_pivoted(system, tol)
    assert naux_piv >= naux_otf
    assert naux_piv <= naux_otf + _rank_margin(naux_otf)


@pytest.mark.parametrize("tol", [1e-4, 1e-6, 1e-10])
def test_pivoted_tolerance_controls_accuracy(tol):
    """Looser tolerance -> fewer vectors, but reconstruction error stays near the tolerance."""
    system = System(
        xyz=WATER, basis_set="cc-pvdz", cholesky_tei="pivoted", cholesky_tol=tol
    )
    nbf = system.nbf
    M = _dense_eri_matrix(system)
    B, naux = cholesky_pivoted(system, tol)
    Bmat = B.reshape(naux, nbf * nbf)
    err = np.linalg.norm(Bmat.T @ Bmat - M)
    assert err < max(1e-8, tol * nbf**2)
    assert naux <= nbf * nbf


def test_step1_pivots_are_valid_and_unique():
    """Step I returns a unique set of packed AO-pair indices, one per numerical-rank vector."""
    tol = 1e-10
    system = System(
        xyz=WATER, basis_set="cc-pvdz", cholesky_tei="pivoted", cholesky_tol=tol
    )
    nbf = system.nbf
    pivots = cholesky_pivots(system, tol)
    # Pivots are packed indices in range and pairwise distinct.
    assert pivots.ndim == 1
    assert np.all((pivots >= 0) & (pivots < nbf * nbf))
    assert len(np.unique(pivots)) == len(pivots)
    # The pivot count is at least the numerical rank recovered by the dense oracle, with a small
    # overshoot from batched greedy pivoting (see test_pivoted_matches_dense_rank_and_span).
    rank = cholesky_wrapper(_dense_eri_matrix(system), tol=tol).shape[0]
    assert len(pivots) >= rank
    assert len(pivots) <= rank + _rank_margin(rank)


def test_step2_vectors_from_given_pivots():
    """Step II alone reconstructs the ERIs from an externally supplied pivot set."""
    tol = 1e-10
    system = System(
        xyz=WATER, basis_set="cc-pvdz", cholesky_tei="pivoted", cholesky_tol=tol
    )
    nbf = system.nbf
    M = _dense_eri_matrix(system)
    layout = _build_shell_pair_layout(system.basis)
    pivots = cholesky_pivots(system, tol, layout=layout)
    B, naux = cholesky_vectors_ri(system, pivots, tol=tol, layout=layout)
    assert naux == len(pivots)
    Bmat = B.reshape(naux, nbf * nbf)
    assert np.linalg.norm(Bmat.T @ Bmat - M) < 1e-8


def test_cholesky_tei_routing_energies_agree():
    """cholesky_tei 'pivoted'/'otf'/'naive' must give identical RHF energies."""
    energies = {}
    for algo in ("pivoted", "otf", "naive"):
        system = System(
            xyz=WATER, basis_set="cc-pvdz", cholesky_tei=algo, cholesky_tol=1e-10
        )
        scf = RHF(charge=0)(system)
        scf.run()
        energies[algo] = scf.E
    assert energies["pivoted"] == pytest.approx(energies["naive"], abs=1e-9)
    assert energies["pivoted"] == pytest.approx(energies["otf"], abs=1e-9)
