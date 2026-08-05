"""
Tests for the on-the-fly (Koch 2003) pivoted Cholesky decomposition of the ERIs.

The on-the-fly factor is validated against the dense reference oracle
(``FockBuilder._build_B_cholesky_dense`` / ``cholesky_wrapper``): it must reconstruct the exact
four-center ERI tensor to the requested tolerance, and routing ``cholesky_tei`` through the SCF must
give an energy identical to the dense path.
"""

import numpy as np
import pytest

from forte2 import System, RHF, integrals
from forte2.integrals.cholesky import cholesky_otf
from forte2.helpers.matrix_functions import cholesky_wrapper

WATER = """
O 0.000000000000 0.000000000000 -0.061664597388
H 0.000000000000 -0.711620616369 0.489330954643
H 0.000000000000 0.711620616369 0.489330954643
"""


def _dense_eri_matrix(system):
    nbf = system.nbf
    return integrals.coulomb_4c(system).reshape((nbf**2,) * 2)


def test_otf_reconstructs_eri():
    """B B^T (packed) == the full ERI matrix to the pivot tolerance."""
    tol = 1e-10
    system = System(xyz=WATER, basis_set="cc-pvdz", cholesky_tei=True, cholesky_tol=tol)
    nbf = system.nbf
    M = _dense_eri_matrix(system)

    B, naux = cholesky_otf(system, tol)
    assert B.shape == (naux, nbf, nbf)
    Bmat = B.reshape(naux, nbf * nbf)
    # sum_J B[J,mn] B[J,rs] must equal (mn|rs)
    assert np.linalg.norm(Bmat.T @ Bmat - M) < 1e-9


def test_otf_diagonal_exact():
    """The reconstructed diagonal must match the (mn|mn) primitive independently."""
    tol = 1e-12
    system = System(xyz=WATER, basis_set="cc-pvdz", cholesky_tei=True, cholesky_tol=tol)
    nbf = system.nbf
    B, naux = cholesky_otf(system, tol)
    Bmat = B.reshape(naux, nbf * nbf)
    diag_reco = np.einsum("Jp,Jp->p", Bmat, Bmat)
    diag_ref = np.asarray(integrals.coulomb_4c_diagonal(system))
    assert np.linalg.norm(diag_reco - diag_ref) < 1e-9


def test_otf_matches_dense_rank_and_span():
    """OTF and dense pivoted Cholesky recover the same rank and the same row space."""
    tol = 1e-10
    system = System(xyz=WATER, basis_set="cc-pvdz", cholesky_tei=True, cholesky_tol=tol)
    nbf = system.nbf
    M = _dense_eri_matrix(system)

    B_otf, naux_otf = cholesky_otf(system, tol)
    B_dense = cholesky_wrapper(M, tol=tol)  # (naux, nbf**2), B.T @ B == M

    # Same number of vectors (both stop at the same numerical rank).
    assert naux_otf == B_dense.shape[0]
    # Both reconstruct the same matrix (the factorization is unique only up to an orthogonal
    # transform, so compare the Gram reconstruction, not the factors themselves).
    Bmat = B_otf.reshape(naux_otf, nbf * nbf)
    assert np.linalg.norm(Bmat.T @ Bmat - B_dense.T @ B_dense) < 1e-10


@pytest.mark.parametrize("tol", [1e-4, 1e-6, 1e-10])
def test_otf_tolerance_controls_accuracy(tol):
    """Looser tolerance -> fewer vectors, but reconstruction error stays near the tolerance."""
    system = System(xyz=WATER, basis_set="cc-pvdz", cholesky_tei=True, cholesky_tol=tol)
    nbf = system.nbf
    M = _dense_eri_matrix(system)
    B, naux = cholesky_otf(system, tol)
    Bmat = B.reshape(naux, nbf * nbf)
    err = np.linalg.norm(Bmat.T @ Bmat - M)
    # The residual max diagonal is <= tol per element; the Frobenius norm scales with nbf**2.
    assert err < max(1e-9, tol * nbf**2)
    assert naux <= nbf * nbf


def test_cholesky_tei_routing_energies_agree():
    """cholesky_tei True/'otf'/'naive' must give identical RHF energies."""
    energies = {}
    for algo in (True, "otf", "naive"):
        system = System(
            xyz=WATER, basis_set="cc-pvdz", cholesky_tei=algo, cholesky_tol=1e-10
        )
        scf = RHF(charge=0)(system)
        scf.run()
        energies[str(algo)] = scf.E
    assert energies["otf"] == pytest.approx(energies["naive"], abs=1e-9)
    assert energies["True"] == pytest.approx(energies["naive"], abs=1e-9)


def test_cholesky_tei_invalid_value():
    """An unknown cholesky_tei string is rejected at construction."""
    with pytest.raises(ValueError):
        System(xyz=WATER, basis_set="sto-3g", cholesky_tei="bogus")
