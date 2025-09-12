import numpy as np

from forte2 import System, RHF, CI, State, RelSlaterRules, hilbert_space, MOSpace
from forte2.jkbuilder import SpinorbitalIntegrals
from forte2.ci.rel_ci import _RelCIBase
from forte2.helpers.comparisons import approx


def compare_rdms(ci: _RelCIBase):
    rdm_threshold = 1e-12

    # test the 1-RDMs
    rdm1_0_sparse = ci.make_1rdm_debug(0)
    rdm1_0_sigma = ci.make_1rdm(0)
    assert (
        np.linalg.norm(rdm1_0_sparse - rdm1_0_sigma) < rdm_threshold
    ), f"Norm of the difference between rdm1_0_sparse and rdm1_0_sigma is too large: {np.linalg.norm(rdm1_0_sparse - rdm1_0_sigma):.12f}."

    rdm1_1_sparse = ci.make_1rdm_debug(1)
    rdm1_1_sigma = ci.make_1rdm(1)
    assert (
        np.linalg.norm(rdm1_1_sparse - rdm1_1_sigma) < rdm_threshold
    ), f"Norm of the difference between rdm1_1_sparse and rdm1_1_sigma is too large: {np.linalg.norm(rdm1_1_sparse - rdm1_1_sigma):.12f}."

    # test the 1-TDMs
    tdm1_01_sparse = ci.make_1rdm_debug(0, 1)
    tdm1_01_sigma = ci.make_1rdm(0, 1)
    assert (
        np.linalg.norm(tdm1_01_sparse - tdm1_01_sigma) < rdm_threshold
    ), f"Norm of the difference between tdm1_01_sparse and tdm1_01_sigma is too large: {np.linalg.norm(tdm1_01_sparse - tdm1_01_sigma):.12f}."

    tdm_10_sparse = ci.make_1rdm_debug(1, 0)
    tdm_10_sigma = ci.make_1rdm(1, 0)
    assert (
        np.linalg.norm(tdm_10_sparse - tdm_10_sigma) < rdm_threshold
    ), f"Norm of the difference between tdm_10_sparse and tdm_10_sigma is too large: {np.linalg.norm(tdm_10_sparse - tdm_10_sigma):.12f}."
    assert np.allclose(tdm_10_sparse, tdm1_01_sparse.T.conj())
    assert np.allclose(tdm_10_sigma, tdm1_01_sigma.T.conj())

    # test the 2-RDMs
    rdm2_0_sparse = ci.make_2rdm_debug(0)
    rdm2_0_sigma = ci.make_2rdm(0)
    assert (
        np.linalg.norm(rdm2_0_sparse - rdm2_0_sigma) < rdm_threshold
    ), f"Norm of the difference between rdm2_0_sparse and rdm2_0_sigma is too large: {np.linalg.norm(rdm2_0_sparse - rdm2_0_sigma):.12f}."
    rdm2_1_sparse = ci.make_2rdm_debug(1)
    rdm2_1_sigma = ci.make_2rdm(1)
    assert (
        np.linalg.norm(rdm2_1_sparse - rdm2_1_sigma) < rdm_threshold
    ), f"Norm of the difference between rdm2_1_sparse and rdm2_1_sigma is too large: {np.linalg.norm(rdm2_1_sparse - rdm2_1_sigma):.12f}."

    # test the 2-TDMs
    tdm2_01_sparse = ci.make_2rdm_debug(0, 1)
    tdm2_01_sigma = ci.make_2rdm(0, 1)
    assert (
        np.linalg.norm(tdm2_01_sparse - tdm2_01_sigma) < rdm_threshold
    ), f"Norm of the difference between tdm2_01_sparse and tdm2_01_sigma is too large: {np.linalg.norm(tdm2_01_sparse - tdm2_01_sigma):.12f}."
    tdm2_10_sparse = ci.make_2rdm_debug(1, 0)
    tdm2_10_sigma = ci.make_2rdm(1, 0)
    assert (
        np.linalg.norm(tdm2_10_sparse - tdm2_10_sigma) < rdm_threshold
    ), f"Norm of the difference between tdm2_10_sparse and tdm2_10_sigma is too large: {np.linalg.norm(tdm2_10_sparse - tdm2_10_sigma):.12f}."
    assert np.allclose(tdm2_10_sparse, tdm2_01_sparse.T.conj())
    assert np.allclose(tdm2_10_sigma, tdm2_01_sigma.T.conj())

    # Test 3-RDMs
    rdm3_0_sparse = ci.make_3rdm_debug(0)
    rdm3_0_sigma = ci.make_3rdm(0)
    assert (
        np.linalg.norm(rdm3_0_sparse - rdm3_0_sigma) < rdm_threshold
    ), f"Norm of the difference between rdm3_0_sparse and rdm3_0_sigma is too large: {np.linalg.norm(rdm3_0_sparse - rdm3_0_sigma):.12f}."
    rdm3_1_sparse = ci.make_3rdm_debug(1)
    rdm3_1_sigma = ci.make_3rdm(1)
    assert (
        np.linalg.norm(rdm3_1_sparse - rdm3_1_sigma) < rdm_threshold
    ), f"Norm of the difference between rdm3_1_sparse and rdm3_1_sigma is too large: {np.linalg.norm(rdm3_1_sparse - rdm3_1_sigma):.12f}."

    # Test 3-TDMs
    tdm3_01_sparse = ci.make_3rdm_debug(0, 1)
    tdm3_01_sigma = ci.make_3rdm(0, 1)
    assert (
        np.linalg.norm(tdm3_01_sparse - tdm3_01_sigma) < rdm_threshold
    ), f"Norm of the difference between tdm3_01_sparse and tdm3_01_sigma is too large: {np.linalg.norm(tdm3_01_sparse - tdm3_01_sigma):.12f}."
    tdm3_10_sparse = ci.make_3rdm_debug(1, 0)
    tdm3_10_sigma = ci.make_3rdm(1, 0)
    assert (
        np.linalg.norm(tdm3_10_sparse - tdm3_10_sigma) < rdm_threshold
    ), f"Norm of the difference between tdm3_10_sparse and tdm3_10_sigma is too large: {np.linalg.norm(tdm3_10_sparse - tdm3_10_sigma):.12f}."
    assert np.allclose(tdm3_10_sparse, tdm3_01_sparse.T.conj())
    assert np.allclose(tdm3_10_sigma, tdm3_01_sigma.T.conj())


def test_ci_rdms_1():
    # reference energy from pyscf:
    # np.random.seed(12)
    # norb = 12
    # h1 = np.random.random((norb,norb)) + 1j * np.random.random((norb,norb))
    # h2 = np.random.random((norb,norb,norb,norb)) + 1j * np.random.random((norb,norb,norb,norb))
    # print(np.linalg.norm(h1), np.linalg.norm(h2))
    # # Restore permutation symmetry
    # h1 = h1 + h1.T.conj()
    # h2 = h2 + h2.transpose(2, 3, 0, 1)
    # h2 = h2 + h2.transpose(1, 0, 3, 2).conj()
    # h2 = h2 + h2.transpose(3, 2, 1, 0).conj()
    # cisolver = pyscf.fci.fci_dhf_slow.FCI()
    # cisolver.max_cycle = 100
    # cisolver.conv_tol = 1e-12
    # e, fcivec = cisolver.kernel(h1, h2, norb, nelec=8, verbose=5)

    eref = -80.43551643145948
    rng = np.random.default_rng(12)
    norb = 12
    h1 = rng.random((norb, norb)) + 1j * rng.random((norb, norb))
    h2 = rng.random((norb, norb, norb, norb)) + 1j * rng.random(
        (norb, norb, norb, norb)
    )
    # Restore permutation symmetry
    h1 = h1 + h1.T.conj()
    # pyscf uses chemist's notation, forte2 uses physicist's notation
    h2 = h2.swapaxes(1, 2)
    h2 = h2 + h2.transpose(2, 3, 0, 1).conj()
    h2 = h2 + h2.transpose(1, 0, 3, 2)
    h2 = h2 + h2.transpose(3, 2, 1, 0).conj()

    fakeints = SpinorbitalIntegrals.__new__(SpinorbitalIntegrals)
    fakeints.E = 0.0
    fakeints.H = h1
    fakeints.V = h2
    mo_space = MOSpace(nmo=norb, active_orbitals=list(range(norb)))
    state = State(nel=8, multiplicity=1, ms=0.0)
    ci = _RelCIBase(
        mo_space=mo_space,
        state=state,
        ints=fakeints,
        nroot=2,
        active_orbsym=[[0] * norb],
        maxiter=200,
        ci_algorithm="hz",
    )
    ci.run()
    assert ci.E[0] == approx(-80.435516431459)
    assert ci.E[1] == approx(-77.139316845036)

    compare_rdms(ci)
