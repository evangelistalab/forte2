import numpy as np

from forte2 import State, MOSpace
from forte2.jkbuilder import SpinorbitalIntegrals
from forte2.ci.ci import _CIBase
from forte2.helpers.comparisons import approx

rdm_threshold = 1e-12


def setup_ci():
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
    ci = _CIBase(
        mo_space=mo_space,
        state=state,
        ints=fakeints,
        nroot=2,
        active_orbsym=[[0] * norb],
        maxiter=200,
        ci_algorithm="hz",
        two_component=True,
    )
    ci.run()
    assert ci.E[0] == approx(-80.435516431459)
    assert ci.E[1] == approx(-77.139316845036)
    return ci


ci = setup_ci()


def test_ci_1rdm():

    # test the 1-RDMs
    rdm1_0_sparse = ci.make_so_1rdm_debug(0)
    rdm1_0_sigma = ci.make_1rdm(0)
    assert (
        np.linalg.norm(rdm1_0_sparse - rdm1_0_sigma) < rdm_threshold
    ), f"Norm of the difference between rdm1_0_sparse and rdm1_0_sigma is too large: {np.linalg.norm(rdm1_0_sparse - rdm1_0_sigma):.12f}."

    rdm1_1_sparse = ci.make_so_1rdm_debug(1)
    rdm1_1_sigma = ci.make_1rdm(1)
    assert (
        np.linalg.norm(rdm1_1_sparse - rdm1_1_sigma) < rdm_threshold
    ), f"Norm of the difference between rdm1_1_sparse and rdm1_1_sigma is too large: {np.linalg.norm(rdm1_1_sparse - rdm1_1_sigma):.12f}."

    # test the 1-TDMs
    tdm1_01_sparse = ci.make_so_1rdm_debug(0, 1)
    tdm1_01_sigma = ci.make_1rdm(0, 1)
    assert (
        np.linalg.norm(tdm1_01_sparse - tdm1_01_sigma) < rdm_threshold
    ), f"Norm of the difference between tdm1_01_sparse and tdm1_01_sigma is too large: {np.linalg.norm(tdm1_01_sparse - tdm1_01_sigma):.12f}."

    tdm_10_sparse = ci.make_so_1rdm_debug(1, 0)
    tdm_10_sigma = ci.make_1rdm(1, 0)
    assert (
        np.linalg.norm(tdm_10_sparse - tdm_10_sigma) < rdm_threshold
    ), f"Norm of the difference between tdm_10_sparse and tdm_10_sigma is too large: {np.linalg.norm(tdm_10_sparse - tdm_10_sigma):.12f}."
    assert np.allclose(tdm_10_sparse, tdm1_01_sparse.T.conj())
    assert np.allclose(tdm_10_sigma, tdm1_01_sigma.T.conj())


def test_ci_2rdm():
    # test the 2-RDMs
    rdm2_0_sparse = ci.make_so_2rdm_debug(0)
    rdm2_0_sigma = ci.make_2rdm(0)
    assert (
        np.linalg.norm(rdm2_0_sparse - rdm2_0_sigma) < rdm_threshold
    ), f"Norm of the difference between rdm2_0_sparse and rdm2_0_sigma is too large: {np.linalg.norm(rdm2_0_sparse - rdm2_0_sigma):.12f}."
    rdm2_1_sparse = ci.make_so_2rdm_debug(1)
    rdm2_1_sigma = ci.make_2rdm(1)
    assert (
        np.linalg.norm(rdm2_1_sparse - rdm2_1_sigma) < rdm_threshold
    ), f"Norm of the difference between rdm2_1_sparse and rdm2_1_sigma is too large: {np.linalg.norm(rdm2_1_sparse - rdm2_1_sigma):.12f}."

    # test the 2-TDMs
    tdm2_01_sparse = ci.make_so_2rdm_debug(0, 1)
    tdm2_01_sigma = ci.make_2rdm(0, 1)
    assert (
        np.linalg.norm(tdm2_01_sparse - tdm2_01_sigma) < rdm_threshold
    ), f"Norm of the difference between tdm2_01_sparse and tdm2_01_sigma is too large: {np.linalg.norm(tdm2_01_sparse - tdm2_01_sigma):.12f}."
    tdm2_10_sparse = ci.make_so_2rdm_debug(1, 0)
    tdm2_10_sigma = ci.make_2rdm(1, 0)
    assert (
        np.linalg.norm(tdm2_10_sparse - tdm2_10_sigma) < rdm_threshold
    ), f"Norm of the difference between tdm2_10_sparse and tdm2_10_sigma is too large: {np.linalg.norm(tdm2_10_sparse - tdm2_10_sigma):.12f}."
    assert np.allclose(tdm2_10_sparse, tdm2_01_sparse.T.conj())
    assert np.allclose(tdm2_10_sigma, tdm2_01_sigma.T.conj())


def test_ci_3rdm():
    # Test 3-RDMs
    rdm3_0_sparse = ci.make_so_3rdm_debug(0)
    rdm3_0_sigma = ci.make_3rdm(0)
    assert (
        np.linalg.norm(rdm3_0_sparse - rdm3_0_sigma) < rdm_threshold
    ), f"Norm of the difference between rdm3_0_sparse and rdm3_0_sigma is too large: {np.linalg.norm(rdm3_0_sparse - rdm3_0_sigma):.12f}."
    rdm3_1_sparse = ci.make_so_3rdm_debug(1)
    rdm3_1_sigma = ci.make_3rdm(1)
    assert (
        np.linalg.norm(rdm3_1_sparse - rdm3_1_sigma) < rdm_threshold
    ), f"Norm of the difference between rdm3_1_sparse and rdm3_1_sigma is too large: {np.linalg.norm(rdm3_1_sparse - rdm3_1_sigma):.12f}."

    # Test 3-TDMs
    tdm3_01_sparse = ci.make_so_3rdm_debug(0, 1)
    tdm3_01_sigma = ci.make_3rdm(0, 1)
    assert (
        np.linalg.norm(tdm3_01_sparse - tdm3_01_sigma) < rdm_threshold
    ), f"Norm of the difference between tdm3_01_sparse and tdm3_01_sigma is too large: {np.linalg.norm(tdm3_01_sparse - tdm3_01_sigma):.12f}."
    tdm3_10_sparse = ci.make_so_3rdm_debug(1, 0)
    tdm3_10_sigma = ci.make_3rdm(1, 0)
    assert (
        np.linalg.norm(tdm3_10_sparse - tdm3_10_sigma) < rdm_threshold
    ), f"Norm of the difference between tdm3_10_sparse and tdm3_10_sigma is too large: {np.linalg.norm(tdm3_10_sparse - tdm3_10_sigma):.12f}."
    assert np.allclose(tdm3_10_sparse, tdm3_01_sparse.T.conj())
    assert np.allclose(tdm3_10_sigma, tdm3_01_sigma.T.conj())


def test_ci_2cumulant():
    lambda2 = ci.make_2cumulant(0)
    rdm1 = ci.make_1rdm(0)
    rdm2 = ci.make_2rdm(0)
    lambda2_ref = (
        rdm2
        - np.einsum("pr,qs->pqrs", rdm1, rdm1)
        + np.einsum("ps,qr->pqrs", rdm1, rdm1)
    )
    assert np.linalg.norm(lambda2 - lambda2_ref) < rdm_threshold


def test_ci_3cumulant():
    # A different and "dumb" way to compute the 3-cumulant, from its definition
    # l^{pqr}_{stu} = g^{pqr}_{stu} - \sum(-1)^p (g^p_s l^{qr}_{tu}) - det(g^p_s g^q_t g^r_u)
    # see eqs (34) and (40a) in 10.1063/1.474405
    lambda3 = ci.make_3cumulant(0)
    rdm1 = ci.make_1rdm(0)
    lambda2 = ci.make_2cumulant(0)
    rdm3 = ci.make_3rdm(0)
    lambda3_ref = rdm3 - (
        +np.einsum("ps,qrtu->pqrstu", rdm1, lambda2, optimize=True)
        - np.einsum("pt,qrsu->pqrstu", rdm1, lambda2, optimize=True)
        - np.einsum("pu,qrts->pqrstu", rdm1, lambda2, optimize=True)
        - np.einsum("qs,prtu->pqrstu", rdm1, lambda2, optimize=True)
        + np.einsum("qt,prsu->pqrstu", rdm1, lambda2, optimize=True)
        + np.einsum("qu,prts->pqrstu", rdm1, lambda2, optimize=True)
        - np.einsum("rs,qptu->pqrstu", rdm1, lambda2, optimize=True)
        + np.einsum("rt,qpsu->pqrstu", rdm1, lambda2, optimize=True)
        + np.einsum("ru,qpts->pqrstu", rdm1, lambda2, optimize=True)
    )
    lambda3_ref -= (
        +np.einsum("ps,qt,ru->pqrstu", rdm1, rdm1, rdm1, optimize=True)
        - np.einsum("pt,qs,ru->pqrstu", rdm1, rdm1, rdm1, optimize=True)
        - np.einsum("ps,qu,rt->pqrstu", rdm1, rdm1, rdm1, optimize=True)
        - np.einsum("pu,qt,rs->pqrstu", rdm1, rdm1, rdm1, optimize=True)
        + np.einsum("pu,qs,rt->pqrstu", rdm1, rdm1, rdm1, optimize=True)
        + np.einsum("pt,qu,rs->pqrstu", rdm1, rdm1, rdm1, optimize=True)
    )

    assert (
        np.linalg.norm(lambda3 - lambda3_ref) < rdm_threshold
    ), f"Norm of the difference between lambda3 and lambda3_ref is too large: {np.linalg.norm(lambda3 - lambda3_ref):.12f}."
