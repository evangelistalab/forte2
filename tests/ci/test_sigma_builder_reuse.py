import numpy as np
import pytest

from forte2.lib.ci_helpers import CIStrings, CISigmaBuilder, RelCISigmaBuilder
from forte2.lib import sparse_ops as sop
from forte2.helpers.comparisons import approx


def _random_real_integrals(rng, norb, symmetric_H):
    H = rng.normal(size=(norb, norb))
    if symmetric_H:
        H = H + H.T
    V = rng.normal(size=(norb, norb, norb, norb))
    V = V + V.transpose(2, 1, 0, 3)  # p <-> r
    V = V + V.transpose(0, 3, 2, 1)  # q <-> s
    V = V + V.transpose(1, 0, 3, 2)  # (p,r) <-> (q,s)
    return H, V


def _random_complex_integrals(rng, norb, symmetric_H):
    H = rng.normal(size=(norb, norb)) + 1j * rng.normal(size=(norb, norb))
    if symmetric_H:
        H = H + H.T.conj()
    V = rng.normal(size=(norb, norb, norb, norb)) + 1j * rng.normal(
        size=(norb, norb, norb, norb)
    )
    V = V + V.transpose(1, 0, 3, 2)  # electron exchange: <pq|rs> = <qp|sr>
    V = V + V.transpose(2, 3, 0, 1).conj()  # Hermiticity: <pq|rs> = <rs|pq>*
    return H, V


@pytest.mark.parametrize("algorithm", ["kh", "hz"])
@pytest.mark.parametrize("symmetric_H", [True, False])
def test_real_sigma_build_against_sparse_ops(algorithm, symmetric_H):
    norb = 6
    lists = CIStrings(2, 2, 0, [[0] * norb], [], [])
    rng = np.random.default_rng(0)
    H, V = _random_real_integrals(rng, norb, symmetric_H)
    E = 0.37

    builder = CISigmaBuilder(lists, E, H, V, 0)
    builder.set_algorithm(algorithm)

    basis = rng.normal(size=lists.ndet)

    sigma_fused = np.zeros(lists.ndet)
    builder.Hamiltonian(basis, sigma_fused)

    sparse_ham = sop.sparse_operator_hamiltonian(E, H, V)
    dets = lists.make_determinants()
    sparse_state = sop.SparseState({det: coeff for det, coeff in zip(dets, basis)})
    sparse_sigma = sop.apply_op(sparse_ham, sparse_state)
    sigma_coeffs = [sparse_sigma[det] for det in dets]
    assert sigma_coeffs == approx(sigma_fused)


@pytest.mark.parametrize("symmetric_H", [True, False])
def test_complex_sigma_build_against_sparse_ops(symmetric_H):
    norb = 12
    lists = CIStrings(4, 0, 0, [[0] * norb], [], [])
    rng = np.random.default_rng(0)
    H, V = _random_complex_integrals(rng, norb, symmetric_H)
    E = 0.37

    builder = RelCISigmaBuilder(lists, E, H, V, 0)
    builder.set_algorithm("hz")

    basis = rng.normal(size=lists.ndet) + 1j * rng.normal(size=lists.ndet)

    sigma_fused = np.zeros(lists.ndet, dtype=complex)
    builder.Hamiltonian(basis, sigma_fused)

    sparse_ham = sop.sparse_operator_hamiltonian(E, H, V)
    dets = lists.make_determinants()
    sparse_state = sop.SparseState({det: coeff for det, coeff in zip(dets, basis)})
    sparse_sigma = sop.apply_op(sparse_ham, sparse_state)
    sigma_coeffs = [sparse_sigma[det] for det in dets]
    assert sigma_coeffs == approx(sigma_fused)


@pytest.mark.parametrize("algorithm", ["kh", "hz"])
@pytest.mark.parametrize("symmetric_H", [True, False])
def test_sigma_one_two_electron_sum_to_hamiltonian(algorithm, symmetric_H):
    """sigma_one_electron(basis) + sigma_two_electron(basis) == Hamiltonian(basis),
    including a non-symmetric H (the case our biorthogonalization generator needs)."""
    norb = 6
    lists = CIStrings(2, 2, 0, [[0] * norb], [], [])
    rng = np.random.default_rng(0)
    H, V = _random_real_integrals(rng, norb, symmetric_H)
    E = 0.37

    builder = CISigmaBuilder(lists, E, H, V, 0)
    builder.set_algorithm(algorithm)

    basis = rng.normal(size=lists.ndet)

    sigma_fused = np.zeros(lists.ndet)
    builder.Hamiltonian(basis, sigma_fused)

    sigma_1e = np.zeros(lists.ndet)
    builder.sigma_one_electron(basis, sigma_1e)
    sigma_2e = np.zeros(lists.ndet)
    builder.sigma_two_electron(basis, sigma_2e)

    np.testing.assert_allclose(sigma_1e + sigma_2e, sigma_fused, rtol=1e-12, atol=1e-12)


def test_sigma_one_electron_matches_hamiltonian_with_zero_V():
    """With V=0, sigma_one_electron must equal the fused Hamiltonian exactly,
    for a non-symmetric H -- the exact usage pattern of the biorthogonalized
    CI-vector transform (a one-body orbital-rotation generator, not a real
    Hamiltonian)."""
    norb = 5
    lists = CIStrings(2, 3, 0, [[0] * norb], [], [])
    rng = np.random.default_rng(1)
    H = rng.normal(size=(norb, norb))  # deliberately not symmetrized
    V = np.zeros((norb, norb, norb, norb))

    builder = CISigmaBuilder(lists, 0.0, H, V, 0)
    basis = rng.normal(size=lists.ndet)

    sigma_fused = np.zeros(lists.ndet)
    builder.Hamiltonian(basis, sigma_fused)
    sigma_1e = np.zeros(lists.ndet)
    builder.sigma_one_electron(basis, sigma_1e)

    np.testing.assert_allclose(sigma_1e, sigma_fused, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("algorithm", ["kh", "hz"])
def test_set_hamiltonian_matches_fresh_builder(algorithm):
    """Swapping in a new (same-norb) Hamiltonian via set_Hamiltonian must give
    identical results to constructing a fresh builder with that Hamiltonian."""
    norb = 6
    lists = CIStrings(2, 2, 0, [[0] * norb], [], [])
    rng = np.random.default_rng(2)
    H_a, V_a = _random_real_integrals(rng, norb, symmetric_H=True)
    H_b, V_b = _random_real_integrals(rng, norb, symmetric_H=False)

    reused_builder = CISigmaBuilder(lists, 0.1, H_a, V_a, 0)
    reused_builder.set_algorithm(algorithm)
    reused_builder.set_Hamiltonian(0.2, H_b, V_b)

    fresh_builder = CISigmaBuilder(lists, 0.2, H_b, V_b, 0)
    fresh_builder.set_algorithm(algorithm)

    basis = rng.normal(size=lists.ndet)
    sigma_reused = np.zeros(lists.ndet)
    reused_builder.Hamiltonian(basis, sigma_reused)
    sigma_fresh = np.zeros(lists.ndet)
    fresh_builder.Hamiltonian(basis, sigma_fresh)

    np.testing.assert_allclose(sigma_reused, sigma_fresh, rtol=1e-12, atol=1e-12)


def test_rel_sigma_one_two_electron_sum_to_hamiltonian():
    norb = 5
    lists = CIStrings(3, 0, 0, [[0] * norb], [], [])
    rng = np.random.default_rng(3)
    H, V = _random_complex_integrals(rng, norb, symmetric_H=True)
    E = -0.11

    builder = RelCISigmaBuilder(lists, E, H, V, 0)
    builder.set_algorithm("hz")

    basis = rng.normal(size=lists.ndet) + 1j * rng.normal(size=lists.ndet)

    sigma_fused = np.zeros(lists.ndet, dtype=complex)
    builder.Hamiltonian(basis, sigma_fused)

    sigma_1e = np.zeros(lists.ndet, dtype=complex)
    builder.sigma_one_electron(basis, sigma_1e)
    sigma_2e = np.zeros(lists.ndet, dtype=complex)
    builder.sigma_two_electron(basis, sigma_2e)

    np.testing.assert_allclose(sigma_1e + sigma_2e, sigma_fused, rtol=1e-12, atol=1e-12)


def test_rel_sigma_one_electron_matches_hamiltonian_with_zero_V():
    norb = 5
    lists = CIStrings(3, 0, 0, [[0] * norb], [], [])
    rng = np.random.default_rng(4)
    H = rng.normal(size=(norb, norb)) + 1j * rng.normal(
        size=(norb, norb)
    )  # non-Hermitian
    V = np.zeros((norb, norb, norb, norb), dtype=complex)

    builder = RelCISigmaBuilder(lists, 0.0, H, V, 0)
    builder.set_algorithm("hz")
    basis = rng.normal(size=lists.ndet) + 1j * rng.normal(size=lists.ndet)

    sigma_fused = np.zeros(lists.ndet, dtype=complex)
    builder.Hamiltonian(basis, sigma_fused)
    sigma_1e = np.zeros(lists.ndet, dtype=complex)
    builder.sigma_one_electron(basis, sigma_1e)

    np.testing.assert_allclose(sigma_1e, sigma_fused, rtol=1e-12, atol=1e-12)


def test_rel_set_hamiltonian_matches_fresh_builder():
    norb = 5
    lists = CIStrings(3, 0, 0, [[0] * norb], [], [])
    rng = np.random.default_rng(5)
    H_a, V_a = _random_complex_integrals(rng, norb, symmetric_H=True)
    H_b, V_b = _random_complex_integrals(rng, norb, symmetric_H=True)

    reused_builder = RelCISigmaBuilder(lists, 0.1, H_a, V_a, 0)
    reused_builder.set_algorithm("hz")
    reused_builder.set_Hamiltonian(0.2, H_b, V_b)

    fresh_builder = RelCISigmaBuilder(lists, 0.2, H_b, V_b, 0)
    fresh_builder.set_algorithm("hz")

    basis = rng.normal(size=lists.ndet) + 1j * rng.normal(size=lists.ndet)
    sigma_reused = np.zeros(lists.ndet, dtype=complex)
    reused_builder.Hamiltonian(basis, sigma_reused)
    sigma_fresh = np.zeros(lists.ndet, dtype=complex)
    fresh_builder.Hamiltonian(basis, sigma_fresh)

    np.testing.assert_allclose(sigma_reused, sigma_fresh, rtol=1e-12, atol=1e-12)
