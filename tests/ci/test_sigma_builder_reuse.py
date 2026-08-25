import numpy as np
import pytest

from forte2.lib.ci_helpers import (
    CIStrings,
    CISigmaBuilder,
    RelCISigmaBuilder,
    SelectedCIHelper,
    RelSelectedCIHelper,
)
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

    builder = CISigmaBuilder(lists, E, H, V, 0, algorithm)

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

    builder = RelCISigmaBuilder(lists, E, H, V, 0, "hz")

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

    builder = CISigmaBuilder(lists, E, H, V, 0, algorithm)

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

    reused_builder = CISigmaBuilder(lists, 0.1, H_a, V_a, 0, algorithm)
    reused_builder.set_Hamiltonian(0.2, H_b, V_b)

    fresh_builder = CISigmaBuilder(lists, 0.2, H_b, V_b, 0, algorithm)

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

    builder = RelCISigmaBuilder(lists, E, H, V, 0, "hz")

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

    builder = RelCISigmaBuilder(lists, 0.0, H, V, 0, "hz")
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

    reused_builder = RelCISigmaBuilder(lists, 0.1, H_a, V_a, 0, "hz")
    reused_builder.set_Hamiltonian(0.2, H_b, V_b)

    fresh_builder = RelCISigmaBuilder(lists, 0.2, H_b, V_b, 0, "hz")

    basis = rng.normal(size=lists.ndet) + 1j * rng.normal(size=lists.ndet)
    sigma_reused = np.zeros(lists.ndet, dtype=complex)
    reused_builder.Hamiltonian(basis, sigma_reused)
    sigma_fresh = np.zeros(lists.ndet, dtype=complex)
    fresh_builder.Hamiltonian(basis, sigma_fresh)

    np.testing.assert_allclose(sigma_reused, sigma_fresh, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("algorithm", ["kh", "hz"])
def test_set_hamiltonian_e_only_update_keeps_h_v(algorithm):
    """E is independent of H/V for both algorithms, so set_Hamiltonian(E=...) alone must work
    and leave H/V untouched, regardless of algorithm."""
    norb = 6
    lists = CIStrings(2, 2, 0, [[0] * norb], [], [])
    rng = np.random.default_rng(6)
    H, V = _random_real_integrals(rng, norb, symmetric_H=True)
    E_a, E_b = 0.1, 0.2

    reused_builder = CISigmaBuilder(lists, E_a, H, V, 0, algorithm)
    reused_builder.set_Hamiltonian(E=E_b)

    fresh_builder = CISigmaBuilder(lists, E_b, H, V, 0, algorithm)

    basis = rng.normal(size=lists.ndet)
    sigma_reused = np.zeros(lists.ndet)
    reused_builder.Hamiltonian(basis, sigma_reused)
    sigma_fresh = np.zeros(lists.ndet)
    fresh_builder.Hamiltonian(basis, sigma_fresh)

    np.testing.assert_allclose(sigma_reused, sigma_fresh, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("which", ["H", "V"])
def test_hz_set_hamiltonian_partial_update_keeps_other_argument(which):
    """Harrison-Zarrabian's h_hz and v_pr_qs/v_pr_qs_a depend on only H or only V respectively,
    so set_Hamiltonian with just one of them given must update only that argument."""
    norb = 6
    lists = CIStrings(2, 2, 0, [[0] * norb], [], [])
    rng = np.random.default_rng(6)
    H_a, V_a = _random_real_integrals(rng, norb, symmetric_H=True)
    H_b, V_b = _random_real_integrals(rng, norb, symmetric_H=False)

    reused_builder = CISigmaBuilder(lists, 0.1, H_a, V_a, 0, "hz")
    reused_builder.set_Hamiltonian(**{which: {"H": H_b, "V": V_b}[which]})

    fresh_builder = CISigmaBuilder(
        lists,
        0.1,
        H_b if which == "H" else H_a,
        V_b if which == "V" else V_a,
        0,
        "hz",
    )

    basis = rng.normal(size=lists.ndet)
    sigma_reused = np.zeros(lists.ndet)
    reused_builder.Hamiltonian(basis, sigma_reused)
    sigma_fresh = np.zeros(lists.ndet)
    fresh_builder.Hamiltonian(basis, sigma_fresh)

    np.testing.assert_allclose(sigma_reused, sigma_fresh, rtol=1e-12, atol=1e-12)


def test_kh_set_hamiltonian_requires_h_and_v_together():
    """Knowles-Handy's h_kh mixes H and V, and neither is cached between calls, so
    set_Hamiltonian must reject H-only or V-only updates and accept H-and-V-together."""
    norb = 6
    lists = CIStrings(2, 2, 0, [[0] * norb], [], [])
    rng = np.random.default_rng(6)
    H_a, V_a = _random_real_integrals(rng, norb, symmetric_H=True)
    H_b, V_b = _random_real_integrals(rng, norb, symmetric_H=False)

    builder = CISigmaBuilder(lists, 0.1, H_a, V_a, 0, "kh")

    with pytest.raises(RuntimeError, match="given together"):
        builder.set_Hamiltonian(H=H_b)
    with pytest.raises(RuntimeError, match="given together"):
        builder.set_Hamiltonian(V=V_b)

    builder.set_Hamiltonian(H=H_b, V=V_b)
    fresh_builder = CISigmaBuilder(lists, 0.1, H_b, V_b, 0, "kh")

    basis = rng.normal(size=lists.ndet)
    sigma_reused = np.zeros(lists.ndet)
    builder.Hamiltonian(basis, sigma_reused)
    sigma_fresh = np.zeros(lists.ndet)
    fresh_builder.Hamiltonian(basis, sigma_fresh)

    np.testing.assert_allclose(sigma_reused, sigma_fresh, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("which", ["E", "H", "V"])
def test_rel_set_hamiltonian_partial_update_keeps_other_arguments(which):
    norb = 5
    lists = CIStrings(3, 0, 0, [[0] * norb], [], [])
    rng = np.random.default_rng(9)
    H_a, V_a = _random_complex_integrals(rng, norb, symmetric_H=True)
    H_b, V_b = _random_complex_integrals(rng, norb, symmetric_H=True)
    E_a, E_b = 0.1, 0.2

    reused_builder = RelCISigmaBuilder(lists, E_a, H_a, V_a, 0, "hz")
    reused_builder.set_Hamiltonian(**{which: {"E": E_b, "H": H_b, "V": V_b}[which]})

    fresh_builder = RelCISigmaBuilder(
        lists,
        E_b if which == "E" else E_a,
        H_b if which == "H" else H_a,
        V_b if which == "V" else V_a,
        0,
        "hz",
    )

    basis = rng.normal(size=lists.ndet) + 1j * rng.normal(size=lists.ndet)
    sigma_reused = np.zeros(lists.ndet, dtype=complex)
    reused_builder.Hamiltonian(basis, sigma_reused)
    sigma_fresh = np.zeros(lists.ndet, dtype=complex)
    fresh_builder.Hamiltonian(basis, sigma_fresh)

    np.testing.assert_allclose(sigma_reused, sigma_fresh, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("which", ["H", "V"])
def test_sci_helper_set_hamiltonian_partial_update_keeps_other_argument(which):
    """SelectedCIHelper.set_Hamiltonian with only H or only V given must not touch the other.
    The diagonal (H0) contribution is cached from construction and is not part of this class's
    reuse contract, so the comparison uses a single-determinant basis vector: this isolates the
    off-diagonal sigma entries, which are rebuilt from the (possibly partially updated) integrals
    on every call."""
    norb = 4
    lists = CIStrings(2, 2, 0, [[0] * norb], [], [])
    dets = lists.make_determinants()
    ndet = len(dets)
    rng = np.random.default_rng(10)
    H_a, V_a = _random_real_integrals(rng, norb, symmetric_H=True)
    H_b, V_b = _random_real_integrals(rng, norb, symmetric_H=False)
    c = np.zeros((ndet, 1))
    c[0, 0] = 1.0

    reused_helper = SelectedCIHelper(norb, dets, c, 0.1, H_a, V_a, 0)
    reused_helper.set_Hamiltonian(**{which: {"H": H_b, "V": V_b}[which]})

    fresh_helper = SelectedCIHelper(
        norb,
        dets,
        c,
        0.1,
        H_b if which == "H" else H_a,
        V_b if which == "V" else V_a,
        0,
    )

    basis = np.zeros(ndet)
    basis[0] = 1.0
    sigma_reused = np.zeros(ndet)
    reused_helper.Hamiltonian(basis, sigma_reused)
    sigma_fresh = np.zeros(ndet)
    fresh_helper.Hamiltonian(basis, sigma_fresh)

    np.testing.assert_allclose(
        sigma_reused[1:], sigma_fresh[1:], rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize("which", ["H", "V"])
def test_rel_sci_helper_set_hamiltonian_partial_update_keeps_other_argument(which):
    norb = 4
    lists = CIStrings(3, 0, 0, [[0] * norb], [], [])
    dets = lists.make_determinants()
    ndet = len(dets)
    rng = np.random.default_rng(11)
    H_a, V_a = _random_complex_integrals(rng, norb, symmetric_H=True)
    H_b, V_b = _random_complex_integrals(rng, norb, symmetric_H=True)
    c = np.zeros((ndet, 1), dtype=complex)
    c[0, 0] = 1.0

    reused_helper = RelSelectedCIHelper(norb, dets, c, 0.1, H_a, V_a, 0)
    reused_helper.set_Hamiltonian(**{which: {"H": H_b, "V": V_b}[which]})

    fresh_helper = RelSelectedCIHelper(
        norb,
        dets,
        c,
        0.1,
        H_b if which == "H" else H_a,
        V_b if which == "V" else V_a,
        0,
    )

    basis = np.zeros(ndet, dtype=complex)
    basis[0] = 1.0
    sigma_reused = np.zeros(ndet, dtype=complex)
    reused_helper.Hamiltonian(basis, sigma_reused)
    sigma_fresh = np.zeros(ndet, dtype=complex)
    fresh_helper.Hamiltonian(basis, sigma_fresh)

    np.testing.assert_allclose(
        sigma_reused[1:], sigma_fresh[1:], rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize("algorithm", ["kh", "hz"])
def test_constructor_algorithm_is_fixed_for_lifetime(algorithm):
    """The algorithm is chosen once, at construction, and get_algorithm() must report it;
    there is no runtime setter to switch it afterwards."""
    norb = 6
    lists = CIStrings(2, 2, 0, [[0] * norb], [], [])
    rng = np.random.default_rng(13)
    H, V = _random_real_integrals(rng, norb, symmetric_H=False)
    E = 0.3

    builder = CISigmaBuilder(lists, E, H, V, 0, algorithm)
    assert builder.get_algorithm() == (
        "Knowles-Handy" if algorithm == "kh" else "Harrison-Zarrabian"
    )
    assert not hasattr(builder, "set_algorithm")

    with pytest.raises(RuntimeError):
        CISigmaBuilder(lists, E, H, V, 0, "not-a-real-algorithm")


def test_rel_constructor_algorithm_defaults_to_hz_and_validates():
    norb = 5
    lists = CIStrings(3, 0, 0, [[0] * norb], [], [])
    rng = np.random.default_rng(14)
    H, V = _random_complex_integrals(rng, norb, symmetric_H=True)

    builder = RelCISigmaBuilder(lists, 0.1, H, V, 0, "hz")
    assert builder.get_algorithm() == "Harrison-Zarrabian"

    with pytest.raises(RuntimeError):
        RelCISigmaBuilder(lists, 0.1, H, V, 0, "kh")
