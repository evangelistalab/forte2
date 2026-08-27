import math

import pytest

from forte2.lib import sparse_ops
from forte2.lib.det import Determinant


def det(value):
    return Determinant(value)


def modes(alpha=(), beta=()):
    result = Determinant.zero()
    for orbital in alpha:
        result.set_na(orbital, True)
    for orbital in beta:
        result.set_nb(orbital, True)
    return result


def coherent_pair(scale=1.0, phase=1.0):
    return sparse_ops.SparseState(
        {
            det("20"): scale * math.sqrt(0.6) * phase,
            det("02"): scale * 1j * math.sqrt(0.4) * phase,
        }
    )


@pytest.mark.parametrize("max_cumulant", (1, 2, 3, 4))
def test_reference_identity_elements_are_one(max_cumulant):
    reference = sparse_ops.CumulantReference(
        coherent_pair(), 2, max_cumulant=max_cumulant
    )
    zero = Determinant.zero()

    assert reference.rdm(zero, zero) == pytest.approx(1.0)
    assert reference.truncated_rdm(zero, zero) == pytest.approx(1.0)
    assert reference.cumulant(zero, zero) == pytest.approx(1.0)


@pytest.mark.parametrize("scale,phase", ((3.7, 1.0), (0.2, 1j), (2.3, -0.6 + 0.8j)))
def test_reference_is_invariant_to_normalization_and_global_phase(scale, phase):
    baseline = sparse_ops.CumulantReference(coherent_pair(), 2, max_cumulant=4)
    transformed = sparse_ops.CumulantReference(
        coherent_pair(scale=scale, phase=phase), 2, max_cumulant=4
    )

    for p in range(2):
        for q in range(2):
            for alpha in (False, True):
                for beta in (False, True):
                    assert transformed.gamma(p, alpha, q, beta) == pytest.approx(
                        baseline.gamma(p, alpha, q, beta), abs=1.0e-13
                    )
    for rank in range(2, 5):
        assert transformed.cumulant_size(rank) == baseline.cumulant_size(rank)
    assert transformed.cumulant(det("22"), det("22")) == pytest.approx(
        baseline.cumulant(det("22"), det("22")), abs=1.0e-13
    )


def test_one_body_density_is_hermitian_and_has_the_particle_number_trace():
    vacuum = sparse_ops.SparseState(
        {
            det("a00"): math.sqrt(0.5),
            det("0a0"): 1j * math.sqrt(0.3),
            det("00a"): -math.sqrt(0.2),
        }
    )
    reference = sparse_ops.CumulantReference(vacuum, 3, max_cumulant=3)

    trace = 0.0
    for p in range(3):
        trace += reference.gamma(p, True, p, True).real
        for q in range(3):
            assert reference.gamma(p, True, q, True) == pytest.approx(
                reference.gamma(q, True, p, True).conjugate(), abs=1.0e-13
            )
            assert reference.gamma(p, True, q, False) == pytest.approx(0.0)
    assert trace == pytest.approx(1.0)


@pytest.mark.parametrize("rank", (2, 3, 4))
def test_cumulants_are_hermitian(rank):
    vacuum = sparse_ops.SparseState(
        {
            det("200"): math.sqrt(0.5),
            det("020"): math.sqrt(0.3),
            det("002"): 1j * math.sqrt(0.2),
        }
    )
    reference = sparse_ops.CumulantReference(vacuum, 3, max_cumulant=4)
    upper = modes(alpha=range(min(rank, 3)), beta=range(max(0, rank - 3)))
    lower = modes(alpha=range(max(0, 3 - rank), 3), beta=range(max(0, rank - 3)))

    assert reference.cumulant(upper, lower) == pytest.approx(
        reference.cumulant(lower, upper).conjugate(), abs=1.0e-13
    )


def test_eta_is_delta_minus_gamma_for_all_spin_blocks():
    reference = sparse_ops.CumulantReference(coherent_pair(), 2)

    for p in range(2):
        for q in range(2):
            for p_alpha in (False, True):
                for q_alpha in (False, True):
                    delta = float(p == q and p_alpha == q_alpha)
                    assert reference.eta(p, p_alpha, q, q_alpha) == pytest.approx(
                        delta - reference.gamma(p, p_alpha, q, q_alpha)
                    )


@pytest.mark.parametrize("max_cumulant", (1, 2, 3, 4))
def test_determinant_moments_reconstruct_exactly_at_higher_rank(max_cumulant):
    reference = sparse_ops.CumulantReference(
        sparse_ops.SparseState({det("2220"): 2.5j}),
        4,
        max_cumulant=max_cumulant,
    )
    occupied = det("2220")
    mixed = modes(alpha=(0, 1, 3), beta=(0, 1, 2))

    assert reference.truncated_rdm(occupied, occupied) == pytest.approx(1.0)
    assert reference.truncated_rdm(mixed, mixed) == pytest.approx(0.0)
    for rank in range(2, max_cumulant + 1):
        assert reference.cumulant_size(rank) == 0


def test_higher_cumulants_have_only_active_indices():
    vacuum = sparse_ops.SparseState(
        {det("2200"): math.sqrt(0.7), det("2020"): math.sqrt(0.3)}
    )
    reference = sparse_ops.CumulantReference(vacuum, 4, max_cumulant=4)
    core_active = det("aa00")
    active_virtual = det("0a0a")

    assert reference.cumulant(core_active, core_active) == pytest.approx(0.0)
    assert reference.cumulant(active_virtual, active_virtual) == pytest.approx(0.0)


@pytest.mark.parametrize("max_cumulant", (2, 3, 4))
def test_too_few_active_modes_produce_no_high_rank_cumulants(max_cumulant):
    vacuum = sparse_ops.SparseState(
        {det("a00"): math.sqrt(0.5), det("0a0"): math.sqrt(0.5)}
    )
    reference = sparse_ops.CumulantReference(vacuum, 3, max_cumulant=max_cumulant)

    for rank in range(2, max_cumulant + 1):
        if 2 * rank > reference.active_modes().count():
            assert reference.cumulant_size(rank) == 0


@pytest.mark.parametrize(
    "factory,match",
    (
        (lambda: sparse_ops.CumulantReference(coherent_pair(), 0), "norb"),
        (
            lambda: sparse_ops.CumulantReference(
                coherent_pair(), Determinant.maxnorb + 1
            ),
            "norb",
        ),
        (
            lambda: sparse_ops.CumulantReference(
                coherent_pair(), 2, screen_thresh=-1.0
            ),
            "screen_thresh",
        ),
        (
            lambda: sparse_ops.CumulantReference(
                sparse_ops.SparseState(
                    {det("20"): 0.5, det("02"): 0.5, det("ab"): 0.5}
                ),
                2,
                screen_thresh=0.6,
            ),
            "significant",
        ),
        (
            lambda: sparse_ops.CumulantReference(
                sparse_ops.SparseState({det("002"): 1.0}), 2
            ),
            "outside norb",
        ),
    ),
)
def test_reference_rejects_invalid_construction_edges(factory, match):
    with pytest.raises((ValueError, IndexError), match=match):
        factory()


def test_reference_rejects_invalid_rdm_and_cumulant_queries():
    reference = sparse_ops.CumulantReference(coherent_pair(), 2, max_cumulant=2)
    zero = Determinant.zero()

    with pytest.raises(ValueError, match="ranks must match"):
        reference.rdm(det("a0"), zero)
    with pytest.raises(ValueError, match="ranks must match"):
        reference.truncated_rdm(det("a0"), zero)
    with pytest.raises(IndexError, match="outside"):
        reference.rdm(det("00a"), det("00a"))
    with pytest.raises(IndexError, match="unavailable"):
        reference.cumulant(det("2a"), det("2a"))
    for rank in (0, 3):
        with pytest.raises(IndexError, match="unavailable"):
            reference.cumulant_size(rank)
