import math

import numpy as np
import pytest

import forte2
from forte2.dsrg.sparse_mrdsrg2 import (
    SparseMRDSRG,
    SparseMRDSRGExcitation,
    canonical_operator_label,
    regularized_denominator,
)
from forte2.lib import sparse_ops
from forte2.lib.det import Determinant


def det(value):
    return Determinant(value)


def vacuum():
    return sparse_ops.SparseState(
        {det("20"): math.sqrt(0.7), det("02"): math.sqrt(0.3)}
    )


def excitation(label="[1a+ 0a-]", denominator=-0.8, rank=1):
    return SparseMRDSRGExcitation(
        sqop=sparse_ops.sqop(label)[0],
        denominator=denominator,
        rank=rank,
        label=label,
    )


def tiny_problem():
    reference = sparse_ops.SparseState({det("a0"): 1.0})
    hamiltonian = sparse_ops.sparse_operator(
        [
            ("[0a+ 0a-]", -0.4),
            ("[1a+ 1a-]", 0.6),
            ("[1a+ 0a-]", 0.02),
            ("[0a+ 1a-]", 0.02),
        ]
    )
    return hamiltonian, reference, (excitation(denominator=-1.0),)


@pytest.mark.parametrize("backend", ("sparse", "cumulant", "validate", "rdm"))
def test_no_excitation_problem_is_well_defined_for_every_backend(backend):
    hamiltonian = sparse_ops.sparse_operator("[]", -1.25)
    result = forte2.solve_sparse_mrdsrg2(
        hamiltonian, vacuum(), 2, [], gno_backend=backend
    )

    assert result.converged
    assert result.iterations == 0
    assert result.energy == pytest.approx(-1.25)
    assert result.scalar_energy == pytest.approx(-1.25)
    assert result.model_space_energies is None
    assert result.max_cumulant == 3
    assert not result.include_four_body_cumulant
    assert len(result.amplitudes) == 0
    assert len(result.history) == 0


@pytest.mark.parametrize(
    "kwargs,expected_include",
    (
        ({}, False),
        ({"include_four_body_cumulant": True}, True),
        ({"max_cumulant": 4}, True),
        ({"max_cumulant": 4, "include_four_body_cumulant": True}, True),
        ({"max_cumulant": 4, "gno_backend": "rdm"}, False),
    ),
)
def test_four_body_option_normalization_and_backward_compatibility(
    kwargs, expected_include
):
    result = forte2.solve_sparse_mrdsrg2(
        sparse_ops.sparse_operator("[]", 0.0), vacuum(), 2, [], **kwargs
    )

    assert result.include_four_body_cumulant is expected_include
    assert result.max_cumulant == (
        4 if kwargs.get("max_cumulant") == 4 or expected_include else 3
    )
    assert result.hbar.max_cumulant() == result.max_cumulant


@pytest.mark.parametrize(
    "kwargs,match",
    (
        ({"gno_backend": "unknown"}, "gno_backend"),
        ({"max_rank": 0}, "max_rank"),
        ({"max_cumulant": -2}, "max_cumulant"),
        ({"gno_validation_tol": -1.0}, "gno_validation_tol"),
        ({"flow_param": -1.0}, "flow_param"),
        ({"screen_thresh": -1.0}, "screen_thresh"),
        ({"commutator_threshold": -1.0}, "commutator_threshold"),
        ({"max_commutators": -1}, "max_commutators"),
        ({"maxiter": -1}, "maxiter"),
        ({"e_tol": -1.0}, "e_tol"),
        ({"r_tol": -1.0}, "r_tol"),
        ({"damping": 0.0}, "damping"),
        ({"damping": 1.01}, "damping"),
        ({"model_space": []}, "model_space"),
        (
            {"include_four_body_cumulant": True, "max_cumulant": 2},
            "include_four_body_cumulant",
        ),
        (
            {"include_four_body_cumulant": True, "max_cumulant": 5},
            "include_four_body_cumulant",
        ),
        (
            {"include_four_body_cumulant": True, "gno_backend": "rdm"},
            "include_four_body_cumulant",
        ),
    ),
)
def test_solver_rejects_invalid_option_edges(kwargs, match):
    with pytest.raises(ValueError, match=match):
        SparseMRDSRG(sparse_ops.sparse_operator("[]", 0.0), vacuum(), 2, [], **kwargs)


@pytest.mark.parametrize(
    "denominator,flow_param",
    ((0.0, 0.0), (0.0, 5.0), (1.0e-16, 2.0), (0.7, 0.0)),
)
def test_regularized_denominator_zero_limits(denominator, flow_param):
    assert regularized_denominator(denominator, flow_param) == pytest.approx(
        flow_param * denominator
    )


@pytest.mark.parametrize("denominator", (0.1, 0.7, 3.0))
def test_regularized_denominator_is_odd(denominator):
    assert regularized_denominator(-denominator, 0.8) == pytest.approx(
        -regularized_denominator(denominator, 0.8)
    )


def test_canonical_operator_label_orders_spin_orbitals_and_directions():
    label = canonical_operator_label(
        [(2, "b"), (1, "a"), (0, "b"), (0, "a")],
        [(0, "a"), (2, "b"), (1, "a"), (0, "b")],
    )

    assert label == "[0a+ 1a+ 0b+ 2b+ 2b- 0b- 1a- 0a-]"


def test_initial_amplitude_shape_is_checked_after_excitation_filtering():
    hamiltonian, reference, excitations = tiny_problem()
    solver = SparseMRDSRG(
        hamiltonian,
        reference,
        2,
        excitations,
        initial_amplitudes=np.zeros(2),
        maxiter=0,
    )

    with pytest.raises(ValueError, match="initial_amplitudes"):
        solver.run()


def test_excitations_above_solver_rank_are_filtered_before_initialization():
    rank1 = excitation(rank=1)
    rank3 = excitation("[0a+ 1a+ 0b+ 1b- 0b- 0a-]", denominator=-2.0, rank=3)
    solver = SparseMRDSRG(
        sparse_ops.sparse_operator("[]", 0.0),
        vacuum(),
        2,
        [rank1, rank3],
        max_rank=2,
        initial_amplitudes=np.zeros(1),
        maxiter=0,
    )

    assert solver.excitations == (rank1,)
    result_excitations = solver.run().excitations
    assert len(result_excitations) == 1
    assert result_excitations[0].rank == 1
    assert result_excitations[0].label == rank1.label


@pytest.mark.parametrize("rank", (2, 3, 4))
def test_fixed_rank_wrappers_reject_a_conflicting_max_rank(rank):
    solver = getattr(forte2, f"solve_sparse_mrdsrg{rank}")
    with pytest.raises(ValueError, match=f"requires max_rank={rank}"):
        solver(
            sparse_ops.sparse_operator("[]", 0.0),
            vacuum(),
            2,
            [],
            max_rank=rank - 1,
        )


def test_zero_commutators_and_zero_maxiter_produce_one_finite_iteration():
    hamiltonian, reference, excitations = tiny_problem()
    result = forte2.solve_sparse_mrdsrg2(
        hamiltonian,
        reference,
        2,
        excitations,
        max_commutators=0,
        maxiter=0,
        initial_amplitudes=np.zeros(1),
    )

    assert not result.converged
    assert result.iterations == 1
    assert len(result.history) == 1
    assert result.history[0].ncomm == 0
    assert np.isfinite(result.energy)
    assert np.all(np.isfinite(result.amplitudes))


def test_repeated_run_resets_history_and_is_reproducible():
    hamiltonian, reference, excitations = tiny_problem()
    solver = SparseMRDSRG(
        hamiltonian,
        reference,
        2,
        excitations,
        max_commutators=2,
        maxiter=1,
        initial_amplitudes=np.zeros(1),
        do_diis=False,
    )

    first = solver.run()
    second = solver.run()

    assert first.iterations == second.iterations == 2
    assert len(solver.history) == 2
    assert second.energy == pytest.approx(first.energy, abs=1.0e-14)
    assert second.amplitudes == pytest.approx(first.amplitudes, abs=1.0e-14)
    assert [item.energy for item in second.history] == pytest.approx(
        [item.energy for item in first.history], abs=1.0e-14
    )


@pytest.mark.parametrize("max_rank", (0, -1))
def test_excitation_enumerator_returns_empty_for_nonpositive_rank(max_rank):
    assert (
        forte2.enumerate_mrdsrg_excitations(
            core_orbitals=[0],
            active_orbitals=[1],
            virtual_orbitals=[2],
            orbital_energies=[-1.0, 0.0, 1.0],
            max_rank=max_rank,
        )
        == ()
    )
