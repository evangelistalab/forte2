import math

import pytest

from forte2.lib import sparse_ops
from forte2.lib.det import Determinant


def det(s):
    return Determinant(s)


def correlated_singlet_vacuum(weight0=0.7):
    """Return sqrt(w)|0^2> + sqrt(1-w)|1^2> in two spatial orbitals."""
    return sparse_ops.SparseState(
        {
            det("20"): math.sqrt(weight0),
            det("02"): math.sqrt(1.0 - weight0),
        }
    )


def one_alpha_delocalized_vacuum():
    return sparse_ops.SparseState(
        {
            det("a0"): 1.0 / math.sqrt(2.0),
            det("0a"): 1.0 / math.sqrt(2.0),
        }
    )


def generalized_normal_order(op, vacuum, norb=2, max_cumulant=2, max_rank=-1):
    return sparse_ops.generalized_normal_order(
        op,
        vacuum,
        norb,
        max_cumulant=max_cumulant,
        max_rank=max_rank,
        screen_thresh=1.0e-12,
    )


def generalized_normal_order_dict(op):
    return {term.str(): coeff for term, coeff in op}


def expectation(op, state):
    return sparse_ops.overlap(state, op.apply_to_state(state))


def assert_sparse_state_close(lhs, rhs, abs=1.0e-12):
    dets = {d for d, _ in lhs.items()} | {d for d, _ in rhs.items()}
    for d in dets:
        assert lhs[d] == pytest.approx(rhs[d], abs=abs)


def test_generalized_one_body_number_operator_uses_fractional_active_density():
    vacuum = correlated_singlet_vacuum(weight0=0.7)
    op = sparse_ops.sparse_operator("[0a+ 0a-]", 1.0)

    no_op = generalized_normal_order(op, vacuum, max_cumulant=1)
    terms = generalized_normal_order_dict(no_op)

    assert terms["[]"] == pytest.approx(0.7)
    assert terms["[0a+ 0a-]"] == pytest.approx(1.0)
    assert expectation(no_op.to_sparse_operator(), vacuum) == pytest.approx(0.7)
    assert no_op.to_sparse_operator() == op


def test_generalized_anti_number_operator_uses_hole_density():
    vacuum = correlated_singlet_vacuum(weight0=0.7)
    op = sparse_ops.sparse_operator("[]", 1.0) - sparse_ops.sparse_operator(
        "[0a+ 0a-]", 1.0
    )

    no_op = generalized_normal_order(op, vacuum, max_cumulant=1)
    terms = generalized_normal_order_dict(no_op)

    assert terms["[]"] == pytest.approx(0.3)
    assert terms["[0a+ 0a-]"] == pytest.approx(-1.0)
    assert expectation(no_op.to_sparse_operator(), vacuum) == pytest.approx(0.3)
    assert no_op.to_sparse_operator() == op


def test_generalized_one_body_operator_uses_offdiagonal_density():
    vacuum = one_alpha_delocalized_vacuum()
    op = sparse_ops.sparse_operator("[0a+ 1a-]", 1.0)

    no_op = generalized_normal_order(op, vacuum, max_cumulant=1)
    terms = generalized_normal_order_dict(no_op)

    assert terms["[]"] == pytest.approx(0.5)
    assert terms["[0a+ 1a-]"] == pytest.approx(1.0)
    assert expectation(no_op.to_sparse_operator(), vacuum) == pytest.approx(0.5)
    assert no_op.to_sparse_operator() == op


def test_generalized_two_body_scalar_and_lower_rank_terms_use_active_rdms():
    vacuum = correlated_singlet_vacuum(weight0=0.7)
    op = sparse_ops.sparse_operator("[0a+ 0b+ 0b- 0a-]", 1.0)

    no_op = generalized_normal_order(op, vacuum, max_cumulant=2)
    terms = generalized_normal_order_dict(no_op)

    assert terms["[]"] == pytest.approx(0.7)
    assert terms["[0a+ 0a-]"] == pytest.approx(0.7)
    assert terms["[0b+ 0b-]"] == pytest.approx(0.7)
    assert terms["[0a+ 0b+ 0b- 0a-]"] == pytest.approx(1.0)

    for term, _ in no_op:
        if not term.is_identity():
            single_term = sparse_ops.GeneralizedNormalOrderedSparseOperator(
                vacuum, 2, 2, term, 1.0
            )
            assert expectation(
                single_term.to_sparse_operator(), vacuum
            ) == pytest.approx(0.0)

    assert expectation(no_op.to_sparse_operator(), vacuum) == pytest.approx(0.7)
    assert no_op.to_sparse_operator() == op


def test_generalized_normal_order_round_trip_and_apply_for_multiterm_operator():
    vacuum = correlated_singlet_vacuum(weight0=0.7)
    op = sparse_ops.sparse_operator(
        [
            ("[]", 0.25),
            ("[0a+ 0a-]", 1.2),
            ("[1a+ 1a-]", -0.4),
            ("[0a+ 0b+ 0b- 0a-]", 0.8),
            ("[1a+ 1b+ 1b- 1a-]", -0.6),
            ("[0a+ 1b+ 1b- 0a-]", 0.2),
        ]
    )
    state = sparse_ops.SparseState(
        {
            det("20"): 0.2,
            det("02"): -0.3,
            det("ab"): 0.4,
            det("ba"): -0.5,
        }
    )

    no_op = generalized_normal_order(op, vacuum, max_cumulant=2)

    assert no_op.to_sparse_operator() == op
    assert_sparse_state_close(no_op.apply_to_state(state), op.apply_to_state(state))
    assert_sparse_state_close(no_op @ state, op.apply_to_state(state))


def test_generalized_normal_order_can_truncate_final_many_body_rank():
    vacuum = correlated_singlet_vacuum(weight0=0.7)
    op = sparse_ops.sparse_operator(
        [
            ("[0a+ 0a-]", 1.0),
            ("[0a+ 0b+ 0b- 0a-]", 1.0),
        ]
    )

    no_op = generalized_normal_order(op, vacuum, max_cumulant=2)
    one_body = generalized_normal_order(op, vacuum, max_cumulant=2, max_rank=1)

    assert all(term.count() <= 2 for term, _ in one_body)
    assert one_body == no_op.truncate(1)
    assert "[0a+ 0b+ 0b- 0a-]" not in generalized_normal_order_dict(one_body)


def test_generalized_normal_order_truncation_keeps_only_supported_contractions():
    vacuum = sparse_ops.SparseState({det("200"): 1.0})
    op = sparse_ops.sparse_operator("[0a+ 2b+ 2b- 0a-]", 1.0)

    no_op = generalized_normal_order(
        op, vacuum, norb=3, max_cumulant=2, max_rank=-1
    )
    one_body = generalized_normal_order(
        op, vacuum, norb=3, max_cumulant=2, max_rank=1
    )

    assert one_body == no_op.truncate(1)
    assert generalized_normal_order_dict(one_body) == {
        "[2b+ 2b-]": pytest.approx(1.0)
    }
    assert no_op.to_sparse_operator() == op
