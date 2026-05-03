import pytest

import forte2


def det(s):
    return forte2.Determinant(s)


def sparse_state_close(lhs, rhs, abs=1e-12):
    dets = {d for d, _ in lhs.items()} | {d for d, _ in rhs.items()}
    for d in dets:
        assert lhs[d] == pytest.approx(rhs[d], abs=abs)


def normal_order_dict(op):
    return {term.str(op.reference()): coeff for term, coeff in op}


def test_normal_order_occupied_number_operator():
    op = forte2.sparse_operator("[0a+ 0a-]", 1.0)
    no_op = forte2.normal_order(op, det("2"))

    terms = normal_order_dict(no_op)
    assert terms["{}"] == pytest.approx(1.0)
    assert terms["{0a- 0a+}"] == pytest.approx(-1.0)
    hole_term = next(term for term, _ in no_op if term.str(no_op.reference()) == "{0a- 0a+}")
    assert hole_term.cre().na(0)
    assert hole_term.ann().na(0)

    assert no_op.reference() == det("2")
    assert no_op.to_sparse_operator() == op


def test_normal_order_virtual_number_operator():
    op = forte2.sparse_operator("[1a+ 1a-]", 1.0)
    no_op = forte2.normal_order(op, det("2"))

    assert normal_order_dict(no_op) == {"{1a+ 1a-}": pytest.approx(1.0)}
    assert no_op.to_sparse_operator() == op


def test_normal_order_beta_and_partially_occupied_reference():
    op = forte2.sparse_operator("[2b+ 2b-]", 1.0)
    no_op = forte2.normal_order(op, det("a0b"))

    terms = normal_order_dict(no_op)
    assert terms["{}"] == pytest.approx(1.0)
    assert terms["{2b- 2b+}"] == pytest.approx(-1.0)
    assert no_op.to_sparse_operator() == op


def test_normal_order_same_mode_anti_number_round_trip():
    number = forte2.sparse_operator("[1a+ 1a-]", 1.0)
    anti_number = forte2.sparse_operator("[]", 1.0) - number

    no_op = forte2.normal_order(anti_number, det("2"))
    terms = normal_order_dict(no_op)
    assert terms["{}"] == pytest.approx(1.0)
    assert terms["{1a+ 1a-}"] == pytest.approx(-1.0)
    assert no_op.to_sparse_operator() == anti_number


def test_normal_order_complex_multiterm_round_trip_and_apply():
    op = forte2.sparse_operator(
        [
            ("[]", 0.25 - 0.1j),
            ("[0a+ 0a-]", 1.5 + 0.2j),
            ("[1a+ 0a-]", -0.3j),
            ("[1a+ 1b+ 0b- 0a-]", 0.7),
        ]
    )
    reference = det("2")
    no_op = forte2.normal_order(op, reference)

    assert no_op.to_sparse_operator() == op

    state = forte2.SparseState({det("20"): 0.5, det("02"): 0.25j, det("ab"): -0.4})
    sparse_result = op.apply_to_state(state)
    normal_result = no_op.apply_to_state(state)
    matmul_result = no_op @ state

    sparse_state_close(normal_result, sparse_result)
    sparse_state_close(matmul_result, sparse_result)


def test_normal_order_many_body_rank_truncation():
    op = forte2.sparse_operator(
        [
            ("[0a+ 0a-]", 1.0),
            ("[1a+ 1a-]", 2.0),
            ("[1a+ 1b+ 0b- 0a-]", 0.7),
        ]
    )
    reference = det("2")
    no_op = forte2.normal_order(op, reference)

    ranks = {term.str(reference): term.many_body_rank() for term, _ in no_op}
    assert ranks["{}"] == 0
    assert ranks["{0a- 0a+}"] == 1
    assert ranks["{1a+ 1a-}"] == 1
    assert ranks["{0a- 1a+ 0b- 1b+}"] == 2

    scalar_only = no_op.truncate(0)
    assert normal_order_dict(scalar_only) == {"{}": pytest.approx(1.0)}

    one_body = no_op.truncate(1)
    assert "{0a- 1a+ 0b- 1b+}" not in normal_order_dict(one_body)
    assert all(term.many_body_rank() <= 1 for term, _ in one_body)

    inline_one_body = forte2.normal_order(op, reference, max_rank=1)
    assert inline_one_body == one_body
    assert no_op.truncate(2) == no_op


def test_rank_screened_commutator_matches_truncated_commutator():
    lhs = forte2.sparse_operator(
        [
            ("[1a+ 0a-]", 0.2),
            ("[2a+ 2b+ 1b- 0a-]", -0.3),
            ("[1a+ 2a+ 1b+ 2b+ 0b- 0a-]", 0.4),
        ]
    )
    rhs = forte2.sparse_operator(
        [
            ("[0a+ 1a-]", 0.5),
            ("[1a+ 1b+ 0b- 0a-]", -0.7),
            ("[1a+ 2a+ 1b+ 2b+ 0b- 0a-]", 0.9),
        ]
    )
    reference = det("200")

    generic = forte2.normal_order(lhs.commutator(rhs), reference, max_rank=2).to_sparse_operator()
    screened = forte2.normal_order(
        lhs.rank_screened_commutator(rhs, max_rank=2), reference, max_rank=2
    ).to_sparse_operator()

    assert screened == generic


def test_normal_ordered_commutator_matches_sparse_commutator():
    lhs = forte2.sparse_operator(
        [
            ("[0a+ 0a-]", 0.4),
            ("[1a+ 0a-]", -0.2),
            ("[1a+ 1b+ 0b- 0a-]", 0.3),
        ]
    )
    rhs = forte2.sparse_operator(
        [
            ("[0a+ 1a-]", 0.5),
            ("[1b+ 0b-]", -0.7),
            ("[1a+ 1b+ 0b- 0a-]", 0.9),
        ]
    )
    reference = det("20")

    lhs_no = forte2.normal_order(lhs, reference, max_rank=2)
    rhs_no = forte2.normal_order(rhs, reference, max_rank=2)
    direct_no = lhs_no.commutator(rhs_no, max_rank=2)
    sparse_no = forte2.normal_order(lhs.commutator(rhs), reference, max_rank=2)

    assert direct_no == sparse_no
    assert direct_no.to_sparse_operator() == sparse_no.to_sparse_operator()


def test_normal_ordered_adjoint_round_trip():
    op = forte2.sparse_operator(
        [
            ("[1a+ 0a-]", 0.2 + 0.3j),
            ("[1a+ 1b+ 0b- 0a-]", -0.5j),
        ]
    )
    reference = det("20")

    no_op = forte2.normal_order(op, reference, max_rank=2)
    assert no_op.adjoint().adjoint() == no_op
    assert no_op.adjoint().to_sparse_operator() == op.adjoint()
