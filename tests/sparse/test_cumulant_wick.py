import math

import pytest

from forte2.lib import sparse_ops
from forte2.lib.det import Determinant


def det(value):
    return Determinant(value)


def correlated_vacuum(weight=0.63):
    return sparse_ops.SparseState(
        {det("20"): math.sqrt(weight), det("02"): math.sqrt(1.0 - weight)}
    )


def gno(op, vacuum, max_rank=-1, max_cumulant=2):
    return sparse_ops.generalized_normal_order(
        op,
        vacuum,
        2,
        max_cumulant=max_cumulant,
        max_rank=max_rank,
        screen_thresh=1.0e-14,
    )


def assert_gno_close(lhs, rhs, abs=1.0e-11):
    lhs_terms = {term.str(): coefficient for term, coefficient in lhs}
    rhs_terms = {term.str(): coefficient for term, coefficient in rhs}
    for term in set(lhs_terms) | set(rhs_terms):
        assert lhs_terms.get(term, 0.0) == pytest.approx(
            rhs_terms.get(term, 0.0), abs=abs
        )


def reference_product(lhs, rhs, vacuum, max_rank):
    bare = sparse_ops.new_product(
        lhs.to_sparse_operator(1.0e-14), rhs.to_sparse_operator(1.0e-14)
    )
    return gno(bare, vacuum, max_rank=max_rank)


@pytest.mark.parametrize("max_rank", [0, 1, 2])
def test_cumulant_wick_product_matches_sparse_route(max_rank):
    vacuum = correlated_vacuum()
    reference = sparse_ops.CumulantReference(vacuum, 2)
    engine = sparse_ops.CumulantWickEngine(reference, max_rank, 1.0e-14)
    lhs = gno(
        sparse_ops.sparse_operator(
            [
                ("[]", 0.2),
                ("[0a+ 1a-]", 0.7 - 0.1j),
                ("[1b+ 0b-]", -0.3j),
            ]
        ),
        vacuum,
    )
    rhs = gno(
        sparse_ops.sparse_operator(
            [
                ("[1a+ 0a-]", -0.4 + 0.2j),
                ("[0b+ 1b-]", 0.6),
            ]
        ),
        vacuum,
    )

    direct = engine.product(lhs, rhs)
    expected = reference_product(lhs, rhs, vacuum, max_rank)

    assert all(term.count() <= 2 * max_rank for term, _ in direct)
    assert_gno_close(direct, expected)


@pytest.mark.parametrize("max_rank", [0, 1, 2])
def test_cumulant_wick_commutator_matches_sparse_route(max_rank):
    vacuum = correlated_vacuum(0.71)
    reference = sparse_ops.CumulantReference(vacuum, 2)
    engine = sparse_ops.CumulantWickEngine(reference, max_rank, 1.0e-14)
    lhs = gno(
        sparse_ops.sparse_operator(
            [
                ("[0a+ 0a-]", 0.7),
                ("[1b+ 0b-]", -0.3 + 0.2j),
                ("[0a+ 1b+ 0b- 1a-]", 0.4),
            ]
        ),
        vacuum,
    )
    rhs = gno(
        sparse_ops.sparse_operator(
            [
                ("[1a+ 0a-]", -0.5j),
                ("[0b+ 1b-]", 0.6),
                ("[1a+ 1b+ 0b- 0a-]", -0.2 + 0.1j),
            ]
        ),
        vacuum,
    )

    direct = engine.commutator(lhs, rhs)
    expected = lhs.commutator(rhs, max_rank=max_rank, screen_thresh=1.0e-14)

    assert_gno_close(direct, expected)


def test_cumulant_wick_handles_gamma_and_eta_orientations():
    vacuum = correlated_vacuum(0.7)
    reference = sparse_ops.CumulantReference(vacuum, 2)
    engine = sparse_ops.CumulantWickEngine(reference, 1, 1.0e-14)
    annihilator = sparse_ops.GeneralizedNormalOrderedSparseOperator(
        vacuum,
        2,
        2,
        sparse_ops.sqop("[0a-]")[0],
        1.0,
    )
    creator = sparse_ops.GeneralizedNormalOrderedSparseOperator(
        vacuum,
        2,
        2,
        sparse_ops.sqop("[0a+]")[0],
        1.0,
    )

    gamma_product = engine.product(creator, annihilator)
    eta_product = engine.product(annihilator, creator)

    assert {term.str(): value for term, value in gamma_product}["[]"] == pytest.approx(
        0.7
    )
    assert {term.str(): value for term, value in eta_product}["[]"] == pytest.approx(
        0.3
    )
    assert_gno_close(
        gamma_product, reference_product(creator, annihilator, vacuum, 1)
    )
    assert_gno_close(eta_product, reference_product(annihilator, creator, vacuum, 1))


def test_cumulant_wick_rejects_incompatible_operands_and_rank_three_terms():
    vacuum = correlated_vacuum()
    reference = sparse_ops.CumulantReference(vacuum, 2)
    engine = sparse_ops.CumulantWickEngine(reference, 2)
    compatible = gno(sparse_ops.sparse_operator("[0a+ 0a-]", 1.0), vacuum)
    other_vacuum = correlated_vacuum(0.5)
    incompatible = gno(
        sparse_ops.sparse_operator("[0a+ 0a-]", 1.0), other_vacuum
    )

    with pytest.raises(ValueError, match="vacua must match"):
        engine.product(compatible, incompatible)

    rank_three = sparse_ops.GeneralizedNormalOrderedSparseOperator(
        vacuum,
        2,
        2,
        sparse_ops.sqop("[0a+ 1a+ 0b+ 1b- 1a- 0a-]")[0],
        1.0,
    )
    with pytest.raises(ValueError, match="rank-two input"):
        engine.product(rank_three, compatible)


def test_cumulant_wick_exhaustive_small_term_commutators():
    vacuum = correlated_vacuum(0.58)
    reference = sparse_ops.CumulantReference(vacuum, 2, max_cumulant=3)
    engine = sparse_ops.CumulantWickEngine(reference, 2, 1.0e-14)
    terms = [
        "[0a+ 0a-]",
        "[0a+ 1a-]",
        "[1b+ 0b-]",
        "[1b+ 1b-]",
        "[0a+ 0b+ 0b- 0a-]",
        "[1a+ 1b+ 1b- 1a-]",
        "[0a+ 1b+ 0b- 1a-]",
        "[1a+ 0b+ 1b- 0a-]",
    ]

    for lhs_string in terms:
        lhs = sparse_ops.GeneralizedNormalOrderedSparseOperator(
            vacuum, 2, 3, sparse_ops.sqop(lhs_string)[0], 1.0
        )
        for rhs_string in terms:
            rhs = sparse_ops.GeneralizedNormalOrderedSparseOperator(
                vacuum, 2, 3, sparse_ops.sqop(rhs_string)[0], 1.0
            )
            assert_gno_close(
                engine.commutator(lhs, rhs),
                lhs.commutator(rhs, max_rank=2, screen_thresh=1.0e-14),
            )


def test_cumulant_wick_complex_reference_cumulants():
    vacuum = sparse_ops.SparseState(
        {det("20"): math.sqrt(0.6), det("02"): 1j * math.sqrt(0.4)}
    )
    reference = sparse_ops.CumulantReference(vacuum, 2, max_cumulant=3)
    engine = sparse_ops.CumulantWickEngine(reference, 2, 1.0e-14)
    lhs = gno(
        sparse_ops.sparse_operator("[0a+ 0b+ 1b- 1a-]", 0.7 - 0.2j),
        vacuum,
        max_cumulant=3,
    )
    rhs = gno(
        sparse_ops.sparse_operator("[1a+ 1b+ 0b- 0a-]", -0.3j),
        vacuum,
        max_cumulant=3,
    )

    assert_gno_close(
        engine.commutator(lhs, rhs),
        lhs.commutator(rhs, max_rank=2, screen_thresh=1.0e-14),
    )
