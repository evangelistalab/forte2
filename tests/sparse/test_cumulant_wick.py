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


def gno(op, vacuum, max_rank=-1, max_cumulant=2, norb=2):
    return sparse_ops.generalized_normal_order(
        op,
        vacuum,
        norb,
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
    return gno(
        bare,
        vacuum,
        max_rank=max_rank,
        max_cumulant=lhs.max_cumulant(),
        norb=lhs.norb(),
    )


def exact_reference_product(lhs, rhs, vacuum, max_rank):
    bare = sparse_ops.new_product(
        lhs.to_sparse_operator(1.0e-14), rhs.to_sparse_operator(1.0e-14)
    )
    return gno(
        bare,
        vacuum,
        max_rank=max_rank,
        max_cumulant=-1,
        norb=lhs.norb(),
    )


def exact_reference_commutator(lhs, rhs, vacuum, max_rank):
    lhs_bare = lhs.to_sparse_operator(1.0e-14)
    rhs_bare = rhs.to_sparse_operator(1.0e-14)
    bare = sparse_ops.new_product(lhs_bare, rhs_bare)
    bare -= sparse_ops.new_product(rhs_bare, lhs_bare)
    return gno(
        bare,
        vacuum,
        max_rank=max_rank,
        max_cumulant=-1,
        norb=lhs.norb(),
    )


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
    assert_gno_close(gamma_product, reference_product(creator, annihilator, vacuum, 1))
    assert_gno_close(eta_product, reference_product(annihilator, creator, vacuum, 1))


def test_cumulant_wick_rejects_incompatible_operands_and_rank_five_terms():
    vacuum = correlated_vacuum()
    reference = sparse_ops.CumulantReference(vacuum, 2)
    engine = sparse_ops.CumulantWickEngine(reference, 2)
    compatible = gno(sparse_ops.sparse_operator("[0a+ 0a-]", 1.0), vacuum)
    other_vacuum = correlated_vacuum(0.5)
    incompatible = gno(sparse_ops.sparse_operator("[0a+ 0a-]", 1.0), other_vacuum)

    with pytest.raises(ValueError, match="vacua must match"):
        engine.product(compatible, incompatible)

    rank_five = sparse_ops.GeneralizedNormalOrderedSparseOperator(
        vacuum,
        2,
        2,
        sparse_ops.sqop("[0a+ 1a+ 2a+ 0b+ 1b+ 1b- 0b- 2a- 1a- 0a-]")[0],
        1.0,
    )
    with pytest.raises(ValueError, match="rank-four input"):
        engine.product(rank_five, compatible)


def test_cumulant_wick_rank_three_inputs_match_sparse_route():
    vacuum = correlated_vacuum(0.61)
    reference = sparse_ops.CumulantReference(vacuum, 2, max_cumulant=3)
    engine = sparse_ops.CumulantWickEngine(reference, 3, 1.0e-14)
    lhs = sparse_ops.GeneralizedNormalOrderedSparseOperator(
        vacuum,
        2,
        3,
        sparse_ops.sqop("[0a+ 1a+ 0b+ 1b- 0b- 0a-]")[0],
        0.7 - 0.2j,
    )
    rhs = sparse_ops.GeneralizedNormalOrderedSparseOperator(
        vacuum,
        2,
        3,
        sparse_ops.sqop("[1a+ 0b+ 1b+ 1b- 1a- 0a-]")[0],
        -0.3 + 0.1j,
    )

    assert_gno_close(engine.product(lhs, rhs), reference_product(lhs, rhs, vacuum, 3))
    assert_gno_close(
        engine.commutator(lhs, rhs),
        lhs.commutator(rhs, max_rank=3, screen_thresh=1.0e-14),
    )


def test_cumulant_wick_contracts_rank_three_cumulant():
    vacuum = correlated_vacuum(0.61)
    lhs_term = sparse_ops.sqop("[0a+ 0a-]")[0]
    rhs_term = sparse_ops.sqop("[1a+ 0b+ 0b- 1a-]")[0]
    scalar_coefficients = {}

    for max_cumulant in (2, 3):
        reference = sparse_ops.CumulantReference(vacuum, 2, max_cumulant=max_cumulant)
        engine = sparse_ops.CumulantWickEngine(reference, 3, 1.0e-14)
        lhs = sparse_ops.GeneralizedNormalOrderedSparseOperator(
            vacuum, 2, max_cumulant, lhs_term, 1.0
        )
        rhs = sparse_ops.GeneralizedNormalOrderedSparseOperator(
            vacuum, 2, max_cumulant, rhs_term, 1.0
        )
        product = engine.product(lhs, rhs)
        scalar_coefficients[max_cumulant] = {
            term.str(): coefficient for term, coefficient in product
        }.get("[]", 0.0)

    reference = sparse_ops.CumulantReference(vacuum, 2, max_cumulant=3)
    lambda3 = reference.cumulant(det("2a"), det("2a"))
    assert scalar_coefficients[3] - scalar_coefficients[2] == pytest.approx(lambda3)


def test_cumulant_wick_rank_three_core_active_virtual_product_is_exact():
    vacuum = sparse_ops.SparseState(
        {
            det("2200"): math.sqrt(0.61),
            det("2020"): math.sqrt(0.39),
        }
    )
    reference = sparse_ops.CumulantReference(vacuum, 4, max_cumulant=3)
    engine = sparse_ops.CumulantWickEngine(reference, 3, 1.0e-14)
    lhs = sparse_ops.GeneralizedNormalOrderedSparseOperator(
        vacuum,
        4,
        3,
        sparse_ops.sqop("[0a+ 3a+ 0b+ 0b- 1a- 0a-]")[0],
        1.0,
    )
    rhs = sparse_ops.GeneralizedNormalOrderedSparseOperator(
        vacuum,
        4,
        3,
        sparse_ops.sqop("[1a+ 1b+ 1b- 3a-]")[0],
        1.0,
    )

    direct = engine.product(rhs, lhs)
    exact = exact_reference_product(rhs, lhs, vacuum, 3)

    assert_gno_close(direct, exact)
    assert_gno_close(
        engine.commutator(lhs, rhs),
        exact_reference_commutator(lhs, rhs, vacuum, 3),
    )
    assert {term.str(): value for term, value in direct}.get(
        "[]", 0.0
    ) == pytest.approx(0.0)


def test_cumulant_wick_rank_four_mixed_space_algebra_is_exact():
    vacuum = sparse_ops.SparseState(
        {
            det("2200"): math.sqrt(0.61),
            det("2020"): math.sqrt(0.39),
        }
    )
    reference = sparse_ops.CumulantReference(vacuum, 4, max_cumulant=4)
    engine = sparse_ops.CumulantWickEngine(reference, 4, 1.0e-14)
    lhs = sparse_ops.GeneralizedNormalOrderedSparseOperator(
        vacuum,
        4,
        4,
        sparse_ops.sqop("[1a+ 2a+ 1b+ 3b+ 2b- 1b- 2a- 1a-]")[0],
        0.4 - 0.1j,
    )
    rhs = sparse_ops.GeneralizedNormalOrderedSparseOperator(
        vacuum,
        4,
        4,
        sparse_ops.sqop("[2b+ 3b+ 3b- 0a-]")[0],
        -0.2 + 0.3j,
    )

    assert_gno_close(
        engine.product(lhs, rhs), exact_reference_product(lhs, rhs, vacuum, 4)
    )
    assert_gno_close(
        engine.commutator(lhs, rhs),
        exact_reference_commutator(lhs, rhs, vacuum, 4),
    )


def test_cumulant_wick_rank_four_inputs_match_sparse_route():
    vacuum = correlated_vacuum(0.61)
    reference = sparse_ops.CumulantReference(vacuum, 2, max_cumulant=4)
    engine = sparse_ops.CumulantWickEngine(reference, 4, 1.0e-14)
    lhs = sparse_ops.GeneralizedNormalOrderedSparseOperator(
        vacuum,
        2,
        4,
        sparse_ops.sqop("[0a+ 1a+ 0b+ 1b+ 1b- 0b- 1a- 0a-]")[0],
        0.4 - 0.1j,
    )
    rhs = sparse_ops.GeneralizedNormalOrderedSparseOperator(
        vacuum,
        2,
        4,
        sparse_ops.sqop("[0a+ 1b+ 1b- 0a-]")[0],
        -0.2 + 0.3j,
    )

    assert_gno_close(engine.product(lhs, rhs), reference_product(lhs, rhs, vacuum, 4))
    assert_gno_close(
        engine.commutator(lhs, rhs),
        lhs.commutator(rhs, max_rank=4, screen_thresh=1.0e-14),
    )


def test_cumulant_wick_contracts_rank_four_cumulant():
    vacuum = correlated_vacuum(0.61)
    lhs_term = sparse_ops.sqop("[0a+ 0a-]")[0]
    rhs_term = sparse_ops.sqop("[1a+ 0b+ 1b+ 1b- 0b- 1a-]")[0]
    scalar_coefficients = {}

    for max_cumulant in (3, 4):
        reference = sparse_ops.CumulantReference(vacuum, 2, max_cumulant=max_cumulant)
        engine = sparse_ops.CumulantWickEngine(reference, 4, 1.0e-14)
        lhs = sparse_ops.GeneralizedNormalOrderedSparseOperator(
            vacuum, 2, max_cumulant, lhs_term, 1.0
        )
        rhs = sparse_ops.GeneralizedNormalOrderedSparseOperator(
            vacuum, 2, max_cumulant, rhs_term, 1.0
        )
        product = engine.product(lhs, rhs)
        scalar_coefficients[max_cumulant] = {
            term.str(): coefficient for term, coefficient in product
        }.get("[]", 0.0)

    reference = sparse_ops.CumulantReference(vacuum, 2, max_cumulant=4)
    lambda4 = reference.cumulant(det("22"), det("22"))
    assert abs(scalar_coefficients[4] - scalar_coefficients[3]) == pytest.approx(
        abs(lambda4)
    )


def test_cumulant_wick_contracts_complex_off_diagonal_rank_four_cumulant():
    vacuum = sparse_ops.SparseState(
        {
            det("200"): math.sqrt(0.5),
            det("020"): math.sqrt(0.3),
            det("002"): 1j * math.sqrt(0.2),
        }
    )
    lhs_term = sparse_ops.SQOperatorString(det("a00"), det("a00"))
    rhs_term = sparse_ops.SQOperatorString(det("b20"), det("b02"))
    products = {}

    for max_cumulant in (3, 4):
        reference = sparse_ops.CumulantReference(vacuum, 3, max_cumulant=max_cumulant)
        engine = sparse_ops.CumulantWickEngine(reference, 4, 1.0e-14)
        lhs = sparse_ops.GeneralizedNormalOrderedSparseOperator(
            vacuum, 3, max_cumulant, lhs_term, 1.0
        )
        rhs = sparse_ops.GeneralizedNormalOrderedSparseOperator(
            vacuum, 3, max_cumulant, rhs_term, 1.0
        )
        products[max_cumulant] = engine.product(lhs, rhs)

    scalar3 = {term.str(): value for term, value in products[3]}.get("[]", 0.0)
    scalar4 = {term.str(): value for term, value in products[4]}.get("[]", 0.0)
    reference = sparse_ops.CumulantReference(vacuum, 3, max_cumulant=4)
    lambda4 = reference.cumulant(det("220"), det("202"))

    assert scalar4 - scalar3 == pytest.approx(lambda4)
    lhs = sparse_ops.GeneralizedNormalOrderedSparseOperator(vacuum, 3, 4, lhs_term, 1.0)
    rhs = sparse_ops.GeneralizedNormalOrderedSparseOperator(vacuum, 3, 4, rhs_term, 1.0)
    assert_gno_close(products[4], reference_product(lhs, rhs, vacuum, 4))


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
