import math

import pytest

from forte2.lib import sparse_ops
from forte2.lib.det import Determinant


def det(value):
    return Determinant(value)


def correlated_vacuum(weight=0.63):
    return sparse_ops.SparseState(
        {det("20"): math.sqrt(weight), det("02"): 1j * math.sqrt(1.0 - weight)}
    )


def gno(vacuum, label=None, coefficient=1.0, *, norb=2, max_cumulant=3):
    if label is None:
        return sparse_ops.GeneralizedNormalOrderedSparseOperator(
            vacuum, norb, max_cumulant
        )
    term = sparse_ops.sqop(label)[0]
    return sparse_ops.GeneralizedNormalOrderedSparseOperator(
        vacuum, norb, max_cumulant, term, coefficient
    )


def terms(operator):
    return {term.str(): coefficient for term, coefficient in operator}


def assert_operator_close(lhs, rhs, atol=1.0e-12):
    lhs_terms = terms(lhs)
    rhs_terms = terms(rhs)
    for term in lhs_terms.keys() | rhs_terms.keys():
        assert lhs_terms.get(term, 0.0) == pytest.approx(
            rhs_terms.get(term, 0.0), abs=atol
        )


def engines(vacuum, max_rank=3, max_cumulant=3, screen_thresh=1.0e-14):
    reference = sparse_ops.CumulantReference(
        vacuum,
        2,
        max_cumulant=max_cumulant,
        screen_thresh=min(screen_thresh, 1.0e-14),
    )
    return (
        sparse_ops.CumulantWickEngine(reference, max_rank, screen_thresh),
        sparse_ops.GeneralizedNormalOrderedProductComputer(
            reference, max_rank, screen_thresh
        ),
    )


@pytest.mark.parametrize("side", ("left", "right"))
def test_empty_operator_product_and_commutator_are_empty(side):
    vacuum = correlated_vacuum()
    direct, alternate = engines(vacuum)
    empty = gno(vacuum)
    operator = gno(vacuum, "[0a+ 1a-]", 0.7 - 0.2j)
    lhs, rhs = (empty, operator) if side == "left" else (operator, empty)

    assert len(direct.product(lhs, rhs)) == 0
    assert len(direct.commutator(lhs, rhs)) == 0
    assert len(alternate.commutator(lhs, rhs)) == 0


@pytest.mark.parametrize("side", ("left", "right"))
def test_identity_product_scales_operand_and_commutes(side):
    vacuum = correlated_vacuum()
    direct, alternate = engines(vacuum)
    identity = gno(vacuum, "[]", -0.4 + 0.3j)
    operator = gno(vacuum, "[0a+ 1b+ 0b-]", 0.7 - 0.2j)
    lhs, rhs = (identity, operator) if side == "left" else (operator, identity)

    assert_operator_close(direct.product(lhs, rhs), operator * (-0.4 + 0.3j))
    assert len(direct.commutator(lhs, rhs)) == 0
    assert len(alternate.commutator(lhs, rhs)) == 0


@pytest.mark.parametrize(
    "lhs_label,rhs_label,should_vanish",
    (
        ("[0a+ 0a-]", "[1b+ 1b-]", True),
        ("[0a+]", "[1b+ 1b-]", True),
        ("[0a+]", "[1b+]", False),
        ("[0a-]", "[1b-]", False),
    ),
)
def test_disjoint_even_and_odd_commutator_rules(lhs_label, rhs_label, should_vanish):
    vacuum = correlated_vacuum()
    direct, alternate = engines(vacuum)
    lhs = gno(vacuum, lhs_label)
    rhs = gno(vacuum, rhs_label)

    direct_result = direct.commutator(lhs, rhs)
    alternate_result = alternate.commutator(lhs, rhs)
    assert_operator_close(direct_result, alternate_result)
    assert (len(direct_result) == 0) is should_vanish


@pytest.mark.parametrize("label", ("[0a+]", "[0a-]"))
def test_repeated_fermion_operator_is_nilpotent(label):
    vacuum = correlated_vacuum()
    direct, _ = engines(vacuum)
    operator = gno(vacuum, label)

    assert len(direct.product(operator, operator)) == 0
    assert len(direct.commutator(operator, operator)) == 0


def test_commutator_is_antisymmetric_for_complex_mixed_rank_operators():
    vacuum = correlated_vacuum(0.57)
    direct, alternate = engines(vacuum)
    lhs = gno(vacuum, "[0a+ 1b+ 0b-]", 0.2 + 0.7j)
    rhs = gno(vacuum, "[1a+ 0b+ 1b- 0a-]", -0.5 + 0.1j)

    assert_operator_close(direct.commutator(lhs, rhs), -direct.commutator(rhs, lhs))
    assert_operator_close(
        alternate.commutator(lhs, rhs), -alternate.commutator(rhs, lhs)
    )


def test_commutator_is_bilinear_with_complex_coefficients():
    vacuum = correlated_vacuum(0.71)
    direct, alternate = engines(vacuum)
    lhs1 = gno(vacuum, "[0a+ 1a-]", 0.2j)
    lhs2 = gno(vacuum, "[1b+ 0b-]", -0.4 + 0.1j)
    rhs = gno(vacuum, "[1a+ 0b+ 1b- 0a-]", 0.3 - 0.2j)

    for engine in (direct, alternate):
        assert_operator_close(
            engine.commutator(lhs1 + lhs2, rhs),
            engine.commutator(lhs1, rhs) + engine.commutator(lhs2, rhs),
        )


@pytest.mark.parametrize("threshold,expected_terms", ((0.0201, 0), (0.0199, 1)))
def test_pair_screening_boundary_is_conservative(threshold, expected_terms):
    vacuum = correlated_vacuum()
    direct, alternate = engines(vacuum, screen_thresh=threshold)
    lhs = gno(vacuum, "[0a+]", 0.1)
    rhs = gno(vacuum, "[1b+]", 0.1)

    direct_result = direct.commutator(lhs, rhs)
    alternate_result = alternate.commutator(lhs, rhs)
    assert len(direct_result) == expected_terms
    assert_operator_close(direct_result, alternate_result)
    if expected_terms:
        assert next(iter(direct_result))[1] == pytest.approx(0.02)


@pytest.mark.parametrize("max_rank", (0, 1, 2, 3))
def test_output_rank_truncation_is_obeyed_by_both_engines(max_rank):
    vacuum = correlated_vacuum(0.68)
    direct, alternate = engines(vacuum, max_rank=max_rank)
    lhs = gno(vacuum, "[0a+ 1b+ 0b-]", 0.3)
    rhs = gno(vacuum, "[1a+ 0b+ 1b- 0a-]", -0.4j)

    direct_result = direct.commutator(lhs, rhs)
    alternate_result = alternate.commutator(lhs, rhs)
    assert all(term.count() <= 2 * max_rank for term, _ in direct_result)
    assert_operator_close(direct_result, alternate_result)


def test_unlimited_direct_output_matches_sufficient_finite_rank():
    vacuum = correlated_vacuum()
    reference = sparse_ops.CumulantReference(vacuum, 2, max_cumulant=3)
    unlimited = sparse_ops.CumulantWickEngine(reference, -1, 1.0e-14)
    finite = sparse_ops.CumulantWickEngine(reference, 4, 1.0e-14)
    lhs = gno(vacuum, "[0a+ 1b+ 0b-]", 0.3)
    rhs = gno(vacuum, "[1a+ 0b+ 1b- 0a-]", -0.4j)

    assert_operator_close(unlimited.product(lhs, rhs), finite.product(lhs, rhs))
    assert_operator_close(unlimited.commutator(lhs, rhs), finite.commutator(lhs, rhs))


def test_repeated_commutator_calls_are_deterministic():
    vacuum = correlated_vacuum(0.59)
    direct, alternate = engines(vacuum, max_rank=2)
    lhs = gno(vacuum, "[0a+ 1a-]", 0.3 + 0.2j)
    lhs += gno(vacuum, "[1b+ 0b-]", -0.4j)
    rhs = gno(vacuum, "[0a+ 1b+ 0b- 1a-]", 0.7)
    rhs += gno(vacuum, "[1a+ 0b+ 1b- 0a-]", -0.2 + 0.1j)

    for engine in (direct, alternate):
        expected = engine.commutator(lhs, rhs)
        for _ in range(3):
            assert terms(engine.commutator(lhs, rhs)) == terms(expected)


@pytest.mark.parametrize("engine_kind", ("direct", "alternate"))
def test_engine_rejects_reference_metadata_mismatch(engine_kind):
    vacuum = correlated_vacuum()
    reference = sparse_ops.CumulantReference(vacuum, 2, max_cumulant=3)
    engine = (
        sparse_ops.CumulantWickEngine(reference, 2)
        if engine_kind == "direct"
        else sparse_ops.GeneralizedNormalOrderedProductComputer(reference, 2)
    )

    cases = (
        gno(vacuum, "[0a+ 0a-]", norb=3),
        gno(vacuum, "[0a+ 0a-]", max_cumulant=2),
        gno(correlated_vacuum(0.5), "[0a+ 0a-]"),
    )
    for incompatible in cases:
        with pytest.raises(ValueError):
            engine.commutator(incompatible, incompatible)


def test_alternate_engine_rejects_operand_metadata_mismatch():
    vacuum = correlated_vacuum()
    reference = sparse_ops.CumulantReference(vacuum, 2, max_cumulant=3)
    engine = sparse_ops.GeneralizedNormalOrderedProductComputer(reference, 2)
    compatible = gno(vacuum, "[0a+ 0a-]")

    with pytest.raises(ValueError, match="norb values must match"):
        engine.commutator(compatible, gno(vacuum, "[0a+ 0a-]", norb=3))
    with pytest.raises(ValueError, match="max_cumulant values must match"):
        engine.commutator(compatible, gno(vacuum, "[0a+ 0a-]", max_cumulant=2))
    with pytest.raises(ValueError, match="vacua must match"):
        engine.commutator(compatible, gno(correlated_vacuum(0.5), "[0a+ 0a-]"))


@pytest.mark.parametrize(
    "factory",
    (
        lambda reference: sparse_ops.CumulantWickEngine(reference, -2),
        lambda reference: sparse_ops.CumulantWickEngine(reference, 2, -1.0),
        lambda reference: sparse_ops.GeneralizedNormalOrderedProductComputer(-1),
        lambda reference: sparse_ops.GeneralizedNormalOrderedProductComputer(2, -1.0),
        lambda reference: sparse_ops.GeneralizedNormalOrderedProductComputer(
            reference, -1
        ),
        lambda reference: sparse_ops.GeneralizedNormalOrderedProductComputer(
            reference, 2, -1.0
        ),
    ),
)
def test_engine_constructors_reject_invalid_limits(factory):
    reference = sparse_ops.CumulantReference(correlated_vacuum(), 2, max_cumulant=3)
    with pytest.raises(ValueError):
        factory(reference)


def test_alternate_engine_reports_its_truncation_semantics():
    vacuum = correlated_vacuum()
    reference = sparse_ops.CumulantReference(vacuum, 2, max_cumulant=3)

    assert not sparse_ops.GeneralizedNormalOrderedProductComputer(
        2
    ).uses_cumulant_truncation()
    assert sparse_ops.GeneralizedNormalOrderedProductComputer(
        reference, 2
    ).uses_cumulant_truncation()
