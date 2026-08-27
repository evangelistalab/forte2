import math

import pytest

from forte2.lib import sparse_ops
from forte2.lib.det import Determinant


def det(value):
    return Determinant(value)


def correlated_reference(weight=0.7):
    return sparse_ops.SparseState(
        {det("220"): math.sqrt(weight), det("202"): math.sqrt(1.0 - weight)}
    )


def test_cumulant_reference_identifies_core_active_and_virtual_modes():
    reference = sparse_ops.CumulantReference(correlated_reference(), 3)

    assert reference.core_modes() == det("200")
    assert reference.active_modes() == det("022")
    assert reference.virtual_modes() == det("000")
    assert reference.max_cumulant() == 2
    assert reference.norb() == 3


def test_cumulant_reference_one_body_density_uses_implicit_orbital_spaces():
    reference = sparse_ops.CumulantReference(correlated_reference(0.7), 4)

    assert reference.gamma(0, True, 0, True) == pytest.approx(1.0)
    assert reference.gamma(1, True, 1, True) == pytest.approx(0.7)
    assert reference.gamma(2, False, 2, False) == pytest.approx(0.3)
    assert reference.gamma(3, True, 3, True) == pytest.approx(0.0)
    assert reference.gamma(1, True, 2, True) == pytest.approx(0.0)
    assert reference.eta(0, True, 0, True) == pytest.approx(0.0)
    assert reference.eta(1, True, 1, True) == pytest.approx(0.3)
    assert reference.eta(3, False, 3, False) == pytest.approx(1.0)


def test_cumulant_reference_two_body_values_match_rdm_decomposition():
    weight = 0.7
    reference = sparse_ops.CumulantReference(correlated_reference(weight), 3)
    pair0 = det("0" + "2" + "0")
    pair1 = det("0" + "0" + "2")

    assert reference.rdm(pair0, pair0) == pytest.approx(weight)
    assert reference.cumulant(pair0, pair0) == pytest.approx(
        weight - weight**2
    )
    assert reference.cumulant(pair0, pair1) == pytest.approx(
        math.sqrt(weight * (1.0 - weight))
    )


def test_determinant_reference_has_zero_higher_cumulants():
    vacuum = sparse_ops.SparseState({det("20"): 2.0})
    reference = sparse_ops.CumulantReference(vacuum, 2)

    assert reference.gamma(0, True, 0, True) == pytest.approx(1.0)
    assert reference.gamma(1, True, 1, True) == pytest.approx(0.0)
    assert reference.cumulant_size(2) == 0


def test_rank_three_cumulants_follow_spin_orbital_index_conventions():
    weight = 0.6
    vacuum = sparse_ops.SparseState(
        {det("20"): math.sqrt(weight), det("02"): 1j * math.sqrt(1.0 - weight)}
    )
    reference = sparse_ops.CumulantReference(vacuum, 2, max_cumulant=3)

    assert reference.cumulant_size(3) == 4
    assert reference.cumulant(det("2a"), det("2a")) == pytest.approx(0.048)
    assert reference.cumulant(det("a2"), det("a2")) == pytest.approx(-0.048)


def test_cumulant_reference_validates_inputs():
    vacuum = sparse_ops.SparseState({det("20"): 1.0})

    with pytest.raises(ValueError, match="max_cumulant"):
        sparse_ops.CumulantReference(vacuum, 2, max_cumulant=4)
    with pytest.raises(ValueError, match="nonzero norm"):
        sparse_ops.CumulantReference(sparse_ops.SparseState(), 2)
    with pytest.raises(IndexError, match="outside"):
        sparse_ops.CumulantReference(vacuum, 2).gamma(2, True, 0, True)
