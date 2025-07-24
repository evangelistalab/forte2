import pytest

from forte2 import *
from forte2.helpers.comparisons import approx


def test_mo_space_invalid():

    with pytest.raises(Exception):
        # repeated active indices in a single GAS
        mospace = MOSpace(active_spaces=[0, 1, 1], core_orbitals=[5, 6])

    with pytest.raises(Exception):
        # repeated active indices across GASes
        mospace = MOSpace(active_spaces=[[0, 1], [1, 2]], core_orbitals=[5, 6])

    with pytest.raises(Exception):
        # repeated core indices
        mospace = MOSpace(core_orbitals=[0, 1, 1], active_spaces=[3, 4])

    with pytest.raises(Exception):
        # overlapping core and active indices
        mospace = MOSpace(core_orbitals=[0, 1, 2], active_spaces=[2, 3])

    with pytest.raises(Exception):
        # unsorted core indices
        mospace = MOSpace(core_orbitals=[1, 0, 2], active_spaces=[3, 4])

    with pytest.raises(Exception):
        # unsorted active indices
        mospace = MOSpace(core_orbitals=[0, 1, 2], active_spaces=[4, 3])

    with pytest.raises(Exception):
        # unsorted active indices across GASes
        mospace = MOSpace(core_orbitals=[0, 2], active_spaces=[[3, 4], [1]])


def test_mo_space_simple_cas():
    mospace = MOSpace(core_orbitals=[0, 1, 2], active_spaces=[3, 4])
    mospace.make_spaces_contiguous(nmo=10)
    assert mospace.ngas == 1
    assert mospace.nactv == 2
    assert mospace.ncore == 3
    assert mospace.active_orbitals == [3, 4]
    assert list(mospace.argsort) == list(range(10))
    assert list(mospace.inv_argsort) == list(range(10))
    assert mospace.core == slice(0, 3)
    assert mospace.actv == slice(3, 5)
    assert mospace.virt == slice(5, 10)


def test_mo_space_interlaced_cas_1():
    mospace = MOSpace(core_orbitals=[0, 3, 5], active_spaces=[1, 4])
    mospace.make_spaces_contiguous(nmo=10)
    assert mospace.ngas == 1
    assert mospace.nactv == 2
    assert mospace.ncore == 3
    assert mospace.active_orbitals == [1, 4]
    assert list(mospace.argsort) == [0, 3, 5, 1, 4, 2, 6, 7, 8, 9]
    assert list(mospace.inv_argsort) == [0, 3, 5, 1, 4, 2, 6, 7, 8, 9]
    assert mospace.core == slice(0, 3)
    assert mospace.actv == slice(3, 5)
    assert mospace.virt == slice(5, 10)


def test_mo_space_interlaced_cas_2():
    mospace = MOSpace(core_orbitals=[0, 1, 5], active_spaces=[2, 3, 4, 6])
    mospace.make_spaces_contiguous(nmo=10)
    assert mospace.ngas == 1
    assert mospace.nactv == 4
    assert mospace.ncore == 3
    assert mospace.active_orbitals == [2, 3, 4, 6]
    assert list(mospace.argsort) == [0, 1, 3, 4, 5, 2, 6, 7, 8, 9]
    assert list(mospace.inv_argsort) == [0, 1, 5, 2, 3, 4, 6, 7, 8, 9]
    assert mospace.core == slice(0, 3)
    assert mospace.actv == slice(3, 7)
    assert mospace.virt == slice(7, 10)


def test_mo_space_simple_gas():
    mospace = MOSpace(core_orbitals=[0], active_spaces=[[1, 2], [5, 6, 7]])
    mospace.make_spaces_contiguous(nmo=10)
    assert mospace.ngas == 2
    assert mospace.nactv == 5
    assert mospace.ncore == 1
    assert mospace.active_orbitals == [1, 2, 5, 6, 7]
    assert list(mospace.argsort) == [0, 1, 2, 6, 7, 3, 4, 5, 8, 9]
    assert list(mospace.inv_argsort) == [0, 1, 2, 5, 6, 7, 3, 4, 8, 9]
    assert mospace.core == slice(0, 1)
    assert mospace.actv[0] == slice(1, 3)
    assert mospace.actv[1] == slice(3, 6)
    assert mospace.virt == slice(6, 10)


def test_mo_space_interlaced_gas_1():
    mospace = MOSpace(core_orbitals=[0, 2], active_spaces=[[1, 3], [4, 5, 6]])
    mospace.make_spaces_contiguous(nmo=10)
    assert mospace.ngas == 2
    assert mospace.nactv == 5
    assert mospace.ncore == 2
    assert mospace.active_orbitals == [1, 3, 4, 5, 6]
    assert list(mospace.argsort) == [0, 2, 1, 3, 4, 5, 6, 7, 8, 9]
    assert list(mospace.inv_argsort) == [0, 2, 1, 3, 4, 5, 6, 7, 8, 9]
    assert mospace.core == slice(0, 2)
    assert mospace.actv[0] == slice(2, 4)
    assert mospace.actv[1] == slice(4, 7)
    assert mospace.virt == slice(7, 10)


def test_mo_space_interlaced_gas_2():
    mospace = MOSpace(core_orbitals=[0, 2], active_spaces=[[1], [3], [5, 6, 7]])
    mospace.make_spaces_contiguous(nmo=10)
    assert mospace.ngas == 3
    assert mospace.nactv == 5
    assert mospace.ncore == 2
    assert mospace.active_orbitals == [1, 3, 5, 6, 7]
    assert list(mospace.argsort) == [0, 2, 1, 3, 7, 4, 5, 6, 8, 9]
    assert list(mospace.inv_argsort) == [0, 2, 1, 3, 5, 6, 7, 4, 8, 9]
    assert mospace.core == slice(0, 2)
    assert mospace.actv[0] == slice(2, 3)
    assert mospace.actv[1] == slice(3, 4)
    assert mospace.actv[2] == slice(4, 7)
    assert mospace.virt == slice(7, 10)
