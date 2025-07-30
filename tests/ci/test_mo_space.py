import pytest

from forte2 import *
from forte2.helpers.comparisons import approx


def test_mo_space_invalid():

    with pytest.raises(Exception):
        # repeated active indices in a single GAS
        mospace = MOSpace(nmo=nmo, active_orbitals=[0, 1, 1], core_orbitals=[5, 6])

    with pytest.raises(Exception):
        # repeated active indices across GASes
        mospace = MOSpace(
            nmo=nmo, active_orbitals=[[0, 1], [1, 2]], core_orbitals=[5, 6]
        )

    with pytest.raises(Exception):
        # illegal list of lists for non-active orbitals
        mospace = MOSpace(
            nmo=nmo, core_orbitals=[[0, 1], [1, 2]], active_orbitals=[5, 6]
        )

    with pytest.raises(Exception):
        # illegal list of lists for non-active orbitals
        mospace = MOSpace(
            nmo=nmo, frozen_virtual_orbitals=[[0, 1], [1, 2]], active_orbitals=[5, 6]
        )

    with pytest.raises(Exception):
        # repeated core indices
        mospace = MOSpace(nmo=nmo, core_orbitals=[0, 1, 1], active_orbitals=[3, 4])

    with pytest.raises(Exception):
        # repeated frozen core indices
        mospace = MOSpace(
            nmo=nmo, frozen_core_orbitals=[0, 1, 1], active_orbitals=[3, 4]
        )

    with pytest.raises(Exception):
        # overlapping core and active indices
        mospace = MOSpace(nmo=nmo, core_orbitals=[0, 1, 2], active_orbitals=[2, 3])

    with pytest.raises(Exception):
        # unsorted core indices
        mospace = MOSpace(nmo=nmo, core_orbitals=[1, 0, 2], active_orbitals=[3, 4])

    with pytest.raises(Exception):
        # unsorted active indices
        mospace = MOSpace(nmo=nmo, core_orbitals=[0, 1, 2], active_orbitals=[4, 3])

    with pytest.raises(Exception):
        # unsorted active indices in one of the GASes
        # note it's acceptable to have [3, 4] in GAS1 and [1] in GAS2, as long as they are individually sorted
        mospace = MOSpace(nmo=nmo, core_orbitals=[0, 2], active_orbitals=[[4, 3], [1]])


def test_mo_space_simple_cas():
    mospace = MOSpace(nmo=10, core_orbitals=[0, 1, 2], active_orbitals=[3, 4])
    assert mospace.ngas == 1
    assert mospace.nactv == 2
    assert mospace.ncore == 3
    assert mospace.active_indices == [3, 4]
    assert list(mospace.contig_to_orig) == list(range(10))
    assert list(mospace.orig_to_contig) == list(range(10))
    assert mospace.core == slice(0, 3)
    assert mospace.actv == slice(3, 5)
    assert mospace.virt == slice(5, 10)


def test_mo_space_interlaced_cas_1():
    mospace = MOSpace(nmo=10, core_orbitals=[0, 3, 5], active_orbitals=[1, 4])
    assert mospace.ngas == 1
    assert mospace.nactv == 2
    assert mospace.ncore == 3
    assert mospace.active_indices == [1, 4]
    assert list(mospace.contig_to_orig) == [0, 3, 5, 1, 4, 2, 6, 7, 8, 9]
    assert list(mospace.orig_to_contig) == [0, 3, 5, 1, 4, 2, 6, 7, 8, 9]
    assert mospace.core == slice(0, 3)
    assert mospace.actv == slice(3, 5)
    assert mospace.virt == slice(5, 10)


def test_mo_space_interlaced_cas_2():
    mospace = MOSpace(
        nmo=10,
        frozen_core_orbitals=[1],
        core_orbitals=[0, 5],
        active_orbitals=[2, 3, 4, 6],
        frozen_virtual_orbitals=[8, 9],
    )
    assert mospace.ngas == 1
    assert mospace.nactv == 4
    assert mospace.ncore == 2
    assert mospace.nvirt == 1
    assert mospace.nfrozen_core == 1
    assert mospace.nfrozen_virtual == 2
    assert mospace.active_indices == [2, 3, 4, 6]
    assert mospace.virtual_indices == [7]
    assert list(mospace.contig_to_orig) == [1, 0, 3, 4, 5, 2, 6, 7, 8, 9]
    assert list(mospace.orig_to_contig) == [1, 0, 5, 2, 3, 4, 6, 7, 8, 9]
    assert mospace.frozen_core == slice(0, 1)
    assert mospace.core == slice(1, 3)
    assert mospace.actv == slice(3, 7)
    assert mospace.virt == slice(7, 8)
    assert mospace.frozen_virt == slice(8, 10)
    assert mospace.corr == slice(1, 8)
    assert mospace.core_corr == slice(0, 2)
    assert mospace.actv_corr == slice(2, 6)
    assert mospace.virt_corr == slice(6, 7)


def test_mo_space_simple_gas():
    mospace = MOSpace(nmo=10, core_orbitals=[0], active_orbitals=[[1, 2], [5, 6, 7]])
    assert mospace.ngas == 2
    assert mospace.nactv == 5
    assert mospace.ncore == 1
    assert mospace.active_indices == [1, 2, 5, 6, 7]
    assert list(mospace.contig_to_orig) == [0, 1, 2, 6, 7, 3, 4, 5, 8, 9]
    assert list(mospace.orig_to_contig) == [0, 1, 2, 5, 6, 7, 3, 4, 8, 9]
    assert mospace.core == slice(0, 1)
    assert mospace.actv == slice(1, 6)
    assert mospace.gas[0] == slice(1, 3)
    assert mospace.gas[1] == slice(3, 6)
    assert mospace.virt == slice(6, 10)


def test_mo_space_interlaced_gas_1():
    mospace = MOSpace(nmo=10, core_orbitals=[0, 2], active_orbitals=[[1, 3], [4, 5, 6]])
    assert mospace.ngas == 2
    assert mospace.nactv == 5
    assert mospace.ncore == 2
    assert mospace.active_indices == [1, 3, 4, 5, 6]
    assert list(mospace.contig_to_orig) == [0, 2, 1, 3, 4, 5, 6, 7, 8, 9]
    assert list(mospace.orig_to_contig) == [0, 2, 1, 3, 4, 5, 6, 7, 8, 9]
    assert mospace.core == slice(0, 2)
    assert mospace.actv == slice(2, 7)
    assert mospace.gas[0] == slice(2, 4)
    assert mospace.gas[1] == slice(4, 7)
    assert mospace.virt == slice(7, 10)


def test_mo_space_interlaced_gas_2():
    mospace = MOSpace(
        nmo=10, core_orbitals=[0, 2], active_orbitals=[[1], [3], [5, 6, 7]]
    )
    assert mospace.ngas == 3
    assert mospace.nactv == 5
    assert mospace.ncore == 2
    assert mospace.active_indices == [1, 3, 5, 6, 7]
    assert list(mospace.contig_to_orig) == [0, 2, 1, 3, 7, 4, 5, 6, 8, 9]
    assert list(mospace.orig_to_contig) == [0, 2, 1, 3, 5, 6, 7, 4, 8, 9]
    assert mospace.core == slice(0, 2)
    assert mospace.actv == slice(2, 7)
    assert mospace.gas[0] == slice(2, 3)
    assert mospace.gas[1] == slice(3, 4)
    assert mospace.gas[2] == slice(4, 7)
    assert mospace.virt == slice(7, 10)


def test_mo_space_interlaced_gas_3():
    mospace = MOSpace(
        nmo=10,
        frozen_core_orbitals=[1],
        core_orbitals=[0, 2],
        active_orbitals=[[4], [3, 6]],
        frozen_virtual_orbitals=[7, 8],
    )
    assert mospace.ngas == 2
    assert mospace.nfrozen_core == 1
    assert mospace.ncore == 2
    assert mospace.nactv == 3
    assert mospace.nvirt == 2
    assert mospace.nfrozen_virtual == 2
    assert mospace.active_indices == [4, 3, 6]
    assert list(mospace.contig_to_orig) == [1, 0, 2, 4, 3, 6, 5, 8, 9, 7]
    assert list(mospace.orig_to_contig) == [1, 0, 2, 4, 3, 6, 5, 9, 7, 8]
    assert mospace.frozen_core == slice(0, 1)
    assert mospace.core == slice(1, 3)
    assert mospace.actv == slice(3, 6)
    assert mospace.gas[0] == slice(3, 4)
    assert mospace.gas[1] == slice(4, 6)
    assert mospace.virt == slice(6, 8)
    assert mospace.frozen_virt == slice(8, 10)
