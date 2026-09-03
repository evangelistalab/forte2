from types import SimpleNamespace

import numpy as np
import pytest

from forte2 import System, RHF, MOSpace
from forte2.orbitals import make_final_orbitals
import forte2.orbitals.final_orbitals as final_orbitals_module
import forte2.orbitals.ibo_align as ibo_align_module
from forte2.orbitals.iao import IBO
from forte2.orbitals.ibo_align import IBOAligner
from forte2.helpers.comparisons import approx
from forte2.system.basis_utils import BasisInfo


def test_make_final_orbitals_centralizes_validation_and_original_mode():
    C = np.eye(3)
    kwargs = {
        "system": None,
        "mo_space": None,
        "irrep_indices": np.zeros(3, dtype=int),
        "C_contig": C,
        "g1_act": None,
    }

    C_original = make_final_orbitals("original", **kwargs)
    np.testing.assert_array_equal(C_original, C)
    assert C_original is not C

    with pytest.raises(ValueError, match="final_orbitals must be one of"):
        make_final_orbitals("invalid", **kwargs)


def test_ibo_water():
    xyz = """
    O
    H 1 1.1
    H 1 1.1 2 104.5
    """

    system = System(xyz=xyz, basis_set="cc-pVTZ", auxiliary_basis_set="cc-pVTZ-JKFIT")
    rhf = RHF(charge=0, e_tol=1e-12)(system)
    rhf.run()
    C_occ = rhf.C[0][:, : rhf.ndocc]
    ibo = IBO(system, C_occ)
    D_ibo = np.einsum("pi,qi->pq", ibo.C_ibo, ibo.C_ibo)
    # IBO should be an equivalent representation of the occupied orbitals
    E = np.einsum("pq,pq->", D_ibo, rhf.F[0] + system.ints_hcore())

    assert E + system.nuclear_repulsion == approx(rhf.E)
    np.testing.assert_allclose(
        ibo.U_ibo.T @ ibo.U_ibo,
        np.eye(rhf.ndocc),
        atol=1.0e-12,
    )
    np.testing.assert_allclose(ibo.C_ibo, C_occ @ ibo.U_ibo, atol=1.0e-12)


def test_ibo_cartesian_alignment_is_not_p_specific(monkeypatch):
    labels = [SimpleNamespace(abs_idx=0, iatom=0, n=1, l=0, m=0)]
    labels += [SimpleNamespace(abs_idx=i + 1, iatom=0, n=2, l=1, m=i) for i in range(3)]
    labels += [SimpleNamespace(abs_idx=i + 4, iatom=0, n=3, l=2, m=i) for i in range(5)]
    monkeypatch.setattr(
        ibo_align_module,
        "BasisInfo",
        lambda system, basis: SimpleNamespace(basis_labels=labels),
    )

    ibo_aligner = object.__new__(IBOAligner)
    ibo_aligner.system = SimpleNamespace(
        minao_basis=SimpleNamespace(center_first_and_last=[(0, 9)])
    )
    ibo_aligner.nocc = 6

    # Include an s orbital with negative phase and a randomly rotated complete
    # d shell. The unused p rows ensure unrelated IAOs can be skipped.
    rng = np.random.default_rng(7)
    d_rotation, _ = np.linalg.qr(rng.normal(size=(5, 5)))
    C_ibo_iao = np.zeros((9, 6))
    C_ibo_iao[0, 0] = -1.0
    C_ibo_iao[4:9, 1:6] = d_rotation

    C_aligned, U_aligned = ibo_aligner._align_cartesian_atomic_orbitals(
        C_ibo_iao, np.eye(6)
    )

    target_rows = [0, 4, 5, 6, 7, 8]
    np.testing.assert_allclose(
        C_aligned[np.ix_(target_rows, range(6))], np.eye(6), atol=1.0e-12
    )
    np.testing.assert_allclose(U_aligned.T @ U_aligned, np.eye(6), atol=1.0e-12)
    assert [
        len(group) for _, group, _ in ibo_aligner._cartesian_alignment_groups
    ] == [6]
    assert [
        assignment.minao_basis_index
        for assignment in ibo_aligner.atomic_orbital_assignments
    ] == [
        0,
        4,
        5,
        6,
        7,
        8,
    ]


def test_ibo_aligns_atom_local_p_orbitals_to_cartesian_iaos():
    system = System(
        xyz="N 0.0 0.0 -0.75\nN 0.0 0.0 0.75",
        basis_set="cc-pVDZ",
        cholesky_tei=True,
        unit="angstrom",
        symmetry=False,
    )
    rhf = RHF(charge=0, e_tol=1.0e-12)(system)
    rhf.run()

    # The six valence orbitals contain a rotationally free px/py pair on each
    # atom. Their raw IBO orientations depend on the starting pi orbitals.
    ibo = IBO(system, rhf.C[0][:, 4:10])
    ibo_aligner = IBOAligner(ibo)
    assert ibo_aligner._cartesian_alignment_groups == []
    ibo_aligner.align_to_atomic_orbitals()
    C_ibo_iao = ibo.C_iao.T @ system.ints_overlap() @ ibo_aligner.C_ibo
    minao_labels = BasisInfo(system, system.minao_basis).basis_labels

    transverse_groups = [
        group
        for group in ibo_aligner._cartesian_alignment_groups
        if len(group[1]) == 2
    ]
    assert len(transverse_groups) == 2
    assert {group[0] for group in transverse_groups} == {0, 1}
    for _, orbital_indices, target_rows in transverse_groups:
        assert [minao_labels[row].label() for row in target_rows] == ["2py", "2px"]
        np.testing.assert_allclose(
            C_ibo_iao[np.ix_(target_rows, orbital_indices)],
            np.eye(2),
            atol=1.0e-10,
        )

    assignments = ibo_aligner.atomic_orbital_assignments
    assert [assignment.atom_index for assignment in assignments[:4]] == [0, 0, 1, 1]
    assert [assignment.label for assignment in assignments[:4]] == [
        "2py",
        "2px",
        "2py",
        "2px",
    ]
    assert assignments[4:] == (None, None)

    np.testing.assert_allclose(
        ibo_aligner.U_ibo.T @ ibo_aligner.U_ibo,
        np.eye(ibo_aligner.nocc),
        atol=1.0e-12,
    )


def test_ibo_atomic_order_follows_atom_and_native_minao_index(monkeypatch):
    labels = []
    for iatom in range(2):
        offset = 5 * iatom
        labels.append(SimpleNamespace(abs_idx=offset, iatom=iatom, n=1, l=0, m=0))
        labels.append(SimpleNamespace(abs_idx=offset + 1, iatom=iatom, n=2, l=0, m=0))
        labels.extend(
            SimpleNamespace(
                abs_idx=offset + 2 + component,
                iatom=iatom,
                n=2,
                l=1,
                m=component,
            )
            for component in range(3)
        )
    monkeypatch.setattr(
        ibo_align_module,
        "BasisInfo",
        lambda system, basis: SimpleNamespace(basis_labels=labels),
    )

    ibo_aligner = object.__new__(IBOAligner)
    ibo_aligner.system = SimpleNamespace(
        minao_basis=SimpleNamespace(center_first_and_last=[(0, 5), (5, 10)])
    )
    ibo_aligner.nocc = 8

    # Start from a deliberately interleaved atomic order. The Procrustes step
    # fixes the Cartesian gauge before the final native-order permutation.
    starting_rows = [6, 1, 9, 4, 7, 2, 8, 3]
    C_ibo_iao = np.eye(10)[:, starting_rows]
    C_aligned, U_aligned = ibo_aligner._align_cartesian_atomic_orbitals(
        C_ibo_iao, np.eye(8)
    )

    expected_rows = [1, 2, 3, 4, 6, 7, 8, 9]
    np.testing.assert_allclose(C_aligned[expected_rows], np.eye(8), atol=1.0e-12)
    np.testing.assert_allclose(U_aligned.T @ U_aligned, np.eye(8), atol=1.0e-12)
    assert [
        assignment.atom_index for assignment in ibo_aligner.atomic_orbital_assignments
    ] == [
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
    ]
    assert [
        assignment.label for assignment in ibo_aligner.atomic_orbital_assignments
    ] == [
        "2s",
        "2py",
        "2pz",
        "2px",
        "2s",
        "2py",
        "2pz",
        "2px",
    ]


@pytest.mark.parametrize("mode", ["ibo", "ibo_atomic"])
def test_ibo_final_orbitals_semicanonicalizes_inactive_space(mode):
    system = System(
        xyz="Li 0.0 0.0 0.0\nH 0.0 0.0 3.0",
        basis_set="sto-3g",
        auxiliary_basis_set="def2-universal-JKFIT",
        unit="bohr",
    )
    rhf = RHF(charge=0, e_tol=1e-12)(system)
    rhf.run()

    # The frozen virtual makes the original-to-contiguous permutation nontrivial.
    mo_space = MOSpace(
        nmo=system.nmo,
        core_orbitals=[0],
        active_orbitals=[1, 2],
        frozen_virtual_orbitals=[3],
    )
    C_original = rhf.mos.C[0].copy()
    C_contig = C_original[:, mo_space.orig_to_contig]
    irrep_indices = np.asarray(rhf.mos.irrep_indices[0])[mo_space.orig_to_contig]

    C_final_contig = make_final_orbitals(
        mode,
        system=system,
        mo_space=mo_space,
        irrep_indices=irrep_indices,
        C_contig=C_contig,
        g1_act=np.zeros((mo_space.nactv, mo_space.nactv)),
    )
    C_final = C_final_contig[:, mo_space.contig_to_orig]

    C_semican_contig = make_final_orbitals(
        "semicanonical",
        system=system,
        mo_space=mo_space,
        irrep_indices=irrep_indices,
        C_contig=C_contig,
        g1_act=np.zeros((mo_space.nactv, mo_space.nactv)),
    )
    C_semican = C_semican_contig[:, mo_space.contig_to_orig]

    inactive = [0, 3, 4, 5]
    np.testing.assert_allclose(C_final[:, inactive], C_semican[:, inactive], atol=1e-12)

    S = system.ints_overlap()
    active_overlap = (
        C_original[:, mo_space.active_indices].T
        @ S
        @ C_final[:, mo_space.active_indices]
    )
    np.testing.assert_allclose(active_overlap.T @ active_overlap, np.eye(2), atol=1e-12)
    assert not np.allclose(np.abs(active_overlap), np.eye(2), atol=1e-3)


@pytest.mark.parametrize(("mode", "aligned"), [("ibo", False), ("ibo_atomic", True)])
def test_ibo_final_orbitals_preserves_gas_blocks(monkeypatch, mode, aligned):
    class _InactiveSemicanonicalizer:
        def __init__(self, *, do_active, **kwargs):
            assert not do_active

        def semi_canonicalize(self, g1, C_contig):
            self.C_semican = C_contig.copy()
            self.C_semican[:, [0, 9]] *= -1

    class _ReversingIBO:
        def __init__(self, system, C, **kwargs):
            self.C_ibo = C[:, ::-1]

    class _PassThroughAligner:
        def __init__(self, ibo):
            self.C_ibo = ibo.C_ibo.copy()

        def align_to_atomic_orbitals(self):
            self.C_ibo *= -1
            return self.C_ibo

    monkeypatch.setattr(final_orbitals_module, "IBO", _ReversingIBO)
    monkeypatch.setattr(final_orbitals_module, "IBOAligner", _PassThroughAligner)
    monkeypatch.setattr(
        final_orbitals_module, "Semicanonicalizer", _InactiveSemicanonicalizer
    )

    mo_space = MOSpace(
        nmo=10,
        core_orbitals=[0],
        active_orbitals=[[1, 2, 3, 4], [5, 6, 7, 8]],
        frozen_virtual_orbitals=[9],
    )
    C = np.eye(10)
    irrep_indices = np.zeros(10, dtype=int)
    C_final = make_final_orbitals(
        mode,
        system=object(),
        mo_space=mo_space,
        irrep_indices=irrep_indices,
        C_contig=C,
        g1_act=np.zeros((mo_space.nactv, mo_space.nactv)),
    )

    expected = C.copy()
    expected[:, [0, 9]] *= -1
    for block in ([1, 2, 3, 4], [5, 6, 7, 8]):
        expected[:, block] = expected[:, block[::-1]]
        if aligned:
            expected[:, block] *= -1
    np.testing.assert_array_equal(C_final, expected)
