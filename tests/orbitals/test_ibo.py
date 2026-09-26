from types import SimpleNamespace

import numpy as np
import pytest

from forte2 import System, RHF, MOSpace
from forte2.orbitals import make_final_orbitals
import forte2.orbitals.final_orbitals as final_orbitals_module
import forte2.orbitals.ibo_align as ibo_align_module
from forte2.orbitals.iao import IBO
from forte2.orbitals.ibo_align import AtomicOrbitalAssignment, IBOAligner
from forte2.helpers.comparisons import approx
from forte2.system.basis_utils import BasisInfo


def _make_test_aligner(
    monkeypatch, labels, center_ranges, nocc, C_minao_projected=None
):
    """Build an aligner without constructing molecular integrals."""

    monkeypatch.setattr(
        ibo_align_module,
        "BasisInfo",
        lambda system, basis: SimpleNamespace(basis_labels=labels),
    )
    nminao = len(labels)
    if C_minao_projected is None:
        C_minao_projected = np.eye(nminao)
    system = SimpleNamespace(
        minao_basis=SimpleNamespace(center_first_and_last=center_ranges)
    )
    C_occ = np.eye(nminao, nocc)
    ibo = SimpleNamespace(
        system=system,
        C_occ=C_occ,
        C_iao=np.eye(nminao),
        C_minao_projected=C_minao_projected,
        S1=np.eye(nminao),
        nocc=nocc,
        U_ibo=np.eye(nocc),
        C_ibo=C_occ.copy(),
    )
    return IBOAligner(ibo)


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


@pytest.mark.parametrize("mode", ["ibo", "ibo_atomic"])
@pytest.mark.parametrize(
    ("two_component", "dtype"),
    [(True, float), (False, complex)],
)
def test_ibo_modes_reject_nonreal_or_relativistic_orbitals_before_localization(
    monkeypatch, mode, two_component, dtype
):
    def unexpected_ibo(*args, **kwargs):
        pytest.fail("IBO localization was reached before validating the input")

    monkeypatch.setattr(final_orbitals_module, "IBO", unexpected_ibo)

    system = SimpleNamespace(point_group="C1", two_component=two_component)
    C = np.eye(2, dtype=dtype)
    with pytest.raises(NotImplementedError, match="real, nonrelativistic orbitals"):
        make_final_orbitals(
            mode,
            system=system,
            mo_space=None,
            irrep_indices=np.zeros(2, dtype=int),
            C_contig=C,
            g1_act=None,
        )


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
    ibo_aligner = _make_test_aligner(monkeypatch, labels, [(0, 9)], nocc=6)

    # Include an s orbital with negative phase and a randomly rotated complete
    # d shell. The unused p rows ensure unrelated IAOs can be skipped.
    rng = np.random.default_rng(7)
    d_rotation, _ = np.linalg.qr(rng.normal(size=(5, 5)))
    C_ibo_iao = np.zeros((9, 6))
    C_ibo_iao[0, 0] = -1.0
    C_ibo_iao[4:9, 1:6] = d_rotation

    _, C_aligned, U_aligned = ibo_aligner._align_cartesian_atomic_orbitals(
        C_ibo_iao.copy(), C_ibo_iao.copy(), np.eye(6)
    )

    target_rows = [0, 4, 5, 6, 7, 8]
    np.testing.assert_allclose(
        C_aligned[np.ix_(target_rows, range(6))], np.eye(6), atol=1.0e-12
    )
    np.testing.assert_allclose(U_aligned.T @ U_aligned, np.eye(6), atol=1.0e-12)
    assert [
        len(alignment_set.orbital_indices)
        for alignment_set in ibo_aligner._alignment_sets
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


def test_projected_minao_targets_are_normalized_per_atom(monkeypatch):
    labels = [
        SimpleNamespace(abs_idx=0, iatom=0, n=1, l=0, m=0),
        SimpleNamespace(abs_idx=1, iatom=0, n=2, l=0, m=0),
        SimpleNamespace(abs_idx=2, iatom=1, n=1, l=0, m=0),
    ]
    C_minao_projected = np.array(
        [[1.0, 0.4, 0.2], [0.0, 0.9, 0.1], [0.0, 0.0, 0.8]]
    )
    ibo_aligner = _make_test_aligner(
        monkeypatch,
        labels,
        [(0, 2), (2, 3)],
        nocc=2,
        C_minao_projected=C_minao_projected,
    )

    for first, last in [(0, 2), (2, 3)]:
        C_atom = ibo_aligner.C_minao_targets[:, first:last]
        np.testing.assert_allclose(
            C_atom.T @ C_atom, np.eye(last - first), atol=1.0e-12
        )


def test_ibo_atomic_alignment_uses_minao_target_gauge(monkeypatch):
    labels = [
        SimpleNamespace(abs_idx=0, iatom=0, n=1, l=0, m=0),
        SimpleNamespace(abs_idx=1, iatom=0, n=2, l=0, m=0),
    ]
    ibo_aligner = _make_test_aligner(monkeypatch, labels, [(0, 2)], nocc=2)

    angle = 0.37
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    S_iao_ibo = np.eye(2)
    S_minao_ibo = np.diag([1.0, 0.8]) @ rotation

    S_iao_aligned, S_minao_aligned, U_aligned = (
        ibo_aligner._align_cartesian_atomic_orbitals(
            S_iao_ibo.copy(), S_minao_ibo.copy(), np.eye(2)
        )
    )

    np.testing.assert_allclose(S_iao_aligned, rotation.T, atol=1.0e-12)
    np.testing.assert_allclose(S_minao_aligned, np.diag([1.0, 0.8]), atol=1.0e-12)
    np.testing.assert_allclose(U_aligned, rotation.T, atol=1.0e-12)


def test_ibo_aligns_atom_local_p_orbitals_to_projected_minao():
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
    assert ibo_aligner._alignment_sets == ()
    ibo_aligner.align_to_atomic_orbitals()
    C_ibo_minao = (
        ibo_aligner.C_minao_targets.T
        @ system.ints_overlap()
        @ ibo_aligner.C_ibo
    )
    minao_labels = BasisInfo(system, system.minao_basis).basis_labels

    transverse_groups = [
        alignment_set
        for alignment_set in ibo_aligner._alignment_sets
        if len(alignment_set.orbital_indices) == 2
    ]
    assert len(transverse_groups) == 2
    assert {alignment_set.atom_index for alignment_set in transverse_groups} == {
        0,
        1,
    }
    for alignment_set in transverse_groups:
        orbital_indices = alignment_set.orbital_indices
        target_rows = alignment_set.target_minao_rows
        assert [minao_labels[row].label() for row in target_rows] == ["2py", "2px"]
        target_overlap = C_ibo_minao[np.ix_(target_rows, orbital_indices)]
        np.testing.assert_allclose(target_overlap, target_overlap.T, atol=1.0e-10)
        assert np.linalg.eigvalsh(target_overlap)[0] > 0.0

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
    ibo_aligner = _make_test_aligner(
        monkeypatch, labels, [(0, 5), (5, 10)], nocc=8
    )

    # Start from a deliberately interleaved atomic order. The Procrustes step
    # fixes the Cartesian gauge before the final native-order permutation.
    starting_rows = [6, 1, 9, 4, 7, 2, 8, 3]
    C_ibo_iao = np.eye(10)[:, starting_rows]
    _, C_aligned, U_aligned = ibo_aligner._align_cartesian_atomic_orbitals(
        C_ibo_iao.copy(), C_ibo_iao.copy(), np.eye(8)
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


def test_ibo_atomic_aligns_a_weak_but_full_rank_target(monkeypatch):
    labels = [
        SimpleNamespace(abs_idx=0, iatom=0, n=1, l=0, m=0),
        SimpleNamespace(abs_idx=1, iatom=0, n=2, l=0, m=0),
        SimpleNamespace(abs_idx=2, iatom=0, n=2, l=1, m=0),
        SimpleNamespace(abs_idx=3, iatom=1, n=1, l=0, m=0),
    ]
    ibo_aligner = _make_test_aligner(
        monkeypatch, labels, [(0, 3), (3, 4)], nocc=2
    )

    # Both orbitals have at least 95% of their IAO norm on atom 0, so the
    # complete set passes the rotation-invariant locality test. The best two
    # target functions have squared singular values 1.0 and 0.85, but are
    # full-rank; target purity is therefore diagnostic and does not reject the
    # alignment.
    S_iao_ibo = np.array(
        [
            [1.0, 0.0],
            [0.0, np.sqrt(0.85)],
            [0.0, np.sqrt(0.10)],
            [0.0, np.sqrt(0.05)],
        ]
    )
    angle = 0.37
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    S_rotated = S_iao_ibo @ rotation

    _, S_aligned, U_aligned = ibo_aligner._align_cartesian_atomic_orbitals(
        S_rotated.copy(), S_rotated.copy(), rotation.copy()
    )

    np.testing.assert_allclose(S_aligned, S_iao_ibo, atol=1.0e-12)
    np.testing.assert_allclose(U_aligned, np.eye(2), atol=1.0e-12)
    assert len(ibo_aligner._alignment_sets) == 1
    assert [
        assignment.minao_basis_index
        for assignment in ibo_aligner.atomic_orbital_assignments
    ] == [0, 1]


def test_ibo_atomic_aligns_atom_local_radial_mixtures(monkeypatch):
    labels = [
        SimpleNamespace(abs_idx=0, iatom=0, n=3, l=2, m=0),
        SimpleNamespace(abs_idx=1, iatom=0, n=4, l=2, m=0),
        SimpleNamespace(abs_idx=2, iatom=0, n=3, l=2, m=1),
        SimpleNamespace(abs_idx=3, iatom=0, n=4, l=2, m=1),
        SimpleNamespace(abs_idx=4, iatom=0, n=5, l=2, m=0),
        SimpleNamespace(abs_idx=5, iatom=0, n=5, l=2, m=1),
    ]
    ibo_aligner = _make_test_aligner(monkeypatch, labels, [(0, 6)], nocc=2)

    # The complete orbital set is exactly atom-local, but each angular
    # direction is distributed over two radial IAOs. The original atomic
    # alignment rejects this because the best individual target overlaps are
    # well below 0.9. The two rows with the largest individual weights are
    # also linearly dependent; pivoted QR must instead select independent
    # angular directions. The atomic alignment treats their populations as
    # confidence diagnostics rather than acceptance thresholds.
    S_iao_ibo = np.array(
        [
            [np.sqrt(0.50), 0.0],
            [np.sqrt(0.40), 0.0],
            [0.0, np.sqrt(0.35)],
            [0.0, np.sqrt(0.34)],
            [np.sqrt(0.10), 0.0],
            [0.0, np.sqrt(0.31)],
        ]
    )
    _, S_aligned, U_aligned = ibo_aligner._align_cartesian_atomic_orbitals(
        S_iao_ibo.copy(), S_iao_ibo.copy(), np.eye(2)
    )

    np.testing.assert_allclose(S_aligned, S_iao_ibo, atol=1.0e-12)
    np.testing.assert_allclose(U_aligned, np.eye(2), atol=1.0e-12)
    assert len(ibo_aligner._alignment_sets) == 1
    assert [
        assignment.minao_basis_index
        for assignment in ibo_aligner.atomic_orbital_assignments
    ] == [0, 2]
    alignment = ibo_aligner._alignment_sets[0]
    np.testing.assert_allclose(
        alignment.locality_eigenvalues, np.ones(2), atol=1.0e-12
    )
    np.testing.assert_allclose(
        alignment.target_singular_values**2, [0.50, 0.35], atol=1.0e-12
    )


def test_ibo_atomic_rejects_a_noninvariant_atom_local_set(monkeypatch):
    labels = [
        SimpleNamespace(abs_idx=0, iatom=0, n=2, l=0, m=0),
        SimpleNamespace(abs_idx=1, iatom=0, n=2, l=1, m=0),
        SimpleNamespace(abs_idx=2, iatom=1, n=1, l=0, m=0),
        SimpleNamespace(abs_idx=3, iatom=1, n=2, l=0, m=0),
    ]
    ibo_aligner = _make_test_aligner(
        monkeypatch, labels, [(0, 2), (2, 4)], nocc=2
    )

    # Both current orbitals have 0.91 population on atom 0, but an allowed
    # rotation exposes a direction with only 0.82 population. The invariant
    # eigenvalue test must therefore reject the proposed atom-local set.
    atom_metric = np.array([[0.91, 0.09], [0.09, 0.91]])
    atom_overlap = np.linalg.cholesky(atom_metric).T
    other_overlap = np.linalg.cholesky(
        np.eye(2) - atom_metric + 1.0e-14 * np.eye(2)
    ).T
    S_iao_ibo = np.vstack((atom_overlap, other_overlap))
    S_iao_aligned, S_minao_aligned, U_aligned = (
        ibo_aligner._align_cartesian_atomic_orbitals(
            S_iao_ibo.copy(), S_iao_ibo.copy(), np.eye(2)
        )
    )

    np.testing.assert_allclose(S_iao_aligned, S_iao_ibo, atol=1.0e-12)
    np.testing.assert_allclose(S_minao_aligned, S_iao_ibo, atol=1.0e-12)
    np.testing.assert_allclose(U_aligned, np.eye(2), atol=1.0e-12)
    assert ibo_aligner._alignment_sets == ()
    assert ibo_aligner.atomic_orbital_assignments == (None, None)


def test_ibo_atomic_summary_reports_main_iao_and_unassigned_orbitals(monkeypatch):
    labels = [
        SimpleNamespace(
            abs_idx=0,
            iatom=0,
            Z=6,
            Zidx=1,
            n=2,
            l=0,
            m=0,
            label=lambda: "2s",
        ),
        SimpleNamespace(
            abs_idx=1,
            iatom=0,
            Z=6,
            Zidx=1,
            n=2,
            l=1,
            m=2,
            label=lambda: "2px",
        ),
    ]
    info_messages = []
    warning_messages = []
    monkeypatch.setattr(ibo_align_module.logger, "log_info1", info_messages.append)
    monkeypatch.setattr(
        ibo_align_module.logger, "log_warning", warning_messages.append
    )

    ibo_aligner = _make_test_aligner(monkeypatch, labels, [(0, 2)], nocc=2)
    # Deliberately use a different MINAO gauge so target and main-IAO
    # populations exercise their separate overlap representations.
    ibo_aligner.C_minao_targets = np.array([[0.0, 1.0], [1.0, 0.0]])
    ibo_aligner.C_ibo = np.array(
        [[np.sqrt(0.6), 0.0], [np.sqrt(0.4), 1.0]]
    )
    ibo_aligner.atomic_orbital_assignments = (
        AtomicOrbitalAssignment(0, 0, 2, 0, 0, "2s"),
        None,
    )

    with pytest.raises(ValueError, match="order must be a permutation"):
        ibo_aligner.log_atomic_alignment_summary(
            gas_number=1,
            order=np.array([0, 0]),
            mo_indices=np.array([3, 4]),
            orbital_energies=np.array([-0.4, 0.2]),
        )

    ibo_aligner.log_atomic_alignment_summary(
        gas_number=1,
        order=np.array([1, 0]),
        mo_indices=np.array([3, 4]),
        orbital_energies=np.array([-0.4, 0.2]),
    )

    summary = info_messages[0]
    assert "IBO atomic-alignment summary for GAS 1" in summary
    assert (
        "  MO      Energy [Eh]   Atomic target   Target pop.   "
        "Main IAO   Main pop."
    ) in summary
    assert "C1 2px" in summary
    assert "C1 2s" in summary
    assert "unassigned" in summary
    assert "-0.40000000" in summary
    assert "0.4000" in summary
    assert "0.6000" in summary
    assert (
        "   4       0.20000000   C1 2s                0.4000   "
        "C1 2s         0.6000"
    ) in summary
    assert warning_messages == [
        "1 of 2 IBO(s) in GAS 1 could not be assigned to a projected MINAO "
        "target. They remain converged localized IBOs.",
        "1 of 2 IBO(s) in GAS 1 have projected MINAO target "
        "populations below 0.900.\nThe reported targets are maximal-overlap labels, "
        "not pure atomic-orbital assignments.\nIBO localization itself succeeded "
        "for every orbital; this warning concerns only atomic-label confidence.",
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


@pytest.mark.parametrize(
    ("mode", "aligned"),
    [("ibo", False), ("ibo_atomic", True)],
)
def test_ibo_final_orbitals_preserves_gas_sets(monkeypatch, mode, aligned):
    info_messages = []

    class _InactiveSemicanonicalizer:
        def __init__(self, *, do_active, **kwargs):
            assert not do_active

        def semi_canonicalize(self, g1, C_contig):
            self.C_semican = C_contig.copy()
            self.C_semican[:, [0, 9]] *= -1
            self.fock_semican = np.diag(np.arange(C_contig.shape[1], dtype=float))

    class _ReversingIBO:
        def __init__(self, system, C, **kwargs):
            self.C_ibo = C[:, ::-1]
            self.U_ibo = np.eye(C.shape[1])[:, ::-1]

    class _PassThroughAligner:
        def __init__(self, ibo):
            self.C_ibo = ibo.C_ibo.copy()
            self.U_ibo = ibo.U_ibo.copy()

        def align_to_atomic_orbitals(self):
            self.C_ibo *= -1
            self.U_ibo *= -1
            return self.C_ibo

        def log_atomic_alignment_summary(self, **kwargs):
            pass

    monkeypatch.setattr(final_orbitals_module, "IBO", _ReversingIBO)
    monkeypatch.setattr(final_orbitals_module, "IBOAligner", _PassThroughAligner)
    monkeypatch.setattr(
        final_orbitals_module.logger, "log_info1", info_messages.append
    )
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
    for gas_indices in ([1, 2, 3, 4], [5, 6, 7, 8]):
        # IBO first reverses each set. The resulting Fock diagonal is also
        # reversed, so ascending-energy ordering restores the original order.
        if aligned:
            expected[:, gas_indices] *= -1
    np.testing.assert_array_equal(C_final, expected)
    assert info_messages == [
        "IBO localization succeeded for all 4 orbital(s) in GAS 1.",
        "IBO localization succeeded for all 4 orbital(s) in GAS 2.",
    ]
