from types import SimpleNamespace

import numpy as np
import pytest

from forte2 import CUHF, RHF, ROHF, UHF, System
from forte2.symmetry import MOSymmetryDetector
from forte2.symmetry.mo_sym_detect import get_symmetry_ops
from forte2.symmetry.sym_utils import CHARACTER_TABLE, COTTON_LABELS, SYMMETRY_OPS
from forte2.system import BasisInfo


def _detector(labels, C, eps):
    system = SimpleNamespace(point_group="D2H", two_component=False)
    detector = MOSymmetryDetector(system, None, np.eye(len(labels)), C, eps)
    characters = np.array([CHARACTER_TABLE["D2H"][label] for label in labels])
    detector.U_ops = {
        op: np.diag(characters[:, i]) for i, op in enumerate(SYMMETRY_OPS["D2H"])
    }
    return detector


def _assert_characters(detector, labels):
    C = detector.C
    np.testing.assert_allclose(
        C.T.conj() @ detector.S @ C, np.eye(C.shape[1]), atol=1e-10
    )
    characters = np.array(
        [CHARACTER_TABLE[detector.system.point_group][label] for label in labels]
    )
    for i, U in enumerate(detector.U_ops.values()):
        np.testing.assert_allclose(
            C.T.conj() @ detector.S @ U @ C,
            np.diag(characters[:, i]),
            atol=detector.tol,
            rtol=0,
        )


def test_noncontiguous_mixed_mos_with_energy_splitting():
    # A core sigma_g/sigma_u pair need not meet the old 1e-6 energy
    # tolerance, and other orbitals may lie between them in the MO array.
    labels = ["ag", "b2u", "b1u"]
    angle = 0.1
    C = np.eye(3)
    C[np.ix_([0, 2], [0, 2])] = [
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)],
    ]
    untouched = C[:, 1].copy()
    detector = _detector(labels, C, np.array([-15.8165, -0.5, -15.8161]))
    assigned, _ = detector._assign_irrep_labels()
    assert assigned == labels
    np.testing.assert_array_equal(C[:, 1], untouched)
    _assert_characters(detector, assigned)


@pytest.mark.parametrize("complex_mos", [False, True])
def test_simultaneous_resolution_of_all_irreps(complex_mos):
    # Resolving all eight D2h irreps requires more than two binary splits.
    # Repeated irreps must retain canonical orbitals within their subspaces.
    labels = list(COTTON_LABELS["D2H"]) + ["ag", "b1u"]
    rng = np.random.default_rng(265)
    matrix = rng.standard_normal((len(labels), len(labels)))
    if complex_mos:
        matrix = matrix + 1j * rng.standard_normal(matrix.shape)
    C, _ = np.linalg.qr(matrix)
    eps = np.linspace(-2, 1, len(labels))
    F = C @ np.diag(eps) @ C.T.conj()
    detector = _detector(labels, C, eps)
    assigned, _ = detector._assign_irrep_labels()
    assert sorted(assigned) == sorted(labels)
    assert np.all(np.diff(eps) >= 0)
    _assert_characters(detector, assigned)
    F_mo = C.T.conj() @ F @ C
    np.testing.assert_allclose(np.diag(F_mo), eps, atol=1e-12)
    for label in set(assigned):
        indices = np.flatnonzero(np.array(assigned) == label)
        np.testing.assert_allclose(
            F_mo[np.ix_(indices, indices)], np.diag(eps[indices]), atol=1e-12
        )


def test_incomplete_symmetry_space_raises():
    # This single MO mixes g/u partners outside the supplied orbital space.
    C = np.array([[np.cos(0.2)], [np.sin(0.2)]])
    original = C.copy()
    detector = _detector(["ag", "b1u"], C, np.array([-1.0]))
    with pytest.raises(RuntimeError, match="not fully symmetrized"):
        detector._assign_irrep_labels()
    np.testing.assert_array_equal(C, original)


def test_invalid_character_vector_raises():
    detector = _detector(["ag", "ag"], np.eye(2), np.zeros(2))
    detector.U_ops["C2z"] = -np.eye(2)
    with pytest.raises(RuntimeError, match="do not match.*character table"):
        detector._assign_irrep_labels()


@pytest.mark.parametrize("distance", [1.8, 2.0, 3.0])
def test_stretched_n2_mo_symmetry(distance):
    system = System(
        xyz=f"N 0 0 0; N 0 0 {distance}",
        basis_set="cc-pvdz",
        auxiliary_basis_set="cc-pvtz-jkfit",
        symmetry=True,
    )
    scf = RHF(charge=0)(system).run()
    assert system.point_group == "D2H"
    labels = scf.mos.irrep_labels[0]
    assert labels[:5] == ["ag", "b1u", "ag", "b1u", "ag"]
    # The order within each degenerate pi pair can vary between eigensolvers.
    assert set(labels[5:7]) == {"b2u", "b3u"}
    assert set(labels[7:9]) == {"b2g", "b3g"}
    assert labels[9] == "b1u"
    _assert_scf_symmetry(scf)


@pytest.mark.parametrize("element,distance", [("H", 3.0), ("Li", 5.0)])
def test_stretched_diatomic_mo_symmetry(element, distance):
    system = System(
        xyz=f"{element} 0 0 0; {element} 0 0 {distance}",
        basis_set="cc-pvdz",
        auxiliary_basis_set="def2-universal-jkfit",
        symmetry=True,
    )
    scf = RHF(charge=0)(system).run()
    assert system.point_group == "D2H"
    assert set(scf.mos.irrep_labels[0][:2]) == {"ag", "b1u"}
    _assert_scf_symmetry(scf)


@pytest.mark.parametrize("method", [UHF, ROHF, CUHF])
def test_open_shell_scf_symmetry(method):
    system = System(
        xyz="H 0 0 0; H 0 0 3.0",
        basis_set="cc-pvdz",
        auxiliary_basis_set="def2-universal-jkfit",
        symmetry=True,
    )
    scf = method(charge=1, ms=0.5)(system).run()
    assert all(labels[0] == "ag" for labels in scf.mos.irrep_labels)
    _assert_scf_symmetry(scf)


def _assert_scf_symmetry(scf):
    for spin, C in enumerate(scf.mos.C):
        detector = MOSymmetryDetector(
            scf.system,
            BasisInfo(scf.system, scf.system.basis),
            scf.system.ints_overlap(),
            C,
            scf.eps[spin],
        )
        detector.U_ops = detector._build_U_matrices(
            get_symmetry_ops(scf.system.point_group)
        )
        labels = scf.mos.irrep_labels[spin]
        _assert_characters(detector, labels)
        assert scf.mos.irrep_indices[spin] == [
            COTTON_LABELS["D2H"][label] for label in labels
        ]
        assert np.all(np.diff(scf.eps[spin]) >= 0)
    for spin, D in enumerate(scf.D):
        C = scf.mos.C[min(spin, len(scf.mos.C) - 1)]
        nocc = (scf.na, scf.nb)[spin]
        np.testing.assert_allclose(D, C[:, :nocc] @ C[:, :nocc].T, atol=1e-12, rtol=0)
