import numpy as np
import pytest

from forte2 import CUHF, RHF, ROHF, UHF, System
from forte2.lib import ints
from forte2.symmetry import SymmetryBasis
from forte2.symmetry.sym_utils import CHARACTER_TABLE, COTTON_LABELS, get_symmetry_ops
from forte2.system import BasisInfo


def _symmetry_basis(system):
    return SymmetryBasis.build(
        system,
        BasisInfo(system, system.basis),
        system.ints_overlap(),
        system.get_Xorth(),
    )


def _assert_scf_symmetry(scf):
    pg = scf.system.point_group
    assert scf.mos.point_group == pg
    basis = _symmetry_basis(scf.system)
    points = np.random.default_rng(265).uniform(-3, 3, (30, 3))
    for spin, (C, labels) in enumerate(zip(scf.mos.C, scf.mos.irrep_labels)):
        # Each orbital must pick up its irrep's character under every operation.
        characters = np.array([CHARACTER_TABLE[pg][label] for label in labels])
        C = np.ascontiguousarray(C)
        values = ints.orbitals_at_points(scf.system.basis, points, C)
        for j, R in enumerate(get_symmetry_ops(pg).values()):
            transformed = ints.orbitals_at_points(scf.system.basis, points @ R.T, C)
            np.testing.assert_allclose(
                transformed, values * characters[:, j], atol=1e-9, rtol=0
            )
        indices = scf.mos.irrep_indices[spin]
        assert indices == [COTTON_LABELS[pg][label] for label in labels]
        assert basis.orbital_irreps(C).tolist() == indices
        assert np.all(np.diff(scf.eps[spin]) >= 0)
    for spin, D in enumerate(scf.D):
        C = scf.mos.C[min(spin, len(scf.mos.C) - 1)]
        nocc = (scf.na, scf.nb)[spin]
        np.testing.assert_allclose(D, C[:, :nocc] @ C[:, :nocc].T, atol=1e-12, rtol=0)


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

    # Mixing the near-degenerate ag and b1u orbitals breaks the symmetry.
    mixed = scf.mos.C[0][:, :2] @ np.array([[1, -1], [1, 1]]) / np.sqrt(2)
    with pytest.raises(RuntimeError, match="single D2H irrep"):
        _symmetry_basis(system).orbital_irreps(mixed)


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
