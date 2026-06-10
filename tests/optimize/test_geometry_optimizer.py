import numpy as np
import pytest

import forte2
from forte2 import GeometryOptimizer, System
from forte2.optimize.geometry_optimizer import _project_previous_occupied_orbitals
from forte2.scf import RHF
from forte2.system import BSE_AVAILABLE


def test_geometry_optimizer_relaxes_stretched_h2():
    system = System(
        xyz="H 0 0 0\nH 0 0 2.4",
        basis_set="sto-3g",
        auxiliary_basis_set="def2-universal-JKFIT",
        unit="bohr",
    )
    initial = RHF(charge=0, e_tol=1.0e-12, d_tol=1.0e-10, maxiter=100)(system).run().E
    rhf = RHF(charge=0, e_tol=1.0e-12, d_tol=1.0e-10, maxiter=100)(system)

    optimizer = GeometryOptimizer(
        maxiter=25,
        g_tol=1.0e-7,
        max_step=0.5,
    )(rhf)
    optimizer.run()

    bond_length = np.linalg.norm(optimizer.coordinates[1] - optimizer.coordinates[0])

    assert rhf.executed
    assert optimizer.converged
    assert optimizer.E < initial
    assert optimizer.E == pytest.approx(-1.117530189001, abs=1.0e-8)
    assert bond_length == pytest.approx(1.34590756, abs=1.0e-6)
    assert np.linalg.norm(optimizer.gradient) < 5.0e-7
    assert optimizer.system is not None
    assert optimizer.method is not None


@pytest.mark.skipif(not BSE_AVAILABLE, reason="basis_set_exchange not installed")
def test_geometry_optimizer_water_cc_pvdz():
    system = System(
        xyz="""
        O 0.000000 0.000000 0.000000
        H 0.000000 0.000000 2.100000
        H 1.900000 0.000000 0.000000
        """,
        basis_set="cc-pVDZ",
        auxiliary_basis_set="def2-universal-JKFIT",
        unit="bohr",
    )
    initial = RHF(charge=0, e_tol=1.0e-10, d_tol=1.0e-6, maxiter=100)(system).run().E
    rhf = RHF(charge=0, e_tol=1.0e-10, d_tol=1.0e-6, maxiter=100)(system)

    optimizer = GeometryOptimizer(
        maxiter=15,
        g_tol=1.0e-7,
        max_step=0.3,
    )(rhf)
    optimizer.run()

    coordinates = optimizer.coordinates
    r_oh1 = np.linalg.norm(coordinates[1] - coordinates[0])
    r_oh2 = np.linalg.norm(coordinates[2] - coordinates[0])
    cos_angle = np.dot(coordinates[1] - coordinates[0], coordinates[2] - coordinates[0])
    cos_angle /= r_oh1 * r_oh2
    angle = np.degrees(np.arccos(cos_angle))

    assert optimizer.converged
    assert np.linalg.norm(optimizer.gradient) < 5.0e-7
    assert r_oh1 == pytest.approx(1.7882104, abs=1.0e-6)
    assert r_oh2 == pytest.approx(1.7882104, abs=1.0e-6)
    assert angle == pytest.approx(104.61747, abs=1.0e-5)
    assert optimizer.E == pytest.approx(-76.027021264399, abs=1.0e-8)


def test_project_previous_occupied_orbitals_to_new_geometry():
    old_system = System(
        xyz="H 0 0 0\nH 0 0 1.7",
        basis_set="sto-3g",
        auxiliary_basis_set="def2-universal-JKFIT",
        unit="bohr",
    )
    new_system = System(
        xyz="H 0 0 0\nH 0 0 1.8",
        basis_set="sto-3g",
        auxiliary_basis_set="def2-universal-JKFIT",
        unit="bohr",
    )
    old_rhf = RHF(charge=0, e_tol=1.0e-12, d_tol=1.0e-10)(old_system)
    old_rhf.run()
    new_rhf = RHF(charge=0, e_tol=1.0e-12, d_tol=1.0e-10)(new_system)

    projected = _project_previous_occupied_orbitals(old_rhf, new_rhf)

    assert projected is not None
    assert len(projected) == 1
    assert projected[0].shape == (new_system.nbf, new_system.nmo)

    S = new_system.ints_overlap()
    # Check that the projected orbitals are orthonormal in the new basis
    np.testing.assert_allclose(
        projected[0].T @ S @ projected[0], np.eye(new_system.nmo), atol=1.0e-10
    )
    # Check that the projected occupied orbital has a large overlap with the old one
    S_cross = forte2.integrals.overlap(new_system, new_system.basis, old_system.basis)
    occupied_overlap = (
        projected[0][:, : new_rhf.na].T @ S_cross @ old_rhf.C[0][:, : old_rhf.na]
    )
    assert abs(occupied_overlap[0, 0]) > 0.99
