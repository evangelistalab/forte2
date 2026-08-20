import numpy as np
import pytest

from forte2 import CISolver, GeometryOptimizer, MCOptimizer, State, System
from forte2.base_classes.params import SelectedCIParams
from forte2.orbitals import mo_overlap, project_occupied_orbitals
from forte2.scf import RHF
from forte2.sci import SelectedCISolver
from forte2.system import BSE_AVAILABLE
from forte2.gradients import FDGradient


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


def test_geometry_optimizer_h2_fd():
    system = System(
        xyz="H 0 0 0\nH 0 0 2.4",
        basis_set="sto-3g",
        auxiliary_basis_set="def2-universal-JKFIT",
        unit="bohr",
    )
    initial = RHF(charge=0, e_tol=1.0e-12, d_tol=1.0e-10, maxiter=100)(system).run().E
    rhf = RHF(charge=0, e_tol=1.0e-12, d_tol=1.0e-10, maxiter=100)(system)
    fd = FDGradient()(rhf)

    optimizer = GeometryOptimizer(
        maxiter=25,
        g_tol=1.0e-7,
        max_step=0.5,
    )(fd)
    optimizer.run()

    bond_length = np.linalg.norm(optimizer.coordinates[1] - optimizer.coordinates[0])

    assert optimizer.E == pytest.approx(-1.117530189001, abs=1.0e-8)
    assert bond_length == pytest.approx(1.34590756, abs=1.0e-6)
    assert np.linalg.norm(optimizer.gradient) < 5.0e-7


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

    projected = project_occupied_orbitals(old_rhf, new_rhf)

    assert projected is not None
    assert len(projected) == 1
    assert projected[0].shape == (new_system.nbf, new_system.nmo)

    # Check that the projected orbitals are orthonormal in the new basis
    np.testing.assert_allclose(
        mo_overlap(projected[0], new_system, projected[0]),
        np.eye(new_system.nmo),
        atol=1.0e-10,
    )
    # Check that the projected occupied orbital has a large overlap with the old one
    occupied_overlap = mo_overlap(
        projected[0][:, : new_rhf.na],
        new_system,
        old_rhf.C[0][:, : old_rhf.na],
        old_system,
    )
    assert abs(occupied_overlap[0, 0]) > 0.99


def _h2_casscf(bond_length):
    system = System(
        xyz=f"H 0 0 0\nH 0 0 {bond_length}",
        basis_set="sto-3g",
        auxiliary_basis_set="def2-universal-JKFIT",
        unit="bohr",
    )
    ci_solver = CISolver(
        State(system=system, multiplicity=1, ms=0.0), active_orbitals=[0, 1]
    )
    rhf = RHF(charge=0, e_tol=1.0e-12, d_tol=1.0e-10, maxiter=100)(system)
    return MCOptimizer(ci_solver, e_tol=1.0e-12, g_tol=1.0e-9)(rhf)


def test_geometry_optimizer_relaxes_a_multistage_casscf_chain():
    # A chained method only acquires `system` when it runs, and rebuilding it at a
    # new geometry means rebuilding every stage including the nested CI solver.
    mc = _h2_casscf(1.7)

    optimizer = GeometryOptimizer(maxiter=25, g_tol=1.0e-7, max_step=0.5)(mc)
    optimizer.run()

    bond_length = np.linalg.norm(optimizer.coordinates[1] - optimizer.coordinates[0])

    assert optimizer.converged
    assert np.linalg.norm(optimizer.gradient) < 5.0e-7
    assert bond_length == pytest.approx(1.38862862, abs=1.0e-6)
    assert optimizer.E == pytest.approx(-1.137332707937, abs=1.0e-8)
    # The optimized chain is a rebuild, not the object that was passed in.
    assert optimizer.method is not mc
    assert isinstance(optimizer.method, MCOptimizer)


def test_geometry_optimizer_casscf_fd():
    # A chained method only acquires `system` when it runs, and rebuilding it at a
    # new geometry means rebuilding every stage including the nested CI solver.
    mc = _h2_casscf(1.7)
    fd = FDGradient()(mc)

    optimizer = GeometryOptimizer(maxiter=25, g_tol=1.0e-7, max_step=0.5)(fd)
    optimizer.run()

    bond_length = np.linalg.norm(optimizer.coordinates[1] - optimizer.coordinates[0])

    assert bond_length == pytest.approx(1.38862862, abs=1.0e-6)
    assert optimizer.E == pytest.approx(-1.137332707937, abs=1.0e-8)


def test_geometry_optimizer_reuses_a_single_chain_across_steps():
    # Regression test for scratch-chain reuse: previously every L-BFGS iterate
    # rebuilt the whole chain from scratch, which is what multiplies out to a
    # memory blowup once FDGradient repeats that per stencil
    # point. Also exercises the SelectedCISolver params-mutation fix: a stale
    # per-state SelectedCIParams reused across geometries would otherwise leak
    # guess determinants from one geometry's solve into the next.
    import forte2.optimize.geometry_optimizer as geometry_optimizer_module

    system = System(
        xyz="H 0 0 0\nH 0 0 1.7",
        basis_set="sto-3g",
        auxiliary_basis_set="def2-universal-JKFIT",
        unit="bohr",
    )
    sci_params = SelectedCIParams(
        ci_algorithm="exact",
        guess_occ_window=1,
        guess_vir_window=1,
        var_threshold=1.0e-8,
        pt2_threshold=0.0,
    )
    ci_solver = SelectedCISolver(
        states=State(nel=2, multiplicity=1, ms=0.0),
        core_orbitals=0,
        active_orbitals=2,
        sci_params=sci_params,
    )
    rhf = RHF(charge=0, e_tol=1.0e-12, d_tol=1.0e-10, maxiter=100)(system)
    mc = MCOptimizer(ci_solver, e_tol=1.0e-12, g_tol=1.0e-9, final_orbitals="original")(
        rhf
    )

    call_count = 0
    original_rebuild = geometry_optimizer_module.rebuild_method_chain

    def counting_rebuild(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original_rebuild(*args, **kwargs)

    geometry_optimizer_module.rebuild_method_chain = counting_rebuild
    try:
        optimizer = GeometryOptimizer(maxiter=25, g_tol=1.0e-7, max_step=0.5)(mc)
        optimizer.run()
    finally:
        geometry_optimizer_module.rebuild_method_chain = original_rebuild

    assert optimizer.converged
    assert optimizer.iter > 1
    assert np.linalg.norm(optimizer.gradient) < 5.0e-7
    # Exactly one fresh chain built for the whole optimization: every
    # subsequent L-BFGS step reuses (rebinds) it instead of rebuilding.
    assert call_count == 1

    # The scratch chain's ci_solver is a fresh copy from the one-time initial
    # build (rebuild_method_chain always copies, per test_rebuild.py), but its
    # sci_params field carries the original object by reference. That original
    # sci_params must come out of many rebind-and-rerun cycles exactly as
    # configured, not carrying guess determinants leaked from an intermediate
    # geometry's solve.
    assert optimizer.method.ci_solver is not ci_solver
    assert optimizer.method.ci_solver.sci_params is sci_params
    assert sci_params.guess_dets == []
