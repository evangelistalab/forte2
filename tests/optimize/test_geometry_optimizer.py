import numpy as np
import pytest

from forte2 import CISolver, GeometryOptimizer, MCOptimizer, State, System
from forte2.scf import RHF
from forte2.system import BSE_AVAILABLE
from forte2.gradients import FDGradient
from forte2.dsrg import DSRG_MRPT2


def test_geometry_optimizer_h2():
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

    assert optimizer.E == pytest.approx(-1.117530189001, abs=1.0e-8)
    assert bond_length == pytest.approx(1.34590756, abs=1.0e-6)


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


def test_geometry_optimizer_selects_matching_root_energy_and_gradient():
    """Use the same absolute root for both parts of an SA objective."""
    system = System(
        xyz="H 0 0 0\nH 0 0 1.4",
        basis_set="sto-3g",
        auxiliary_basis_set="def2-universal-JKFIT",
        symmetry=False,
    )
    built_methods = []

    class RootResolvedMethod:
        def __init__(self, current_system):
            self.system = current_system
            self.executed = False
            self.E = None
            self.E_ci = None
            self.gradient_roots = []

        def run(self):
            self.E = -1.0
            self.E_ci = np.array([-0.9, -0.4])
            self.executed = True
            return self

        def gradient(self, root=None):
            self.gradient_roots.append(root)
            return np.full((2, 3), float(root))

    def build_method(current_system):
        method = RootResolvedMethod(current_system)
        built_methods.append(method)
        return method

    optimizer = GeometryOptimizer(
        method_factory=build_method,
        root=1,
        project_orbitals=False,
    )
    objective, coordinates = optimizer._build_objective(system)

    assert objective.evaluate(coordinates) == pytest.approx(-0.4)
    assert objective.gradient(coordinates) == pytest.approx(np.ones(6))
    assert built_methods[0].gradient_roots == [1]

    with pytest.raises(TypeError, match="integer or None"):
        GeometryOptimizer(root=True)
    with pytest.raises(ValueError, match="nonnegative"):
        GeometryOptimizer(root=-1)


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


def test_geometry_optimizer_casscf():
    # A chained method only acquires `system` when it runs, and rebuilding it at a
    # new geometry means rebuilding every stage including the nested CI solver.
    mc = _h2_casscf(1.7)

    optimizer = GeometryOptimizer(maxiter=25, g_tol=1.0e-7, max_step=0.5)(mc)
    optimizer.run()

    bond_length = np.linalg.norm(optimizer.coordinates[1] - optimizer.coordinates[0])

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


def test_fd_gradient_sa_casscf():
    system = System(
        xyz="H 0 0 0\nH 0 0 2.0",
        basis_set="sto-3g",
        auxiliary_basis_set="def2-universal-JKFIT",
        unit="bohr",
    )
    rhf = RHF(charge=0, e_tol=1.0e-12, d_tol=1.0e-10, maxiter=100)(system)
    ci_solver = CISolver(
        State(system=system, multiplicity=1, ms=0.0),
        active_orbitals=[0, 1],
        nroots=2,
    )
    mc = MCOptimizer(
        ci_solver, e_tol=1.0e-12, g_tol=1.0e-10, final_orbitals="original"
    )(rhf)

    # optmize wrt to the second root
    energy_to_differentiate = lambda method: method.ci_solver.E[1]
    fd = FDGradient(
        step=1.0e-3,
        npoints=4,
        energy_accessor=energy_to_differentiate,
    )(mc)

    optimizer = GeometryOptimizer(
        maxiter=25,
        g_tol=1.0e-7,
        max_step=0.5,
    )(fd)
    optimizer.run()


@pytest.mark.slow
def test_fd_gradient_dsrg_mrpt2():
    system = System(
        xyz="H 0 0 0\nH 0 0 2.0",
        basis_set="6-31g",
        auxiliary_basis_set="def2-universal-JKFIT",
        unit="bohr",
    )
    rhf = RHF(charge=0, e_tol=1.0e-12, d_tol=1.0e-10, maxiter=100)(system)
    ci_solver = CISolver(
        State(system=system, multiplicity=1, ms=0.0),
        active_orbitals=[0, 1],
    )
    mc = MCOptimizer(
        ci_solver,
        e_tol=1.0e-12,
        g_tol=1.0e-10,
        final_orbitals="semicanonical",
    )(rhf)
    dsrg = DSRG_MRPT2(flow_param=0.5, relax_reference=True)(mc)

    fd = FDGradient(
        step=1.0e-3,
        npoints=4,
        energy_accessor=lambda method: method.E_relaxed_ref,
    )(dsrg)

    optimizer = GeometryOptimizer(
        maxiter=25,
        g_tol=1.0e-7,
        max_step=0.5,
    )(fd)
    optimizer.run()

    bond_length = np.linalg.norm(optimizer.coordinates[1] - optimizer.coordinates[0])

    assert optimizer.converged
    assert optimizer.E == pytest.approx(-1.1492336391828861, abs=1.0e-7)
    assert bond_length == pytest.approx(1.4140893807, abs=1.0e-6)
