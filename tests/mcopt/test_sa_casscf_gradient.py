import numpy as np
import pytest

from forte2 import CISolver, MCOptimizer, RHF, State, System
from forte2.mcopt.orbital_optimizer import OrbOptimizer
from tests.gradient_test_utils import (
    four_point_central_difference_gradient_component,
    xyz_string,
)


def _sa_casscf(symbols, coordinates):
    system = System(
        xyz=xyz_string(symbols, coordinates),
        basis_set="sto-3g",
        auxiliary_basis_set="def2-universal-JKFIT",
        unit="bohr",
    )
    rhf = RHF(charge=0, e_tol=1.0e-12, d_tol=1.0e-10, maxiter=100)(system)
    ci_solver = CISolver(
        State(system=system, multiplicity=1, ms=0.0),
        core_orbitals=[0],
        active_orbitals=[1, 2],
        nroots=2,
        weights=[0.5, 0.5],
    )
    mc = MCOptimizer(
        ci_solver,
        e_tol=1.0e-12,
        g_tol=1.0e-9,
        maxiter=30,
        final_orbitals="original",
    )(rhf)
    mc.run()
    return mc


def _sa_casscf_root_energies(symbols, coordinates):
    return _sa_casscf(symbols, coordinates).E_ci


def _orbital_gradient_at_displacement(orbital_optimizer, displacement):
    """Evaluate the fixed-RDM orbital gradient from the current orbitals."""
    trial = OrbOptimizer(
        orbital_optimizer.C.copy(),
        (orbital_optimizer.core, orbital_optimizer.actv, orbital_optimizer.virt),
        orbital_optimizer.fock_builder,
        orbital_optimizer.hcore,
        orbital_optimizer.e_nuc,
        orbital_optimizer.nrr.copy(),
        compute_active_hessian=orbital_optimizer.compute_active_hessian,
    )
    # These RDMs are already in the orbital optimizer's internal convention.
    trial.g1 = orbital_optimizer.g1.copy()
    trial.g2 = orbital_optimizer.g2.copy()
    trial.evaluate(displacement)
    return trial.gradient(displacement)


def test_sa_casscf_gradient_lih_finite_difference_reference():
    """Establish root-specific LiH gradient references at SA-CASSCF orbitals."""
    symbols = ["Li", "H"]
    coordinates = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 3.0]])

    numerical = four_point_central_difference_gradient_component(
        _sa_casscf_root_energies,
        symbols,
        coordinates,
        1,
        2,
    )

    assert numerical == pytest.approx(
        np.array([0.015192021141382, -0.016999054963958]),
        abs=1.0e-8,
    )


def test_sa_casscf_orbital_orbital_response_lih():
    """Check the dense and matrix-free fixed-RDM orbital response."""
    symbols = ["Li", "H"]
    coordinates = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 3.0]])
    orbital_optimizer = _sa_casscf(symbols, coordinates).orb_opt

    direction = np.arange(1, orbital_optimizer.nrot + 1, dtype=float)
    direction /= np.linalg.norm(direction)
    product = orbital_optimizer.compute_orbital_hessian_vector_product(direction)

    step = 1.0e-4
    gradient_plus = _orbital_gradient_at_displacement(
        orbital_optimizer, step * direction
    )
    gradient_minus = _orbital_gradient_at_displacement(
        orbital_optimizer, -step * direction
    )
    finite_difference = (gradient_plus - gradient_minus) / (2.0 * step)

    assert product == pytest.approx(finite_difference, abs=1.0e-7)

    hessian = orbital_optimizer.compute_orbital_hessian()
    assert hessian @ direction == pytest.approx(product, abs=1.0e-11)
    assert hessian == pytest.approx(hessian.T, abs=1.0e-8)
