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


def _orbital_gradient_at_wavefunction_displacement(
    mc,
    orbital_displacement,
    ci_displacement,
    state_averaged,
):
    """Evaluate an orbital gradient from explicitly displaced C and CI vectors."""
    layout = mc.get_ci_response_layout()
    nact = mc.mo_space.nactv
    g1 = np.zeros((nact,) * 2)
    g2 = np.zeros((nact,) * 4)
    total_norm = 0.0

    for absolute_root, state_index, root_in_state, coefficient_slice in layout:
        sub_solver = mc.ci_solver.sub_solvers[state_index]
        reference = sub_solver.evecs[:, root_in_state]
        if state_averaged:
            weight = mc.ci_solver.weights_flat[absolute_root]
            assert weight > 0.0
            coefficient = reference + ci_displacement[coefficient_slice] / weight
            density_weight = weight
        else:
            coefficient = reference + ci_displacement[coefficient_slice]
            density_weight = 1.0

        coefficient_det = sub_solver.csf_C_to_det_C(coefficient)
        sigma_builder = sub_solver.ci_sigma_builder
        total_norm += density_weight * np.dot(coefficient, coefficient)
        g1 += density_weight * sigma_builder.sf_1rdm(coefficient_det, coefficient_det)
        g2 += density_weight * sigma_builder.sf_2rdm(coefficient_det, coefficient_det)

    orbital_optimizer = mc.orb_opt
    trial = OrbOptimizer(
        orbital_optimizer.C.copy(),
        (orbital_optimizer.core, orbital_optimizer.actv, orbital_optimizer.virt),
        orbital_optimizer.fock_builder,
        orbital_optimizer.hcore,
        orbital_optimizer.e_nuc,
        orbital_optimizer.nrr.copy(),
        compute_active_hessian=orbital_optimizer.compute_active_hessian,
    )
    trial.set_rdms(g1, g2)
    trial._update_orbitals(orbital_displacement)
    trial._compute_Fcore()
    trial.get_eri_gaaa()
    trial._compute_orbgrad()

    # OrbOptimizer assumes a normalized CI state. Restore the scalar core
    # contribution for the deliberately unnormalized finite-difference vectors.
    trial.A_pq[:, trial.core] += 2.0 * (total_norm - 1.0) * trial.Fcore[:, trial.core]
    gradient = 2.0 * (trial.A_pq - trial.A_pq.T)
    return trial._mat_to_vec(gradient)


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


def test_sa_casscf_orbital_ci_response_lih():
    """Check the CI action, dense block, and combined orbital response."""
    symbols = ["Li", "H"]
    coordinates = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 3.0]])
    mc = _sa_casscf(symbols, coordinates)
    orbital_optimizer = mc.orb_opt
    layout = mc.get_ci_response_layout()
    nci = layout[-1][-1].stop

    ci_direction = np.arange(1, nci + 1, dtype=float)
    ci_direction /= np.linalg.norm(ci_direction)
    overlap_response = 0.0
    for _, state_index, root_in_state, coefficient_slice in layout:
        reference = mc.ci_solver.sub_solvers[state_index].evecs[:, root_in_state]
        overlap_response += 2.0 * np.dot(ci_direction[coefficient_slice], reference)
    assert abs(overlap_response) > 1.0e-3

    ci_product = mc.compute_orbital_ci_hessian_vector_product(ci_direction)

    step = 1.0e-5
    zero_orbital = np.zeros(orbital_optimizer.nrot)
    gradient_plus = _orbital_gradient_at_wavefunction_displacement(
        mc,
        zero_orbital,
        step * ci_direction,
        state_averaged=False,
    )
    gradient_minus = _orbital_gradient_at_wavefunction_displacement(
        mc,
        zero_orbital,
        -step * ci_direction,
        state_averaged=False,
    )
    finite_difference = (gradient_plus - gradient_minus) / (2.0 * step)
    assert ci_product == pytest.approx(finite_difference, abs=1.0e-8)

    orbital_ci_hessian = mc.compute_orbital_ci_hessian()
    assert orbital_ci_hessian.shape == (orbital_optimizer.nrot, nci)
    assert orbital_ci_hessian @ ci_direction == pytest.approx(ci_product, abs=1.0e-11)

    orbital_direction = np.arange(1, orbital_optimizer.nrot + 1, dtype=float)
    orbital_direction /= np.linalg.norm(orbital_direction)
    combined = mc.compute_orbital_response_vector_product(
        orbital_direction, ci_direction
    )
    gradient_plus = _orbital_gradient_at_wavefunction_displacement(
        mc,
        step * orbital_direction,
        step * ci_direction,
        state_averaged=True,
    )
    gradient_minus = _orbital_gradient_at_wavefunction_displacement(
        mc,
        -step * orbital_direction,
        -step * ci_direction,
        state_averaged=True,
    )
    finite_difference = (gradient_plus - gradient_minus) / (2.0 * step)
    assert combined == pytest.approx(finite_difference, abs=1.0e-7)
