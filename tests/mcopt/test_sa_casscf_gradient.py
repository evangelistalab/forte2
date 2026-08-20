import numpy as np
import pytest

from forte2 import CISolver, MCOptimizer, RHF, State, System
from forte2.lib.ci_helpers import CISigmaBuilder
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


def _sa_casscf_c2_ccpvdz(symbols, coordinates):
    """Run a compact two-root C2 SA-CASSCF calculation in cc-pVDZ."""
    system = System(
        xyz=xyz_string(symbols, coordinates),
        basis_set="cc-pVDZ",
        auxiliary_basis_set="cc-pVTZ-JKFIT",
        unit="bohr",
    )
    rhf = RHF(charge=0, e_tol=1.0e-12, d_tol=1.0e-10, maxiter=100)(system)
    ci_solver = CISolver(
        State(system=system, multiplicity=1, ms=0.0),
        core_orbitals=[0, 1, 2, 3, 4],
        active_orbitals=[5, 6],
        nroots=2,
        weights=[0.5, 0.5],
    )
    mc = MCOptimizer(
        ci_solver,
        e_tol=1.0e-11,
        g_tol=1.0e-9,
        maxiter=40,
        final_orbitals="original",
    )(rhf)
    mc.run()
    return mc


def _sa_casscf_c2_ccpvdz_root_energies(symbols, coordinates):
    return _sa_casscf_c2_ccpvdz(symbols, coordinates).E_ci


def _sa_gasscf_h2_ccpvdz(symbols, coordinates):
    """Run a two-root H2 SA-GASSCF calculation with two GAS spaces."""
    system = System(
        xyz=xyz_string(symbols, coordinates),
        basis_set="cc-pVDZ",
        auxiliary_basis_set="cc-pVTZ-JKFIT",
        unit="bohr",
    )
    rhf = RHF(charge=0, e_tol=1.0e-12, d_tol=1.0e-10, maxiter=100)(system)
    ci_solver = CISolver(
        State(
            system=system,
            multiplicity=1,
            ms=0.0,
            gas_min=[1],
            gas_max=[1],
        ),
        active_orbitals=[[0], [1, 2]],
        nroots=2,
        weights=[0.5, 0.5],
    )
    mc = MCOptimizer(
        ci_solver,
        e_tol=1.0e-11,
        g_tol=1.0e-8,
        maxiter=30,
        final_orbitals="original",
    )(rhf)
    mc.run()
    return mc


def _sa_gasscf_h2_ccpvdz_root_energies(symbols, coordinates):
    return _sa_gasscf_h2_ccpvdz(symbols, coordinates).E_ci


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


def _ci_gradient_at_orbital_displacement(mc, orbital_displacement):
    """Evaluate 2 w_alpha H(C) c_alpha at explicitly displaced orbitals."""
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
    trial._update_orbitals(orbital_displacement)
    trial._compute_Fcore()
    trial.get_eri_gaaa()

    scalar = trial.Ecore + trial.e_nuc
    one_body = np.ascontiguousarray(trial.Fcore[trial.actv, trial.actv])
    two_body = np.ascontiguousarray(trial.get_active_space_ints())
    layout = mc.get_ci_response_layout()
    gradient = np.empty(layout[-1][-1].stop)
    builders = {}

    for absolute_root, state_index, root_in_state, coefficient_slice in layout:
        sub_solver = mc.ci_solver.sub_solvers[state_index]
        if state_index not in builders:
            builder = CISigmaBuilder(
                sub_solver.ci_strings,
                scalar,
                one_body,
                two_body,
                sub_solver.log_level,
            )
            algorithm = sub_solver.ci_params.ci_algorithm.lower()
            builder.set_algorithm("kh" if algorithm == "exact" else algorithm)
            builders[state_index] = builder

        reference = sub_solver.evecs[:, root_in_state]
        reference_det = sub_solver.csf_C_to_det_C(reference)
        sigma_det = np.empty(sub_solver.ndet)
        builders[state_index].Hamiltonian(reference_det, sigma_det)
        sigma_csf = np.empty(sub_solver.basis_size)
        sub_solver.spin_adapter.det_C_to_csf_C(sigma_det, sigma_csf)
        weight = mc.ci_solver.weights_flat[absolute_root]
        gradient[coefficient_slice] = 2.0 * weight * sigma_csf

    return gradient


def _root_energy_at_orbital_displacement(mc, root, orbital_displacement):
    """Evaluate a fixed-CI root energy at explicitly displaced orbitals."""
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
    trial._update_orbitals(orbital_displacement)
    trial._compute_Fcore()
    trial.get_eri_gaaa()

    absolute_root, state_index, root_in_state, _ = mc.get_ci_response_layout()[root]
    assert absolute_root == root
    sub_solver = mc.ci_solver.sub_solvers[state_index]
    scalar = trial.Ecore + trial.e_nuc
    one_body = np.ascontiguousarray(trial.Fcore[trial.actv, trial.actv])
    two_body = np.ascontiguousarray(trial.get_active_space_ints())
    builder = CISigmaBuilder(
        sub_solver.ci_strings,
        scalar,
        one_body,
        two_body,
        sub_solver.log_level,
    )
    algorithm = sub_solver.ci_params.ci_algorithm.lower()
    builder.set_algorithm("kh" if algorithm == "exact" else algorithm)

    reference = sub_solver.evecs[:, root_in_state]
    reference_det = sub_solver.csf_C_to_det_C(reference)
    sigma_det = np.empty(sub_solver.ndet)
    builder.Hamiltonian(reference_det, sigma_det)
    sigma_csf = np.empty(sub_solver.basis_size)
    sub_solver.spin_adapter.det_C_to_csf_C(sigma_det, sigma_csf)
    return np.dot(reference, sigma_csf)


def test_sa_casscf_gradient_lih_finite_difference():
    """Validate both relaxed root gradients against four-point differences."""
    symbols = ["Li", "H"]
    coordinates = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 3.0]])

    mc = _sa_casscf(symbols, coordinates)
    gradients = np.array([mc.gradient(root=root) for root in range(2)])
    numerical = four_point_central_difference_gradient_component(
        _sa_casscf_root_energies,
        symbols,
        coordinates,
        1,
        2,
    )

    assert gradients[:, 1, 2] == pytest.approx(numerical, abs=1.0e-7)
    assert numerical == pytest.approx(
        np.array([0.015192021141382, -0.016999054963958]),
        abs=1.0e-8,
    )
    assert gradients.sum(axis=1) == pytest.approx(np.zeros((2, 3)), abs=1.0e-10)
    with pytest.raises(ValueError, match="root must be specified"):
        mc.gradient()
    with pytest.raises(ValueError, match=r"root in \[0, 2\)"):
        mc.gradient(root=2)
    with pytest.raises(TypeError, match="root must be an integer"):
        mc.gradient(root=0.0)


def test_sa_casscf_gradient_c2_ccpvdz_finite_difference():
    """Validate a non-minimal-basis root gradient on a compact C2 problem."""
    symbols = ["C", "C"]
    coordinates = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.3]])

    mc = _sa_casscf_c2_ccpvdz(symbols, coordinates)
    gradient = mc.gradient(root=0)
    numerical = four_point_central_difference_gradient_component(
        _sa_casscf_c2_ccpvdz_root_energies,
        symbols,
        coordinates,
        1,
        2,
    )[0]

    assert mc.system.nbf == 28
    assert gradient[1, 2] == pytest.approx(numerical, abs=1.0e-7)
    assert gradient[1, 2] == pytest.approx(-0.153859975336, abs=1.0e-8)
    assert gradient.sum(axis=0) == pytest.approx(np.zeros(3), abs=1.0e-10)


def test_sa_gasscf_gradient_h2_ccpvdz_finite_difference():
    """Validate both roots for a partitioned, occupation-restricted SA-GASSCF."""
    symbols = ["H", "H"]
    coordinates = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.7]])

    mc = _sa_gasscf_h2_ccpvdz(symbols, coordinates)
    gradients = np.array([mc.gradient(root=root) for root in range(2)])
    numerical = four_point_central_difference_gradient_component(
        _sa_gasscf_h2_ccpvdz_root_energies,
        symbols,
        coordinates,
        1,
        2,
    )

    state = mc.ci_solver.sub_solvers[0].state
    assert mc.mo_space.ngas == 2
    assert mc.mo_space.active_orbitals == [[0], [1, 2]]
    assert state.gas_min == [1]
    assert state.gas_max == [1]
    assert gradients[:, 1, 2] == pytest.approx(numerical, abs=1.0e-7)
    assert numerical == pytest.approx(np.array([-0.05776565, 0.11036714]), abs=1.0e-8)
    assert gradients.sum(axis=1) == pytest.approx(np.zeros((2, 3)), abs=1.0e-10)


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


def test_sa_casscf_ci_orbital_response_lih():
    """Check the orbital action in the CI row and the weighted transpose."""
    symbols = ["Li", "H"]
    coordinates = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 3.0]])
    mc = _sa_casscf(symbols, coordinates)
    orbital_optimizer = mc.orb_opt
    layout = mc.get_ci_response_layout()
    nci = layout[-1][-1].stop

    orbital_direction = np.arange(1, orbital_optimizer.nrot + 1, dtype=float)
    orbital_direction /= np.linalg.norm(orbital_direction)
    ci_product = mc.compute_ci_orbital_hessian_vector_product(orbital_direction)

    step = 1.0e-5
    gradient_plus = _ci_gradient_at_orbital_displacement(mc, step * orbital_direction)
    gradient_minus = _ci_gradient_at_orbital_displacement(mc, -step * orbital_direction)
    finite_difference = (gradient_plus - gradient_minus) / (2.0 * step)
    assert ci_product == pytest.approx(finite_difference, abs=1.0e-8)

    ci_orbital_hessian = mc.compute_ci_orbital_hessian()
    assert ci_orbital_hessian.shape == (nci, orbital_optimizer.nrot)
    assert ci_orbital_hessian @ orbital_direction == pytest.approx(
        ci_product, abs=1.0e-11
    )

    coefficient_weights = np.empty(nci)
    for absolute_root, _, _, coefficient_slice in layout:
        coefficient_weights[coefficient_slice] = mc.ci_solver.weights_flat[
            absolute_root
        ]
    orbital_ci_hessian = mc.compute_orbital_ci_hessian()
    assert ci_orbital_hessian == pytest.approx(
        coefficient_weights[:, None] * orbital_ci_hessian.T,
        abs=1.0e-9,
    )


def test_sa_casscf_ci_ci_response_lih():
    """Check the matrix-free and dense raw CI--CI response block."""
    symbols = ["Li", "H"]
    coordinates = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 3.0]])
    mc = _sa_casscf(symbols, coordinates)
    layout = mc.get_ci_response_layout()
    nci = layout[-1][-1].stop

    ci_direction = np.arange(1, nci + 1, dtype=float)
    ci_direction /= np.linalg.norm(ci_direction)
    product = mc.compute_ci_ci_hessian_vector_product(ci_direction)

    expected = np.empty(nci)
    hamiltonians = {}
    for absolute_root, state_index, _, coefficient_slice in layout:
        sub_solver = mc.ci_solver.sub_solvers[state_index]
        if state_index not in hamiltonians:
            hamiltonians[state_index] = sub_solver.ci_sigma_builder.form_H_csf(
                sub_solver.dets, sub_solver.spin_adapter
            )
        root_hamiltonian = hamiltonians[state_index]
        root_direction = ci_direction[coefficient_slice]
        expected[coefficient_slice] = (
            2.0
            * (root_hamiltonian - mc.E_ci[absolute_root] * np.eye(root_direction.size))
            @ root_direction
        )
    assert product == pytest.approx(expected, abs=1.0e-11)

    ci_ci_hessian = mc.compute_ci_ci_hessian()
    assert ci_ci_hessian.shape == (nci, nci)
    assert ci_ci_hessian @ ci_direction == pytest.approx(product, abs=1.0e-11)
    assert ci_ci_hessian == pytest.approx(ci_ci_hessian.T, abs=1.0e-11)

    for root, (_, _, _, row_slice) in enumerate(layout):
        for other_root, (_, _, _, column_slice) in enumerate(layout):
            if root != other_root:
                assert ci_ci_hessian[row_slice, column_slice] == pytest.approx(0.0)

    references = np.empty(nci)
    for _, state_index, root_in_state, coefficient_slice in layout:
        references[coefficient_slice] = mc.ci_solver.sub_solvers[state_index].evecs[
            :, root_in_state
        ]
    assert mc.compute_ci_ci_hessian_vector_product(references) == pytest.approx(
        0.0, abs=1.0e-9
    )

    orbital_direction = np.arange(1, mc.orb_opt.nrot + 1, dtype=float)
    orbital_direction /= np.linalg.norm(orbital_direction)
    combined = mc.compute_ci_response_vector_product(orbital_direction, ci_direction)
    assert combined == pytest.approx(
        mc.compute_ci_orbital_hessian_vector_product(orbital_direction) + product,
        abs=1.0e-11,
    )


def test_sa_casscf_orbital_response_b_vector_lih():
    """Check the target-root orbital b vector and its SA cancellation."""
    symbols = ["Li", "H"]
    coordinates = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 3.0]])
    mc = _sa_casscf(symbols, coordinates)
    layout = mc.get_ci_response_layout()
    nci = layout[-1][-1].stop

    b_vectors = [mc.compute_orbital_response_b_vector(root) for root in range(2)]
    assert np.linalg.norm(b_vectors[0]) > 1.0e-3

    orbital_direction = np.arange(1, mc.orb_opt.nrot + 1, dtype=float)
    orbital_direction /= np.linalg.norm(orbital_direction)
    step = 1.0e-5
    for root, b_vector in enumerate(b_vectors):
        energy_plus = _root_energy_at_orbital_displacement(
            mc, root, step * orbital_direction
        )
        energy_minus = _root_energy_at_orbital_displacement(
            mc, root, -step * orbital_direction
        )
        finite_difference = (energy_plus - energy_minus) / (2.0 * step)
        assert np.dot(b_vector, orbital_direction) == pytest.approx(
            finite_difference, abs=1.0e-9
        )

        half_reference = np.zeros(nci)
        _, state_index, root_in_state, coefficient_slice = layout[root]
        half_reference[coefficient_slice] = (
            0.5 * mc.ci_solver.sub_solvers[state_index].evecs[:, root_in_state]
        )
        assert mc.compute_orbital_ci_hessian_vector_product(
            half_reference
        ) == pytest.approx(b_vector, abs=1.0e-10)

    weighted_average = sum(
        mc.ci_solver.weights_flat[root] * b_vectors[root] for root in range(2)
    )
    assert weighted_average == pytest.approx(0.0, abs=1.0e-8)

    with pytest.raises(ValueError, match="target response root"):
        mc.compute_orbital_response_b_vector(2)


def test_sa_casscf_ci_response_b_vector_lih():
    """Check the raw target CI gradient and its response-space projection."""
    symbols = ["Li", "H"]
    coordinates = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 3.0]])
    mc = _sa_casscf(symbols, coordinates)
    layout = mc.get_ci_response_layout()
    nci = layout[-1][-1].stop

    hamiltonians = {}
    for target_root in range(2):
        raw_b = mc._compute_raw_ci_response_b_vector(target_root, layout)
        expected_raw = np.zeros(nci)
        _, state_index, root_in_state, coefficient_slice = layout[target_root]
        sub_solver = mc.ci_solver.sub_solvers[state_index]
        if state_index not in hamiltonians:
            hamiltonians[state_index] = sub_solver.ci_sigma_builder.form_H_csf(
                sub_solver.dets, sub_solver.spin_adapter
            )
        reference = sub_solver.evecs[:, root_in_state]
        expected_raw[coefficient_slice] = 2.0 * hamiltonians[state_index] @ reference

        assert np.linalg.norm(raw_b) > 1.0
        assert raw_b == pytest.approx(expected_raw, abs=1.0e-11)
        assert raw_b[coefficient_slice] == pytest.approx(
            2.0 * mc.E_ci[target_root] * reference, abs=1.0e-9
        )

        expected_projected = expected_raw.copy()
        for _, block_state, _, block_slice in layout:
            solved_roots = mc.ci_solver.sub_solvers[block_state].evecs
            block = expected_projected[block_slice]
            expected_projected[block_slice] = block - solved_roots @ (
                solved_roots.T @ block
            )

        b_vector = mc.compute_ci_response_b_vector(target_root)
        assert b_vector == pytest.approx(expected_projected, abs=1.0e-11)
        assert b_vector == pytest.approx(0.0, abs=1.0e-9)

    trial = np.arange(1, nci + 1, dtype=float)
    projected = mc.project_ci_response_vector(trial)
    assert mc.project_ci_response_vector(projected) == pytest.approx(
        projected, abs=1.0e-11
    )
    for _, state_index, _, coefficient_slice in layout:
        solved_roots = mc.ci_solver.sub_solvers[state_index].evecs
        assert solved_roots.T @ projected[coefficient_slice] == pytest.approx(
            0.0, abs=1.0e-11
        )

    with pytest.raises(TypeError, match="target response root"):
        mc.compute_ci_response_b_vector(0.0)


def test_sa_casscf_solve_orbital_response_lih():
    """Solve the projected coupled response equations for both LiH roots."""
    symbols = ["Li", "H"]
    coordinates = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 3.0]])
    mc = _sa_casscf(symbols, coordinates)
    layout = mc.get_ci_response_layout()
    nrot = mc.orb_opt.nrot
    nci = layout[-1][-1].stop

    orbital_work, density_work, hamiltonian_work = (
        mc.orb_opt._build_coupled_response_intermediates()
    )
    B_ga = orbital_work[2]
    assert B_ga is density_work[1]
    assert B_ga is hamiltonian_work[2]
    assert B_ga.shape == (
        mc.system.naux,
        mc.mo_space.nmo,
        mc.mo_space.nactv,
    )

    ci_projector = np.zeros((nci, nci))
    for _, state_index, _, coefficient_slice in layout:
        solved_roots = mc.ci_solver.sub_solvers[state_index].evecs
        block_projector = np.eye(solved_roots.shape[0]) - (
            solved_roots @ solved_roots.T
        )
        ci_projector[coefficient_slice, coefficient_slice] = block_projector
    ci_complement = np.eye(nci) - ci_projector

    orbital_orbital = mc.orb_opt.compute_orbital_hessian()
    orbital_ci = mc.compute_orbital_ci_hessian()
    ci_orbital = mc.compute_ci_orbital_hessian()
    ci_ci = mc.compute_ci_ci_hessian()
    dense_operator = np.block(
        [
            [orbital_orbital, orbital_ci @ ci_projector],
            [
                ci_projector @ ci_orbital,
                ci_projector @ ci_ci @ ci_projector + ci_complement,
            ],
        ]
    )

    trial = np.arange(1, nrot + nci + 1, dtype=float)
    orbital_product, ci_product = mc.compute_projected_response_vector_product(
        trial[:nrot], trial[nrot:]
    )
    assert np.concatenate((orbital_product, ci_product)) == pytest.approx(
        dense_operator @ trial, abs=1.0e-9
    )

    solutions = []
    for root in range(2):
        orbital_b = mc.compute_orbital_response_b_vector(root)
        ci_b = mc.compute_ci_response_b_vector(root)
        rhs = -np.concatenate((orbital_b, ci_b))
        dense_solution = np.linalg.solve(dense_operator, rhs)

        orbital_response, ci_response = mc.solve_state_specific_response(
            root, r_tol=1.0e-11
        )
        solution = np.concatenate((orbital_response, ci_response))
        assert solution == pytest.approx(dense_solution, abs=1.0e-8)
        assert dense_operator @ solution == pytest.approx(rhs, abs=1.0e-9)
        assert ci_projector @ ci_response == pytest.approx(ci_response, abs=1.0e-10)
        solutions.append(solution)

    assert solutions[0] == pytest.approx(-solutions[1], abs=1.0e-7)
    assert mc.solve_orbital_response_vector(0, r_tol=1.0e-11) == pytest.approx(
        solutions[0][:nrot], abs=1.0e-9
    )


def test_sa_casscf_response_omega_lih():
    """Build the relaxed overlap multiplier for both LiH target roots."""
    symbols = ["Li", "H"]
    coordinates = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 3.0]])
    mc = _sa_casscf(symbols, coordinates)
    layout = mc.get_ci_response_layout()
    orbital_optimizer = mc.orb_opt

    density_intermediates = orbital_optimizer._build_ci_orbital_response_intermediates()
    average_A = orbital_optimizer._build_orbital_lagrangian_from_rdms(
        1.0,
        mc.make_average_1rdm(),
        mc.make_average_2rdm(),
        density_intermediates,
    )
    orbital_intermediates = orbital_optimizer._build_orbital_response_intermediates()

    omegas = []
    for root in range(2):
        orbital_response, ci_response = mc.solve_state_specific_response(
            root, r_tol=1.0e-11
        )
        omega = mc.compute_omega(root, orbital_response, ci_response)
        assert omega.shape == (mc.mo_space.nmo, mc.mo_space.nmo)
        assert omega == pytest.approx(omega.T, abs=1.0e-13)
        assert mc.compute_omega(root, r_tol=1.0e-11) == pytest.approx(
            omega, abs=1.0e-10
        )

        target_A = orbital_optimizer._build_orbital_lagrangian_from_rdms(
            1.0,
            mc.make_sf_1rdm(root),
            mc.make_sf_2rdm(root),
            density_intermediates,
        )
        ci_A = orbital_optimizer._build_orbital_lagrangian_from_rdms(
            *mc._compute_ci_response_rdms(ci_response, layout),
            density_intermediates,
        )
        directional_A = orbital_optimizer._compute_orbital_lagrangian_response(
            orbital_response, orbital_intermediates
        )
        Z = orbital_optimizer._vec_to_mat(orbital_response)
        orbital_A = directional_A + Z @ average_A - average_A @ Z
        Omega = target_A + ci_A + orbital_A

        assert omega == pytest.approx(0.5 * (Omega + Omega.T), abs=1.0e-11)
        stationarity = orbital_optimizer._mat_to_vec(2.0 * (Omega - Omega.T))
        assert stationarity == pytest.approx(np.zeros_like(stationarity), abs=1.0e-9)
        assert omega != pytest.approx(0.5 * (target_A + target_A.T), abs=1.0e-4)

        if root == 0:
            # The symmetric commutator contribution cannot be inferred from
            # the orbital Hessian action. Check it against a full coefficient
            # derivative of z.T @ g for one sizeable LiH matrix element.
            def orbital_multiplier_value(C):
                trial = OrbOptimizer(
                    C,
                    (
                        orbital_optimizer.core,
                        orbital_optimizer.actv,
                        orbital_optimizer.virt,
                    ),
                    orbital_optimizer.fock_builder,
                    orbital_optimizer.hcore,
                    orbital_optimizer.e_nuc,
                    orbital_optimizer.nrr.copy(),
                    compute_active_hessian=orbital_optimizer.compute_active_hessian,
                )
                trial.g1 = orbital_optimizer.g1.copy()
                trial.g2 = orbital_optimizer.g2.copy()
                trial._compute_Fcore()
                trial.get_eri_gaaa()
                gradient = trial._compute_orbgrad()
                return orbital_response @ trial._mat_to_vec(gradient)

            def coefficient_derivative(p, q):
                step = 1.0e-5
                C_plus = orbital_optimizer.C.copy()
                C_minus = orbital_optimizer.C.copy()
                C_plus[:, q] += step * orbital_optimizer.C[:, p]
                C_minus[:, q] -= step * orbital_optimizer.C[:, p]
                return (
                    orbital_multiplier_value(C_plus) - orbital_multiplier_value(C_minus)
                ) / (2.0 * step)

            p, q = 5, 1
            numerical_symmetric = 0.25 * (
                coefficient_derivative(p, q) + coefficient_derivative(q, p)
            )
            assert 0.5 * (orbital_A[p, q] + orbital_A[q, p]) == pytest.approx(
                numerical_symmetric, abs=1.0e-9
            )
            assert 0.5 * (directional_A[p, q] + directional_A[q, p]) != pytest.approx(
                numerical_symmetric, abs=1.0e-4
            )
        omegas.append(omega)

    average_omega = sum(
        weight * omega
        for weight, omega in zip(mc.ci_solver.weights_flat, omegas, strict=True)
    )
    assert average_omega == pytest.approx(
        orbital_optimizer.compute_orbital_lagrangian(), abs=1.0e-9
    )
