import numpy as np
import pytest

from forte2 import CISolver, MCOptimizer, RHF, State, System, X2CParams
from forte2.gradients import finite_difference
from forte2.mcopt.mc_optimizer_grad import (
    _backtransform_df_hole_response,
    _build_sa_casscf_relaxed_one_body_density,
    _transform_df_hole_response,
)
from forte2.mcopt.mc_optimizer_response import (
    _build_ci_reference_det_vectors,
    _build_coupled_response_intermediates,
    _build_orbital_lagrangian_from_rdms,
    _compute_orbital_lagrangian_response,
    _compute_ci_response_rdms,
    _project_ci_response_vector,
    compute_omega,
    compute_projected_response_vector_product,
    get_ci_response_layout,
    solve_state_specific_response,
)
from forte2.mcopt.orbital_optimizer import OrbOptimizer
from tests.gradient_test_utils import xyz_string


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


def _sa_casscf_mixed_spin(symbols, coordinates):
    """Run an equal-weight singlet/triplet LiH SA-CASSCF calculation."""
    system = System(
        xyz=xyz_string(symbols, coordinates),
        basis_set="sto-3g",
        auxiliary_basis_set="def2-universal-JKFIT",
        unit="bohr",
    )
    rhf = RHF(charge=0, e_tol=1.0e-12, d_tol=1.0e-10, maxiter=100)(system)
    ci_solver = CISolver(
        [
            State(system=system, multiplicity=1, ms=0.0),
            State(system=system, multiplicity=3, ms=0.0),
        ],
        core_orbitals=[0],
        active_orbitals=[1, 2],
        nroots=[1, 1],
        weights=[[0.5], [0.5]],
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


def _sa_gasscf_h2_ccpvdz(symbols, coordinates, spin_free_x2c=False):
    """Run a two-root H2 SA-GASSCF calculation with two GAS spaces."""
    system_options = {}
    if spin_free_x2c:
        system_options = {
            "x2c": X2CParams(x2c_type="sf", x2c_model="1e"),
            "minao_basis_set": None,
        }
    system = System(
        xyz=xyz_string(symbols, coordinates),
        basis_set="cc-pVDZ",
        auxiliary_basis_set="cc-pVTZ-JKFIT",
        unit="bohr",
        **system_options,
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


def test_sa_casscf_relaxed_one_body_density_kernel():
    """Check the target density plus the average-density orbital response."""
    rng = np.random.default_rng(7)
    Ccore = rng.standard_normal((6, 2))
    Cact = rng.standard_normal((6, 3))
    Ccore_response = rng.standard_normal(Ccore.shape)
    Cact_response = rng.standard_normal(Cact.shape)
    base_g1 = rng.standard_normal((3, 3))
    average_g1 = rng.standard_normal((3, 3))

    actual = _build_sa_casscf_relaxed_one_body_density(
        Ccore,
        Cact,
        Ccore_response,
        Cact_response,
        base_g1,
        average_g1,
    )
    base = 2.0 * Ccore @ Ccore.T + Cact @ base_g1 @ Cact.T
    response = finite_difference(
        lambda scale: 2.0
        * (Ccore + scale * Ccore_response)
        @ (Ccore + scale * Ccore_response).T
        + (Cact + scale * Cact_response)
        @ average_g1
        @ (Cact + scale * Cact_response).T,
        0.0,
        step=1.0e-5,
        npoints=2,
    )
    assert actual == pytest.approx(base + response, abs=1.0e-9)


def test_sa_casscf_df_hole_response_kernel():
    """Check the directional hole-space transformation of the AO DF tensor."""
    rng = np.random.default_rng(11)
    Ch = rng.standard_normal((5, 3))
    Ch_response = rng.standard_normal(Ch.shape)
    Z_ao = rng.standard_normal((4, 5, 5))
    Z_ao += Z_ao.transpose(0, 2, 1)

    Z_h, Z_h_response = _transform_df_hole_response(Z_ao, Ch, Ch_response)
    expected = np.einsum("mi,Pmn,nj->Pij", Ch, Z_ao, Ch, optimize=True)
    expected_response = finite_difference(
        lambda scale: np.einsum(
            "mi,Pmn,nj->Pij",
            Ch + scale * Ch_response,
            Z_ao,
            Ch + scale * Ch_response,
            optimize=True,
        ),
        0.0,
        step=1.0e-5,
        npoints=2,
    )
    assert Z_h == pytest.approx(expected, abs=1.0e-12)
    assert Z_h_response == pytest.approx(expected_response, abs=1.0e-9)


def test_sa_casscf_df_hole_backtransform_kernel():
    """Check all three product-rule terms in the relaxed AO DF weight."""
    rng = np.random.default_rng(13)
    Ch = rng.standard_normal((5, 3))
    Ch_response = rng.standard_normal(Ch.shape)
    W3_h_base = rng.standard_normal((4, 3, 3))
    W3_h_average = rng.standard_normal((4, 3, 3))
    W3_h_average_response = rng.standard_normal((4, 3, 3))

    actual = _backtransform_df_hole_response(
        Ch,
        Ch_response,
        W3_h_base,
        W3_h_average,
        W3_h_average_response,
    )
    base = np.einsum("mi,Pij,nj->Pmn", Ch, W3_h_base, Ch, optimize=True)
    response = finite_difference(
        lambda scale: np.einsum(
            "mi,Pij,nj->Pmn",
            Ch + scale * Ch_response,
            W3_h_average + scale * W3_h_average_response,
            Ch + scale * Ch_response,
            optimize=True,
        ),
        0.0,
        step=1.0e-5,
        npoints=2,
    )
    assert actual == pytest.approx(base + response, abs=1.0e-9)


def test_sa_casscf_gradient_lih_finite_difference():
    """Validate both relaxed root gradients against four-point differences."""
    symbols = ["Li", "H"]
    coordinates = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 3.0]])

    mc = _sa_casscf(symbols, coordinates)
    gradients = np.array([mc.gradient(root=root) for root in range(2)])
    numerical = finite_difference(
        lambda displaced: _sa_casscf(symbols, displaced).E_ci,
        coordinates,
        step=1.0e-3,
        npoints=4,
        components=[(1, 2)],
    )[0]

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


def test_sa_casscf_gradient_mixed_spin_lih_finite_difference():
    """Validate roots whose singlet and triplet CI blocks have different sizes."""
    symbols = ["Li", "H"]
    coordinates = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 3.0]])

    mc = _sa_casscf_mixed_spin(symbols, coordinates)
    layout = get_ci_response_layout(mc)
    block_sizes = [
        coefficient_slice.stop - coefficient_slice.start
        for *_, coefficient_slice in layout
    ]
    assert block_sizes == [3, 1]
    assert [solver.basis_size for solver in mc.ci_solver.sub_solvers] == [3, 1]

    gradients = np.array([mc.gradient(root=root) for root in range(2)])
    numerical = finite_difference(
        lambda displaced: _sa_casscf_mixed_spin(symbols, displaced).E_ci,
        coordinates,
        step=1.0e-3,
        npoints=4,
        components=[(1, 2)],
    )[0]

    assert gradients[:, 1, 2] == pytest.approx(numerical, abs=1.0e-7)
    assert numerical == pytest.approx(
        np.array([0.006221111355472, -0.029709654781810]),
        abs=5.0e-8,
    )
    assert gradients.sum(axis=1) == pytest.approx(np.zeros((2, 3)), abs=1.0e-10)


def test_sa_casscf_gradient_c2_ccpvdz_finite_difference():
    """Validate a non-minimal-basis root gradient on a compact C2 problem."""
    symbols = ["C", "C"]
    coordinates = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.3]])

    mc = _sa_casscf_c2_ccpvdz(symbols, coordinates)
    gradient = mc.gradient(root=0)
    numerical = finite_difference(
        lambda displaced: _sa_casscf_c2_ccpvdz(symbols, displaced).E_ci,
        coordinates,
        step=1.0e-3,
        npoints=4,
        components=[(1, 2)],
    )[0, 0]

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
    numerical = finite_difference(
        lambda displaced: _sa_gasscf_h2_ccpvdz(symbols, displaced).E_ci,
        coordinates,
        step=1.0e-3,
        npoints=4,
        components=[(1, 2)],
    )[0]

    state = mc.ci_solver.sub_solvers[0].state
    assert mc.mo_space.ngas == 2
    assert mc.mo_space.active_orbitals == [[0], [1, 2]]
    assert state.gas_min == [1]
    assert state.gas_max == [1]
    assert gradients[:, 1, 2] == pytest.approx(numerical, abs=1.0e-7)
    assert numerical == pytest.approx(np.array([-0.05776565, 0.11036714]), abs=1.0e-8)
    assert gradients.sum(axis=1) == pytest.approx(np.zeros((2, 3)), abs=1.0e-10)


def test_sf_x2c_sa_gasscf_gradient_h2_ccpvdz_finite_difference():
    """Validate a target-root spin-free X2C SA-GASSCF gradient."""
    symbols = ["H", "H"]
    coordinates = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.7]])

    mc = _sa_gasscf_h2_ccpvdz(symbols, coordinates, spin_free_x2c=True)
    gradient = mc.gradient(root=1)
    numerical = finite_difference(
        lambda displaced: _sa_gasscf_h2_ccpvdz(
            symbols, displaced, spin_free_x2c=True
        ).E_ci,
        coordinates,
        step=1.0e-3,
        npoints=4,
        components=[(1, 2)],
    )[0, 1]

    assert mc.system.x2c_type == "sf"
    assert mc.mo_space.ngas == 2
    assert gradient[1, 2] == pytest.approx(numerical, abs=1.0e-7)
    assert gradient.sum(axis=0) == pytest.approx(np.zeros(3), abs=1.0e-10)


def test_sa_casscf_solve_orbital_response_lih():
    """Solve the projected coupled response equations for both LiH roots."""
    symbols = ["Li", "H"]
    coordinates = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 3.0]])
    mc = _sa_casscf(symbols, coordinates)
    layout = get_ci_response_layout(mc)
    nrot = mc.orb_opt.nrot
    nci = layout[-1][-1].stop

    orbital_work, density_work, hamiltonian_work = (
        _build_coupled_response_intermediates(mc.orb_opt)
    )
    B_ga = orbital_work[2]
    assert B_ga is density_work[1]
    assert B_ga is hamiltonian_work[2]
    assert B_ga.shape == (
        mc.system.naux,
        mc.mo_space.nmo,
        mc.mo_space.nactv,
    )

    trial = np.arange(1, nrot + nci + 1, dtype=float)
    orbital_product, ci_product = compute_projected_response_vector_product(
        mc, trial[:nrot], trial[nrot:]
    )
    projected_ci = _project_ci_response_vector(mc, trial[nrot:], layout)
    assert orbital_product.shape == (nrot,)
    assert ci_product.shape == (nci,)
    assert ci_product - _project_ci_response_vector(
        mc, ci_product, layout
    ) == pytest.approx(trial[nrot:] - projected_ci, abs=1.0e-10)

    solutions = []
    for root in range(2):
        orbital_response, ci_response = solve_state_specific_response(
            mc, root, r_tol=1.0e-11
        )
        solution = np.concatenate((orbital_response, ci_response))
        assert _project_ci_response_vector(mc, ci_response, layout) == pytest.approx(
            ci_response, abs=1.0e-10
        )
        solutions.append(solution)

    assert solutions[0] == pytest.approx(-solutions[1], abs=1.0e-7)


def test_sa_casscf_cached_ci_reference_dets_lih(monkeypatch):
    """Reuse determinant-form reference roots in transition RDMs."""
    mc = _sa_casscf(["Li", "H"], np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 3.0]]))
    layout = get_ci_response_layout(mc)
    trial = np.arange(1, layout[-1][-1].stop + 1, dtype=float)
    expected = _compute_ci_response_rdms(mc, trial, layout)
    reference_dets = _build_ci_reference_det_vectors(mc, layout)

    sub_solver = mc.ci_solver.sub_solvers[0]
    original_transform = sub_solver.csf_C_to_det_C
    calls = 0

    def count_transform(vector):
        nonlocal calls
        calls += 1
        return original_transform(vector)

    monkeypatch.setattr(sub_solver, "csf_C_to_det_C", count_transform)
    result = _compute_ci_response_rdms(mc, trial, layout, reference_dets)
    assert calls == len(layout)
    for actual, reference in zip(result, expected):
        assert actual == pytest.approx(reference, abs=1.0e-12)


def test_sa_casscf_response_omega_lih():
    """Build the relaxed overlap multiplier for both LiH target roots."""
    symbols = ["Li", "H"]
    coordinates = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 3.0]])
    mc = _sa_casscf(symbols, coordinates)
    layout = get_ci_response_layout(mc)
    orbital_optimizer = mc.orb_opt

    orbital_intermediates, density_intermediates, _ = (
        _build_coupled_response_intermediates(orbital_optimizer)
    )
    average_A = _build_orbital_lagrangian_from_rdms(
        orbital_optimizer,
        1.0,
        mc.make_average_1rdm(),
        mc.make_average_2rdm(),
        density_intermediates,
    )
    omegas = []
    for root in range(2):
        orbital_response, ci_response = solve_state_specific_response(
            mc, root, r_tol=1.0e-11
        )
        omega = compute_omega(mc, root, orbital_response, ci_response)
        assert omega.shape == (mc.mo_space.nmo, mc.mo_space.nmo)
        assert omega == pytest.approx(omega.T, abs=1.0e-13)

        target_A = _build_orbital_lagrangian_from_rdms(
            orbital_optimizer,
            1.0,
            mc.make_sf_1rdm(root),
            mc.make_sf_2rdm(root),
            density_intermediates,
        )
        ci_A = _build_orbital_lagrangian_from_rdms(
            orbital_optimizer,
            *_compute_ci_response_rdms(mc, ci_response, layout),
            density_intermediates,
        )
        directional_A = _compute_orbital_lagrangian_response(
            orbital_optimizer, orbital_response, orbital_intermediates
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
