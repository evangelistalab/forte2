"""Projector-penalty derivatives and their integration into CASSCF."""

import numpy as np
import pytest
from scipy.linalg import block_diag, expm

from forte2 import (
    AVAS,
    CISolver,
    MCOptimizer,
    RHF,
    RelCISolver,
    SpinorUpcaster,
    State,
    System,
)
from forte2.helpers.comparisons import approx
from forte2.mcopt.orbital_optimizer import OrbOptimizer, RelOrbOptimizer


def _optimizer(complex_orbitals=False, gas=False, lambda_penalty=0.7):
    """Use the production lower-triangular core/active/virtual mask."""
    nmo = 6
    nrr = np.zeros((nmo, nmo), dtype=bool)
    nrr[2:4, :2] = True
    nrr[4:, :4] = True
    if gas:
        nrr[3, 2] = True
    cls = RelOrbOptimizer if complex_orbitals else OrbOptimizer
    return cls(
        C=np.eye(nmo, dtype=complex if complex_orbitals else float),
        extents=[slice(0, 2), [2, 3] if gas else slice(2, 4), slice(4, 6)],
        fock_builder=None,
        hcore=None,
        e_nuc=0.0,
        nrr=nrr,
        compute_active_hessian=gas,
        lambda_penalty=lambda_penalty,
    )


def _move_away_from_reference(opt):
    """Several noncommuting increments expose a gradient in the wrong frame."""
    rng = np.random.default_rng(481)
    x = opt.R.copy()
    for _ in range(3):
        step = 0.3 * rng.normal(size=opt.nrot)
        if np.iscomplexobj(x):
            step = step + 0.3j * rng.normal(size=opt.nrot)
        x = x + step
        opt._update_orbitals(x)


def _restore_rotation(opt, U, R):
    opt.U = U.copy()
    opt.R = R.copy()
    opt.C = opt.C0 @ opt.U
    opt.Cgen = opt.C
    opt.Ccore = opt.C[:, opt.core]
    opt.Cact = opt.C[:, opt.actv]


@pytest.mark.parametrize("complex_orbitals", [False, True])
@pytest.mark.parametrize("gas", [False, True])
def test_penalty_current_frame_derivatives(complex_orbitals, gas):
    opt = _optimizer(complex_orbitals=complex_orbitals, gas=gas)
    _move_away_from_reference(opt)
    U0, R0 = opt.U.copy(), opt.R.copy()
    gradient = opt._penalty_gradient_matrix()[opt.nrr]
    hessian = opt._penalty_hessian_diagonal()[opt.nrr]
    energy0 = opt._penalty_energy()

    # Principal angles give an independent check of the penalty energy.
    overlap = opt.U[2:4, 2:4]
    singular_values = np.linalg.svd(overlap, compute_uv=False)
    expected = 2 * opt.lambda_penalty * np.sum(1 - singular_values**2)
    assert energy0 == pytest.approx(expected, abs=1e-13)

    for coordinate in range(opt.nrot):
        for phase in ([1.0, 1.0j] if complex_orbitals else [1.0]):
            direction = np.zeros_like(opt.R)
            direction[coordinate] = phase

            def displaced_energy(step):
                # Differentiate U0 exp(step*K), not exp(R0 + step*K).
                _restore_rotation(opt, U0, R0)
                opt._update_orbitals(R0 + step * direction)
                return opt._penalty_energy()

            h = 1e-6
            numerical_gradient = (displaced_energy(h) - displaced_energy(-h)) / (
                2 * h
            )
            component = (
                gradient[coordinate].real
                if phase == 1.0
                else gradient[coordinate].imag
            )
            assert numerical_gradient == pytest.approx(component, abs=2e-9)

            h = 2e-4
            numerical_curvature = (
                displaced_energy(h) + displaced_energy(-h) - 2 * energy0
            ) / h**2
            assert numerical_curvature == pytest.approx(
                hessian[coordinate].real, abs=3e-7
            )

    _restore_rotation(opt, U0, R0)


@pytest.mark.parametrize("complex_orbitals", [False, True])
def test_penalty_alignment_and_internal_rotation_invariance(complex_orbitals):
    opt = _optimizer(complex_orbitals=complex_orbitals, gas=True)
    assert opt._penalty_energy() == 0.0
    np.testing.assert_allclose(opt._penalty_gradient_matrix(), 0.0, atol=1e-14)

    curvature = opt._penalty_hessian_diagonal()
    np.testing.assert_allclose(curvature[2:4, :2], 4 * opt.lambda_penalty)
    np.testing.assert_allclose(curvature[4:, 2:4], 4 * opt.lambda_penalty)
    np.testing.assert_allclose(curvature[4:, :2], 0.0, atol=1e-14)
    assert curvature[3, 2] == 0.0  # Inter-GAS rotation preserves their union.

    _move_away_from_reference(opt)
    energy0 = opt._penalty_energy()
    deviation0 = opt.active_space_deviation()
    projector0 = opt._build_active_projector()
    rng = np.random.default_rng(731)
    rotations = []
    for _ in range(3):
        matrix = rng.normal(size=(2, 2))
        if complex_orbitals:
            matrix = matrix + 1j * rng.normal(size=(2, 2))
        rotations.append(expm(matrix - matrix.conj().T))
    opt.U = opt.U @ block_diag(*rotations)

    np.testing.assert_allclose(opt._build_active_projector(), projector0, atol=1e-14)
    assert opt._penalty_energy() == pytest.approx(energy0, abs=1e-13)
    assert opt.active_space_deviation() == pytest.approx(deviation0, abs=1e-13)
    assert opt._penalty_gradient_matrix()[3, 2] == 0.0
    assert opt._penalty_hessian_diagonal()[3, 2] == 0.0


@pytest.mark.parametrize("complex_orbitals", [False, True])
def test_zero_penalty_at_displaced_orbitals(complex_orbitals):
    opt = _optimizer(complex_orbitals=complex_orbitals, lambda_penalty=0.0)
    _move_away_from_reference(opt)
    assert opt.active_space_deviation() > 0.0
    assert opt._penalty_energy() == 0.0
    np.testing.assert_array_equal(opt._penalty_gradient_matrix(), 0.0)
    np.testing.assert_array_equal(opt._penalty_hessian_diagonal(), 0.0)


@pytest.fixture(scope="module")
def cyclopropene_calculations():
    """Matched starting orbitals for the original small branch benchmark."""
    system = System(
        xyz="""
        H   0.912650   0.000000   1.457504
        H  -0.912650   0.000000   1.457504
        H   0.000000  -1.585659  -1.038624
        H   0.000000   1.585659  -1.038624
        C   0.000000   0.000000   0.859492
        C   0.000000  -0.651229  -0.499559
        C   0.000000   0.651229  -0.499559
        """,
        basis_set="sto-3g",
        auxiliary_basis_set="def2-universal-JKFIT",
    )
    rhf = RHF(charge=0, e_tol=1e-6)(system)
    avas = AVAS(
        subspace=["C(2p)"],
        subspace_pi_planes=[["C1-3"]],
        selection_method="total",
        num_active=3,
    )(rhf)
    avas.run()
    starting_orbitals = avas.mos.C[0].copy()
    results = {}
    for name, options in (
        ("default", {}),
        ("zero", {"lambda_penalty": 0.0}),
        ("penalty", {"lambda_penalty": 0.5}),
    ):
        ci = CISolver(State(nel=rhf.nel, multiplicity=1, ms=0.0))
        mc = MCOptimizer(ci, **options)(avas)
        mc.run()
        assert mc.converged
        np.testing.assert_array_equal(avas.mos.C[0], starting_orbitals)
        results[name] = mc
    return results


def test_casscf_zero_penalty_regression(cyclopropene_calculations):
    default = cyclopropene_calculations["default"]
    zero = cyclopropene_calculations["zero"]
    assert default.E == approx(-114.440831983407)
    assert zero.E == pytest.approx(default.E, abs=1e-12)
    assert zero.iter == default.iter
    for mc in (default, zero):
        assert mc.E_penalty == 0.0
        assert mc.E_objective == pytest.approx(mc.E, abs=1e-12)


def test_casscf_penalty_energy_and_subspace_response(cyclopropene_calculations):
    zero = cyclopropene_calculations["zero"]
    penalized = cyclopropene_calculations["penalty"]
    assert 0.0 < penalized.delta_act < zero.delta_act
    assert penalized.E >= zero.E - 1e-9
    assert penalized.E_penalty == pytest.approx(
        penalized.lambda_penalty * penalized.delta_act**2, abs=1e-12
    )
    assert penalized.E_objective == pytest.approx(
        penalized.E + penalized.E_penalty, abs=1e-12
    )
    assert penalized.E == pytest.approx(penalized.ci_solver.compute_average_energy())


def test_total_orbital_objective_gradient_fd(cyclopropene_calculations):
    """Check the combined physical/penalty objective at fixed CI coefficients."""
    opt = cyclopropene_calculations["penalty"].orb_opt
    original_U, original_R = opt.U.copy(), opt.R.copy()
    rng = np.random.default_rng(319)
    try:
        # A displaced point prevents cancellation of two nonzero gradients at
        # the converged penalized solution from making this test too easy.
        x = original_R + 0.03 * rng.normal(size=opt.nrot)
        opt.evaluate(x)
        U0, R0 = opt.U.copy(), opt.R.copy()
        gradient = opt.gradient(x).copy()
        assert np.linalg.norm(gradient) > 1e-4
        for _ in range(3):
            direction = rng.normal(size=opt.nrot)
            direction /= np.linalg.norm(direction)
            energies = []
            h = 1e-5
            for sign in (1, -1):
                _restore_rotation(opt, U0, R0)
                energies.append(opt.evaluate(R0 + sign * h * direction))
            numerical = (energies[0] - energies[1]) / (2 * h)
            assert numerical == pytest.approx(np.dot(gradient, direction), abs=2e-7)
    finally:
        _restore_rotation(opt, original_U, original_R)
        opt._compute_Fcore()
        opt.get_eri_gaaa()
        opt.gradient(original_R)


@pytest.mark.parametrize("lambda_penalty", [-1e-8, np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("complex_orbitals", [False, True])
def test_invalid_penalty_parameter(lambda_penalty, complex_orbitals):
    with pytest.raises(ValueError, match="finite and non-negative"):
        _optimizer(
            complex_orbitals=complex_orbitals,
            lambda_penalty=lambda_penalty,
        )

    solver = (
        RelCISolver(nel=2, active_orbitals=4)
        if complex_orbitals
        else CISolver(State(nel=2, multiplicity=1, ms=0.0), active_orbitals=2)
    )
    with pytest.raises(ValueError, match="finite and non-negative"):
        MCOptimizer(solver, lambda_penalty=lambda_penalty)


def test_rel_total_orbital_objective_gradient_fd():
    """Exercise the complex physical gradient and penalty through shared hooks."""
    system = System(
        xyz="Li 0 0 0\nH 0 0 3.0",
        unit="bohr",
        basis_set="sto-3g",
        auxiliary_basis_set="def2-universal-JKFIT",
    )
    rhf = RHF(charge=0)(system)
    parent = SpinorUpcaster(apply_random_phase=True, rng=11)(rhf)
    solver = RelCISolver(
        nel=4,
        core_orbitals=[0, 1],
        active_orbitals=[2, 3, 4, 5],
    )
    mc = MCOptimizer(solver, lambda_penalty=0.5, final_orbitals="original")(parent)
    mc.run()
    assert mc.converged
    assert isinstance(mc.orb_opt, RelOrbOptimizer)
    assert mc.E_penalty > 0.0
    assert mc.E_penalty == pytest.approx(mc.lambda_penalty * mc.delta_act**2)
    assert mc.E_objective == pytest.approx(mc.E + mc.E_penalty)

    opt = mc.orb_opt
    rng = np.random.default_rng(430)
    displacement = 0.03 * (
        rng.normal(size=opt.nrot) + 1j * rng.normal(size=opt.nrot)
    )
    x = opt.R + displacement
    opt.evaluate(x)
    U0, R0 = opt.U.copy(), opt.R.copy()
    gradient = opt.gradient(x).copy()
    assert np.linalg.norm(gradient.real) > 1e-4
    assert np.linalg.norm(gradient.imag) > 1e-4
    assert np.linalg.norm(opt._penalty_gradient_matrix()[opt.nrr]) > 1e-3

    # Pure real and pure imaginary directions independently check both parts
    # of the complex gradient, including its conjugation convention.
    for phase in (1.0, 1.0j):
        direction = phase * rng.normal(size=opt.nrot)
        direction /= np.linalg.norm(direction)
        energies = []
        h = 1e-5
        for sign in (1, -1):
            _restore_rotation(opt, U0, R0)
            energies.append(opt.evaluate(R0 + sign * h * direction))
        numerical = ((energies[0] - energies[1]) / (2 * h)).real
        analytic = np.vdot(gradient, direction).real
        assert numerical == pytest.approx(analytic, abs=2e-7)
