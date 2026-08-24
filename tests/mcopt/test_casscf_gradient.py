import numpy as np
import pytest

from forte2 import System, RHF, MCOptimizer, State, CISolver, X2CParams
from forte2.gradients import FDGradient
from forte2.integrals import LIBCINT_AVAILABLE
from tests.gradient_test_utils import xyz_string


def _system(symbols, coordinates, **kwargs):
    return System(
        xyz=xyz_string(symbols, coordinates),
        basis_set=kwargs.pop("basis_set", "sto-3g"),
        auxiliary_basis_set=kwargs.pop("auxiliary_basis_set", "def2-universal-JKFIT"),
        unit="bohr",
        **kwargs,
    )


def _casscf(
    symbols,
    coordinates,
    *,
    active_orbitals,
    core_orbitals=None,
    final_orbitals="original",
    maxiter=30,
):
    system = _system(symbols, coordinates)
    rhf = RHF(charge=0, e_tol=1.0e-12, d_tol=1.0e-10, maxiter=100)(system)
    ci_solver = CISolver(
        State(system=system, multiplicity=1, ms=0.0),
        active_orbitals=active_orbitals,
        core_orbitals=[] if core_orbitals is None else core_orbitals,
    )
    mc = MCOptimizer(
        ci_solver,
        e_tol=1.0e-12,
        g_tol=1.0e-9,
        maxiter=maxiter,
        final_orbitals=final_orbitals,
    )(rhf)
    mc.run()
    return mc


def _gasscf_h2(symbols, coordinates):
    system = _system(symbols, coordinates)
    rhf = RHF(charge=0, e_tol=1.0e-12, d_tol=1.0e-10, maxiter=100)(system)
    ci_solver = CISolver(
        State(
            system=system,
            multiplicity=1,
            ms=0.0,
            gas_min=[1],
            gas_max=[1],
        ),
        active_orbitals=[[0], [1]],
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


def _gasscf_n2_three_gas(symbols, coordinates):
    system = _system(symbols, coordinates)
    rhf = RHF(charge=0, e_tol=1.0e-12, d_tol=1.0e-10, maxiter=100)(system)
    ci_solver = CISolver(
        State(
            system=system,
            multiplicity=1,
            ms=0.0,
            gas_min=[2, 2, 0],
            gas_max=[4, 4, 2],
        ),
        core_orbitals=[0, 1, 2, 3],
        active_orbitals=[[4, 9], [5, 6], [7, 8]],
    )
    mc = MCOptimizer(
        ci_solver,
        e_tol=1.0e-10,
        g_tol=1.0e-8,
        maxiter=80,
        final_orbitals="original",
    )(rhf)
    mc.run()
    return mc


def test_casscf_gradient_h2_full_active_finite_difference_and_translation():
    """Validate the all-active state-specific CASSCF gradient by finite differences."""
    symbols = ["H", "H"]
    coordinates = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.7]])

    casscf = _casscf(symbols, coordinates, active_orbitals=2)
    analytical = casscf.gradient()
    fd = FDGradient()(casscf)
    numerical = fd.gradient()

    assert analytical == pytest.approx(numerical, abs=1.0e-7)
    assert analytical.sum(axis=0) == pytest.approx(np.zeros(3), abs=1.0e-10)


def test_casscf_gradient_lih_core_active_selected_finite_difference():
    """Validate a CASSCF gradient component with inactive core and active orbitals."""
    symbols = ["Li", "H"]
    coordinates = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 3.0]])
    kwargs = {"core_orbitals": [0], "active_orbitals": [1, 2]}

    casscf = _casscf(symbols, coordinates, **kwargs)
    analytical = casscf.gradient()
    fd = FDGradient()(casscf)
    numerical = fd.gradient()

    assert analytical == pytest.approx(numerical, abs=1.0e-7)
    assert analytical.sum(axis=0) == pytest.approx(np.zeros(3), abs=1.0e-10)


def test_gasscf_gradient_h2_two_gas_finite_difference_and_translation():
    """Validate a state-specific GASSCF gradient with an explicit two-GAS space.

    The active space is split as ``[[0], [1]]`` and the state requires exactly
    one electron in GAS1.  This exercises the GASSCF RDM path while keeping all
    inter-GAS orbital rotations optimized, which is the stationarity condition
    assumed by the current analytic gradient implementation.
    """
    symbols = ["H", "H"]
    coordinates = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.7]])

    gasscf = _gasscf_h2(symbols, coordinates)
    analytical = gasscf.gradient()
    fd = FDGradient()(gasscf)
    numerical = fd.gradient()

    assert analytical == pytest.approx(numerical, abs=1.0e-7)
    assert analytical.sum(axis=0) == pytest.approx(np.zeros(3), abs=1.0e-10)


def test_gasscf_gradient_n2_three_gas_selected_finite_difference():
    """Validate a three-GAS N2 gradient with nontrivial occupation restrictions.

    This test uses 6 active electrons in 6 active orbitals split into three
    two-orbital GAS spaces: ``[4, 9]`` for sigma orbitals, ``[5, 6]`` for pi
    orbitals, and ``[7, 8]`` for pi* orbitals.  The restrictions
    ``gas_min=[2, 2, 0]`` and ``gas_max=[4, 4, 2]`` allow multiple occupation
    configurations while constraining the electron distribution across all
    three GAS spaces.
    """
    symbols = ["N", "N"]
    coordinates = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]])

    mc = _gasscf_n2_three_gas(symbols, coordinates)
    analytical = mc.gradient()

    assert mc.mo_space.ngas == 3
    assert mc.mo_space.active_orbitals == [[4, 9], [5, 6], [7, 8]]
    assert mc.ci_solver.sub_solvers[0].state.gas_min == [2, 2, 0]
    assert mc.ci_solver.sub_solvers[0].state.gas_max == [4, 4, 2]
    assert len(mc.ci_solver.sub_solvers[0].ci_strings.gas_occupations) > 1

    fd = FDGradient()(mc)
    numerical = fd.gradient()

    assert analytical == pytest.approx(numerical, abs=1.0e-6)
    assert analytical.sum(axis=0) == pytest.approx(np.zeros(3), abs=1.0e-10)


def test_casscf_gradient_auto_runs_and_reuses_executed_object():
    """Ensure MCOptimizer.gradient() runs CASSCF on demand and is repeatable."""
    symbols = ["H", "H"]
    coordinates = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.7]])
    system = _system(symbols, coordinates)
    rhf = RHF(charge=0, e_tol=1.0e-12, d_tol=1.0e-10, maxiter=100)(system)
    ci_solver = CISolver(
        State(system=system, multiplicity=1, ms=0.0),
        active_orbitals=2,
    )
    mc = MCOptimizer(
        ci_solver,
        e_tol=1.0e-12,
        g_tol=1.0e-9,
        maxiter=30,
    )(rhf)

    assert not mc.executed
    gradient1 = mc.gradient()
    energy1 = mc.E

    assert mc.executed
    gradient2 = mc.gradient()
    gradient3 = mc.gradient(root=0)

    assert mc.E == pytest.approx(energy1)
    assert gradient1 == pytest.approx(gradient2, abs=1.0e-12)
    assert gradient1 == pytest.approx(gradient3, abs=1.0e-12)
    assert gradient1.shape == (system.natoms, 3)


def test_casscf_gradient_rejects_unconverged_wavefunction(monkeypatch):
    """Require both orbital and CI stationarity before using the gradient."""
    mc = _casscf(
        ["H", "H"],
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.7]]),
        active_orbitals=2,
    )

    mc.converged = False
    with pytest.raises(RuntimeError, match="converged orbital optimization"):
        mc.gradient()

    mc.converged = True
    monkeypatch.setattr(mc.ci_solver, "get_convergence_status", lambda: [False])
    with pytest.raises(RuntimeError, match="converged CI roots"):
        mc.gradient()


def test_casscf_gradient_reuses_orbital_optimizer_intermediates(monkeypatch):
    """Avoid rebuilding orbital intermediates when the final MO basis is unchanged."""
    mc = _casscf(
        ["H", "H"],
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.7]]),
        active_orbitals=2,
        final_orbitals="original",
    )

    def fail(*args, **kwargs):
        raise AssertionError("CASSCF gradient rebuilt converged orbital intermediates")

    monkeypatch.setattr(type(mc.orb_opt), "__init__", fail)
    monkeypatch.setattr(mc.orb_opt, "_compute_Fcore", fail)
    monkeypatch.setattr(mc.orb_opt, "get_eri_gaaa", fail)

    gradient = mc.gradient()
    assert gradient.shape == (mc.system.natoms, 3)


def test_casscf_gradient_requires_root_for_state_average():
    """Require an explicit absolute root for an SA-CASSCF gradient."""
    system = _system(["H", "H"], np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.7]]))
    rhf = RHF(charge=0)(system)
    ci_solver = CISolver(
        State(system=system, multiplicity=1, ms=0.0),
        active_orbitals=2,
        nroots=2,
    )
    mc = MCOptimizer(ci_solver, final_orbitals="original")(rhf)

    with pytest.raises(ValueError, match="root must be specified"):
        mc.gradient()


def test_sa_gasscf_gradient_rejects_frozen_inter_gas_rotations():
    """Reject SA gradients when inter-GAS rotations were not optimized."""
    system = _system(["H", "H"], np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.7]]))
    rhf = RHF(charge=0)(system)
    ci_solver = CISolver(
        State(
            system=system,
            multiplicity=1,
            ms=0.0,
            gas_min=[1],
            gas_max=[1],
        ),
        active_orbitals=[[0], [1]],
        nroots=2,
        weights=[0.5, 0.5],
    )
    mc = MCOptimizer(
        ci_solver,
        freeze_inter_gas_rots=True,
        final_orbitals="original",
    )(rhf)

    with pytest.raises(NotImplementedError, match="frozen inter-GAS rotations"):
        mc.gradient(root=0)

    assert not mc.executed


def test_casscf_gradient_rejects_frozen_core_orbitals():
    """Reject frozen core orbitals until the CASSCF Z-vector path is added."""
    system = _system(["Li", "H"], np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 3.0]]))
    rhf = RHF(charge=0)(system)
    ci_solver = CISolver(
        State(system=system, multiplicity=1, ms=0.0),
        frozen_core_orbitals=[0],
        active_orbitals=[1, 2],
    )
    mc = MCOptimizer(ci_solver, final_orbitals="original")(rhf)

    with pytest.raises(NotImplementedError, match="frozen core"):
        mc.gradient()


def test_casscf_gradient_rejects_frozen_virtual_orbitals():
    """Reject frozen virtual orbitals until their response terms are added."""
    system = _system(["Li", "H"], np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 3.0]]))
    rhf = RHF(charge=0)(system)
    ci_solver = CISolver(
        State(system=system, multiplicity=1, ms=0.0),
        core_orbitals=[0],
        active_orbitals=[1, 2],
        frozen_virtual_orbitals=[5],
    )
    mc = MCOptimizer(ci_solver, final_orbitals="original")(rhf)

    with pytest.raises(NotImplementedError, match="frozen virtual"):
        mc.gradient()


def test_casscf_gradient_rejects_cholesky_tei():
    """Reject Cholesky ERIs because this gradient path is DF-integral based."""
    system = System(
        xyz="H 0 0 0\nH 0 0 1.7",
        basis_set="sto-3g",
        cholesky_tei=True,
        unit="bohr",
    )
    rhf = RHF(charge=0)(system)
    ci_solver = CISolver(
        State(system=system, multiplicity=1, ms=0.0),
        active_orbitals=2,
    )
    mc = MCOptimizer(ci_solver, final_orbitals="original")(rhf)

    with pytest.raises(NotImplementedError, match="density fitting"):
        mc.gradient()


@pytest.mark.skipif(not LIBCINT_AVAILABLE, reason="Libcint is not available")
def test_casscf_gradient_gaussian_nuclear_charges_finite_difference():
    """Validate the Gaussian nuclear model in the CASSCF gradient."""
    symbols = ["H", "H"]
    coordinates = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.7]])

    system = _system(symbols, coordinates, use_gaussian_charges=True)
    rhf = RHF(charge=0, e_tol=1.0e-12, d_tol=1.0e-10)(system)
    ci_solver = CISolver(
        State(system=system, multiplicity=1, ms=0.0),
        active_orbitals=2,
    )
    mc = MCOptimizer(
        ci_solver,
        e_tol=1.0e-12,
        g_tol=1.0e-9,
        final_orbitals="original",
    )(rhf)

    analytical = mc.gradient()
    fd = FDGradient()(mc)
    numerical = fd.gradient()

    assert analytical == pytest.approx(numerical, abs=1.0e-8)
    assert analytical.sum(axis=0) == pytest.approx(np.zeros(3), abs=1.0e-10)


def test_sf_x2c_casscf_gradient_finite_difference():
    """Validate scalar-X2C CASSCF through the shared X2C hcore derivative."""
    symbols = ["H", "H"]
    coordinates = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.7]])

    system = _system(
        symbols,
        coordinates,
        x2c=X2CParams(x2c_type="sf", x2c_model="1e"),
        minao_basis_set=None,
    )
    rhf = RHF(charge=0, e_tol=1.0e-12, d_tol=1.0e-10)(system)
    ci_solver = CISolver(
        State(system=system, multiplicity=1, ms=0.0),
        active_orbitals=2,
    )
    mc = MCOptimizer(
        ci_solver,
        e_tol=1.0e-12,
        g_tol=1.0e-9,
        final_orbitals="original",
    )(rhf)

    analytical = mc.gradient()
    fd = FDGradient()(mc)
    numerical = fd.gradient()

    assert analytical == pytest.approx(numerical, abs=1.0e-8)
