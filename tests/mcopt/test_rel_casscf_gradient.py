import numpy as np
import pytest

from forte2 import (
    CISolver,
    GHF,
    MCOptimizer,
    RHF,
    RelCISolver,
    RelState,
    SpinorUpcaster,
    State,
    System,
)
from tests.gradient_test_utils import (
    four_point_central_difference_gradient_component,
    xyz_string,
)


def _system(symbols, coordinates, x2c_type=None, snso_type=None):
    return System(
        xyz=xyz_string(symbols, coordinates),
        basis_set="sto-3g",
        auxiliary_basis_set="def2-universal-JKFIT",
        unit="bohr",
        x2c_type=x2c_type,
        snso_type=snso_type,
        minao_basis_set=None,
    )


def _rel_casscf(
    symbols,
    coordinates,
    *,
    nel,
    active_orbitals,
    core_orbitals=None,
    x2c_type=None,
    snso_type=None,
    apply_random_phase=False,
    gas_min=None,
    gas_max=None,
):
    system = _system(
        symbols,
        coordinates,
        x2c_type=x2c_type,
        snso_type=snso_type,
    )
    if x2c_type == "so":
        parent = GHF(charge=0, e_tol=1.0e-10, d_tol=1.0e-6)(system)
    else:
        rhf = RHF(charge=0, e_tol=1.0e-12, d_tol=1.0e-10)(system)
        parent = SpinorUpcaster(
            apply_random_phase=apply_random_phase,
            rng=11,
        )(rhf)

    solver_options = dict(
        core_orbitals=[] if core_orbitals is None else core_orbitals,
        active_orbitals=active_orbitals,
    )
    if gas_min is None and gas_max is None:
        ci_solver = RelCISolver(nel=nel, **solver_options)
    else:
        ci_solver = RelCISolver(
            RelState(
                nel=nel,
                gas_min=[] if gas_min is None else gas_min,
                gas_max=[] if gas_max is None else gas_max,
            ),
            **solver_options,
        )
    return MCOptimizer(
        ci_solver,
        e_tol=1.0e-11,
        g_tol=1.0e-8,
        maxiter=50,
        final_orbitals="original",
    )(parent)


def _rel_casscf_energy(symbols, coordinates, **kwargs):
    return _rel_casscf(symbols, coordinates, **kwargs).run().E.real


def test_rel_casscf_gradient_upcast_limit_and_phase_invariance():
    symbols = ["Li", "H"]
    coordinates = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 3.0]])

    system = _system(symbols, coordinates)
    rhf = RHF(charge=0, e_tol=1.0e-12, d_tol=1.0e-10)(system)
    ci_solver = CISolver(
        State(system=system, multiplicity=1, ms=0.0),
        core_orbitals=[0],
        active_orbitals=[1, 2],
    )
    nonrel = MCOptimizer(
        ci_solver,
        e_tol=1.0e-11,
        g_tol=1.0e-8,
        maxiter=50,
        final_orbitals="original",
    )(rhf)
    reference = nonrel.gradient()

    options = dict(
        nel=4,
        core_orbitals=[0, 1],
        active_orbitals=[2, 3, 4, 5],
    )
    unphased = _rel_casscf(symbols, coordinates, **options).gradient()
    phased = _rel_casscf(
        symbols,
        coordinates,
        apply_random_phase=True,
        **options,
    ).gradient()

    assert unphased == pytest.approx(reference, abs=1.0e-10)
    assert phased == pytest.approx(reference, abs=1.0e-10)


def test_sf_x2c_rel_casscf_gradient_finite_difference_and_translation():
    symbols = ["H", "H"]
    coordinates = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.7]])
    options = dict(
        nel=2,
        active_orbitals=4,
        x2c_type="sf",
        apply_random_phase=True,
    )

    gradient = _rel_casscf(symbols, coordinates, **options).gradient()
    numerical = four_point_central_difference_gradient_component(
        _rel_casscf_energy,
        symbols,
        coordinates,
        1,
        2,
        **options,
    )

    assert gradient[1, 2] == pytest.approx(numerical, abs=1.0e-8)
    assert gradient.sum(axis=0) == pytest.approx(np.zeros(3), abs=1.0e-10)


def test_so_x2c_rel_casscf_gradient_finite_difference_and_translation():
    symbols = ["O", "H"]
    coordinates = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.8]])
    options = dict(
        nel=9,
        core_orbitals=list(range(8)),
        active_orbitals=list(range(8, 12)),
        x2c_type="so",
        snso_type="row-dependent",
    )

    gradient = _rel_casscf(symbols, coordinates, **options).gradient()
    numerical = four_point_central_difference_gradient_component(
        _rel_casscf_energy,
        symbols,
        coordinates,
        1,
        2,
        step=5.0e-4,
        **options,
    )

    assert gradient[1, 2] == pytest.approx(numerical, abs=5.0e-7)
    assert gradient.sum(axis=0) == pytest.approx(np.zeros(3), abs=1.0e-9)


def test_snso_x2c_rel_casscf_gradient_triatomic_finite_difference():
    """Exercise correlated two-component CASSCF response with SNSO."""
    symbols = ["S", "H", "H"]
    coordinates = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.70, 1.80],
            [0.0, -1.55, 1.95],
        ]
    )
    options = dict(
        nel=18,
        core_orbitals=list(range(14)),
        active_orbitals=list(range(14, 22)),
        x2c_type="so",
        snso_type="row-dependent",
    )

    gradient = _rel_casscf(symbols, coordinates, **options).gradient()
    numerical = four_point_central_difference_gradient_component(
        _rel_casscf_energy,
        symbols,
        coordinates,
        1,
        1,
        step=5.0e-4,
        **options,
    )

    assert gradient[1, 1] == pytest.approx(numerical, abs=2.0e-7)
    assert np.linalg.norm(gradient) > 1.0e-3
    assert gradient.sum(axis=0) == pytest.approx(np.zeros(3), abs=1.0e-8)


def test_snso_x2c_rel_gasscf_gradient_finite_difference():
    """Validate two-component GASSCF with optimized inter-GAS rotations."""
    symbols = ["Li", "H"]
    coordinates = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 3.0]])
    options = dict(
        nel=4,
        core_orbitals=[0, 1],
        active_orbitals=[[2, 3], [4, 5]],
        gas_min=[1],
        gas_max=[2],
        x2c_type="so",
        snso_type="row-dependent",
    )

    mc = _rel_casscf(symbols, coordinates, **options)
    gradient = mc.gradient()
    numerical = four_point_central_difference_gradient_component(
        _rel_casscf_energy,
        symbols,
        coordinates,
        1,
        2,
        step=5.0e-4,
        **options,
    )

    assert mc.mo_space.ngas == 2
    assert not mc.freeze_inter_gas_rots
    assert gradient[1, 2] == pytest.approx(numerical, abs=5.0e-7)
    assert gradient.sum(axis=0) == pytest.approx(np.zeros(3), abs=1.0e-9)


def test_rel_casscf_gradient_rejects_state_average_before_auto_run():
    symbols = ["H", "H"]
    coordinates = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.7]])
    system = _system(symbols, coordinates)
    parent = SpinorUpcaster()(RHF(charge=0)(system))
    mc = MCOptimizer(
        RelCISolver(nel=2, nroots=2, active_orbitals=4),
        final_orbitals="original",
    )(parent)

    with pytest.raises(NotImplementedError, match="state-specific"):
        mc.gradient()

    assert not mc.executed
