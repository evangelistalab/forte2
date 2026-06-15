import numpy as np
import pytest

from forte2 import System, RHF, MCOptimizer, State, CISolver


def _xyz(symbols, coordinates):
    return "\n".join(
        f"{symbol} {xyz[0]:.16f} {xyz[1]:.16f} {xyz[2]:.16f}"
        for symbol, xyz in zip(symbols, coordinates)
    )


def _system(symbols, coordinates, **kwargs):
    return System(
        xyz=_xyz(symbols, coordinates),
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
    final_orbital="original",
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
        final_orbital=final_orbital,
    )(rhf)
    mc.run()
    return mc


def _casscf_energy(symbols, coordinates, **kwargs):
    return _casscf(symbols, coordinates, **kwargs).E


def _casscf_gradient(symbols, coordinates, **kwargs):
    return _casscf(symbols, coordinates, **kwargs).gradient()


def _four_point_central_difference_component(
    energy_fn, symbols, coordinates, atom, cart, *args, step=1.0e-3, **kwargs
):
    coordinates = np.asarray(coordinates, dtype=float)

    def shifted_energy(scale):
        shifted_coordinates = coordinates.copy()
        shifted_coordinates[atom, cart] += scale * step
        return energy_fn(symbols, shifted_coordinates, *args, **kwargs)

    return (
        -shifted_energy(2.0)
        + 8.0 * shifted_energy(1.0)
        - 8.0 * shifted_energy(-1.0)
        + shifted_energy(-2.0)
    ) / (12.0 * step)


def test_casscf_gradient_h2_full_active_finite_difference_and_translation():
    """Validate the all-active state-specific CASSCF gradient by finite differences."""
    symbols = ["H", "H"]
    coordinates = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.7]])
    kwargs = {"active_orbitals": 2}

    gradient = _casscf_gradient(symbols, coordinates, **kwargs)

    for atom in range(2):
        for cart in range(3):
            numerical = _four_point_central_difference_component(
                _casscf_energy, symbols, coordinates, atom, cart, **kwargs
            )
            assert gradient[atom, cart] == pytest.approx(numerical, abs=1.0e-7)

    assert gradient.sum(axis=0) == pytest.approx(np.zeros(3), abs=1.0e-10)


def test_casscf_gradient_lih_core_active_selected_finite_difference():
    """Validate a CASSCF gradient component with inactive core and active orbitals."""
    symbols = ["Li", "H"]
    coordinates = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 3.0]])
    kwargs = {"core_orbitals": [0], "active_orbitals": [1, 2]}

    gradient = _casscf_gradient(symbols, coordinates, **kwargs)
    numerical = _four_point_central_difference_component(
        _casscf_energy, symbols, coordinates, 1, 2, **kwargs
    )

    assert gradient[1, 2] == pytest.approx(numerical, abs=1.0e-7)
    assert gradient.sum(axis=0) == pytest.approx(np.zeros(3), abs=1.0e-10)


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

    assert mc.E == pytest.approx(energy1)
    assert gradient1 == pytest.approx(gradient2, abs=1.0e-12)
    assert gradient1.shape == (system.natoms, 3)


def test_casscf_gradient_rejects_state_average():
    """Reject SA-CASSCF because V1 implements only state-specific gradients."""
    system = _system(["H", "H"], np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.7]]))
    rhf = RHF(charge=0)(system)
    ci_solver = CISolver(
        State(system=system, multiplicity=1, ms=0.0),
        active_orbitals=2,
        nroots=2,
    )
    mc = MCOptimizer(ci_solver, final_orbital="original")(rhf)

    with pytest.raises(NotImplementedError, match="state-specific"):
        mc.gradient()


def test_casscf_gradient_rejects_frozen_core_orbitals():
    """Reject frozen core orbitals until the CASSCF Z-vector path is added."""
    system = _system(["Li", "H"], np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 3.0]]))
    rhf = RHF(charge=0)(system)
    ci_solver = CISolver(
        State(system=system, multiplicity=1, ms=0.0),
        frozen_core_orbitals=[0],
        active_orbitals=[1, 2],
    )
    mc = MCOptimizer(ci_solver, final_orbital="original")(rhf)

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
    mc = MCOptimizer(ci_solver, final_orbital="original")(rhf)

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
    mc = MCOptimizer(ci_solver, final_orbital="original")(rhf)

    with pytest.raises(NotImplementedError, match="density fitting"):
        mc.gradient()


def test_casscf_gradient_rejects_gaussian_nuclear_charges():
    """Reject Gaussian nuclear charges until their derivative terms are added."""
    system = _system(
        ["H", "H"],
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.7]]),
        use_gaussian_charges=True,
    )
    rhf = RHF(charge=0)(system)
    ci_solver = CISolver(
        State(system=system, multiplicity=1, ms=0.0),
        active_orbitals=2,
    )
    mc = MCOptimizer(ci_solver, final_orbital="original")(rhf)

    with pytest.raises(NotImplementedError, match="Gaussian nuclear charges"):
        mc.gradient()


def test_casscf_gradient_rejects_x2c():
    """Reject X2C CASSCF gradients until relativistic derivative terms are added."""
    system = _system(
        ["H", "H"],
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.7]]),
        x2c_type="sf",
    )
    rhf = RHF(charge=0)(system)
    ci_solver = CISolver(
        State(system=system, multiplicity=1, ms=0.0),
        active_orbitals=2,
    )
    mc = MCOptimizer(ci_solver, final_orbital="original")(rhf)

    with pytest.raises(NotImplementedError, match="X2C"):
        mc.gradient()
