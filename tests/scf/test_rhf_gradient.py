import numpy as np
import pytest

import forte2
from forte2 import System
from forte2.gradients import nuclear_repulsion_deriv
from forte2.scf import RHF

forte2.set_verbosity_level(0)


def _xyz(symbols, coordinates):
    return "\n".join(
        f"{symbol} {xyz[0]:.16f} {xyz[1]:.16f} {xyz[2]:.16f}"
        for symbol, xyz in zip(symbols, coordinates)
    )


def _system(symbols, coordinates):
    return System(
        xyz=_xyz(symbols, coordinates),
        basis_set="sto-3g",
        auxiliary_basis_set="def2-universal-JKFIT",
        unit="bohr",
    )


def _rhf_energy(symbols, coordinates):
    rhf = RHF(charge=0, e_tol=1.0e-12, d_tol=1.0e-10, maxiter=100)(
        _system(symbols, coordinates)
    )
    rhf.run()
    return rhf.E


def _rhf_gradient(symbols, coordinates):
    rhf = RHF(charge=0, e_tol=1.0e-12, d_tol=1.0e-10, maxiter=100)(
        _system(symbols, coordinates)
    )
    rhf.run()
    return rhf.gradient()


def _finite_difference_gradient_component(
    symbols, coordinates, atom, cart, delta=1.0e-4
):
    coords_plus = coordinates.copy()
    coords_minus = coordinates.copy()
    coords_plus[atom, cart] += delta
    coords_minus[atom, cart] -= delta
    return (_rhf_energy(symbols, coords_plus) - _rhf_energy(symbols, coords_minus)) / (
        2.0 * delta
    )


def test_nuclear_repulsion_deriv_finite_difference():
    symbols = ["O", "H", "H"]
    coordinates = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.8],
            [1.6, 0.0, 0.0],
        ]
    )
    delta = 1.0e-5
    atom = 2
    cart = 0

    system0 = _system(symbols, coordinates)
    analytical = nuclear_repulsion_deriv(system0.atoms)

    coords_plus = coordinates.copy()
    coords_minus = coordinates.copy()
    coords_plus[atom, cart] += delta
    coords_minus[atom, cart] -= delta
    numerical = (
        forte2.integrals.nuclear_repulsion(_system(symbols, coords_plus))
        - forte2.integrals.nuclear_repulsion(_system(symbols, coords_minus))
    ) / (2.0 * delta)

    np.testing.assert_allclose(analytical[atom, cart], numerical, atol=1.0e-9)


def test_rhf_gradient_h2_finite_difference():
    symbols = ["H", "H"]
    coordinates = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.7]])

    gradient = _rhf_gradient(symbols, coordinates)
    numerical = _finite_difference_gradient_component(symbols, coordinates, 1, 2)

    np.testing.assert_allclose(gradient[1, 2], numerical, atol=1.0e-7)


def test_rhf_gradient_h2o_finite_difference_and_translation():
    symbols = ["O", "H", "H"]
    coordinates = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.8],
            [1.6, 0.0, 0.0],
        ]
    )

    gradient = _rhf_gradient(symbols, coordinates)
    numerical = _finite_difference_gradient_component(symbols, coordinates, 2, 0)

    np.testing.assert_allclose(gradient[2, 0], numerical, atol=1.0e-7)
    np.testing.assert_allclose(gradient.sum(axis=0), np.zeros(3), atol=1.0e-10)


def test_rhf_gradient_auto_runs_and_reuses_executed_object():
    system = _system(["H", "H"], np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.7]]))
    rhf = RHF(charge=0, e_tol=1.0e-12, d_tol=1.0e-10, maxiter=100)(system)

    assert not rhf.executed
    gradient1 = rhf.gradient()
    energy1 = rhf.E

    assert rhf.executed
    gradient2 = rhf.gradient()

    assert rhf.E == pytest.approx(energy1)
    np.testing.assert_allclose(gradient1, gradient2, atol=1.0e-12)
    assert gradient1.shape == (system.natoms, 3)


def test_rhf_gradient_rejects_cholesky_tei():
    system = System(
        xyz="H 0 0 0\nH 0 0 1.7",
        basis_set="sto-3g",
        cholesky_tei=True,
        unit="bohr",
    )
    rhf = RHF(charge=0, e_tol=1.0e-12, d_tol=1.0e-10, maxiter=100)(system)
    rhf.run()

    with pytest.raises(NotImplementedError, match="density fitting"):
        rhf.gradient()

def test_rhf_gradient_with_df_ortho_rtol():
    # this test asserts that the df_ortho_rtol codepath runs
    system = System(
        xyz="""
        H 0 0 0
        H 0 0 1.7
        """,
        basis_set="sto-3g",
        auxiliary_basis_set="def2-universal-JKFIT",
        unit="bohr",
        df_ortho_rtol=1e-8,
    )
    rhf = RHF(charge=0, e_tol=1.0e-12, d_tol=1.0e-10, maxiter=100)(system)
    _ = rhf.gradient()
