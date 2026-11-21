import numpy as np
import pytest

import forte2
from forte2.gradients import (
    central_stencil,
    finite_difference,
    nuclear_repulsion_deriv,
)
from forte2.system import System, coords_to_xyz


def test_finite_difference_of_scalar_functions():
    # f: R -> R
    assert finite_difference(np.sin, 0.7) == pytest.approx(np.cos(0.7), abs=1.0e-10)

    # f: R -> R^{2x2}
    def f(t):
        return np.array([[t**2, np.sin(t)], [np.exp(t), 1.0]])

    t = 0.3
    derivative = finite_difference(f, t, step=1.0e-3)
    expected = np.array([[2 * t, np.cos(t)], [np.exp(t), 0.0]])

    np.testing.assert_allclose(derivative, expected, atol=1.0e-9)


def test_finite_difference_of_vector_functions():
    def f(v):
        return float(v[0] ** 2 * v[1] + np.sin(v[2]))

    x = np.array([1.3, -0.7, 0.4])
    original = x.copy()
    expected = np.array([2 * x[0] * x[1], x[0] ** 2, np.cos(x[2])])

    gradient = finite_difference(f, x, step=1.0e-3, npoints=6)

    np.testing.assert_allclose(gradient, expected, atol=1.0e-9)
    np.testing.assert_array_equal(x, original)  # x shouldn't be modified

    # x of shape (2, 3) and f -> R^4 gives a (2, 3, 4) derivative.
    rng = np.random.default_rng(0)
    A = rng.standard_normal((4, 6))

    def linear(v):
        return A @ v.reshape(-1)

    x2 = rng.standard_normal((2, 3))
    derivative = finite_difference(linear, x2, npoints=2)

    assert derivative.shape == (2, 3, 4)
    np.testing.assert_allclose(derivative.reshape(6, 4), A.T, atol=1.0e-8)


def test_components_selects_a_subset_of_indices():
    def f(v):
        return float(v @ v)

    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).reshape(-1)
    full = finite_difference(f, x)
    subset = finite_difference(f, x, components=[1, 4])

    assert subset.shape == (2,)
    np.testing.assert_allclose(subset, [full[1], full[4]], atol=1.0e-10)

    # Multidimensional indices are also accepted.
    def cubic(v):
        return float(np.sum(v**3))

    x2 = np.array([[1.0, 2.0], [3.0, 4.0]])
    derivative = finite_difference(cubic, x2, components=[(1, 0)], npoints=6)

    assert derivative.shape == (1,)
    assert derivative[0] == pytest.approx(3 * 3.0**2, abs=1.0e-6)


def test_nuclear_repulsion_gradient_matches_the_analytic_derivative():
    charges = [8, 1, 1]
    coordinates = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.8], [1.6, 0.0, 0.0]])

    def energy(xyz):
        system = System(
            xyz=coords_to_xyz(charges, xyz),
            basis_set="sto-3g",
            auxiliary_basis_set="def2-universal-JKFIT",
            unit="bohr",
        )
        return forte2.integrals.nuclear_repulsion(system)

    numerical = finite_difference(energy, coordinates, step=1.0e-4)

    system = System(
        xyz=coords_to_xyz(charges, coordinates),
        basis_set="sto-3g",
        auxiliary_basis_set="def2-universal-JKFIT",
        unit="bohr",
    )
    analytical = nuclear_repulsion_deriv(system.atoms)

    assert numerical.shape == (3, 3)
    np.testing.assert_allclose(numerical, analytical, atol=1.0e-9)
    # Translational invariance: the net force must vanish.
    np.testing.assert_allclose(numerical.sum(axis=0), np.zeros(3), atol=1.0e-9)
