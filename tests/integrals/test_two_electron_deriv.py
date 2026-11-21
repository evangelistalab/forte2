import numpy as np
import pytest

import forte2
from forte2.lib import ints
import forte2.integrals
from forte2.data.atom_data import ANGSTROM_TO_BOHR
from forte2.gradients import finite_difference
from forte2.helpers.comparisons import approx_abs

rng = np.random.default_rng(seed=20240603)


def _make_system(xyz):
    return forte2.System(
        xyz=xyz,
        basis_set="sto-3g",
        auxiliary_basis_set="cc-pvtz-jkfit",
    )


def _h2_systems(delta=1.0e-5):
    """Reference system and a factory shifting one coordinate by `t` angstrom."""

    def displaced(t=0.0):
        return _make_system(f"H 0.0 0.0 0.0\nH 0.0 0.0 {0.74 + t:.12f}")

    return displaced(), displaced, 5, delta


def _h2o_systems(delta=1.0e-5):
    """Reference system and a factory shifting one coordinate by `t` angstrom."""

    def displaced(t=0.0):
        return _make_system(f"O 0.0 {t:.12f} 0.0\nH 0.0 0.0 1.0\nH 0.0 1.0 0.0")

    return displaced(), displaced, 1, delta


def _contracted_derivative(integral, displaced, weights, subscripts, delta):
    """Derivative of ``einsum(subscripts, integral(system), weights)`` in Eh/Bohr.

    The displacement is parameterized in angstrom, so the result is converted to
    the atomic units the analytic derivatives use.
    """

    def contracted(t):
        return np.einsum(subscripts, integral(displaced(t)), weights)

    return (
        float(finite_difference(contracted, 0.0, step=delta, npoints=2))
        / ANGSTROM_TO_BOHR
    )


def _random_w3(system, complex_weights=False):
    shape = (system.auxiliary_basis.size, system.basis.size, system.basis.size)
    weights = rng.standard_normal(size=shape)
    if complex_weights:
        weights = weights + 1j * rng.standard_normal(size=shape)
    return weights


def _random_w2(system, complex_weights=False):
    shape = (system.auxiliary_basis.size, system.auxiliary_basis.size)
    weights = rng.standard_normal(size=shape)
    if complex_weights:
        weights = weights + 1j * rng.standard_normal(size=shape)
    return weights


def test_coulomb_3c_deriv_rejects_bad_shape_and_centers():
    system = _make_system("H 0.0 0.0 0.0\nH 0.0 0.0 1.0")
    bad_shape = np.zeros(
        (system.auxiliary_basis.size + 1, system.basis.size, system.basis.size)
    )

    with pytest.raises(ValueError, match="W3 has incorrect shape"):
        ints.coulomb_3c_deriv(
            system.auxiliary_basis,
            system.basis,
            system.basis,
            bad_shape,
            system.atoms,
        )

    W3 = np.zeros((system.auxiliary_basis.size, system.basis.size, system.basis.size))
    with pytest.raises(
        ValueError, match="basis1 center 0 does not match charges center 0"
    ):
        ints.coulomb_3c_deriv(
            system.auxiliary_basis,
            system.basis,
            system.basis,
            W3,
            list(reversed(system.atoms)),
        )


def test_coulomb_2c_deriv_rejects_bad_shape_and_centers():
    system = _make_system("H 0.0 0.0 0.0\nH 0.0 0.0 1.0")
    bad_shape = np.zeros((system.auxiliary_basis.size + 1, system.auxiliary_basis.size))

    with pytest.raises(ValueError, match="W2 has incorrect shape"):
        ints.coulomb_2c_deriv(
            system.auxiliary_basis,
            system.auxiliary_basis,
            bad_shape,
            system.atoms,
        )

    W2 = np.zeros((system.auxiliary_basis.size, system.auxiliary_basis.size))
    with pytest.raises(
        ValueError, match="basis1 center 0 does not match charges center 0"
    ):
        ints.coulomb_2c_deriv(
            system.auxiliary_basis,
            system.auxiliary_basis,
            W2,
            list(reversed(system.atoms)),
        )


@pytest.mark.parametrize("systems", [_h2_systems, _h2o_systems])
def test_coulomb_3c_deriv_finite_difference_real_weights(systems):
    system0, displaced, component, delta = systems()
    W3 = _random_w3(system0)

    analytical = ints.coulomb_3c_deriv(
        system0.auxiliary_basis, system0.basis, system0.basis, W3, system0.atoms
    )
    wrapper = forte2.integrals.coulomb_3c_deriv(system0, W3)
    assert np.linalg.norm(analytical - wrapper) < 1.0e-12

    numerical = _contracted_derivative(
        forte2.integrals.coulomb_3c, displaced, W3, "Pmn,Pmn->", delta
    )

    assert analytical[component] == approx_abs(numerical, atol=1.0e-6)


@pytest.mark.parametrize("systems", [_h2_systems, _h2o_systems])
def test_coulomb_2c_deriv_finite_difference_real_weights(systems):
    system0, displaced, component, delta = systems()
    W2 = _random_w2(system0)

    analytical = ints.coulomb_2c_deriv(
        system0.auxiliary_basis, system0.auxiliary_basis, W2, system0.atoms
    )
    wrapper = forte2.integrals.coulomb_2c_deriv(system0, W2)
    assert np.linalg.norm(analytical - wrapper) < 1.0e-12

    numerical = _contracted_derivative(
        forte2.integrals.coulomb_2c, displaced, W2, "PQ,PQ->", delta
    )

    assert analytical[component] == approx_abs(numerical, atol=1.0e-6)


def test_coulomb_derivs_accept_complex_weights_and_use_real_part():
    system0, displaced, component, delta = _h2_systems()

    W3 = _random_w3(system0, complex_weights=True)
    grad3 = ints.coulomb_3c_deriv(
        system0.auxiliary_basis, system0.basis, system0.basis, W3, system0.atoms
    )
    num3 = _contracted_derivative(
        forte2.integrals.coulomb_3c, displaced, W3.real, "Pmn,Pmn->", delta
    )
    assert np.isrealobj(grad3)
    assert grad3[component] == approx_abs(num3, atol=1.0e-6)

    W2 = _random_w2(system0, complex_weights=True)
    grad2 = ints.coulomb_2c_deriv(
        system0.auxiliary_basis, system0.auxiliary_basis, W2, system0.atoms
    )
    num2 = _contracted_derivative(
        forte2.integrals.coulomb_2c, displaced, W2.real, "PQ,PQ->", delta
    )
    assert np.isrealobj(grad2)
    assert grad2[component] == approx_abs(num2, atol=1.0e-6)


def test_coulomb_deriv_translation_invariance():
    system = _make_system("O 0.0 0.0 0.0\nH 0.0 0.0 1.0\nH 0.0 1.0 0.0")

    grad3 = forte2.integrals.coulomb_3c_deriv(system, _random_w3(system))
    grad2 = forte2.integrals.coulomb_2c_deriv(system, _random_w2(system))

    assert np.linalg.norm(grad3.reshape(-1, 3).sum(axis=0)) < 1.0e-7
    assert np.linalg.norm(grad2.reshape(-1, 3).sum(axis=0)) < 1.0e-7
