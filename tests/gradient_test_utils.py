"""Shared helpers for the analytic-gradient tests.

The finite differences themselves come from ``forte2.gradients.finite_difference``;
these wrappers only adapt it to the ``energy_fn(symbols, coordinates, ...)``
signature the gradient tests are written against.
"""

import numpy as np

from forte2 import System
from forte2.data import ATOM_SYMBOL_TO_Z
from forte2.gradients import finite_difference
from forte2.system import coords_to_xyz


def xyz_string(symbols, coordinates):
    """Format symbols and Cartesian coordinates as an XYZ geometry string."""
    charges = [ATOM_SYMBOL_TO_Z[symbol.upper()] for symbol in symbols]
    return coords_to_xyz(charges, coordinates)


def make_test_system(symbols, coordinates):
    """Build the common density-fitted system used by SCF gradient tests."""
    return System(
        xyz=xyz_string(symbols, coordinates),
        basis_set="cc-pVDZ",
        auxiliary_basis_set="def2-universal-JKFIT",
        unit="bohr",
    )


def _central_difference_gradient_component(
    energy_fn, symbols, coordinates, atom, cart, npoints, *args, step, **kwargs
):
    coordinates = np.asarray(coordinates, dtype=float)

    def energy(displaced):
        return energy_fn(symbols, displaced, *args, **kwargs)

    derivative = finite_difference(
        energy,
        coordinates,
        step=step,
        npoints=npoints,
        components=[(atom, cart)],
    )
    return float(derivative[0])


def four_point_central_difference_gradient_component(
    energy_fn, symbols, coordinates, atom, cart, *args, step=1.0e-3, **kwargs
):
    """Compute one Cartesian gradient component with a four-point stencil."""
    return _central_difference_gradient_component(
        energy_fn, symbols, coordinates, atom, cart, 4, *args, step=step, **kwargs
    )


def six_point_central_difference_gradient_component(
    energy_fn, symbols, coordinates, atom, cart, *args, step=1.0e-3, **kwargs
):
    """Compute one Cartesian gradient component with a six-point stencil."""
    return _central_difference_gradient_component(
        energy_fn, symbols, coordinates, atom, cart, 6, *args, step=step, **kwargs
    )
