import numpy as np

from forte2 import System

_FOUR_POINT_STENCIL = ((-2.0, 1.0), (-1.0, -8.0), (1.0, 8.0), (2.0, -1.0))
_SIX_POINT_STENCIL = (
    (-3.0, -1.0),
    (-2.0, 9.0),
    (-1.0, -45.0),
    (1.0, 45.0),
    (2.0, -9.0),
    (3.0, 1.0),
)


def xyz_string(symbols, coordinates):
    """Format symbols and Cartesian coordinates as an XYZ geometry string."""
    return "\n".join(
        f"{symbol} {xyz[0]:.16f} {xyz[1]:.16f} {xyz[2]:.16f}"
        for symbol, xyz in zip(symbols, coordinates)
    )


def make_test_system(symbols, coordinates):
    """Build the common density-fitted system used by SCF gradient tests."""
    return System(
        xyz=xyz_string(symbols, coordinates),
        basis_set="cc-pVDZ",
        auxiliary_basis_set="def2-universal-JKFIT",
        unit="bohr",
    )


def _central_difference_gradient_component(
    energy_fn,
    symbols,
    coordinates,
    atom,
    cart,
    stencil,
    denominator,
    *args,
    step,
    **kwargs,
):
    coordinates = np.asarray(coordinates, dtype=float)

    def shifted_energy(scale):
        shifted_coordinates = coordinates.copy()
        shifted_coordinates[atom, cart] += scale * step
        return energy_fn(symbols, shifted_coordinates, *args, **kwargs)

    return sum(weight * shifted_energy(scale) for scale, weight in stencil) / (
        denominator * step
    )


def four_point_central_difference_gradient_component(
    energy_fn, symbols, coordinates, atom, cart, *args, step=1.0e-3, **kwargs
):
    """Compute one Cartesian gradient component with a four-point stencil."""
    return _central_difference_gradient_component(
        energy_fn,
        symbols,
        coordinates,
        atom,
        cart,
        _FOUR_POINT_STENCIL,
        12.0,
        *args,
        step=step,
        **kwargs,
    )


def six_point_central_difference_gradient_component(
    energy_fn, symbols, coordinates, atom, cart, *args, step=1.0e-3, **kwargs
):
    """Compute one Cartesian gradient component with a six-point stencil."""
    return _central_difference_gradient_component(
        energy_fn,
        symbols,
        coordinates,
        atom,
        cart,
        _SIX_POINT_STENCIL,
        60.0,
        *args,
        step=step,
        **kwargs,
    )
