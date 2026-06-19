import numpy as np

from forte2 import System


def _xyz(symbols, coordinates):
    return "\n".join(
        f"{symbol} {xyz[0]:.16f} {xyz[1]:.16f} {xyz[2]:.16f}"
        for symbol, xyz in zip(symbols, coordinates)
    )


def _system(symbols, coordinates):
    return System(
        xyz=_xyz(symbols, coordinates),
        basis_set="cc-pVDZ",
        auxiliary_basis_set="def2-universal-JKFIT",
        unit="bohr",
    )


def four_point_central_difference_gradient_component(
    energy_fn, symbols, coordinates, atom, cart, *args, step=1.0e-3, **kwargs
):
    """Compute one Cartesian gradient component with a four-point central stencil."""
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


def six_point_central_difference_gradient_component(
    energy_fn, symbols, coordinates, atom, cart, *args, step=1.0e-3, **kwargs
):
    """Compute one Cartesian gradient component with a six-point central stencil."""
    coordinates = np.asarray(coordinates, dtype=float)

    def shifted_energy(scale):
        shifted_coordinates = coordinates.copy()
        shifted_coordinates[atom, cart] += scale * step
        return energy_fn(symbols, shifted_coordinates, *args, **kwargs)

    return (
        shifted_energy(3.0)
        - 9.0 * shifted_energy(2.0)
        + 45.0 * shifted_energy(1.0)
        - 45.0 * shifted_energy(-1.0)
        + 9.0 * shifted_energy(-2.0)
        - shifted_energy(-3.0)
    ) / (60.0 * step)
