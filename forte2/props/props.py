import numpy as np
import scipy.linalg
from typing import Callable

import forte2
from forte2.helpers import logger


def get_1e_property(system, g1_sf, property_name, origin=None, unit="debye"):
    """
    Calculate a one-electron property using AO-basis quantities.

    Parameters
    ----------
    system : System
        The system for which the property is calculated.
    g1_sf : NDArray
        The 1-particle spin-free density matrix (dm_aa + dm_bb) in the AO basis.
    property_name : str
        The name of the property to calculate (e.g., "kinetic_energy", "nuclear_attraction_energy", "electric_dipole").
    origin: list[float], optional
        The origin point for properties that depend on it (e.g., electric dipole moment).
    unit: str, optional, default="debye"
        The unit for the property value, either "debye" or "au". Default is "debye".
        Only used for multipole moments. For quadrupole moments, "debye" stands for debye * angstrom, etc.

    Returns
    -------
    float or NDArray
        The calculated property value.
    """

    def _origin_check(origin):
        if origin is None:
            origin = [0.0, 0.0, 0.0]
            logger.log_info1(f"Origin not provided, using zero origin: {origin}")
        assert len(origin) == 3, "Origin must be a 3-element vector."
        return origin

    spin_independent_properties = [
        "kinetic_energy",
        "nuclear_attraction_energy",
        "electric_dipole",
        "dipole",
        "electric_quadrupole",
        "quadrupole",
    ]
    assert (
        property_name in spin_independent_properties
    ), f"Property '{property_name}' is not supported, must be one of {spin_independent_properties}."
    factor = 1.0

    match property_name:
        case "kinetic_energy":
            ints = forte2.ints.kinetic(system.basis)
        case "nuclear_attraction_energy":
            ints = forte2.ints.nuclear(system.basis, system.atoms)
        case "electric_dipole":
            origin = _origin_check(origin)
            _, *ints = forte2.ints.emultipole1(system.basis, origin=origin)
            factor = -1.0 / forte2.atom_data.DEBYE_TO_AU if unit == "debye" else -1.0
        case "dipole":
            e_dip = get_1e_property(
                system, g1_sf, "electric_dipole", origin=origin, unit=unit
            )
            nuc_dip = system.nuclear_dipole(origin=origin, unit=unit)
            return e_dip + nuc_dip
        case "electric_quadrupole":
            origin = _origin_check(origin)
            *_, xx, xy, xz, yy, yz, zz = forte2.ints.emultipole2(
                system.basis, origin=origin
            )
            ints = [xx, xy, xz, yy, yz, zz]
            factor = (
                -1.0
                / (forte2.atom_data.DEBYE_TO_AU * forte2.atom_data.ANGSTROM_TO_BOHR)
                if unit == "debye"
                else -1.0
            )
        case "quadrupole":
            xx, xy, xz, yy, yz, zz = get_1e_property(
                system, g1_sf, "electric_quadrupole", origin=origin, unit=unit
            )
            e_quad = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
            e_quad = 0.5 * (3 * e_quad - np.trace(e_quad) * np.eye(3))
            nuc_quad = system.nuclear_quadrupole(origin=origin, unit=unit)
            return e_quad + nuc_quad
        case _:
            raise ValueError(f"Property '{property_name}' is not supported.")

    if not isinstance(ints, list):
        return np.einsum("pq,qp->", g1_sf, ints) * factor
    return np.array([np.einsum("pq,qp->", g1_sf, _) for _ in ints]) * factor


def mulliken_population(system, g1_sf):
    """
    Perform Mulliken population analysis on the system using the given method.

    Parameters
    ----------
    system : System
        The system for which the Mulliken population is calculated.
    g1_sf : NDArray
        The 1-particle spin-free density matrix (dm_aa + dm_bb).

    Returns
    -------
    tuple(NDArray, NDArray)
        The Mulliken population for each basis function and the atomic charges.

    Notes
    -----
    See eq 3.196 in Szabo and Ostlund.
    """
    ovlp = forte2.ints.overlap(system.basis)
    psdiag = np.einsum("pq,qp->p", g1_sf, ovlp)
    center_first_and_last = system.basis.center_first_and_last
    charges = system.atomic_charges
    pop = np.array([psdiag[_[0] : _[1]].sum() for _ in center_first_and_last])
    return (psdiag, charges - pop)
