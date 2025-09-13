import numpy as np

from forte2 import ints
from forte2.system.atom_data import DEBYE_TO_AU, ANGSTROM_TO_BOHR
from forte2.helpers.matrix_functions import block_diag_2x2


def get_1e_property(system, g1, property_name, origin=None, unit="debye"):
    """
    Calculate a one-electron property using AO-basis quantities.

    Parameters
    ----------
    system : System
        The system for which the property is calculated.
    g1 : NDArray
        The 1-particle density matrix in the AO basis.
        Should be the spin-free density matrix (dm_aa + dm_bb) for the non-relativistic case.
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

    if system.two_component:
        assert (
            g1.shape[0] == 2 * system.nbf
        ), f"g1 shape {g1.shape[0]} does not match the number of basis functions, {2 * system.nbf} in the system."
    else:
        assert (
            g1.shape[0] == system.nbf
        ), f"g1 shape {g1.shape[0]} does not match the number of basis functions, {system.nbf} in the system."

    def _origin_check(origin):
        if origin is None:
            origin = [0.0, 0.0, 0.0]
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
            oei = ints.kinetic(system.basis)
        case "nuclear_attraction_energy":
            oei = ints.nuclear(system.basis, system.atoms)
        case "electric_dipole":
            origin = _origin_check(origin)
            _, *oei = ints.emultipole1(system.basis, origin=origin)
            factor = -1.0 / DEBYE_TO_AU if unit == "debye" else -1.0
        case "dipole":
            e_dip = get_1e_property(
                system, g1, "electric_dipole", origin=origin, unit=unit
            )
            nuc_dip = system.nuclear_dipole(origin=origin, unit=unit)
            return e_dip + nuc_dip
        case "electric_quadrupole":
            origin = _origin_check(origin)
            *_, xx, xy, xz, yy, yz, zz = ints.emultipole2(system.basis, origin=origin)
            oei = [xx, xy, xz, yy, yz, zz]
            factor = (
                -1.0 / (DEBYE_TO_AU * ANGSTROM_TO_BOHR) if unit == "debye" else -1.0
            )
        case "quadrupole":
            xx, xy, xz, yy, yz, zz = get_1e_property(
                system, g1, "electric_quadrupole", origin=origin, unit=unit
            )
            e_quad = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
            e_quad = 0.5 * (3 * e_quad - np.trace(e_quad) * np.eye(3))
            nuc_quad = system.nuclear_quadrupole(origin=origin, unit=unit)
            return e_quad + nuc_quad
        case _:
            raise ValueError(f"Property '{property_name}' is not supported.")

    if system.two_component:
        if isinstance(oei, list):
            oei = [(block_diag_2x2(_)) for _ in oei]
        else:
            oei = block_diag_2x2(oei)

    if not isinstance(oei, list):
        return np.einsum("pq,pq->", g1, oei) * factor
    return np.array([np.einsum("pq,pq->", g1, _) for _ in oei]) * factor


def mulliken_population(system, g1):
    """
    Perform Mulliken population analysis on the system using the given method.

    Parameters
    ----------
    system : System
        The system for which the Mulliken population is calculated.
    g1 : NDArray
        The 1-particle spin-free density matrix (dm_aa + dm_bb).

    Returns
    -------
    tuple(NDArray, NDArray)
        The Mulliken population for each basis function and the atomic charges.

    Notes
    -----
    See eq 3.196 in Szabo and Ostlund.
    """
    ovlp = ints.overlap(system.basis)
    psdiag = np.einsum("pq,qp->p", g1, ovlp)
    center_first_and_last = system.basis.center_first_and_last
    charges = system.atomic_charges
    pop = np.array([psdiag[_[0] : _[1]].sum() for _ in center_first_and_last])
    return (psdiag, charges - pop)


def iao_partial_charge(system, g1_iao):
    """
    Perform partial charge analysis using IAOs.

    Parameters
    ----------
    system : System
        The system for which the partial charge is calculated.
    g1_iao : NDArray
        The 1-particle spin-free density matrix in the IAO basis.
        Calulated using `forte2.orbitlas.iao.IAO.make_sf_1rdm`.

    Returns
    -------
    tuple(NDArray, NDArray)
        The diagonal elements of the 1-particle density matrix in the IAO basis and the
        partial charges for each atom.
    """
    g1diag = np.diag(g1_iao)
    center_first_and_last = system.minao_basis.center_first_and_last
    charges = system.atomic_charges
    pop = np.array([g1diag[_[0] : _[1]].sum() for _ in center_first_and_last])
    return (g1diag, charges - pop)
