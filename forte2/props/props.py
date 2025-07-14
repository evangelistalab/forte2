import forte2
import numpy as np
import scipy.linalg


def _spin_independent_property_1e(dm_tot, ints, factor=1.0):
    """
    Calculate a one-electron spin-independent property using the total density matrix and integrals.

    Parameters
    ----------
        dm_tot : NDArray
            The total density matrix (dm_aa + dm_bb)
        ints : NDArray or list[NDArray]
            The integrals for the property.
        factor : float, optional, default=1.0
            A scaling factor for the property value.

    Returns
    -------
        float or NDArray
            The calculated property value, either a single value or an array if multiple integrals are provided.
    """
    if not isinstance(ints, list):
        return np.einsum("pq,qp->", dm_tot, ints) * factor
    return np.array([np.einsum("pq,qp->", dm_tot, _) for _ in ints]) * factor


def get_property(method, property_name, origin=None, unit="debye"):
    """
    Calculate a property of the system using the given method.

    Parameters
    ----------
    method : object
        A method object that has been run and contains the necessary data.
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
            if method.charge != 0:
                print(
                    "Warning: Electric multipole moment for a charged system is "
                    "origin-dependent. Using center of mass as origin."
                )
                origin = method.system.center_of_mass()
                print(f"Center-of-mass origin: {origin}")
        assert len(origin) == 3, "Origin must be a 3-element vector."
        return origin

    if not method.executed:
        method.run()

    spin_independent_properties = [
        "kinetic_energy",
        "nuclear_attraction_energy",
        "electric_dipole",
        "dipole",
        "electric_quadrupole",
        "quadrupole",
    ]
    if property_name in spin_independent_properties:
        propfunc = _spin_independent_property_1e
        assert hasattr(method, "_build_total_density_matrix"), (
            "Method must have a '_build_total_density_matrix' method to calculate "
            f"the property '{property_name}'."
        )
        dm = method._build_total_density_matrix()

    factor = 1.0

    match property_name:
        case "kinetic_energy":
            ints = forte2.ints.kinetic(method.system.basis)
        case "nuclear_attraction_energy":
            ints = forte2.ints.nuclear(method.system.basis, method.system.atoms)
        case "electric_dipole":
            origin = _origin_check(origin)
            _, *ints = forte2.ints.emultipole1(method.system.basis, origin=origin)
            factor = -1.0 / forte2.atom_data.DEBYE_TO_AU if unit == "debye" else -1.0
        case "dipole":
            e_dip = get_property(method, "electric_dipole", origin=origin, unit=unit)
            nuc_dip = method.system.nuclear_dipole(origin=origin, unit=unit)
            return e_dip + nuc_dip
        case "electric_quadrupole":
            origin = _origin_check(origin)
            *_, xx, xy, xz, yy, yz, zz = forte2.ints.emultipole2(
                method.system.basis, origin=origin
            )
            ints = [xx, xy, xz, yy, yz, zz]
            factor = (
                -1.0
                / (forte2.atom_data.DEBYE_TO_AU * forte2.atom_data.ANGSTROM_TO_BOHR)
                if unit == "debye"
                else -1.0
            )
        case "quadrupole":
            xx, xy, xz, yy, yz, zz = get_property(
                method, "electric_quadrupole", origin=origin, unit=unit
            )
            e_quad = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
            e_quad = 0.5 * (3 * e_quad - np.trace(e_quad) * np.eye(3))
            nuc_quad = method.system.nuclear_quadrupole(origin=origin, unit=unit)
            return e_quad + nuc_quad
        case _:
            raise ValueError(f"Property '{property_name}' is not supported.")

    return propfunc(dm, ints, factor=factor)


def mulliken_population(method):
    """
    Perform Mulliken population analysis on the system using the given method.

    Parameters
    ----------
    method : object
        A method object that has a ``_build_total_density_matrix`` method.

    Returns
    -------
    tuple(NDArray, NDArray)
        The Mulliken population for each basis function and the atomic charges.

    Notes
    -----
    See eq 3.196 in Szabo and Ostlund.
    """
    if not method.executed:
        method.run()
    assert hasattr(method, "_build_total_density_matrix"), (
        "Method must have a '_build_total_density_matrix' method to calculate "
        "Mulliken population."
    )
    system = method.system
    dm = method._build_total_density_matrix()
    ovlp = forte2.ints.overlap(system.basis)
    psdiag = np.einsum("pq,qp->p", dm, ovlp)
    center_first_and_last = system.basis.center_first_and_last
    charges = system.atomic_charges()
    pop = np.array([psdiag[_[0] : _[1]].sum() for _ in center_first_and_last])
    return (psdiag, charges - pop)
