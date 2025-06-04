import forte2
import numpy as np
import scipy.linalg


def _spin_independent_property_1e(dm_tot, ints):
    """Calculate a one-electron spin-independent property using the total density matrix and integrals."""
    if not isinstance(ints, list):
        return np.einsum("pq,qp->", dm_tot, ints)
    return np.array([np.einsum("pq,qp->", dm_tot, _) for _ in ints])


def get_property(method, property_name, origin=None):
    """
    Calculate a property of the system using the given method.
    Args:
        method: A method object that has been run and contains the necessary data.
        property_name: The name of the property to calculate (e.g., "kinetic_energy", "nuclear_attraction_energy", "electric_dipole").
        origin: Optional; the origin point for properties that depend on it (e.g., electric dipole moment).
    Returns:
        The calculated property value.
    Raises:
        ValueError: If the property name is not supported or if the method has not been executed.
    """
    if not method.executed:
        method.run()

    spin_independent_properties = [
        "kinetic_energy",
        "nuclear_attraction_energy",
        "electric_dipole",
    ]
    if property_name in spin_independent_properties:
        propfunc = _spin_independent_property_1e
        assert hasattr(method, "_build_total_density_matrix"), (
            "Method must have a '_build_total_density_matrix' method to calculate "
            f"the property '{property_name}'."
        )
        dm = method._build_total_density_matrix()

    match property_name:
        case "kinetic_energy":
            ints = forte2.ints.kinetic(method.system.basis)
        case "nuclear_attraction_energy":
            ints = forte2.ints.nuclear(method.system.basis, method.system.atoms)
        case "electric_dipole":
            if origin is None:
                origin = [0.0, 0.0, 0.0]
                if method.charge != 0:
                    print(
                        "Warning: Electric dipole moment for a charged system is "
                        "origin-dependent. Using center of mass as origin."
                    )
                    origin = method.system.center_of_mass()
                    print(f"Center-of-mass origin: {origin}")
            assert len(origin) == 3, "Origin must be a 3-element vector."
            _, *ints = forte2.ints.emultipole1(method.system.basis, origin=origin)
        case "dipole":
            e_dip = get_property(method, "electric_dipole", origin=origin)
            nuc_dip = method.system.nuclear_dipole(origin=origin)
            return (-e_dip + nuc_dip) / forte2.atom_data.DEBYE_TO_AU
        case _:
            raise ValueError(f"Property '{property_name}' is not supported.")

    return propfunc(dm, ints)


def mulliken_population(method):
    """
    Perform Mulliken population analysis on the system using the given method.
    See eq 3.196 in Szabo and Ostlund.
    Args:
        method: A method object that has been run and contains the necessary data.
    Returns:
        Tuple(np.ndarray, np.ndarray): The Mulliken population for each basis function and the atomic charges.
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
    charges = np.array([atom[0] for atom in system.atoms])
    pop = np.array([psdiag[_[0] : _[1]].sum() for _ in center_first_and_last])
    return (psdiag, charges - pop)
