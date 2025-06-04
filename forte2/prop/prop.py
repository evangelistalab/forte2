from dataclasses import dataclass, field
import time
import numpy as np
import scipy as sp
import copy

import forte2


def _spin_independent_property(dm_tot, ints):
    """Calculate a spin-independent property using the total density matrix and integrals."""
    if not isinstance(ints, list):
        return np.einsum("pq,qp->", dm_tot, ints)
    return np.array([np.einsum("pq,qp->", dm_tot, _) for _ in ints])


def get_property(method, property_name, origin=None):
    if not method.executed:
        method.run()

    spin_independent_properties = [
        "kinetic_energy",
        "nuclear_attraction_energy",
        "electric_dipole",
    ]
    if property_name in spin_independent_properties:
        propfunc = _spin_independent_property
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
            return (-e_dip + nuc_dip)/ forte2.atom_data.DEBYE_TO_AU
        case _:
            raise ValueError(f"Property '{property_name}' is not supported.")

    return propfunc(dm, ints)
