import numpy as np

from forte2 import ints
from forte2.helpers.matrix_functions import block_diag_2x2


def parse_basis_args_1e(system, basis1, basis2):
    # 2 possible cases:
    # 1. both basis sets are None -> set both to system.basis
    # 2. basis1 is provided, basis2 is None -> set basis2 to basis1
    if basis1 is None and basis2 is None:
        basis1 = system.basis
        basis2 = system.basis
    elif basis1 is not None and basis2 is None:
        basis2 = basis1
    elif basis1 is None and basis2 is not None:
        raise ValueError("If basis2 is provided, basis1 must also be provided.")
    return basis1, basis2


def parse_basis_args_2c2e(system, basis1, basis2):
    # 2 possible cases:
    # 1. both basis sets are None -> set both to system.auxiliary_basis
    # 2. basis1 is provided, basis2 is None -> set basis2 to basis1
    if basis1 is None and basis2 is None:
        basis1 = system.auxiliary_basis
        basis2 = system.auxiliary_basis
    elif basis1 is not None and basis2 is None:
        basis2 = basis1
    elif basis1 is None and basis2 is not None:
        raise ValueError("If basis2 is provided, basis1 must also be provided.")
    return basis1, basis2


def parse_basis_args_3c2e(system, basis1, basis2, basis3):
    # 3 possible cases:
    # 1. all basis sets are None -> set basis1 to system.auxiliary, basis2 and basis3 to system.basis
    # 2. basis1 is provided, basis2 and basis3 are None -> set basis2 and basis3 to system.basis
    # 3. all basis sets are provided
    if basis1 is None:
        basis1 = system.auxiliary_basis
    if basis2 is None and basis3 is not None:
        raise ValueError("If basis3 is provided, basis2 must also be provided.")
    if basis2 is None:
        basis2 = system.basis
    if basis3 is None:
        basis3 = system.basis

    return basis1, basis2, basis3


def parse_basis_args_4c2e(system, basis1, basis2, basis3, basis4):
    # 3 possible cases:
    # 1. all basis sets are None -> set all to system.basis
    # 2. basis1 is provided, others are None -> set others to basis1
    # 3. all basis sets are provided
    if basis1 is None and any(basis is not None for basis in [basis2, basis3, basis4]):
        raise ValueError(
            "If any of basis2, basis3, or basis4 is provided, basis1 must also be provided."
        )
    if basis1 is None:
        basis1 = basis2 = basis3 = basis4 = system.basis
    elif basis1 is not None and all(
        basis is None for basis in [basis2, basis3, basis4]
    ):
        basis2 = basis3 = basis4 = basis1
    return basis1, basis2, basis3, basis4


def nuclear_repulsion(system):
    if system.use_gaussian_charges:
        ints2c = ints.coulomb_2c(
            system.gaussian_charge_basis, system.gaussian_charge_basis
        )
        ints2c -= np.diag(np.diag(ints2c))  # remove self-interaction terms
        enuc = 0.5 * np.einsum(
            "IJ,I,J->", ints2c, system.atomic_charges, system.atomic_charges
        )
        return enuc
    else:
        return ints.nuclear_repulsion(system.atoms)


def overlap(system, basis1=None, basis2=None):
    basis1, basis2 = parse_basis_args_1e(system, basis1, basis2)
    return ints.overlap(basis1, basis2)


def kinetic(system, basis1=None, basis2=None):
    basis1, basis2 = parse_basis_args_1e(system, basis1, basis2)
    return ints.kinetic(basis1, basis2)


def nuclear(system, basis1=None, basis2=None):
    basis1, basis2 = parse_basis_args_1e(system, basis1, basis2)
    if system.use_gaussian_charges:
        int3c = ints.coulomb_3c(system.gaussian_charge_basis, basis1, basis2)
        V = -np.einsum("Zpq,Z->pq", int3c, system.atomic_charges)
        return V
    else:
        return ints.nuclear(basis1, basis2, system.atoms)


def emultipole1(system, basis1=None, basis2=None, origin=None):
    basis1, basis2 = parse_basis_args_1e(system, basis1, basis2)
    if origin is None:
        origin = [0.0, 0.0, 0.0]
    return ints.emultipole1(basis1, basis2, origin)


def emultipole2(system, basis1=None, basis2=None, origin=None):
    basis1, basis2 = parse_basis_args_1e(system, basis1, basis2)
    if origin is None:
        origin = [0.0, 0.0, 0.0]
    return ints.emultipole2(basis1, basis2, origin)


def emultipole3(system, basis1=None, basis2=None, origin=None):
    basis1, basis2 = parse_basis_args_1e(system, basis1, basis2)
    if origin is None:
        origin = [0.0, 0.0, 0.0]
    return ints.emultipole3(basis1, basis2, origin)


def opVop(system, basis1=None, basis2=None):
    basis1, basis2 = parse_basis_args_1e(system, basis1, basis2)
    return ints.opVop(basis1, basis2, system.atoms)


def erf_nuclear(system, omega, basis1=None, basis2=None):
    basis1, basis2 = parse_basis_args_1e(system, basis1, basis2)
    return ints.erf_nuclear(basis1, basis2, (omega, system.atoms))


def erfc_nuclear(system, omega, basis1=None, basis2=None):
    basis1, basis2 = parse_basis_args_1e(system, basis1, basis2)
    return ints.erfc_nuclear(basis1, basis2, (omega, system.atoms))


def coulomb_4c(system, basis1=None, basis2=None, basis3=None, basis4=None):
    basis1, basis2, basis3, basis4 = parse_basis_args_4c2e(
        system, basis1, basis2, basis3, basis4
    )
    return ints.coulomb_4c(basis1, basis2, basis3, basis4)


def coulomb_3c(system, basis1=None, basis2=None, basis3=None):
    basis1, basis2, basis3 = parse_basis_args_3c2e(system, basis1, basis2, basis3)
    return ints.coulomb_3c(basis1, basis2, basis3)


def coulomb_2c(system, basis1=None, basis2=None):
    basis1, basis2 = parse_basis_args_2c2e(system, basis1, basis2)
    return ints.coulomb_2c(basis1, basis2)


def erf_coulomb_3c(system, omega, basis1=None, basis2=None, basis3=None):
    basis1, basis2, basis3 = parse_basis_args_3c2e(system, basis1, basis2, basis3)
    return ints.erf_coulomb_3c(basis1, basis2, basis3, omega)


def erf_coulomb_2c(system, omega, basis1=None, basis2=None):
    basis1, basis2 = parse_basis_args_2c2e(system, basis1, basis2)
    return ints.erf_coulomb_2c(basis1, basis2, omega)


def erfc_coulomb_3c(system, omega, basis1=None, basis2=None, basis3=None):
    basis1, basis2, basis3 = parse_basis_args_3c2e(system, basis1, basis2, basis3)
    return ints.erfc_coulomb_3c(basis1, basis2, basis3, omega)


def erfc_coulomb_2c(system, omega, basis1=None, basis2=None):
    basis1, basis2 = parse_basis_args_2c2e(system, basis1, basis2)
    return ints.erfc_coulomb_2c(basis1, basis2, omega)
