from dataclasses import dataclass, field

import forte2
from forte2.helpers import logger
from forte2.x2c import get_hcore_x2c
from .build_basis import build_basis
from .parse_xyz import parse_xyz
from .atom_data import ATOM_DATA

import numpy as np
from numpy.typing import NDArray

from copy import deepcopy


@dataclass
class System:
    xyz: str
    basis: str | dict
    auxiliary_basis: str | dict = None
    auxiliary_basis_corr: str | dict = None
    atoms: list[tuple[float, tuple[float, float, float]]] = None
    minao_basis: str = None
    x2c_type: str = None
    unit: str = "angstrom"

    def __post_init__(self):
        assert self.unit in [
            "angstrom",
            "bohr",
        ], f"Invalid unit: {self.unit}. Use 'angstrom' or 'bohr'."
        self.atoms = parse_xyz(self.xyz, self.unit)
        self.basis = build_basis(self.basis, self.atoms)
        self.auxiliary_basis = (
            build_basis(self.auxiliary_basis, self.atoms)
            if self.auxiliary_basis is not None
            else None
        )
        if self.auxiliary_basis_corr is not None:
            logger.log_warning(f"Using a separate auxiliary basis is not recommended!")
            self.auxiliary_basis_corr = build_basis(
                self.auxiliary_basis_corr, self.atoms
            )
        else:
            self.auxiliary_basis_corr = self.auxiliary_basis
        self.minao_basis = (
            build_basis("cc-pvtz-minao", self.atoms)
            if self.minao_basis is not None
            else None
        )
        print(
            f"Parsed {len(self.atoms)} atoms with basis set of {self.basis.size} functions."
        )

        self.Zsum = np.sum([x[0] for x in self.atoms])

        self._init_x2c()

    def _init_x2c(self):
        if self.x2c_type is not None:
            assert self.x2c_type in [
                "sf",
                "so",
            ], f"x2c_type {self.x2c_type} is not supported. Use None, 'sf' or 'so'."
        else:
            return
        if self.x2c_type == "sf":
            self.ints_hcore = lambda: get_hcore_x2c(self, x2c_type="sf")
        elif self.x2c_type == "so":
            self.ints_hcore = lambda: get_hcore_x2c(self, x2c_type="so")

    def __repr__(self):
        return f"System(atoms={self.atoms}, basis={self.basis}, auxiliary_basis={self.auxiliary_basis})"

    def nbf(self):
        """
        Get the number of basis functions in the system.

        Returns:
            int: Number of basis functions.
        """
        return self.basis.size

    def naux(self):
        """
        Get the number of auxiliary basis functions in the system.

        Returns:
            int: Number of auxiliary basis functions.
        """
        return self.auxiliary_basis.size if self.auxiliary_basis else 0

    def nminao(self):
        """
        Get the number of minao basis functions in the system.

        Returns:
            int: Number of minao basis functions.
        """
        return self.minao_basis.size if self.minao_basis else 0

    def decontract(self):
        """
        Decontract the basis set.

        Returns:
            forte2.ints.Basis: Decontracted basis set.
        """
        return build_basis(
            self.basis.name,
            self.atoms,
            embed_normalization_into_coefficients=True,
            decontract=True,
        )

    def ints_overlap(self):
        """
        Return the overlap integrals for the system.
        Returns:
            NDArray: Overlap integrals matrix.
        """
        return forte2.ints.overlap(self.basis)

    def ints_hcore(self):
        """
        Return the core Hamiltonian integrals for the system.
        Returns:
            NDArray: Core Hamiltonian integrals matrix.
        """
        T = forte2.ints.kinetic(self.basis)
        V = forte2.ints.nuclear(self.basis, self.atoms)
        return T + V

    def nuclear_repulsion_energy(self):
        """
        Return the nuclear repulsion energy for the system.
        Returns:
            float: Nuclear repulsion energy.
        """
        return forte2.ints.nuclear_repulsion(self.atoms)

    def atomic_charges(self):
        """
        Return the atomic charges for the system.
        Returns:
            NDArray: Array of atomic charges, shape (N,) where N is the number of atoms.
        """
        return np.array([atom[0] for atom in self.atoms])

    def atomic_masses(self):
        """
        Return the average atomic masses for the system.
        Returns:
            NDArray: Array of atomic masses, shape (N,) where N is the number of atoms.
        """
        return np.array([ATOM_DATA[atom[0]]["mass"] for atom in self.atoms])

    def atomic_positions(self):
        """
        Return the atomic positions (in bohr) for the system.
        Returns:
            NDArray: Array of atomic positions, shape (N, 3) where N is the number of atoms.
        """
        return np.array([atom[1] for atom in self.atoms])

    def center_of_mass(self):
        """
        Calculate the center of mass of the system.
        Uses average atomic masses for the calculation.

        Returns:
            tuple: Center of mass coordinates (x, y, z).
        """
        masses = self.atomic_masses()
        positions = np.array([atom[1] for atom in self.atoms])
        return np.einsum("a,ax->x", masses, positions) / np.sum(masses)

    def nuclear_dipole(self, origin=None, unit="debye"):
        assert unit in ["debye", "au"], f"Invalid unit: {unit}. Use 'debye' or 'au'."
        charges = self.atomic_charges()
        positions = self.atomic_positions()
        if origin is not None:
            assert len(origin) == 3, "Origin must be a 3-element vector."
            positions -= np.array(origin)[np.newaxis, :]
        conversion_factor = (
            1.0 / forte2.atom_data.DEBYE_TO_AU if unit == "debye" else 1.0
        )
        return np.einsum("a,ax->x", charges, positions) * conversion_factor

    def nuclear_quadrupole(self, origin=None, unit="debye"):
        assert unit in ["debye", "au"], f"Invalid unit: {unit}. Use 'debye' or 'au'."
        charges = self.atomic_charges()
        positions = self.atomic_positions()
        if origin is not None:
            assert len(origin) == 3, "Origin must be a 3-element vector."
            positions -= np.array(origin)[np.newaxis, :]
        nuc_quad = np.einsum("a,ax,ay->xy", charges, positions, positions)
        nuc_quad = 0.5 * (3 * nuc_quad - np.eye(3) * nuc_quad.trace())
        conversion_factor = (
            1.0 / forte2.atom_data.DEBYE_ANGSTROM_TO_AU if unit == "debye" else 1.0
        )
        return nuc_quad * conversion_factor


@dataclass
class ModelSystem:
    """
    A base class for model systems.
    One needs to specify the overlap, hcore, and eri tensors.
    The number of electrons needs be set by setting charge to -nel at runtime.
    """

    overlap: NDArray = field(init=False)
    hcore: NDArray = field(init=False)
    eri: NDArray = field(init=False)

    def __post_init__(self):
        self.Zsum = 0  # total nuclear charge, here set to zero, so charge can be set to -nel later
        self.x2c_type = None
        self.nuclear_repulsion = 0.0

    def ints_overlap(self):
        return self.overlap

    def ints_hcore(self):
        return self.hcore

    def nuclear_repulsion_energy(self):
        return self.nuclear_repulsion

    def nbf(self):
        return self.hcore.shape[0]

    def naux(self):
        return 0


@dataclass
class HubbardModel1D(ModelSystem):
    """
    A 1D Hubbard model system.
    H = -t * (c_{i,sigma}^+ c_{i+1,sigma} + c_{i+1,sigma}^+ c_{i,sigma}) + U * n_{i,alpha} n_{i,beta}
    Attributes:
        t (float): Hopping parameter.
        U (float): On-site interaction strength.
        nsites (int): Number of sites in the 1D chain.
        pbc (bool): Whether to apply 1D periodic boundary conditions.
    """

    t: float
    U: float
    nsites: int
    pbc: bool = False

    def __post_init__(self):
        super().__post_init__()

        self.hcore = np.zeros((self.nsites, self.nsites))
        for i in range(self.nsites - 1):
            self.hcore[i, i + 1] = self.hcore[i + 1, i] = -self.t
        # periodic boundary conditions
        if self.pbc:
            self.hcore[0, self.nsites - 1] = self.hcore[self.nsites - 1, 0] = -self.t
        self.overlap = np.eye(self.nsites)

        self.eri = np.zeros((self.nsites,) * 4)
        for i in range(self.nsites):
            self.eri[i, i, i, i] = self.U
