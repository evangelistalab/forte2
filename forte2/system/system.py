import scipy as sp
from dataclasses import dataclass, field

import forte2
from forte2.helpers import logger
from forte2.helpers.matrix_functions import invsqrt_matrix, canonical_orth
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
    # If the min/max eigenvalue of the overlap matrix falls below
    # this trigger, linear dependency will be removed
    linear_dep_trigger: float = 1e-10
    # This is the threshold below which the eigenvalues of
    # the overlap matrix will be removed
    ortho_thresh: float = 1e-8

    # Non-init attributes
    Zsum: float = field(init=False, default=None)
    nbf: int = field(init=False, default=None)
    nmo: int = field(init=False, default=None)
    naux: int = field(init=False, default=None)
    nminao: int = field(init=False, default=None)
    Xorth: NDArray = field(init=False, default=None)

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
        self.nbf = self.basis.size
        self.naux = self.auxiliary_basis.size if self.auxiliary_basis else 0
        self.nminao = self.minao_basis.size if self.minao_basis else 0

        self._init_x2c()
        self.check_linear_dependencies()

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

    def check_linear_dependencies(self):
        S = self.ints_overlap()
        e, _ = np.linalg.eigh(S)
        self._eigh = sp.linalg.eigh
        self.nmo = self.nbf
        if min(e) / max(e) < self.linear_dep_trigger:
            logger.log_warning(f"Linear dependencies detected in overlap matrix S!")
            logger.log_debug(
                f"Max eigenvalue: {np.max(e):.2e}. \n"
                f"Min eigenvalue: {np.min(e):.2e}. \n"
                f"Condition number: {max(e)/min(e):.2e}. \n"
                f"Removing linear dependencies with threshold {self.ortho_thresh:.2e}."
            )
            self.nmo -= np.sum(e < self.ortho_thresh)
            if self.nmo < self.nbf:
                logger.log_warning(
                    f"Reduced number of basis functions from {self.nbf} to {self.nmo} due to linear dependencies."
                )
            else:
                logger.log_warning(
                    f"Linear dependencies detected, but no basis functions were removed. Consider changing linear_dep_trigger or ortho_thresh."
                )
            self.Xorth = canonical_orth(S, tol=self.ortho_thresh)
        else:
            # no linear dependencies, use symmetric orthogonalization
            self.Xorth = invsqrt_matrix(S, tol=1e-13)


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
        self.nbf = self.hcore.shape[0]
        self.nmo = self.nbf
        self.naux = 0
        self.Xorth = invsqrt_matrix(self.ints_overlap(), tol=1e-13)

    def ints_overlap(self):
        return self.overlap

    def ints_hcore(self):
        return self.hcore

    def nuclear_repulsion_energy(self):
        return self.nuclear_repulsion


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

        super().__post_init__()
