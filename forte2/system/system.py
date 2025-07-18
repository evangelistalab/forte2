import scipy as sp
from dataclasses import dataclass, field

import forte2
from forte2.helpers import logger
from forte2.helpers.matrix_functions import invsqrt_matrix, canonical_orth
from forte2.x2c import get_hcore_x2c
from .build_basis import build_basis
from .parse_xyz import parse_xyz
from .atom_data import ATOM_DATA, Z_TO_ATOM_SYMBOL

import numpy as np
from numpy.typing import NDArray

from copy import deepcopy


@dataclass
class System:
    """
    A class to represent a quantum chemical system.

    Parameters
    ----------
    xyz : str
        A XYZ string representing the atomic coordinates.
    basis_set : str | dict
        The basis set to be used, either as a string (e.g. "cc-pvdz") or as a dictionary
        assigning potentially different basis sets to each atom (e.g. {"H": "sto-3g", "O": "cc-pvdz"}).
    auxiliary_basis_set : str | dict, optional
        The auxiliary basis set, either as a string or a dictionary (see `basis`).
    auxiliary_basis_set_corr : str | dict, optional
        A separate auxiliary basis set for all correlated calculations, either as a string or a dictionary (see `basis`).
    minao_basis : str | dict, optional, default="cc-pvtz-minao"
        The minimal atomic orbital basis set, used in IAO calculations, either as a string or a dictionary (see `basis`).
    x2c_type : str, optional
        The type of X2C transformation to be used. Options are "sf" for scalar
        relativistic effects or "so" for spin-orbit coupling. If None, no X2C transformation is applied.
    unit : str, optional, default="angstrom"
        The unit for the atomic coordinates. Can be "angstrom" or "bohr".
    linear_dep_trigger : float, optional, default=1e-10
        The trigger for detecting linear dependencies in the overlap matrix. If the ratio of the minimum to
        maximum eigenvalue of the overlap matrix falls below this value, linear dependencies will be removed.
    ortho_thresh : float, optional, default=1e-8
        Linear combinations of AO basis functions with overlap eigenvalues below this threshold will be removed
        during orthogonalization.

    Attributes
    ----------
    atoms : list[tuple[float, tuple[float, float, float]]]
        A list of tuples representing the atoms in the system, where each tuple contains the atomic charge and a tuple of coordinates (x, y, z).
    natoms : int
        The number of atoms in the system.
    atomic_charges : NDArray
        An array of atomic charges, shape (N,) where N is the number of atoms.
    atomic_positions : NDArray
        An array of atomic positions, shape (N, 3) where N is the number of atoms.
    center_of_mass : NDArray
        The center of mass of the system, shape (3,).
    centroid : NDArray
        The centroid (geometric center) of the system, shape (3,).
    nuclear_repulsion : float
        The nuclear repulsion energy of the system.
    atom_counts : dict[int : int]
        A dictionary mapping atomic numbers to their numbers in the system.
    atom_to_center : dict[int : list[int]]
        A dictionary mapping atomic numbers to a list of (0-based) indices of atoms of that type in the system.
    basis : forte2.ints.Basis
        The basis set for the system, built from the provided `basis_set`.
    auxiliary_basis : forte2.ints.Basis
        The auxiliary basis set for the system, built from the provided `auxiliary_basis_set`.
    auxiliary_basis_set_corr : forte2.ints.Basis
        The auxiliary basis set for correlated calculations, built from the provided `auxiliary_basis_set_corr`.
    minao_basis : forte2.ints.Basis
        The minimal atomic orbital basis set, built from the provided `minao_basis_set`.
    Zsum : float
        The total nuclear charge of the system, calculated as the sum of atomic charges.
    nbf : int
        The number of basis functions in the system.
    nmo : int
        The number of linearly independent combinations of atomic orbitals in the system.
    naux : int
        The number of auxiliary basis functions in the system.
    nminao : int
        The number of minimal atomic orbital basis functions in the system.
    Xorth : NDArray
        The orthogonalization matrix for the basis functions.

    """

    xyz: str
    basis_set: str | dict
    auxiliary_basis_set: str | dict = None
    auxiliary_basis_set_corr: str | dict = None
    minao_basis_set: str | dict = "cc-pvtz-minao"
    x2c_type: str = None
    unit: str = "angstrom"
    linear_dep_trigger: float = 1e-10
    ortho_thresh: float = 1e-8

    ### Non-init attributes
    atoms: list[tuple[float, tuple[float, float, float]]] = field(
        init=False, default=None
    )
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
        self.natoms = len(self.atoms)
        self.atomic_charges = np.array([atom[0] for atom in self.atoms])
        self.atomic_masses = np.array(
            [ATOM_DATA[Z]["mass"] for Z in self.atomic_charges]
        )
        self.atomic_positions = np.array([atom[1] for atom in self.atoms])
        self.centroid = np.mean(self.atomic_positions, axis=0)
        self.nuclear_repulsion = forte2.ints.nuclear_repulsion(self.atoms)

        self.center_of_mass = np.einsum(
            "a,ax->x", self.atomic_masses, self.atomic_positions
        ) / np.sum(self.atomic_masses)

        self.atom_counts = {}
        for atom in self.atoms:
            if atom[0] not in self.atom_counts:
                self.atom_counts[atom[0]] = 0
            self.atom_counts[atom[0]] += 1

        self.atom_to_center = {}
        for i, atom in enumerate(self.atoms):
            if atom[0] not in self.atom_to_center:
                self.atom_to_center[atom[0]] = []
            self.atom_to_center[atom[0]].append(i)

        self.basis = build_basis(self.basis_set, self.atoms)
        self.auxiliary_basis = (
            build_basis(self.auxiliary_basis_set, self.atoms)
            if self.auxiliary_basis_set is not None
            else None
        )
        if self.auxiliary_basis_set_corr is not None:
            logger.log_warning(f"Using a separate auxiliary basis is not recommended!")
            self.auxiliary_basis_set_corr = build_basis(
                self.auxiliary_basis_set_corr, self.atoms
            )
        else:
            self.auxiliary_basis_set_corr = self.auxiliary_basis
        self.minao_basis = (
            build_basis(self.minao_basis_set, self.atoms)
            if self.minao_basis_set is not None
            else None
        )
        logger.log_info1(
            f"Parsed {len(self.atoms)} atoms with basis set of {self.basis.size} functions."
        )

        self.Zsum = np.sum([x[0] for x in self.atoms])
        self.nbf = self.basis.size
        self.naux = self.auxiliary_basis.size if self.auxiliary_basis else 0
        self.nminao = self.minao_basis.size if self.minao_basis else 0

        self._init_x2c()
        self._check_linear_dependencies()

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
        return f"System(atoms={self.atoms}, basis_set={self.basis}, auxiliary_basis_set={self.auxiliary_basis})"

    def decontract(self):
        """
        Decontract the basis set.

        Returns
        -------
        forte2.ints.Basis
            Decontracted basis set.
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

        Returns
        -------
        NDArray
            Overlap integrals matrix.
        """
        return forte2.ints.overlap(self.basis)

    def ints_hcore(self):
        """
        Return the core Hamiltonian integrals for the system.

        Returns
        -------
        NDArray
            Core Hamiltonian integrals matrix.
        """
        T = forte2.ints.kinetic(self.basis)
        V = forte2.ints.nuclear(self.basis, self.atoms)
        return T + V

    def nuclear_dipole(self, origin=None, unit="debye"):
        """
        Calculate the nuclear dipole moment of the system.

        Args
        ----
        origin : tuple[float, float, float], optional
            The origin point for the dipole calculation. If None, the center of mass of the system is used.
        unit : str, optional, default="debye"
            The unit for the dipole moment. Can be "debye" or "au".

        Returns
        -------
        NDArray
            Nuclear dipole moment vector, shape (3,).
        """
        assert unit in ["debye", "au"], f"Invalid unit: {unit}. Use 'debye' or 'au'."
        charges = self.atomic_charges
        positions = self.atomic_positions
        if origin is not None:
            assert len(origin) == 3, "Origin must be a 3-element vector."
            positions -= np.array(origin)[np.newaxis, :]
        conversion_factor = (
            1.0 / forte2.atom_data.DEBYE_TO_AU if unit == "debye" else 1.0
        )
        return np.einsum("a,ax->x", charges, positions) * conversion_factor

    def nuclear_quadrupole(self, origin=None, unit="debye"):
        """
        Calculate the nuclear quadrupole moment of the system.

        Args
        ----
        origin : tuple[float, float, float], optional
            The origin point for the quadrupole calculation. If None, the center of mass of the system is used.
        unit : str, optional, default="debye"
            The unit for the quadrupole moment. Can be "debye" or "au".

        Returns
        -------
        NDArray
            Nuclear quadrupole moment tensor, shape (3, 3).
        """
        assert unit in ["debye", "au"], f"Invalid unit: {unit}. Use 'debye' or 'au'."
        charges = self.atomic_charges
        positions = self.atomic_positions
        if origin is not None:
            assert len(origin) == 3, "Origin must be a 3-element vector."
            positions -= np.array(origin)[np.newaxis, :]
        nuc_quad = np.einsum("a,ax,ay->xy", charges, positions, positions)
        nuc_quad = 0.5 * (3 * nuc_quad - np.eye(3) * nuc_quad.trace())
        conversion_factor = (
            1.0 / forte2.atom_data.DEBYE_ANGSTROM_TO_AU if unit == "debye" else 1.0
        )
        return nuc_quad * conversion_factor

    def _check_linear_dependencies(self):
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
    r"""
    A 1D Hubbard model system.

    .. math::
        \hat{H} =
        -t \sum_i\sum_{\sigma\in\{\alpha,\beta\}}(\hat{a}_{i,\sigma}^{\dagger} \hat{a}_{i+1,\sigma}
            + \hat{a}_{i+1,\sigma}^{\dagger} \hat{a}_{i,\sigma})
        + U \sum_i \hat{n}_{i,\alpha} \hat{n}_{i,\beta},

    where :math:`\hat{a}_{i,\sigma}^{\dagger}` and :math:`\hat{a}_{i,\sigma}` and :math:`\hat{n}_{i,\sigma} = \hat{a}_{i,\sigma}^{\dagger} \hat{a}_{i,\sigma}` are the creation, annihilation, and number operator for spin :math:`\sigma` at site :math:`i`.

    Parameters
    ----------
        t : float
            Hopping parameter.
        U : float
            On-site interaction strength.
        nsites : int
            Number of sites in the 1D chain.
        pbc : bool, optional, default=False
            Whether to apply 1D periodic boundary conditions.
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

@dataclass
class HubbardModel2D(ModelSystem):
    """
    A 2D Hubbard model system.

    Parameters
    ----------
        t : float
            Hopping parameter.
        U : float
            On-site interaction strength.
        nsites_x : int
            Number of sites in the x-direction.
        nsites_y : int
            Number of sites in the y-direction.
        pbc_x : bool, optional, default=False
            Whether to apply periodic boundary conditions in the x-direction.
        pbc_y : bool, optional, default=False
            Whether to apply periodic boundary conditions in the y-direction.
    """

    t: float
    U: float
    nsites_x: int
    nsites_y: int
    pbc_x: bool = False
    pbc_y: bool = False

    def __post_init__(self):
        self.nsites = self.nsites_x * self.nsites_y

        # helper to map 2D coordinates to 1D index
        def site_index(i, j):
            return i * self.nsites_y + j
        
        # Hopping 
        self.hcore = np.zeros((self.nsites, self.nsites))
        for i in range(self.nsites_x):
            for j in range(self.nsites_y):
                idx = site_index(i, j)
                if i < self.nsites_x - 1:
                    right_idx = site_index(i + 1, j)
                    self.hcore[idx, right_idx] = self.hcore[right_idx, idx] = -self.t
                if j < self.nsites_y - 1:
                    up_idx = site_index(i, j + 1)
                    self.hcore[idx, up_idx] = self.hcore[up_idx, idx] = -self.t

        # periodic boundary conditions, x-direction
        if self.pbc_x:
            for j in range(self.nsites_y):
                left_idx = site_index(self.nsites_x - 1, j)
                right_idx = site_index(0, j)
                self.hcore[left_idx, right_idx] = self.hcore[right_idx, left_idx] = -self.t

        # periodic boundary conditions, y-direction
        if self.pbc_y:
            for i in range(self.nsites_x):
                down_idx = site_index(i, self.nsites_y - 1)
                up_idx = site_index(i, 0)
                self.hcore[down_idx, up_idx] = self.hcore[up_idx, down_idx] = -self.t

        # Overlap
        self.overlap = np.eye(self.nsites)
        
        # On-site interaction
        self.eri = np.zeros((self.nsites,) * 4)
        for i in range(self.nsites):
            self.eri[i, i, i, i] = self.U

        super().__post_init__()