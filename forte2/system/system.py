from dataclasses import dataclass, field
import numpy as np
from numpy.typing import NDArray

from forte2 import integrals
from forte2.data import DEBYE_TO_AU, DEBYE_ANGSTROM_TO_AU, Z_TO_ATOM_SYMBOL
from forte2.helpers import logger, invsqrt_matrix, canonical_orth, block_diag_2x2
from forte2.x2c import X2CHelper
from forte2.jkbuilder import FockBuilder
from .build_basis import build_basis
from .geom_utils import GeometryHelper, parse_geometry


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
    x2c_type : str, optional, default=None
        The type of X2C transformation to be used. Options are "sf" for scalar
        relativistic effects or "so" for spin-orbit coupling. If None, no X2C transformation is applied.
    snso_type : str, optional, default="row-dependent"
        The type of screened nuclear spin-orbit coupling scaling scheme to use.
        Only relevant if `x2c_type` is "so".
        Options are None, "boettger", "dc", "dcb", or "row-dependent".
    unit : str, optional, default="angstrom"
        The unit for the atomic coordinates. Can be "angstrom" or "bohr".
    ortho_thresh : float, optional, default=1e-8
        Linear combinations of AO basis functions with overlap eigenvalues below this threshold will be removed
        during orthogonalization.
    cholesky_tei : bool, optional, default=False
        If True, auxiliary basis sets (if any) will be disregarded, and the B tensor will be built using the Cholesky decomposition of the 4D ERI tensor instead.
    cholesky_tol : float, optional, default=1e-6
        The tolerance for the Cholesky decomposition of the 4D ERI tensor. Only used if `cholesky_tei` is True.
    symmetry : bool, optional, default=False
        Whether to automatically detect the largest Abelian point group symmetry of the molecule.
        This will center the molecule at its center of mass and reorient it along its principal axes of inertia.
    symmetry_tol : float, optional, default=1e-4
        The tolerance for detecting symmetry.
    use_gaussian_charges : bool, optional, default=False
        Whether to use Gaussian nuclear charge distributions instead of point charges.

    Attributes
    ----------
    atoms : list[list[float, list[float, float, float]]]
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
    basis : ints.Basis
        The basis set for the system, built from the provided `basis_set`.
    auxiliary_basis : ints.Basis
        The auxiliary basis set for the system, built from the provided `auxiliary_basis_set`.
    auxiliary_basis_corr : ints.Basis
        The auxiliary basis set for correlated calculations, built from the provided `auxiliary_basis_set_corr`.
    minao_basis : ints.Basis
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
    snso_type: str = "row-dependent"
    unit: str = "angstrom"
    ortho_thresh: float = 1e-8
    cholesky_tei: bool = False
    cholesky_tol: float = 1e-6
    symmetry: bool = False
    symmetry_tol: float = 1e-4
    use_gaussian_charges: bool = False

    ### Non-init attributes
    atoms: list[list[float, list[float, float, float]]] = field(
        init=False, default=None
    )
    Zsum: float = field(init=False, default=None)
    nbf: int = field(init=False, default=None)
    nmo: int = field(init=False, default=None)
    naux: int = field(init=False, default=None)
    nminao: int = field(init=False, default=None)
    Xorth: NDArray = field(init=False, default=None)
    two_component: bool = field(init=False, default=False)

    def __post_init__(self):
        assert self.unit in [
            "angstrom",
            "bohr",
        ], f"Invalid unit: {self.unit}. Use 'angstrom' or 'bohr'."

        self._init_geometry()
        self._init_basis()
        self.nuclear_repulsion = integrals.nuclear_repulsion(self)
        self._init_x2c()
        _S = integrals.overlap(self)
        self.Xorth, _, info = canonical_orth(_S, self.ortho_thresh, print_info=True)
        self.nmo = info["n_kept"]
        self.fock_builder = FockBuilder(self)
        # The B tensors here are lazily evaluated, so no overhead if not used
        if self.auxiliary_basis_set_corr is not None:
            logger.log_warning(
                "Building separate auxiliary basis for correlated calculations!"
            )
            self.fock_builder_corr = FockBuilder(self, use_aux_corr=True)
        else:
            self.fock_builder_corr = self.fock_builder

    def _init_geometry(self):
        self.atoms = parse_geometry(self.xyz, self.unit)
        self.geom_helper = GeometryHelper(
            self.atoms, symmetry=self.symmetry, tol=self.symmetry_tol
        )
        self.atoms = self.geom_helper.atoms
        self.Zsum = self.geom_helper.Zsum
        self.natoms = self.geom_helper.natoms
        self.atomic_charges = self.geom_helper.atomic_charges
        self.atomic_masses = self.geom_helper.atomic_masses
        self.atomic_positions = self.geom_helper.atomic_positions
        self.centroid = self.geom_helper.centroid
        self.center_of_mass = self.geom_helper.center_of_mass
        self.atom_counts = self.geom_helper.atom_counts
        self.atom_to_center = self.geom_helper.atom_to_center
        self.prin_atomic_positions = self.geom_helper.prin_atomic_positions
        self.point_group = self.geom_helper.point_group

        logger.log_info1("Principal Atomic Positions (a.u.):")
        for i in range(self.natoms):
            logger.log_info1(
                f"   {Z_TO_ATOM_SYMBOL[self.atoms[i][0]]}   {self.prin_atomic_positions[i, 0]:<.8f}   {self.prin_atomic_positions[i, 1]:<.8f}   {self.prin_atomic_positions[i, 2]:<.8f}"
            )

    def _init_basis(self):
        self.basis = build_basis(self.basis_set, self.geom_helper)
        logger.log_info1(
            f"Parsed {self.natoms} atoms with basis set of {self.basis.size} functions."
        )

        if not self.cholesky_tei:
            if self.auxiliary_basis_set is not None:
                self.auxiliary_basis = build_basis(
                    self.auxiliary_basis_set, self.geom_helper
                )
            else:
                self.auxiliary_basis = None

            if self.auxiliary_basis_set_corr is not None:
                logger.log_warning(
                    "Using a separate auxiliary basis is not recommended!"
                )
                self.auxiliary_basis_corr = build_basis(
                    self.auxiliary_basis_set_corr,
                    self.geom_helper,
                )
            else:
                self.auxiliary_basis_corr = self.auxiliary_basis
        else:
            if (
                self.auxiliary_basis_set is not None
                or self.auxiliary_basis_set_corr is not None
            ):
                logger.log_warning(
                    "Ignoring provided auxiliary basis sets since cholesky_tei=True!"
                )
            self.auxiliary_basis = None
            self.auxiliary_basis_corr = None

        if self.minao_basis_set is not None:
            self.minao_basis = build_basis(self.minao_basis_set, self.geom_helper)
        else:
            self.minao_basis = None

        self.nbf = self.basis.size
        self.naux = self.auxiliary_basis.size if self.auxiliary_basis else None
        self.nminao = self.minao_basis.size if self.minao_basis else None

        self.gaussian_charge_basis = None
        if self.use_gaussian_charges:
            self.gaussian_charge_basis = build_basis(
                "gaussian_charge",
                self.geom_helper,
                embed_normalization_into_coefficients=False,
            )

    def _init_x2c(self):
        if self.x2c_type is not None:
            assert self.x2c_type in [
                "sf",
                "so",
            ], f"x2c_type {self.x2c_type} is not supported. Use None, 'sf' or 'so'."
            self.x2c_helper = X2CHelper(self, ortho_thresh=self.ortho_thresh)
        else:
            return
        if self.x2c_type == "so":
            self.two_component = True

    def __repr__(self):
        return f"System(atoms={self.atoms}, basis_set={self.basis}, auxiliary_basis_set={self.auxiliary_basis})"

    def ints_overlap(self):
        """
        Return the overlap integrals for the system.

        Returns
        -------
        NDArray
            Overlap integrals matrix.
        """
        S = integrals.overlap(self)
        if self.two_component:
            S = block_diag_2x2(S)
        return S

    def ints_hcore(self):
        """
        Return the core Hamiltonian integrals for the system.

        Returns
        -------
        NDArray
            Core Hamiltonian integrals matrix.
        """
        if self.x2c_type in ["sf", "so"]:
            H = self.x2c_helper.hcore_x2c
        else:
            T = integrals.kinetic(self)
            V = integrals.nuclear(self)
            H = T + V
        if self.x2c_type in [None, "sf"] and self.two_component:
            H = block_diag_2x2(H)
        return H

    def get_Xorth(self):
        """
        Return the orthonormalization matrix for the basis functions.

        Returns
        -------
        NDArray
            Orthonormalization matrix.
        """
        if self.two_component:
            return block_diag_2x2(self.Xorth)
        else:
            return self.Xorth

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
        conversion_factor = 1.0 / DEBYE_TO_AU if unit == "debye" else 1.0
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
        conversion_factor = 1.0 / DEBYE_ANGSTROM_TO_AU if unit == "debye" else 1.0
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

    two_component: bool = field(init=False, default=False)

    def __post_init__(self):
        self.Zsum = 0  # total nuclear charge, here set to zero, so charge can be set to -nel later
        self.x2c_type = None
        self.nuclear_repulsion = 0.0
        self.nbf = self.hcore.shape[0]
        self.nmo = self.nbf
        self.naux = 0
        self.Xorth = invsqrt_matrix(self.ints_overlap(), tol=1e-13)
        self.symmetry = False
        self.point_group = "C1"
        self.fock_builder = FockBuilder(self)
        self.fock_builder_corr = self.fock_builder

    def ints_overlap(self):
        return self.overlap

    def ints_hcore(self):
        return self.hcore

    def nuclear_repulsion_energy(self):
        return self.nuclear_repulsion

    def get_Xorth(self):
        return self.Xorth


@dataclass
class HubbardModel(ModelSystem):
    r"""
    A Hubbard model system.
    For a 1D chain, the Hamiltonian is given by:

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
    nsites : int | tuple[int, int]
        Number of sites in the 1D chain, or a tuple for number of sites in the x and y directions.
    pbc : bool | tuple[bool, bool], optional, default=False
        Whether to apply 1D periodic boundary conditions. If a bool is provided but a 2D system is specified, it will be applied in both directions.
    """

    t: float
    U: float
    nsites: int | tuple[int, int]
    pbc: bool | tuple[bool, bool] = False

    def __post_init__(self):
        if isinstance(self.nsites, int):
            self.ndim = 1
        elif isinstance(self.nsites, tuple) and len(self.nsites) == 2:
            self.ndim = 2
        else:
            raise ValueError(
                "nsites must be an int for 1D or a tuple of two ints for 2D."
            )

        if self.ndim == 1:
            self._init_1d_hubbard_ints()
        elif self.ndim == 2:
            self._init_2d_hubbard_ints()

        super().__post_init__()

    def _init_1d_hubbard_ints(self):
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

    def _init_2d_hubbard_ints(self):
        nsites_x, nsites_y = self.nsites
        nsites_flat = nsites_x * nsites_y

        if isinstance(self.pbc, bool):
            pbc_x = pbc_y = self.pbc
        elif isinstance(self.pbc, tuple) and len(self.pbc) == 2:
            pbc_x, pbc_y = self.pbc
        else:
            raise ValueError(
                "pbc must be a bool or a tuple of two bools for 2D systems."
            )

        # helper to map 2D coordinates to 1D index
        def site_index(i, j):
            return i * nsites_y + j

        # Hopping
        self.hcore = np.zeros((nsites_flat, nsites_flat))
        for i in range(nsites_x):
            for j in range(nsites_y):
                idx = site_index(i, j)
                if i < nsites_x - 1:
                    right_idx = site_index(i + 1, j)
                    self.hcore[idx, right_idx] = self.hcore[right_idx, idx] = -self.t
                if j < nsites_y - 1:
                    up_idx = site_index(i, j + 1)
                    self.hcore[idx, up_idx] = self.hcore[up_idx, idx] = -self.t

        # periodic boundary conditions, x-direction
        if pbc_x:
            for j in range(nsites_y):
                left_idx = site_index(nsites_x - 1, j)
                right_idx = site_index(0, j)
                self.hcore[left_idx, right_idx] = self.hcore[right_idx, left_idx] = (
                    -self.t
                )

        # periodic boundary conditions, y-direction
        if pbc_y:
            for i in range(nsites_x):
                down_idx = site_index(i, nsites_y - 1)
                up_idx = site_index(i, 0)
                self.hcore[down_idx, up_idx] = self.hcore[up_idx, down_idx] = -self.t

        # Overlap
        self.overlap = np.eye(nsites_flat)

        # On-site interaction
        self.eri = np.zeros((nsites_flat,) * 4)
        for i in range(nsites_flat):
            self.eri[i, i, i, i] = self.U
