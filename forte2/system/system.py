from dataclasses import dataclass, field

import forte2
from .build_basis import build_basis
from .parse_xyz import parse_xyz
from forte2.x2c import get_hcore_x2c

import numpy as np
from numpy.typing import NDArray


@dataclass
class System:
    xyz: str
    basis: str
    auxiliary_basis: str = None
    atoms: list[tuple[float, tuple[float, float, float]]] = None
    minao_basis: str = None
    x2c_type: str = None

    def __post_init__(self):
        self.atoms = parse_xyz(self.xyz)
        self.basis = build_basis(self.basis, self.atoms)
        self.auxiliary_basis = (
            build_basis(self.auxiliary_basis, self.atoms)
            if self.auxiliary_basis is not None
            else None
        )
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


@dataclass
class ModelSystem:
    """
    A base class for model systems.
    One needs to spefify the overlap, hcore, and eri tensors.
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
