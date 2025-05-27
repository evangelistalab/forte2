from dataclasses import dataclass, field

import forte2
from .build_basis import build_basis
from .parse_xyz import parse_xyz

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

        if self.x2c_type is not None:
            assert self.x2c_type in [
                "sf",
                "so",
            ], f"x2c_type {self.x2c_type} is not supported. Use None, 'sf' or 'so'."

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
        return forte2.ints.overlap(self.basis)

    def ints_hcore(self):
        T = forte2.ints.kinetic(self.basis)
        V = forte2.ints.nuclear(self.basis, self.atoms)
        return T + V

    def nuclear_repulsion_energy(self):
        return forte2.ints.nuclear_repulsion(self.atoms)


@dataclass
class ModelSystem:
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
        """
        Get the number of basis functions in the system.

        Returns:
            int: Number of basis functions.
        """
        return self.hcore.shape[0]

    def naux(self):
        return 0


@dataclass
class HubbardModel1D(ModelSystem):
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
