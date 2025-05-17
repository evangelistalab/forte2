from dataclasses import dataclass

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

    def get_ints(self, int_type):
        """
        Get the integrals of the specified type.

        Args:
            int_type (str): Type of integrals to compute. Options are "overlap", "kinetic", "nuclear", "coulomb".

        Returns:
            np.ndarray: Computed integrals.
        """
        if int_type == "overlap":
            return forte2.ints.overlap(self.basis)
        elif int_type == "hcore":
            return forte2.ints.kinetic(self.basis) + forte2.ints.nuclear(
                self.basis, self.atoms
            )
        elif int_type == "nuclear_repulsion":
            return forte2.ints.nuclear_repulsion(self.atoms)
        else:
            raise ValueError(f"Unknown integral type: {int_type}")


@dataclass
class ModelSystem:
    model_name: str
    hcore: NDArray
    overlap: NDArray
    eri: NDArray
    nuclear_repulsion: float = 0.0

    def __post_init__(self):
        self.Zsum = 0  # total nuclear charge, here set to zero, so charge can be set to -nel later
        self.x2c_type = None

    def get_ints(self, int_type):
        """
        Get the integrals of the specified type.

        Args:
            int_type (str): Type of integrals to compute. Options are "overlap", "kinetic", "nuclear", "coulomb".

        Returns:
            np.ndarray: Computed integrals.
        """
        if int_type == "overlap":
            return self.overlap
        elif int_type == "hcore":
            return self.hcore
        elif int_type == "eri":
            return self.eri
        elif int_type == "nuclear_repulsion":
            return self.nuclear_repulsion
        else:
            raise ValueError(f"Unknown integral type: {int_type}")

    def nbf(self):
        """
        Get the number of basis functions in the system.

        Returns:
            int: Number of basis functions.
        """
        return self.hcore.shape[0]

    def naux(self):
        return 0
