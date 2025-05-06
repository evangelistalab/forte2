from dataclasses import dataclass, field
import forte2

from .build_basis import build_basis
from .parse_xyz import parse_xyz


@dataclass
class System:
    xyz: str
    basis_name: str
    auxiliary_basis_name: str = None
    atoms: list[tuple[float, tuple[float, float, float]]] = None
    minao_basis_name: str = None

    def __post_init__(self):
        self.atoms = parse_xyz(self.xyz)
        self.basis = build_basis(self.basis_name, self.atoms)
        self.auxiliary_basis = (
            build_basis(self.auxiliary_basis_name, self.atoms)
            if self.auxiliary_basis_name is not None
            else None
        )
        self.minao_basis = (
            build_basis("cc-pvtz-minao", self.atoms)
            if self.minao_basis_name is not None
            else None
        )
        print(
            f"Parsed {len(self.atoms)} atoms with basis set of {self.basis.size} functions."
        )

    def __repr__(self):
        return f"System(atoms={self.atoms}, basis={self.basis_name}, auxiliary_basis={self.auxiliary_basis_name})"

    def nao(self):
        """
        Get the number of atomic orbitals in the system.

        Returns:
            int: Number of atomic orbitals.
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
            self.basis_name,
            self.atoms,
            embed_normalization_into_coefficients=True,
            decontract=True,
        )
