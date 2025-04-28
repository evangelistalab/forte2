from dataclasses import dataclass
import forte2

from .build_basis import build_basis
from .parse_xyz import parse_xyz


@dataclass
class System:
    xyz: str
    basis: forte2.ints.Basis
    auxiliary_basis: forte2.ints.Basis
    atoms: list[tuple[float, tuple[float, float, float]]] = None
    charge: int = 0

    def __post_init__(self):
        self.atoms = parse_xyz(self.xyz)
        self.basis = build_basis(self.basis, self.atoms)
        self.auxiliary_basis = build_basis(self.auxiliary_basis, self.atoms)
        print(
            f"Parsed {len(self.atoms)} atoms with basis set of {self.basis.size} and auxiliary basis set of {self.auxiliary_basis.size} functions."
        )

    def __repr__(self):
        return f"System(atoms={self.atoms}, basis={self.basis}, auxiliary_basis={self.auxiliary_basis})"
