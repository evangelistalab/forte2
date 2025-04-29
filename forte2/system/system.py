from dataclasses import dataclass, field
import forte2

from .build_basis import build_basis
from .parse_xyz import parse_xyz


@dataclass
class System:
    xyz: str
    basis: forte2.ints.Basis
    auxiliary_basis: forte2.ints.Basis = None
    atoms: list[tuple[float, tuple[float, float, float]]] = None
    minao_basis: forte2.ints.Basis = None

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

    def __repr__(self):
        return f"System(atoms={self.atoms}, basis={self.basis}, auxiliary_basis={self.auxiliary_basis})"
