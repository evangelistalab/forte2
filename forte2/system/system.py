import numpy as np
import forte2

from .atom_data import ATOM_SYMBOL_TO_Z, ANGSTROM_TO_BOHR
from .build_basis import build_basis


class System:
    def __init__(self, xyz, basis):
        self.atoms = self.parse_xyz(xyz)
        self.basis = build_basis(basis, self.atoms)
        print(
            f"Parsed {len(self.atoms)} atoms with basis set {basis}. Basis size: {self.basis.size}"
        )

    def __repr__(self):
        return f"System(atoms={self.atoms})"

    def parse_xyz(self, xyz):
        # Parse the XYZ string into a list of atoms
        lines = xyz.split("\n")
        atoms = []
        for line in lines:
            # Skip empty lines
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            atomic_number = ATOM_SYMBOL_TO_Z[parts[0].upper()]
            coords = np.array([float(x) * ANGSTROM_TO_BOHR for x in parts[1:]])
            atoms.append((atomic_number, coords))
        return atoms
