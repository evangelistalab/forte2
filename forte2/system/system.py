import numpy as np
import json
import forte2
import itertools

from .atom_data import ATOM_SYMBOL_TO_Z


class System:
    def __init__(self, xyz, basis):
        self.parse_xyz(xyz)
        self.parse_basis(basis)

    def __repr__(self):
        return f"System(atoms={self.atoms})"

    def parse_xyz(self, xyz):
        # Parse the XYZ string into a list of atoms
        lines = xyz.split("\n")
        self.atoms = []
        for line in lines:
            parts = line.split()
            if len(parts) < 4:
                continue
            atomic_number = ATOM_SYMBOL_TO_Z[parts[0].upper()]
            coords = np.array([float(x) for x in parts[1:]])
            self.atoms.append((atomic_number, coords))

    def parse_basis(self, basis):
        self.basis = forte2.Basis()
        with open(f"{basis}.json", "r") as f:
            basis = json.load(f)
            for atomic_number, xyz in self.atoms:
                # we need to check if the atom is in the basis set
                atom_basis = basis["elements"][f"{atomic_number}"]["electron_shells"]
                for shell in atom_basis:
                    angular_momentum = list(map(int, shell["angular_momentum"]))
                    exponents = list(map(float, shell["exponents"]))

                    for l, subshell_coefficients in itertools.zip_longest(
                        angular_momentum,
                        shell["coefficients"],
                        fillvalue=angular_momentum[-1],
                    ):
                        coefficients = list(map(float, subshell_coefficients))
                        self.basis.add(forte2.Shell(l, exponents, coefficients, xyz))
