import numpy as np
import json
import forte2
import itertools

from .atom_data import ATOM_SYMBOL_TO_Z


class System:
    def __init__(self, xyz, basis):
        self.atoms = self.parse_xyz(xyz)
        self.basis = self.parse_basis(basis)
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
            coords = np.array([float(x) for x in parts[1:]])
            atoms.append((atomic_number, coords))
        return atoms

    def parse_basis(self, basis_name):
        basis = forte2.ints.Basis()
        try:
            with open(f"{basis_name.lower()}.json", "r") as f:
                basis_json = json.load(f)
        except OSError:
            raise Exception(
                f"[forte2] Basis file {basis_name.lower()}.json could not be opened."
            )
        for atomic_number, xyz in self.atoms:
            # we need to check if the atom is in the basis set
            atom_basis = basis_json["elements"][f"{atomic_number}"]["electron_shells"]
            for shell in atom_basis:
                angular_momentum = list(map(int, shell["angular_momentum"]))
                exponents = list(map(float, shell["exponents"]))

                for l, subshell_coefficients in itertools.zip_longest(
                    angular_momentum,
                    shell["coefficients"],
                    fillvalue=angular_momentum[-1],
                ):
                    coefficients = list(map(float, subshell_coefficients))
                    basis.add(forte2.ints.Shell(l, exponents, coefficients, xyz))
        return basis
