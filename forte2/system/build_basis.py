import forte2
import json
import itertools
from importlib import resources


def build_basis(basis_name, atoms):
    basis = forte2.ints.Basis()
    try:
        with resources.path("forte2.data.basis", f"{basis_name}.json") as p:
            basis_json = json.loads(p.read_text())
        # with open(f"{basis_name.lower()}.json", "r") as f:
        #     basis_json = json.load(f)
    except OSError:
        raise Exception(
            f"[forte2] Basis file {basis_name.lower()}.json could not be opened."
        )
    for atomic_number, xyz in atoms:
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
