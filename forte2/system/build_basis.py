import forte2
import json
import itertools
from importlib import resources


def parse_basis_json(basis_name: str) -> dict:
    """
    Get the basis set JSON file from the package resources.
    Args:
        basis_name (str): The name of the basis set.
    Returns:
        dict: The basis set JSON data.
    Raises:
        Exception: If the basis set file cannot be opened.
    """
    # Convert basis name to lowercase
    basis_name = basis_name.lower()

    try:
        with resources.files("forte2.data.basis").joinpath(f"{basis_name}.json").open(
            "r"
        ) as f:
            basis_json = json.load(f)
    except OSError:
        raise Exception(
            f"[forte2] Basis file {basis_name.lower()}.json could not be opened."
        )
    return basis_json


def assemble_basis(
    basis_name: str,
    basis_json: dict,
    atoms: list[tuple[int, tuple[float, float, float]]],
) -> forte2.ints.Basis:
    """
    Assemble the basis set from the JSON data and the list of atoms.
    Args:
        basis_name (str): The name of the basis set.
        basis_json (dict): The basis set JSON data.
        atoms (list[tuple[int, list[float]]]): A list of tuples containing atomic numbers and coordinates.
    Returns:
        forte2.ints.Basis: The basis set.
    Raises:
        Exception: If the basis set file cannot be opened or if an element is not in the basis set.
    """
    basis = forte2.ints.Basis()

    for atomic_number, xyz in atoms:
        # check if the atom is in the basis set
        if str(atomic_number) not in basis_json["elements"]:
            raise Exception(
                f"[forte2] Basis set {basis_name} does not contain element {atomic_number}."
            )
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


def build_basis(
    basis_name: str, atoms: list[tuple[int, tuple[float, float, float]]]
) -> forte2.ints.Basis:
    """
    Build a basis set from a basis name and a list of atoms.
    Args:
        basis_name (str): The name of the basis set.
        atoms (list[tuple[int, tuple[float, float, float]]]): A list of tuples containing atomic numbers and coordinates.
    Returns:
        forte2.ints.Basis: The basis set.
    Raises:
        Exception: If the basis set file cannot be opened or if an element is not in the basis set.
    """

    basis_json = parse_basis_json(basis_name)

    return assemble_basis(basis_name, basis_json, atoms)
