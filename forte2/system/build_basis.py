import forte2
import json
import itertools
from importlib import resources

try:
    import basis_set_exchange as bse

    BSE_AVAILABLE = True
except ImportError:
    BSE_AVAILABLE = False


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
            return json.load(f)
    except FileNotFoundError:
        if BSE_AVAILABLE:
            print(
                f"[forte2] Basis {basis_name.lower()} not found locally. Using Basis Set Exchange."
            )
            return None
        else:
            raise Exception(
                f"[forte2] Basis file {basis_name.lower()}.json could not be opened."
            )
    return basis_json


def assemble_basis(
    basis_name: str,
    atoms: list[tuple[int, tuple[float, float, float]]],
    embed_normalization_into_coefficients: bool = True,
    decontract: bool = False,
) -> forte2.ints.Basis:
    """
    Assemble the basis set from JSON data or Basis Set Exchange, depending on availability.
    Caches BSE lookups per atomic number to avoid repeated queries.
    Args:
        basis_name (str): The name of the basis set.
        atoms (list[tuple[int, list[float]]]): A list of tuples containing atomic numbers and coordinates.
    Returns:
        forte2.ints.Basis: The basis set.
    Raises:
        Exception: If the basis set file cannot be opened or if an element is not in the basis set.
    """
    basis = forte2.ints.Basis()
    basis_json = parse_basis_json(basis_name)

    # Cache for BSE queries to avoid repeated downloads
    bse_cache = {}

    for atomic_number, xyz in atoms:

        if basis_json is None:
            # Fetch and cache element data from Basis Set Exchange
            if atomic_number not in bse_cache:
                # bse will throw a KeyError if this fails
                try:
                    element_data = bse.get_basis(basis_name, elements=atomic_number)
                except KeyError:
                    raise Exception(
                        f"[forte2] Basis Set Exchange does not have data for element Z={atomic_number} in basis set {basis_name}!"
                    )
                bse_cache[atomic_number] = element_data["elements"][str(atomic_number)]
            atom_basis = bse_cache[atomic_number]["electron_shells"]
        else:
            # Load element data from local JSON
            elements = basis_json["elements"]
            # check if the atom is in the basis set
            if str(atomic_number) not in basis_json["elements"]:
                raise Exception(
                    f"[forte2] Basis set {basis_name} does not contain element {atomic_number}."
                )
            atom_basis = basis_json["elements"][f"{atomic_number}"]["electron_shells"]

        if decontract:
            for shell in atom_basis:
                angular_momentum = list(map(int, shell["angular_momentum"]))
                exponents = list(map(float, shell["exponents"]))

                for l in angular_momentum:
                    for alpha in exponents:
                        basis.add(
                            forte2.ints.Shell(
                                l,
                                [alpha],
                                [1.0],
                                xyz,
                                embed_normalization_into_coefficients=embed_normalization_into_coefficients,
                            )
                        )
        else:
            for shell in atom_basis:
                angular_momentum = list(map(int, shell["angular_momentum"]))
                exponents = list(map(float, shell["exponents"]))

                for l, subshell_coefficients in itertools.zip_longest(
                    angular_momentum,
                    shell["coefficients"],
                    fillvalue=angular_momentum[-1],
                ):
                    coefficients = list(map(float, subshell_coefficients))
                    basis.add(
                        forte2.ints.Shell(
                            l,
                            exponents,
                            coefficients,
                            xyz,
                            embed_normalization_into_coefficients=embed_normalization_into_coefficients,
                        )
                    )
    return basis


def build_basis(
    basis_name: str,
    atoms: list[tuple[int, tuple[float, float, float]]],
    embed_normalization_into_coefficients: bool = True,
    decontract: bool = False,
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
    return assemble_basis(
        basis_name, atoms, embed_normalization_into_coefficients, decontract
    )
