import forte2
import json
import itertools
from importlib import resources
from collections import defaultdict

try:
    import basis_set_exchange as bse

    BSE_AVAILABLE = True
except ImportError:
    BSE_AVAILABLE = False


def get_atom_basis(basis_per_atom: dict) -> dict:
    """
    Get the basis set data for each atom based on the provided basis set names.

    Parameters
    ----------
    basis_per_atom : dict
        A dictionary mapping atomic numbers to basis set names.

    Returns
    -------
    dict
        The basis set data.

    Raises
    ------
    RuntimeError
        If the basis set file cannot be opened / found on Basis Set Exchange (if avaialble) or if an element is not in the basis set.
    """

    # invert the basis_per_atom dictionary to get a list of atoms per basis set
    # E.g. {1:"cc-pvdz", 2:"cc-pvdz", 8:"sto-6g"} -> {"cc-pvdz":[1, 2], "sto-6g":[8]}
    atoms_per_basis = defaultdict(list)

    for atomic_number, basis_name in basis_per_atom.items():
        atoms_per_basis[basis_name.lower()].append(atomic_number)

    # stores the basis set data for each atom
    atom_basis = {}
    for basis_name in atoms_per_basis.keys():
        # check if this basis is locally available
        if resources.is_resource("forte2.data.basis", f"{basis_name}.json"):
            with resources.files("forte2.data.basis").joinpath(
                f"{basis_name}.json"
            ).open("r") as f:
                bfile = json.load(f)
                for atomic_number in atoms_per_basis[basis_name]:
                    # check if the atomic number is in the basis set
                    assert (
                        str(atomic_number) in bfile["elements"]
                    ), f"Element {atomic_number} not found in basis set {basis_name}."
                    atom_basis[atomic_number] = bfile["elements"][str(atomic_number)][
                        "electron_shells"
                    ]
        else:
            if BSE_AVAILABLE:
                print(
                    f"[forte2] Basis {basis_name} not found locally. Using Basis Set Exchange."
                )
                for atomic_number in atoms_per_basis[basis_name]:
                    try:
                        bse_basis = bse.get_basis(basis_name, elements=atomic_number)
                    except KeyError:
                        raise RuntimeError(
                            f"[forte2] Basis Set Exchange does not have data for element Z={atomic_number} in basis set {basis_name}!"
                        )
                    atom_basis[atomic_number] = bse_basis["elements"][
                        str(atomic_number)
                    ]["electron_shells"]
            else:
                raise RuntimeError(
                    f"[forte2] Basis file {basis_name}.json could not be found, and Basis Set Exchange is not available. "
                )
    return atom_basis


def parse_custom_basis_assignment(unique_atoms: set, basis_assignment: dict) -> dict:
    """
    Parse a custom basis assignment dictionary

    Parameters
    ----------
    unique_atoms : set
        A set of unique atomic numbers in the system.
    basis_assignment : dict
        A dictionary mapping atom symbols to basis set names.

    Returns
    -------
    basis_per_atom : dict
        A dictionary mapping all unique atomic numbers to their assigned basis set names.
    """

    default_basis = basis_assignment.pop("default", None)

    # convert atom symbols to atomic numbers
    basis_assignment = {
        forte2.ATOM_SYMBOL_TO_Z[atom.upper()]: basis
        for atom, basis in basis_assignment.items()
    }

    # ensure all unique atoms in the system have a basis set assigned
    provided_atoms = set([atom_number for atom_number in basis_assignment.keys()])
    if not provided_atoms.issuperset(unique_atoms) and default_basis is None:
        raise RuntimeError(
            f"[forte2] Basis set {basis_assignment} does not contain all elements in the system. "
            f"Provided: {list(provided_atoms)}, Required: {list(unique_atoms)}."
            "Please provide them or supply a default basis set."
        )

    # dictionary for all unique atoms with their basis set assignments
    basis_per_atom = {
        atom_number: basis_assignment.get(atom_number, default_basis)
        for atom_number in unique_atoms
    }
    return basis_per_atom


def build_basis(
    basis_assignment: str | dict,
    atoms: list[tuple[int, tuple[float, float, float]]],
    embed_normalization_into_coefficients: bool = True,
    decontract: bool = False,
) -> forte2.ints.Basis:
    """
    Assemble the basis set from JSON data or Basis Set Exchange, depending on availability.

    Parameters
    ----------
    basis_assignment : str or dict
        The basis set name or a dictionary with per-atom basis assignments.
        The dictionary should be in the format::

            {
                "H": "cc-pvtz",
                "O": "cc-pvqz",
                "default": "cc-pvdz"
            }

        where ``"default"`` is used for atoms not explicitly listed.
    atoms : list[tuple(int, list[float])]
        A list of tuples containing atomic numbers and coordinates.

    Returns
    -------
    basis : forte2.ints.Basis
        The basis set.
    """
    basis = forte2.ints.Basis()
    prefix = "decon-" if decontract else ""
    unique_atoms = set([atomic_number for atomic_number, _ in atoms])

    # get the mapping of atomic numbers to basis set names
    if isinstance(basis_assignment, str):
        basis_name = basis_assignment
        basis_per_atom = {atom: basis_name for atom in unique_atoms}
    elif isinstance(basis_assignment, dict):
        basis_name = "custom_basis"
        basis_per_atom = parse_custom_basis_assignment(unique_atoms, basis_assignment)
    else:
        raise TypeError(
            f"[forte2] Invalid basis assignment type: {type(basis_assignment)}. "
            "Expected str or dict."
        )

    # convert the basis set names to lowercase
    basis_per_atom = {
        atom: basis_name.lower() for atom, basis_name in basis_per_atom.items()
    }
    # set the basis name
    basis.set_name(prefix + basis_name)
    # get the basis set data for each atom
    atom_basis = get_atom_basis(basis_per_atom)

    if decontract:
        # the decontracted is simply a double loop over l and alpha for each shell, with unit coefficients
        for atomic_number, xyz in atoms:
            for shell in atom_basis[atomic_number]:
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
        for atomic_number, xyz in atoms:
            for shell in atom_basis[atomic_number]:
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
