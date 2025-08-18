import json
import itertools
from importlib import resources
import regex as re

from forte2 import Basis, Shell
from forte2.system.atom_data import ATOM_SYMBOL_TO_Z
from forte2.helpers import logger
from .parse_geometry import _GeometryHelper

try:
    import basis_set_exchange as bse

    BSE_AVAILABLE = True
except ImportError:
    BSE_AVAILABLE = False


def build_basis(
    basis_assignment: str | dict,
    atoms,
    embed_normalization_into_coefficients: bool = True,
    decontract: bool = False,
    return_basis_data: bool = False,
) -> Basis:
    """
    Assemble the basis set from JSON data or Basis Set Exchange, depending on availability.

    Parameters
    ----------
    basis_assignment : str or dict
        The basis set name or a dictionary with per-atom basis assignments.
        It is also possible to assign different basis sets to specific atomic indices (e.g. "O2" for the second oxygen atom).
        The dictionary should be in the format::

            {
                "H": "cc-pvtz",
                "O2": "cc-pvqz",
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
    geometry = _GeometryHelper(atoms)

    prefix = "decon-" if decontract else ""
    natoms = geometry.natoms

    # get the mapping of atomic index to basis set names
    if isinstance(basis_assignment, str):
        basis_name = basis_assignment
        basis_per_atom = [basis_name for _ in range(natoms)]
    elif isinstance(basis_assignment, dict):
        basis_name = "custom_basis"
        basis_per_atom = _parse_custom_basis_assignment(geometry, basis_assignment)
    else:
        raise TypeError(
            f"[forte2] Invalid basis assignment type: {type(basis_assignment)}. "
            "Expected str or dict."
        )

    # convert the basis set names to lowercase
    basis_per_atom = [basis_name.lower() for basis_name in basis_per_atom]

    if_decontract_atom_basis = ["decon-" in b or decontract for b in basis_per_atom]
    # remove the "decon-" prefix to prepare for fetching of the basis set data
    basis_per_atom = [b.replace("decon-", "") for b in basis_per_atom]

    # get the 'shopping list' for _get_atom_basis
    # e.g. {"cc-pvdz": {1, 2}, "sto-6g": {8}} (fetch H, He from cc-pvdz and O from sto-6g)
    fetch_map = {}
    for i, b in enumerate(basis_per_atom):
        if b not in fetch_map:
            fetch_map[b] = set()
        Z = geometry.atoms[i][0]  # atomic number
        fetch_map[b].add(Z)

    # set the basis name
    basis = Basis()
    basis.set_name(prefix + basis_name)
    # get the unique basis set data
    atom_basis = _get_atom_basis(fetch_map)
    for (Z, coords), decon in zip(geometry.atoms, if_decontract_atom_basis):
        if decon:
            basis = _add_atom_basis_to_basis_decontracted(
                basis,
                atom_basis[Z],
                coords,
                embed_normalization_into_coefficients,
            )
        else:
            basis = _add_atom_basis_to_basis(
                basis,
                atom_basis[Z],
                coords,
                embed_normalization_into_coefficients,
            )
    if return_basis_data:
        return basis, atom_basis
    else:
        return basis


def _parse_custom_basis_assignment(geometry, basis_assignment):
    default_basis = basis_assignment.pop("default", None)
    atom_to_center = geometry.atom_to_center
    atom_counts = geometry.atom_counts

    assigned_basis = {}
    for k, v in basis_assignment.items():
        # "could be "O" or "O2" or "H2-12"
        m = re.match(r"([a-zA-Z]{1,2})([0-9]+)?-?([0-9]+)?", k).groups()
        # m[0]: atom symbol
        # m[1]: atom index start (optional)
        # m[2]: atom index end (optional)
        Z = ATOM_SYMBOL_TO_Z[m[0].upper()]
        if m[1] is None:
            assert (
                m[2] is None
            ), f"Basis assignment {k} has an end index without a start index."
            for i in atom_to_center[Z]:
                assigned_basis[i] = v
        elif m[2] is None:
            # single atom index
            i = int(m[1]) - 1
            assert i < atom_counts[Z], f"Atom {k} not found in geometry."
            assigned_basis[atom_to_center[Z][i]] = v
        else:
            # range of atom indices
            start = int(m[1]) - 1
            end = int(m[2])
            assert end <= atom_counts[Z], f"Atom range {k} is out of bounds."
            for i in range(start, end):
                assigned_basis[atom_to_center[Z][i]] = v

    if default_basis is None:
        assert (
            len(assigned_basis) == geometry.natoms
        ), "Not all atoms have a basis set assigned. Provide a default basis set or assign a basis set to all atoms."
    else:
        # ensure all atoms have a basis set assigned
        for i in range(geometry.natoms):
            if i not in assigned_basis:
                assigned_basis[i] = default_basis

    basis_per_atom = []
    for i in range(geometry.natoms):
        basis_per_atom.append(assigned_basis[i])

    return basis_per_atom


def _get_atom_basis(fetch_map):
    atom_basis = {}
    for basis_name, atoms in fetch_map.items():
        if resources.is_resource("forte2.data.basis", f"{basis_name}.json"):
            with resources.files("forte2.data.basis").joinpath(
                f"{basis_name}.json"
            ).open("r") as f:
                bfile = json.load(f)
                for Z in atoms:
                    # check if the atomic number is in the basis set
                    assert (
                        str(Z) in bfile["elements"]
                    ), f"Element {Z} not found in basis set {basis_name}."
                    atom_basis[Z] = bfile["elements"][str(Z)]["electron_shells"]
        else:
            if BSE_AVAILABLE:
                logger.log_info1(
                    f"[forte2] Basis {basis_name} not found locally. Using Basis Set Exchange."
                )
                for Z in atoms:
                    try:
                        bse_basis = bse.get_basis(basis_name, elements=Z)
                    except KeyError:
                        raise RuntimeError(
                            f"[forte2] Basis Set Exchange does not have data for element Z={Z} in basis set {basis_name}!"
                        )
                    atom_basis[Z] = bse_basis["elements"][str(Z)]["electron_shells"]
            else:
                raise RuntimeError(
                    f"[forte2] Basis file {basis_name}.json could not be found, and Basis Set Exchange is not available. "
                )
    return atom_basis


def _add_atom_basis_to_basis_decontracted(
    basis, atom_basis, coords, embed_normalization_into_coefficients
):
    for shell in atom_basis:
        angular_momentum = list(map(int, shell["angular_momentum"]))
        exponents = list(map(float, shell["exponents"]))
        for l in angular_momentum:
            for alpha in exponents:
                basis.add(
                    Shell(
                        l,
                        [alpha],
                        [1.0],
                        coords,
                        embed_normalization_into_coefficients=embed_normalization_into_coefficients,
                    )
                )
    return basis


def _add_atom_basis_to_basis(
    basis, atom_basis, coords, embed_normalization_into_coefficients
):
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
                Shell(
                    l,
                    exponents,
                    coefficients,
                    coords,
                    embed_normalization_into_coefficients=embed_normalization_into_coefficients,
                )
            )
    return basis
