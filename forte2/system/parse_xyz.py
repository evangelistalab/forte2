import regex as re

import numpy as np

from .atom_data import ATOM_SYMBOL_TO_Z, ANGSTROM_TO_BOHR


def parse_xyz(xyz, unit):
    # Parse an XYZ string into a list of atoms
    atoms = []
    for line in xyz.split("\n"):
        # look for lines of th form "Li 0.0 0.0 0.0" or "N 10.0 0.0 0.0" and capture the element symbol and coordinates
        # Use regex to match the expected format
        m = re.match(
            r"^\s*([A-Z][a-z]?)\s+([-+]?\d*\.\d+|\d+)\s+([-+]?\d*\.\d+|\d+)\s+([-+]?\d*\.\d+|\d+)\s*$",
            line,
        )
        # Skip lines that do not match the expected format
        if not m:
            # Test if one or two coordinates are missing, e.g., "Li 0.0 0.0" or "Li 0.0"
            # This regex captures the element symbol and up to three coordinates
            check_missing_coordinate = re.match(
                r"^\s*([A-Z][a-z]?)\s+([-+]?\d*\.\d+|\d+)(?:\s+([-+]?\d*\.\d+|\d+))?(?:\s+([-+]?\d*\.\d+|\d+))?\s*$",
                line,
            )
            if check_missing_coordinate:
                n = len(check_missing_coordinate.groups()) - 2
                raise ValueError(
                    f"Invalid line in XYZ file: {line}. Expected 3 coordinates, found {n}."
                )
            continue

        parts = m.groups()
        atomic_number = ATOM_SYMBOL_TO_Z[parts[0].upper()]
        conv = 1.0 if unit == "bohr" else ANGSTROM_TO_BOHR
        coords = np.array([float(x) * conv for x in parts[1:]])
        atoms.append((atomic_number, coords))

    return atoms
