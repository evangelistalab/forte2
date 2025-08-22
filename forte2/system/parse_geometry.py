import regex as re
from dataclasses import dataclass
import numpy as np

from forte2 import ints
from forte2.helpers import logger
from .atom_data import ATOM_DATA, ATOM_SYMBOL_TO_Z, ANGSTROM_TO_BOHR


def rotation_mat(axis, theta):
    """
    Euler-Rodrigues formula for rotation matrix

    Parameters
    ----------
    axis : ndarray of shape (3,)
        The axis to rotate around. Will be normalized internally.
    theta : float
        Rotation angle

    Returns
    -------
    R : ndarray of shape (3, 3)
        The rotation matrix.
    """
    axis = axis / np.linalg.norm(axis)
    a = np.cos(theta / 2)
    b, c, d = -axis * np.sin(theta / 2)

    R = np.array(
        [
            [a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
            [2 * (b * c + a * d), a * a + c * c - b * b - d * d, 2 * (c * d - a * b)],
            [2 * (b * d - a * c), 2 * (c * d + a * b), a * a + d * d - b * b - c * c],
        ]
    )
    return R


def reflection_mat(plane):
    """
    Return the 3x3 reflection matrix for the mirror plane σ(x_i, x_j).

    Parameters
    ----------
    plane: tuple (i, j) with i != j and i,j ∈ {0,1,2} mapping to x,y,z.
           Example: (0,1) -> σ_xy (flip z).

    Returns
    -------
    R : ndarray of shape (3, 3)
    The reflection matrix.
    """
    i, j = plane
    if i == j:
        raise ValueError("plane must use two distinct axes")
    R = np.eye(3)
    k = 3 - i - j  # the axis not in the plane (since {0,1,2})
    R[k, k] = -1.0
    return R


def parse_geometry(geom, unit):
    """
    Parse a geometry string (XYZ or Z-matrix) into a list of atoms.

    Parameters
    ----------
    geom : str
        Geometry string in XYZ or Z-matrix format.
    unit : str
        Coordinate unit, "bohr" or "angstrom".

    Returns
    -------
    atoms : list[tuple(int, NDArray)]
        List of (Z, coords) tuples in Bohr.
    """
    lines = [line for line in geom.strip().splitlines() if line.strip()]
    if not lines:
        raise ValueError("Empty geometry string provided.")

    if re.match(r"^([A-Z][a-z]?)$", lines[0]):
        return parse_zmatrix(geom, unit)
    else:
        return parse_xyz(geom, unit)


def parse_xyz(xyz, unit):
    r"""
    Parse an XYZ string into a list of atoms.

    Parameters
    ----------
    xyz : str
        The XYZ formatted string containing atom symbols and coordinates.
    unit : str
        The unit of the coordinates, either "bohr" or "angstrom".

    Returns
    -------
    atoms : list[tuple(int, NDArray)]
        A list of tuples, each containing the atomic number and a numpy array of coordinates.

    Raises
    ------
    ValueError
        If a line in the XYZ string does not match the expected format or has an incorrect number of coordinates.

    Examples
    --------
    >>> xyz = "Li 0.0 0.0 0.0\nN -10 0 0\n"
    >>> parse_xyz(xyz, "bohr")
    [(3, array([0., 0., 0.])), (7, array([-10., 0., 0.]))]

    >>> xyz = "Li 0.0 0.0\nN -10 0\n"
    >>> parse_xyz(xyz, "angstrom")
    Traceback (most recent call last):
        ... ValueError: Invalid line in XYZ file: Li 0.0 0.0. Expected 3 coordinates, found 2.

    """
    atoms = []
    for line in xyz.split("\n"):
        # look for lines of the form "Li 0.0 0.0 0.0" or "N -10 0 0" and capture the element symbol and coordinates
        # Use regex to match the expected format
        m = re.match(
            r"^\s*([A-Z][a-z]?)\s+([-+]?\d*\.\d+|[-+]?\d+)\s+([-+]?\d*\.\d+|[-+]?\d+)\s+([-+]?\d*\.\d+|[-+]?\d+)\s*$",
            line,
        )
        # Skip lines that do not match the expected format
        if not m:
            # Test if one or two coordinates are missing, e.g., "Li 0.0 0.0" or "Li 0.0"
            # This regex captures the element symbol and up to three coordinates
            check_missing_coordinate = re.match(
                r"^\s*([A-Z][a-z]?)\s+([-+]?\d*\.\d+|[-+]?\d+)(?:\s+([-+]?\d*\.\d+|[-+]?\d+))?(?:\s+([-+]?\d*\.\d+|[-+]?\d+))?\s*$",
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
        atoms.append([atomic_number, coords])

    return atoms


def parse_zmatrix(zmat, unit):
    """Parse a Z-matrix string into a list of atoms.

    Parameters
    ----------
        zmat (str): The Z-matrix formatted string containing atom symbols and coordinates.
        unit (str): The unit of the coordinates, either "bohr" or "angstrom".

    Returns
    -------
    atoms : list[tuple(int, NDArray)]
        A list of tuples, each containing the atomic number and a numpy array of coordinates.

    Raises
    ------
    ValueError
        If a line in the Z-matrix string does not match the expected format.

    Examples
    --------
    Unit bohr:
    H
    O 1 0.9
    O 2 1.4 1 105.0
    H 3 0.9 2 105.0 1 120.0

    Should return:
    [[ 0.          0.          0.        ]
    [ 1.70075351  0.          0.        ]
    [ 2.38548947  0.         -2.55546938]
    [ 3.29283067  1.42270804 -2.76806356]]

    """

    atoms = []
    coords = []
    lines = [line.strip() for line in zmat.strip().splitlines() if line.strip()]
    conv = 1.0 if unit == "bohr" else ANGSTROM_TO_BOHR

    for i, line in enumerate(lines):
        if i == 0:
            # Atom with no internal coordinates
            m = re.match(r"^([A-Z][a-z]?)$", line)
            if not m:
                raise ValueError(f"Invalid Z-matrix line {i+1}: {line}")
            symbol = m.group(1)
            coord = np.array([0.0, 0.0, 0.0])
        elif i == 1:
            # Atom with bond length
            m = re.match(r"^([A-Z][a-z]?)\s+(\d+)\s+([-+]?\d*\.\d+|[-+]?\d+)$", line)
            if not m:
                raise ValueError(f"Invalid Z-matrix line {i+1}: {line}")
            symbol, i1, r = m.groups()
            coord = np.array([float(r) * conv, 0.0, 0.0])
        elif i == 2:
            # Atom with bond length and angle
            m = re.match(
                r"^([A-Z][a-z]?)\s+(\d+)\s+([-+]?\d*\.\d+|[-+]?\d+)\s+(\d+)\s+([-+]?\d*\.\d+|[-+]?\d+)$",
                line,
            )
            if not m:
                raise ValueError(f"Invalid Z-matrix line {i+1}: {line}")
            symbol, i1, r, i2, angle = m.groups()
            theta = float(angle) / 180 * np.pi
            i1, i2 = int(i1) - 1, int(i2) - 1

            p1 = coords[i1]
            p2 = coords[i2]
            # Vector from p1 to p2
            v1 = p2 - p1
            v1 = v1 / np.linalg.norm(v1)
            if not np.allclose(v1[:2], 0):
                v_rot = np.cross(v1, np.array((0.0, 0.0, 1.0)))
            else:
                v_rot = np.array((0.0, 0.0, 1.0))
            rmat = rotation_mat(v_rot, theta)
            c = np.dot(rmat, v1) * float(r) * conv
            coord = c + p1
        else:
            m = re.match(
                r"^([A-Z][a-z]?)\s+(\d+)\s+([-+]?\d*\.\d+|[-+]?\d+)"
                r"\s+(\d+)\s+([-+]?\d*\.\d+|[-+]?\d+)"
                r"\s+(\d+)\s+([-+]?\d*\.\d+|[-+]?\d+)$",
                line,
            )
            if not m:
                raise ValueError(f"Invalid Z-matrix line {i+1}: {line}")
            symbol, i1, r, i2, angle, i3, dihedral = m.groups()
            theta = float(angle) / 180 * np.pi
            phi = float(dihedral) / 180 * np.pi

            i1, i2, i3 = int(i1) - 1, int(i2) - 1, int(i3) - 1
            p1, p2, p3 = coords[i1], coords[i2], coords[i3]
            # Vector from p1 to p2
            v1 = p2 - p1
            # Vector from p3 to p2
            v2 = p2 - p3
            c = conv * float(r) * v1 / np.linalg.norm(v1)
            normal = np.cross(v1, v2)
            c = np.dot(rotation_mat(normal, theta), c)
            c = np.dot(rotation_mat(v1, phi), c)
            coord = c + p1
        atomic_number = ATOM_SYMBOL_TO_Z[symbol.upper()]
        atoms.append([atomic_number, coord])
        coords.append(coord)

    return atoms


@dataclass
class GeometryHelper:
    """Helper class to process geometry data."""

    atoms: list[tuple[int, np.ndarray]]
    tol: float = 1.0e-06
    reorient: bool = True

    def __post_init__(self):
        self.Zsum = round(np.sum([x[0] for x in self.atoms]))
        self.natoms = len(self.atoms)
        self.atomic_charges = np.array([atom[0] for atom in self.atoms])
        self.atomic_masses = np.array(
            [ATOM_DATA[Z]["mass"] for Z in self.atomic_charges]
        )
        self.atomic_positions = np.array([atom[1] for atom in self.atoms])
        self.centroid = np.mean(self.atomic_positions, axis=0)
        self.nuclear_repulsion = ints.nuclear_repulsion(self.atoms)

        self.center_of_mass = np.einsum(
            "a,ax->x", self.atomic_masses, self.atomic_positions
        ) / np.sum(self.atomic_masses)

        self.atom_counts = {}
        for atom in self.atoms:
            if atom[0] not in self.atom_counts:
                self.atom_counts[atom[0]] = 0
            self.atom_counts[atom[0]] += 1

        self.atom_to_center = {}
        for i, atom in enumerate(self.atoms):
            if atom[0] not in self.atom_to_center:
                self.atom_to_center[atom[0]] = []
            self.atom_to_center[atom[0]].append(i)

        self.inertia_matrix = np.zeros((3, 3))
        for m, r in zip(self.atomic_masses, self.atomic_positions):
            x = r - self.center_of_mass
            r2 = np.dot(x, x)
            self.inertia_matrix += m * ((r2 * np.eye(3)) - np.outer(x, x))

        if self.reorient:
            # compute principle moments of inertia. These will be sorted in ascending order
            moi, moi_vectors = np.linalg.eigh(self.inertia_matrix)

            # check if moi_vectors is right-handed
            # if np.linalg.det(moi_vectors) < 0:
            #     moi_vectors[:, 0] *= -1.0  # flip one axis to fix handedness

            shifted_atomic_positions = (
                self.atomic_positions - self.center_of_mass[None, :]
            )

            axis_order = []
            for i in range(3):
                R = rotation_mat(moi_vectors[:, i], np.pi)
                rotated_positions = (R @ shifted_atomic_positions.T).T
                all_match = True
                for x in shifted_atomic_positions:
                    found = False
                    for y in rotated_positions:
                        if np.linalg.norm(x - y) < self.tol:
                            found = True
                    if not found:
                        all_match = False

                axis_order.append(2 if all_match else 1)

            sorted_axis = sorted(
                zip(axis_order, moi, range(3)),
                key=lambda x: (
                    x[0],
                    -x[1],
                ),  # sort axes by Cn order first, then by descending MOI
            )
            logger.log_debug("Sorted Axis Order:")
            for ax in sorted_axis:
                n, I, idx = ax
                logger.log_debug(
                    f"Axis: {moi_vectors[:, idx]}   Cn: {n}   MOI: {I}   Axis Assignment: {idx}"
                )

            self.prinrot = np.concatenate(
                [moi_vectors[:, n][:, np.newaxis] for _, _, n in sorted_axis], axis=1
            ).T  # for rotated water
            self.prin_atomic_positions = np.array(
                [
                    self.prinrot @ (r - self.center_of_mass)
                    for r in self.atomic_positions
                ]
            )

            # Overwrite original inputted atomic positions with principle atomic positions
            self.atomic_positions = self.prin_atomic_positions.copy()
            for i in range(self.natoms):
                self.atoms[i][1] = self.prin_atomic_positions[i, :]
            logger.log_info1(
                "Original atomic coordinates overwritten with principle atomic positions"
            )

            # self.center_of_mass = np.einsum(
            #     "a,ax->x", self.atomic_masses, self.atomic_positions
            # ) / np.sum(self.atomic_masses)
            # print(f"NEW COM: {self.center_of_mass}")
        else:
            self.prinrot = np.eye(3)
            self.prin_atomic_positions = self.atomic_positions.copy()
