from dataclasses import dataclass
import numpy as np
import re

import forte2
from forte2.helpers import logger
from forte2.helpers.mixins import MOsMixin, SystemMixin
from forte2.system.atom_data import Z_TO_ATOM_SYMBOL, ATOM_SYMBOL_TO_Z


def parse_subspace_pi_planes(system, planes_expr):
    """
    Parse the option "SUBSPACE_PI_PLANES" and return a direction vector on each plane atoms.
    :param molecule: a Psi4 Molecule object
    :param planes_expr: a list of plane expressions
    :param debug: debug flag for printing intermediate steps
    :return: a map from atom (atomic number, relative index in the molecule) to the direction vector

    This function parse a list of planes where each plane is defined by a list of atoms.
    The acceptable expressions for atoms include:
      - "C": all carbon atoms
      - "C2": the second carbon atom of the molecule
      - "C1-4": the first to fourth carbon atoms
    Examples for planes expressions:
      - [['C', 'H', 'O']]: only one plane consisting all C, H, and O atoms of the molecule.
      - [['C1-6'], ['N1-2', 'C9-11']]: plane 1 with the first six C atoms of the molecule,
                                       plane 2 with C9, C10, C11, N1 and N2 atoms.
      - [['C1-4'], ['C1-2', 'C5-6']]: plane 1 with the first four C atoms of the molecule,
                                      plane 2 with C1, C2, C5, and C6 atoms. Two planes share C1 and C2.

    Motivations:
      This function detects the directions of π orbitals for atoms forming the planes.
      The direction suggests how the atomic p orbitals should be linearly combined,
      such that the resulting orbital is perpendicular to the plane.
      This function can be useful for AVAS subspace selections on complicated molecules when:
        - the molecular xyz frame does not align with absolute xyz frame
        - the molecule contains multiple π systems
        - the plane is slightly distorted but an approximate averaged plane is desired

    Implementations:
      - Each plane is characterized by the plane unit normal.
        This normal is attached to all atoms that defines this plane.
      - If multiple planes share the same atom,
        the direction is obtained by summing over all unit normals of the planes and then normalize it.
      - The convention for the direction of a plane normal is defined such that the angle between
        the molecular centroid to the plane centroid and the the plane normal is acute.
      - The plane unit normal is computed as the smallest principal axis of the plane xyz coordinates.
        For a real plane, the singular value is zero.
        For an approximate plane, the singular value should be close to zero.

    PR #261: https://github.com/evangelistalab/forte/pull/261
    """
    # return empty dictionary if no planes are defined
    if not planes_expr:
        return {}

    # test input
    if not isinstance(system, forte2.System):
        raise ValueError("Invalid argument for system!")

    if not isinstance(planes_expr, list):
        raise ValueError("Invalid plane expressions: layer 1 not a list!")
    else:
        for plane_atoms in planes_expr:
            if not isinstance(plane_atoms, list):
                raise ValueError("Invalid plane expressions: layer 2 not a list!")
            else:
                if not all(isinstance(i, str) for i in plane_atoms):
                    raise ValueError(
                        "Invalid plane expressions: atom expressions not string!"
                    )

    # print requested planes
    logger.log_info1("\n  ==> List of Planes Requested <==\n")
    for i, plane in enumerate(planes_expr):
        logger.log_info1(f"\n    Plane {i + 1:2d}")
        for j, atom in enumerate(plane):
            if j % 10 == 0:
                logger.log_info1("\n    ")
            logger.log_info1(f"{atom:>8s}")

    # create index map {'C': [absolute indices in molecule], 'BE': [...], ...}
    charges = system.atomic_charges()
    abs_indices = {}
    for i in range(system.natoms):
        try:
            abs_indices[Z_TO_ATOM_SYMBOL[charges[i]].upper()].append(i)
        except KeyError:
            abs_indices[Z_TO_ATOM_SYMBOL[charges[i]].upper()] = [i]
    logger.log_debug(f"Index map: {abs_indices}")

    # put molecular geometry (Bohr) in numpy array format
    xyz = system.atomic_positions()

    # centroid (geometric center) of the molecule
    centroid = np.mean(xyz, axis=0)
    logger.log_debug(f"Molecule centroid (Bohr): {centroid}")

    # parse planes
    atom_dirs = {}
    atom_regex = r"([A-Za-z]{1,2})\s*(\d*)\s*-?\s*(\d*)"

    for n, plane_atoms in enumerate(planes_expr):
        logger.log_debug(f"Process plane {n + 1}")

        plane = []  # absolute index for atoms forming the plane
        plane_z = []  # pair of atomic number and relative index

        # parse each plane entry
        for atom_expr in plane_atoms:
            atom_expr = atom_expr.upper()

            m = re.match(atom_regex, atom_expr)
            if not m:
                raise ValueError("Invalid expression of atoms!")

            atom, start_str, end_str = m.groups()
            if atom not in abs_indices:
                raise ValueError(f"Atom '{atom}' not in molecule!")

            start = 1
            end = int(end_str) if end_str else len(abs_indices[atom])
            if start_str:
                start = int(start_str)
                end = int(end_str) if end_str else start

            z = ATOM_SYMBOL_TO_Z[atom]
            for i in range(start - 1, end):
                plane.append(abs_indices[atom][i])
                plane_z.append((z, i))
            logger.log_debug(f"  parsed entry: {atom:2s} {start:>3d} - {end:d}")

        logger.log_debug(f"  atom indices of the plane: {plane}")

        # compute the plane unit normal (smallest principal axis)
        plane_xyz = xyz[plane]
        plane_centroid = np.mean(plane_xyz, axis=0)
        plane_xyz = plane_xyz - plane_centroid
        logger.log_debug(f"  plane centroid (Bohr): {plane_centroid}")
        logger.log_debug(f"  shifted plane xyz (Bohr):")
        for x, y, z in plane_xyz:
            logger.log_debug(f"    {x:13.10f}  {y:13.10f}  {z:13.10f}")

        # SVD the xyz coordinate
        u, s, vh = np.linalg.svd(plane_xyz)

        # fix phase
        p = plane_centroid - centroid
        plane_normal = vh[2] if np.inner(vh[2], p) >= 0.0 else vh[2] * -1.0
        logger.log_debug(f"  singular values: {s}")
        logger.log_debug(f"  plane unit normal: {plane_normal}")

        # attach each atom to the unit normal
        for z_i in plane_z:
            if z_i in atom_dirs:
                atom_dirs[z_i] = atom_dirs[z_i] + plane_normal
            else:
                atom_dirs[z_i] = plane_normal

    # normalize the directions on each requested atom
    atom_dirs = {z_i: n / np.linalg.norm(n) for z_i, n in atom_dirs.items()}
    logger.log_debug(
        "Averaged vector perpendicular to the requested planes on each atom"
    )
    for z, i in sorted(atom_dirs.keys()):
        n_str = " ".join(f"{i:15.10f}" for i in atom_dirs[(z, i)])
        logger.log_debug(
            f"  Atom Z: {z:3d}, relative index: {i:3d}, direction: {n_str}"
        )

    return atom_dirs

def make_ao_space_projector()


@dataclass
class AVAS(MOsMixin, SystemMixin):
    """
    Automatic valence active space (AVAS) method for selecting active orbitals for multi-reference calculations.

    Parameters
    ----------
    subspace : list
        The subspace of orbitals to be considered for AVAS selection.
    subspace_pi_planes : list, optional
        A list of planes defined by atoms in the molecule, used to determine π orbitals.
    diagonalize : bool, optional, default=True
        Whether to diagonalize the occupied and virtual space overlap matrices.
    sigma : float, optional, default=0.98
        Cumulative cutoff for the eigenvalues of the overlap matrix, controlling the size of the active space.
    cutoff : float, optional, default=1.0
        Cutoff for the eigenvalues of the overlap matrix; eigenvalues greater than this value are considered active.
    evals_threshold : float, optional, default=1.0e-6
        Threshold below which an eigenvalue of the projected overlap is considered zero.
    num_active : int, optional, default=0
        Total number of active orbitals. If set, it takes priority over threshold-based selections.
    num_active_occ : int, optional, default=0
        Number of active occupied orbitals. If set, it takes priority over cutoff-based selections and
        that based on the total number of active orbitals.
    num_active_vir : int, optional, default=0
        Number of active virtual orbitals. If set, it takes priority over cutoff-based selections and
        that based on the total number of active orbitals.
    """

    subspace: list
    subspace_pi_planes: list = None
    diagonalize: bool = True
    sigma: float = 0.98
    cutoff: float = 1.0
    evals_threshold: float = 1.0e-6
    num_active: int = 0
    num_active_occ: int = 0
    num_active_vir: int = 0

    def __call__(self, parent_method):
        self.parent_method = parent_method
        self._check_parameters()

    def _check_parameters(self):
        if self.num_active_occ + self.num_active_vir > 0:
            self.avas_selection = "separate"
        elif self.num_active > 0:
            self.avas_selection = "total"
        elif 1.0 - self.cutoff > self.evals_threshold:
            self.avas_selection = "cutoff"
        else:
            raise ValueError("Invalid AVAS selection criteria.")

    def run(self):
        if self.subspace_pi_planes is not None:
            # parse the subspace pi planes and get the direction vectors
            self.pi_planes = parse_subspace_pi_planes(
                self.parent_method.system, self.subspace_pi_planes
            )
        else:
            self.pi_planes = {}
        
