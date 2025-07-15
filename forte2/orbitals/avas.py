from dataclasses import dataclass
import numpy as np
import re

import forte2
from forte2.helpers import logger
from forte2.helpers.mixins import MOsMixin, SystemMixin
from forte2.system.atom_data import Z_TO_ATOM_SYMBOL, ATOM_SYMBOL_TO_Z


@dataclass
class AVAS(MOsMixin, SystemMixin):
    """
    Atomic valence active space (AVAS) method for selecting active orbitals for multi-reference calculations.

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

    Notes
    -----
    The allow subspace specification is a list of strings, non-exhaustive examples:
    - ["C"]              # all carbon atoms
    - ["C","N"]          # all carbon and nitrogen atoms
    - ["C1"]             # carbon atom #1
    - ["C1-3"]           # carbon atoms #1, #2, #3
    - ["C(2p)"]          # the 2p subset of all carbon atoms
    - ["C(1s)","C(2s)"]  # the 1s/2s subsets of all carbon atoms
    - ["C1-3(2s)"]       # the 2s subsets of carbon atoms #1, #2, #3
    - ["Ce(4fzx2-zy2)"]  # the 4f zxx-zyy orbital of all Ce atoms

    See `J. Chem. Theory Comput. 2017, 13, 4063-4078 <https://doi.org/10.1021/acs.jctc.7b00128>`_ for details on the AVAS method.
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

    # def _process_minao_basis(self):
    #     basis = self.system.minao_basis
    #     charges = self.system.atomic_charges()
    #     self._atom_counts = {}
    #     for i in charges:
    #         if i in self._atom_counts:
    #             self._atom_counts[i] += 1
    #         else:
    #             self._atom_counts[i] = 1
    #     xyz = self.system.atomic_positions()
    #     shell_first_and_size = basis.shell_first_and_size

    #     ao_info = []
    #     atom_am_to_f = {}
    #     element_count = {}
    #     count = 0
    #     for iatom in range(self.system.natoms):
    #         Z = charges[iatom]
    #         first = shell_first_and_size[iatom][0]
    #         size = shell_first_and_size[iatom][1]
    #         ao_list = []
    #         for ishell in range(first, first + size):
    #             shell = basis[ishell]
    #             nprim = shell.nprim
    #             l = shell.l
    #             for m in range(nprim):
    #                 ao_info.append((iatom, Z, element_count.get(Z, 0), l, m))
    #                 ao_list.append(count)
    #                 count += 1

    def _process_minao_basis(self):
        pass

    def _parse_subspace(self, ss_str):
        m = re.match(
            "([a-zA-Z]{1,2})([0-9]+)?-?([0-9]+)?\\(?((?:\\/?[1-9]{1}[SPDFGHspdfgh]{1}[a-zA-Z0-9-]*)*)\\)?",
            ss_str,
        ).groups()

        basis = self.system.minao_basis
        center_first_and_last = basis.center_first_and_last()
        # m[0] is the element symbol
        try:
            Z = ATOM_SYMBOL_TO_Z[m[0].upper()]
        except KeyError:
            raise ValueError(
                f"Invalid element symbol in subspace specification: {m[0]}"
            )

        # m[1] is the start index, m[2] is the end index
        if m[1] is None and m[2] is None:
            # no index specified, use all atoms of the element
            start = 0
            end = self._atom_counts[Z]
        elif m[1] is None and m[2] is not None:
            # catches the edge case of "C-3"
            raise ValueError(
                "Invalid subspace specification: start index is not specified but end index is."
            )
        else:
            # if only start is specified, only one atom is selected
            # if both start and end are specified, use the range
            start = 0 if m[1] is None else int(m[1]) - 1
            end = start + 1 if m[2] is None else int(m[2])

        # m[3] contains the subset of AOs e.g. "2p", "2pz", "3dz2" etc.
        if m[3] is None:
            # select all AOs of the element, subject to subspace_planes
            ...
        else:
            ...

        pass
        # re.match("([a-zA-Z]{1,2})([0-9]+)?-?([0-9]+)?\\(?((?:\\/?[1-9]{1}[SPDFGH]{1}[a-zA-Z0-9-]*)*)\\)?")

    def _check_parameters(self):
        for subspace in self.subspace:
            m = re.match(
                "([a-zA-Z]{1,2})([0-9]+)?-?([0-9]+)?\\(?((?:\\/?[1-9]{1}[SPDFGHspdfgh]{1}[a-zA-Z0-9-]*)*)\\)?",
                subspace,
            )
            if not m:
                raise ValueError(f"Invalid subspace specification: {subspace}")

        if self.num_active_occ + self.num_active_vir > 0:
            self.avas_selection = "separate"
        elif self.num_active > 0:
            self.avas_selection = "total"
        elif 1.0 - self.cutoff > self.evals_threshold:
            self.avas_selection = "cutoff"
        else:
            raise ValueError("Invalid AVAS selection criteria.")

    def _startup(self):
        l_labels = ["S", "P", "D", "F", "G", "H", "I", "K", "L", "M"]
        lm_labels_spherical = [
            ["S"],
            ["PZ", "PX", "PY"],
            ["DZ2", "DXZ", "DYZ", "DX2-Y2", "DXY"],
            ["FZ3", "FXZ2", "FYZ2", "FZX2-ZY2", "FXYZ", "FX3-3XY2", "F3X2Y-Y3"],
            ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9"],
            ["H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8", "H9", "H10", "H11"],
        ]
        self.labels_spherical_to_lm = {}
        for l in range(len(lm_labels_spherical)):
            for m in range(len(lm_labels_spherical[l])):
                self.labels_spherical_to_lm[lm_labels_spherical[l][m]] = (l, m)

    def run(self):
        if self.subspace_pi_planes is not None:
            # parse the subspace pi planes and get the direction vectors
            self.pi_planes = self._parse_subspace_pi_planes()
        else:
            self.pi_planes = {}

    def _parse_subspace_pi_planes(self):
        """
        Parse the "subplace_pi_planes" argument and return a direction vector on each plane atoms.

        Returns
        -------
        dict
            A map from atom (atomic number, relative index in the molecule) to the direction vector

        Notes
        -----
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
        planes_expr = self.subspace_pi_planes
        system = self.parent_method.system

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

    def _make_ao_space_projector(self):
        # Overlap Matrix in Minimal AO Basis
        Smm = forte2.ints.overlap(self.system.minao_basis)
        nbf_m = Smm.shape[0]
        # Build Cms: minimal AO basis to subspace AO basis
        nbf_s = self.subspace_counter 
        Cms = np.zeros((nbf_m, nbf_s))
        for m, s, c in self.subspace:
            # m is the index of the minimal AO
            # s is the index of the subspace AO
            # c is the coefficient of the subspace AO in the minimal AO basis
            Cms[m, s] = c
        # Subspace overlap matrix 
        Sss = Cms.T @ (Smm @ Cms)
        # Orthogonalize Sss: Xss = Sss^(-1/2)
        evals, evecs = np.linalg.eigh(Sss)
        Xss = evecs @ np.diag(evals ** (-0.5)) @ evecs.T
        # Build overlap matrix between subspace and large basis  
        Sml = forte2.ints.overlap(self.system.minao_basis, self.system.large_basis)
        # Project into subspace
        Ssl = Cms.T @ Sml
        # AO projector 
        # Pao = Ssl^T Sss^-1 Ssl = (Cms^T Sml)^T (Xss^T Xss) (Cms^T Sml)
        Xsl = Xss @ Ssl 
        Pao = Xsl.T @ Xsl
        return Pao



