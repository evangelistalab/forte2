from dataclasses import dataclass, field
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
    selection_method : str, optional, default="cumulative"
        The method for selecting active orbitals. Options are "cumulative", "cutoff", "separate", and "total".
        - "cumulative": Selects orbitals based on cumulative eigenvalues of the overlap matrix. Must define `sigma`.
        - "cutoff": Selects orbitals based on a cutoff value for eigenvalues of the overlap matrix. Must define `cutoff`.
        - "separate": Selects occupied and virtual orbitals separately. Must define `num_active_occ` and `num_active_vir`.
        - "total": Selects a total number of active orbitals. Must define `num_active`.
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
    selection_method: str = "cumulative"
    diagonalize: bool = True
    sigma: float = 0.98
    cutoff: float = 1.0
    evals_threshold: float = 1.0e-6
    num_active: int = 0
    num_active_occ: int = 0
    num_active_vir: int = 0

    executed: bool = field(init=False, default=False)

    def __post_init__(self):
        self._regex = "([a-zA-Z]{1,2})([0-9]+)?-?([0-9]+)?\\(?((?:\\/?[1-9]{1}[spdfgh]{1}[a-zA-Z0-9-]*)*)\\)?"
        self._check_parameters()

    def __call__(self, parent_method):
        assert isinstance(
            parent_method, (forte2.scf.RHF, forte2.scf.ROHF)
        ), f"Parent method must be RHF or ROHF, got {type(parent_method)}"
        self.parent_method = parent_method
        return self

    def run(self):
        if not self.parent_method.executed:
            self.parent_method.run()
        SystemMixin.copy_from_upstream(self, self.parent_method)
        MOsMixin.copy_from_upstream(self, self.parent_method)

        minao_info = forte2.basis_utils.BasisInfo(self.system, self.system.minao_basis)
        self.minao_labels = minao_info.basis_labels
        self.atom_to_aos = minao_info.atom_to_aos

        self.atom_normals = self._parse_subspace_pi_planes()
        self.subspace_counter = 0
        self.minao_subspace = []
        for subspace in self.subspace:
            self._parse_subspace(subspace)
        print(self.minao_subspace)

        self.ao_projector = self._make_ao_space_projector()
        self._make_avas_orbitals()
        self.executed = True
        return self

    def _check_parameters(self):
        self.selection_method = self.selection_method.lower()
        assert self.selection_method in [
            "cumulative",
            "cutoff",
            "separate",
            "total",
        ], f"Invalid selection method: {self.selection_method}"

        if self.selection_method == "cumulative":
            assert (
                self.sigma >= 0.0 and self.sigma <= 1.0 + 1e-10
            ), f"Sigma must be in [0, 1], got {self.sigma}"
        elif self.selection_method == "cutoff":
            assert self.cutoff > 0.0, f"Cutoff must be positive, got {self.cutoff}"
            assert (
                1.0 - self.cutoff > self.evals_threshold
            ), f"Cutoff {self.cutoff} is smaller than 1-evals_threshold, {1-self.evals_threshold}, no orbitals will be selected."
        elif self.selection_method == "separate":
            assert (
                self.num_active_occ > 0 and self.num_active_vir > 0
            ), "Number of active occupied and virtual orbitals must be positive."
        elif self.selection_method == "total":
            raise NotImplementedError(
                "'Total' AVAS selection is not implemented yet. Use 'cumulative', 'cutoff', or 'separate'."
            )
            assert self.num_active > 0, "Number of active orbitals must be positive."

    def _parse_subspace(self, ss_str):
        mgroups = re.match(self._regex, ss_str).groups()

        # m[0] is the element symbol
        try:
            Z = ATOM_SYMBOL_TO_Z[mgroups[0].upper()]
        except KeyError:
            raise ValueError(
                f"Invalid element symbol in subspace specification: {mgroups[0]}"
            )

        # m[1] is the start index, m[2] is the end index
        if mgroups[1] is None and mgroups[2] is None:
            # no index specified, use all atoms of the element
            start = 1
            end = self.system.atom_counts[Z] + 1
        elif mgroups[1] is None and mgroups[2] is not None:
            # catches the edge case of "C-3"
            raise ValueError(
                "Invalid subspace specification: start index is not specified but end index is."
            )
        else:
            # if only start is specified, only one atom is selected
            # if both start and end are specified, use the range
            start = 1 if mgroups[1] is None else int(mgroups[1])
            end = start + 1 if mgroups[2] is None else int(mgroups[2]) + 1

        # mgroups[3] contains the subset of AOs e.g. "2p", "2pz", "3dz2" etc.
        if mgroups[3] is None:
            # select all AOs of the element, subject to subspace_planes
            for A in range(start, end):
                in_plane = (Z, A) in self.atom_normals
                for pos in self.atom_to_aos[Z][A]:
                    if self.minao_labels[pos].l == 1 and in_plane:
                        raise NotImplementedError(
                            "Subspace selection for p orbitals in a plane is not implemented yet."
                        )
                    else:
                        self.minao_subspace.append((pos, self.subspace_counter, 1.0))
                        self.subspace_counter += 1
        else:
            n = int(mgroups[3][0])
            for A in range(start, end):
                if mgroups[3][1:].lower() == "p" and (Z, A) in self.atom_normals:
                    raise NotImplementedError(
                        "Subspace selection for p orbitals in a plane is not implemented yet."
                    )
                else:
                    lm = forte2.basis_utils.shell_label_to_lm(mgroups[3][1:].lower())
                    for l, m in lm:
                        for pos in self.atom_to_aos[Z][A]:
                            if (
                                self.minao_labels[pos].n == n
                                and self.minao_labels[pos].l == l
                                and self.minao_labels[pos].m == m
                            ):
                                self.minao_subspace.append(
                                    (pos, self.subspace_counter, 1.0)
                                )
                                self.subspace_counter += 1

    def _make_ao_space_projector(self):
        # Overlap Matrix in Minimal AO Basis
        Smm = forte2.ints.overlap(self.system.minao_basis)
        nbf_m = Smm.shape[0]
        # Build Cms: minimal AO basis to subspace AO basis
        nbf_s = self.subspace_counter
        Cms = np.zeros((nbf_m, nbf_s))
        for m, s, c in self.minao_subspace:
            # m is the index of the minimal AO
            # s is the index of the subspace AO
            # c is the coefficient of the subspace AO in the minimal AO basis
            Cms[m, s] = c
        # Subspace overlap matrix
        Sss = Cms.T @ Smm @ Cms
        # Orthogonalize Sss: Xss = Sss^(-1/2)
        evals, evecs = np.linalg.eigh(Sss)
        Xss = evecs @ np.diag((1.0 / np.sqrt(evals))) @ evecs.T
        # Build overlap matrix between subspace and computational (large) basis
        Sml = forte2.ints.overlap(self.system.minao_basis, self.system.basis)
        # Project into subspace
        Ssl = Cms.T @ Sml
        # AO projector
        # Pao = Ssl^T Sss^-1 Ssl = (Cms^T Sml)^T (Xss^T Xss) (Cms^T Sml)
        Xsl = Xss @ Ssl
        Pao = Xsl.T @ Xsl
        return Pao

    def _make_avas_orbitals(self):
        ndocc = self.parent_method.ndocc
        nsocc = (
            0 if not hasattr(self.parent_method, "nsocc") else self.parent_method.nsocc
        )
        nuocc = self.parent_method.nuocc
        nmo = self.system.nmo

        CpsC = self.parent_method.C[0].T @ self.ao_projector @ self.parent_method.C[0]

        logger.log_info1("MOs with significant overlap with the subspace (> 1.00e-3):")
        for i in range(nmo):
            if CpsC[i, i] > 1.0e-3:
                logger.log_info1(f"{i:3d} {CpsC[i, i]:.6f}")

        docc_sl = slice(0, ndocc)
        uocc_sl = slice(ndocc + nsocc, nmo)

        if self.diagonalize:
            U = np.zeros((nmo, nmo))
            s_docc, Udocc = np.linalg.eigh(CpsC[docc_sl, docc_sl])
            U[docc_sl, docc_sl] = Udocc
            s_uocc, Uuocc = np.linalg.eigh(CpsC[uocc_sl, uocc_sl])
            U[uocc_sl, uocc_sl] = Uuocc
        else:
            s_docc = CpsC[docc_sl, docc_sl].diagonal()
            s_uocc = CpsC[uocc_sl, uocc_sl].diagonal()
            U = np.eye(nmo)

        s_sum = np.sum(s_docc) + np.sum(s_uocc)
        logger.log_info1(f"Sum of eigenvalues of the projected overlap: {s_sum:.6f}")
        argsort = np.argsort(np.concatenate((s_docc, s_uocc)))[::-1]
        s_all = np.zeros((ndocc + nuocc, 3), dtype=float)
        s_all[:, 0] = np.concatenate((s_docc, s_uocc))
        s_all[:, 1] = np.concatenate(([1] * ndocc, [0] * nuocc))
        s_all[:, 2] = np.concatenate(
            (np.arange(ndocc), np.arange(nuocc) + ndocc + nsocc)
        )
        s_all = s_all[argsort]

        act_docc = []
        act_uocc = []
        inact_docc = []
        inact_uocc = []

        s_act_sum = 0.0

        if self.selection_method == "separate":
            nact_docc = nact_uocc = 0

            for imo in s_all:
                if imo[1] == 1:  # occupied
                    if nact_docc < self.num_active_occ:
                        act_docc.append(imo[2])
                        s_act_sum += imo[0]
                        nact_docc += 1
                    else:
                        inact_docc.append(imo[2])
                else:  # unoccupied
                    if nact_uocc < self.num_active_vir:
                        act_uocc.append(imo[2])
                        s_act_sum += imo[0]
                        nact_uocc += 1
                    else:
                        inact_uocc.append(imo[2])
        elif self.selection_method == "cutoff":
            for imo in s_all:
                sig = imo[0]
                if sig > self.cutoff and sig > self.evals_threshold:
                    if imo[1] == 1:
                        act_docc.append(imo[2])
                    else:
                        act_uocc.append(imo[2])
                    s_act_sum += sig
                else:
                    if imo[1] == 1:
                        inact_docc.append(imo[2])
                    else:
                        inact_uocc.append(imo[2])
        elif self.selection_method == "cumulative":
            for imo in s_all:
                sig = imo[0]
                if s_act_sum / s_sum <= self.sigma and sig >= self.evals_threshold:
                    if imo[1] == 1:
                        act_docc.append(imo[2])
                    else:
                        act_uocc.append(imo[2])
                    s_act_sum += sig
                else:
                    if imo[1] == 1:
                        inact_docc.append(imo[2])
                    else:
                        inact_uocc.append(imo[2])

        inact_docc = np.array(inact_docc, dtype=int)
        inact_uocc = np.array(inact_uocc, dtype=int)
        act_docc = np.array(act_docc, dtype=int)
        act_uocc = np.array(act_uocc, dtype=int)

        logger.log_info1(f"AVAS covers {100*s_act_sum/s_sum:.2f}% of the subspace.")

        logger.log_info1("Chosen active orbitals:")
        for i in act_docc:
            logger.log_info1(f"  {i:3d} {s_docc[i]:.6f} (occ)")
        for i in act_uocc:
            logger.log_info1(f"  {i:3d} {s_uocc[i - ndocc - nsocc]:.6f} (virt)")

        # reminder that C_tilde will have zero SOCC coefficients, if ROHF
        C_tilde = self.C[0] @ U
        fock = self.parent_method.F[0]
        C_inact_docc = self._canonicalize_block(fock, C_tilde, inact_docc)
        C_inact_uocc = self._canonicalize_block(fock, C_tilde, inact_uocc)
        C_act_docc = self._canonicalize_block(fock, C_tilde, act_docc)
        C_act_uocc = self._canonicalize_block(fock, C_tilde, act_uocc)

        # fill the C matrix as follows:
        # [C_inact_docc, C_act_docc, !!C_socc!!, C_act_uocc, C_inact_uocc]
        # !! C_socc has not been changed since parent_method.
        n_inact_docc = len(inact_docc)
        n_act_docc = len(act_docc)
        n_act_uocc = len(act_uocc)
        id_sl = slice(0, n_inact_docc)
        ad_sl = slice(id_sl.stop, n_inact_docc + n_act_docc)
        au_sl = slice(ad_sl.stop + nsocc, ad_sl.stop + nsocc + n_act_uocc)
        iu_sl = slice(au_sl.stop, nmo)

        self.C[0][:, id_sl] = C_inact_docc
        self.C[0][:, ad_sl] = C_act_docc
        self.C[0][:, au_sl] = C_act_uocc
        self.C[0][:, iu_sl] = C_inact_uocc

    def _canonicalize_block(self, F, C, mos):
        C_sub = C[:, mos]
        F_sub = C_sub.T @ F @ C_sub
        _, U_sub = np.linalg.eigh(F_sub)
        return C_sub @ U_sub

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
        system = self.system

        # return empty dictionary if no planes are defined
        if not planes_expr:
            return {}
        else:
            raise NotImplementedError("Subspace pi-planes are not implemented yet.")

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
