from dataclasses import dataclass, field
import numpy as np
import re

from forte2 import ints
from forte2.scf import RHF, ROHF, GHF
from forte2.state import MOSpace
from forte2.helpers import logger, invsqrt_matrix, block_diag_2x2
from forte2.base_classes.mixins import MOsMixin, SystemMixin, MOSpaceMixin
from forte2.system import System
from forte2.system.basis_utils import BasisInfo, shell_label_to_lm
from forte2.data import ATOM_SYMBOL_TO_Z


@dataclass
class AVAS(MOsMixin, SystemMixin, MOSpaceMixin):
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
        - "separate": Selects occupied and virtual orbitals separately. Must define `num_active_docc` and `num_active_uocc`.
        - "total": Selects a total number of active orbitals. Must define `num_active`.
    diagonalize : bool, optional, default=True
        Whether to diagonalize the occupied and virtual space overlap matrices.
    sigma : float, optional, default=0.98
        Cumulative cutoff for the eigenvalues of the overlap matrix, controlling the size of the active space.
    cutoff : float, optional, default=1.0
        Cutoff for the eigenvalues of the overlap matrix; eigenvalues greater than this value are considered active.
    evals_threshold : float, optional, default=1.0e-6
        Threshold below which an eigenvalue of the projected overlap is considered zero.
    num_active_docc : int, optional, default=0
        Number of active doubly occupied orbitals. This is on top of any singly occupied orbitals if an ROHF reference is used.
    num_active_uocc : int, optional, default=0
        Number of active unoccupied orbitals.
    num_active : int, optional, default=0
        Total number of active orbitals. Again, this is on top of any singly occupied orbitals if an ROHF reference is used.


    Notes
    -----
    The allow subspace specification is a list of strings, non-exhaustive examples::

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
    num_active_docc: int = 0
    num_active_uocc: int = 0

    executed: bool = field(init=False, default=False)

    def __post_init__(self):
        self._regex = "([a-zA-Z]{1,2})([0-9]+)?-?([0-9]+)?\\(?((?:\\/?[1-9]{1}[spdfgh]{1}[a-zA-Z0-9-]*)*)\\)?"
        self._check_parameters()

    def __call__(self, parent_method):
        assert isinstance(
            parent_method, (RHF, ROHF, GHF)
        ), f"Parent method must be RHF, ROHF, or GHF, got {type(parent_method)}"
        if isinstance(parent_method, ROHF):
            logger.log_info1(
                "*** AVAS will take all singly occupied orbitals to be active! ***"
            )
        self.parent_method = parent_method
        return self

    def run(self):
        if not self.parent_method.executed:
            self.parent_method.run()
        SystemMixin.copy_from_upstream(self, self.parent_method)
        MOsMixin.copy_from_upstream(self, self.parent_method)
        self.nmo = self.C[0].shape[1]
        self.two_component = self.system.two_component
        self.dtype = float if not self.two_component else complex

        self.basis_info = BasisInfo(self.system, self.system.basis)
        minao_info = BasisInfo(self.system, self.system.minao_basis)
        self.minao_labels = minao_info.basis_labels
        self.atom_to_aos = minao_info.atom_to_aos

        logger.log_info1("\nEntering Atomic Valence Active Space (AVAS) procedure")
        logger.log_info1("\n1. Parsing the subspace specification")
        self.atom_normals = self._parse_subspace_pi_planes()
        self.subspace_counter = 0
        self.minao_subspace = []
        for subspace in self.subspace:
            self._parse_subspace(subspace)
        self._print_subspace_info()

        logger.log_info1("\n2. Building the AVAS projector")
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
                self.num_active_docc > 0 and self.num_active_uocc > 0
            ), "Number of active occupied and virtual orbitals must be positive."
        elif self.selection_method == "total":
            assert self.num_active > 0, "Number of active orbitals must be positive."

    def _parse_subspace(self, ss_str):
        found = False
        mgroups = re.match(self._regex, ss_str).groups()

        # m[0] is the element symbol
        try:
            Z = ATOM_SYMBOL_TO_Z[mgroups[0].upper()]
        except KeyError:
            raise ValueError(
                f"Invalid element symbol in subspace specification: {mgroups[0]}"
            )
        if Z not in self.system.atom_counts:
            raise ValueError(f"Element {mgroups[0]} is not present in the system.")

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
            # no subset specified (e.g. "C")
            # select all AOs of the element, subject to subspace_planes
            for A in range(start, end):
                in_plane = (Z, A) in self.atom_normals
                for pos in self.atom_to_aos[Z][A]:
                    if self.minao_labels[pos].l == 1 and in_plane:
                        # plane normal is defined [x,y,z], AOs are defined [y,z,x]
                        m = (self.minao_labels[pos].m + 1) % 3
                        c = self.atom_normals[(Z, A)][m]
                        self.minao_subspace.append((pos, self.subspace_counter, c))
                        found = True
                        # Only add 1 to subspace_counter for an entire set of p orbitals
                        # here we use m=2 arbitrarily
                        self.subspace_counter += (
                            1 if self.minao_labels[pos].m == 2 else 0
                        )
                    else:
                        self.minao_subspace.append((pos, self.subspace_counter, 1.0))
                        found = True
                        self.subspace_counter += 1
        else:
            n = int(mgroups[3][0])
            for A in range(start, end):
                # if the subset is e.g., "C(2p)" and subspace_pi_planes is specified,
                # we need to check if the atom is in the plane,
                # and only add one linear combination of the p orbitals
                if mgroups[3][1:].lower() == "p" and (Z, A) in self.atom_normals:
                    for pos in self.atom_to_aos[Z][A]:
                        if (
                            self.minao_labels[pos].n == n
                            and self.minao_labels[pos].l == 1
                        ):
                            # plane normal is defined [x,y,z], AOs are defined [y,z,x]
                            m = (self.minao_labels[pos].m + 1) % 3
                            c = self.atom_normals[(Z, A)][m]
                            self.minao_subspace.append((pos, self.subspace_counter, c))
                            found = True
                    # Only add 1 to subspace_counter for an entire set of p orbitals
                    self.subspace_counter += 1
                # if the subset is completely specified, e.g., "C2-3(3dz2)",
                # we disregard the planes (if any), we only add the specified AO
                else:
                    lm = shell_label_to_lm(mgroups[3][1:].lower())
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
                                found = True
                                self.subspace_counter += 1

        if not found:
            raise ValueError(
                f"Subspace specification {ss_str} does not match any AOs in the minao_basis."
            )

    def _print_subspace_info(self):
        logger.log_info1(
            "The following AOs from the minimal AO basis will be used to build the AVAS projector:"
        )
        logger.log_info1(
            "(AO and atom labels are 0-indexed, "
            "whereas relative atom indices are 1-indexed, e.g., C1 is the first carbon atom.)"
        )
        logger.log_info1("=" * 41)
        logger.log_info1(
            f"{'AO':<5} {'Atom':<5} {'Label':<5} {'AO label':<10} {'Coefficient':<12}"
        )
        logger.log_info1("-" * 41)
        for i in self.minao_subspace:
            label = self.minao_labels[i[0]]
            logger.log_info1(f"{label} {i[2]:<12.6f}")
        logger.log_info1("-" * 41)
        expl = self.subspace_counter < len(self.minao_subspace)
        logger.log_info1(
            f"Number of subspace orbitals: {self.subspace_counter} {'*' if expl else ''}"
        )
        logger.log_info1("=" * 41)
        if self.subspace_counter < len(self.minao_subspace):
            logger.log_info1(
                "* Number of subspace orbitals is smaller than the number of minimal AO basis functions due \n"
                "  to the specification of the subspace_pi_planes. Fixed linear combinations of p orbitals \n"
                "  specified by the coefficients will be used to build the AVAS projector."
            )

    def _make_ao_space_projector(self):
        # Overlap Matrix in Minimal AO Basis
        Smm = ints.overlap(self.system.minao_basis)
        nbf_m = Smm.shape[0]
        # Build Cms: minimal AO basis to subspace AO basis
        nbf_s = self.subspace_counter
        Cms = np.zeros((nbf_m, nbf_s), dtype=self.dtype)
        for m, s, c in self.minao_subspace:
            # m is the index of the minimal AO
            # s is the index of the subspace AO
            # c is the coefficient of the subspace AO in the minimal AO basis
            Cms[m, s] = c
        # Subspace overlap matrix
        Sss = Cms.T.conj() @ Smm @ Cms
        # Orthogonalize Sss: Xss = Sss^(-1/2)
        Xss = invsqrt_matrix(Sss)
        # Build overlap matrix between subspace and computational (large) basis
        Sml = ints.overlap(self.system.minao_basis, self.system.basis)
        # Project into subspace
        Ssl = Cms.T.conj() @ Sml
        # AO projector
        # Pao = Ssl^T Sss^-1 Ssl = (Cms^T Sml)^T (Xss^T Xss) (Cms^T Sml)
        Xsl = Xss @ Ssl
        Pao = Xsl.T.conj() @ Xsl
        if self.two_component:
            Pao = block_diag_2x2(Pao)
        return Pao

    def _make_avas_orbitals(self):
        if self.two_component:
            # GHF has the "nocc" attribute instead of "ndocc"
            ndocc = self.parent_method.nocc
        else:
            ndocc = self.parent_method.ndocc
        nsocc = (
            getattr(self.parent_method, "nsocc", 0)
        )
        nuocc = self.parent_method.nuocc

        CpsC = self.C[0].T.conj() @ self.ao_projector @ self.C[0]

        logger.log_info1(
            "\nMOs with significant overlap with the subspace (> 1.00e-3):"
        )
        logger.log_info1("(MOs are 0-indexed)")
        logger.log_info1(
            "These are the diagonal elements of C.T @ Pao @ C (the projected overlap matrix),\n"
            "which will be used to select the AVAS orbitals if diagonalize=False, \n"
            "otherwise the eigenvalues of the projected overlap matrix will be used.)\n"
        )
        logger.log_info1("=" * 18)
        logger.log_info1(f"{'# MO':<5} {'<phi|P|phi>':<12}")
        logger.log_info1("-" * 18)
        print_mos = []
        for i in range(self.nmo):
            if CpsC[i, i] > 1.0e-3:
                print_mos.append(i)
                logger.log_info1(f"{i:<5d} {CpsC[i, i].real:<12.6f}")
        logger.log_info1("=" * 18)
        logger.log_info1("AO Composition of MOs with significant overlap:")
        self.basis_info.print_ao_composition(
            self.C[0],
            print_mos,
            nprint=5,
            thres=1.0e-3,
            spinorbital=True,
        )

        docc_sl = slice(0, ndocc)
        uocc_sl = slice(ndocc + nsocc, self.nmo)

        if self.diagonalize:
            logger.log_info1(
                "\ndiagonalize=True, diagonalizing the projected overlap matrix"
            )
            logger.log_info1(
                "The eigenvalues of the projected overlap matrix will be used to select the AVAS orbitals."
            )
            U = np.zeros((self.nmo, self.nmo), dtype=self.dtype)
            s_docc, Udocc = np.linalg.eigh(CpsC[docc_sl, docc_sl])
            U[docc_sl, docc_sl] = Udocc
            s_uocc, Uuocc = np.linalg.eigh(CpsC[uocc_sl, uocc_sl])
            U[uocc_sl, uocc_sl] = Uuocc
            sigma_type = "eigen"
        else:
            logger.log_info1(
                "\ndiagonalize=False, collecting the diagonal elements of the projected overlap matrix"
            )
            logger.log_info1(
                "The diagonal elements of the projected overlap matrix will be used to select the AVAS orbitals."
            )
            s_docc = np.real(CpsC[docc_sl, docc_sl].diagonal())
            s_uocc = np.real(CpsC[uocc_sl, uocc_sl].diagonal())
            U = np.eye(self.nmo, dtype=self.dtype)
            sigma_type = "diagonal "

        s_sum = np.sum(s_docc) + np.sum(s_uocc)
        logger.log_info1(
            f"Sum of {sigma_type}values of the projected overlap: {s_sum:.6f}"
        )
        argsort = np.argsort(np.concatenate((s_docc, s_uocc)))[::-1]
        sigmas = np.concatenate((s_docc, s_uocc))[argsort]
        occupations = np.concatenate(([1] * ndocc, [0] * nuocc), dtype=int)[argsort]
        indices = np.concatenate(
            (np.arange(ndocc), np.arange(nuocc) + ndocc + nsocc), dtype=int
        )[argsort]
        nsig = len(sigmas)

        act_docc = []
        act_uocc = []
        inact_docc = []
        inact_uocc = []

        s_act_sum = 0.0

        logger.log_info1(
            f"\n3. Constructing AVAS orbitals using the {self.selection_method} selection method"
        )
        if self.selection_method == "separate":
            nact_docc = nact_uocc = 0
            for imo in range(nsig):
                if occupations[imo] == 1:  # doubly occupied
                    if nact_docc < self.num_active_docc:
                        act_docc.append(indices[imo])
                        s_act_sum += sigmas[imo]
                        nact_docc += 1
                    else:
                        inact_docc.append(indices[imo])
                else:  # unoccupied
                    if nact_uocc < self.num_active_uocc:
                        act_uocc.append(indices[imo])
                        s_act_sum += sigmas[imo]
                        nact_uocc += 1
                    else:
                        inact_uocc.append(indices[imo])
        elif self.selection_method == "cutoff":
            for imo in range(nsig):
                sig = sigmas[imo]
                if sig > self.cutoff and sig > self.evals_threshold:
                    if occupations[imo] == 1:
                        act_docc.append(indices[imo])
                    else:
                        act_uocc.append(indices[imo])
                    s_act_sum += sig
                else:
                    if occupations[imo] == 1:
                        inact_docc.append(indices[imo])
                    else:
                        inact_uocc.append(indices[imo])
        elif self.selection_method == "total":
            for imo in range(self.num_active):
                if occupations[imo] == 1:
                    act_docc.append(indices[imo])
                else:
                    act_uocc.append(indices[imo])
                s_act_sum += sigmas[imo]
            for imo in range(self.num_active, nsig):
                if occupations[imo] == 1:
                    inact_docc.append(indices[imo])
                else:
                    inact_uocc.append(indices[imo])
        elif self.selection_method == "cumulative":
            for imo in range(nsig):
                sig = sigmas[imo]
                if s_act_sum / s_sum <= self.sigma and sig >= self.evals_threshold:
                    if occupations[imo] == 1:
                        act_docc.append(indices[imo])
                    else:
                        act_uocc.append(indices[imo])
                    s_act_sum += sig
                else:
                    if occupations[imo] == 1:
                        inact_docc.append(indices[imo])
                    else:
                        inact_uocc.append(indices[imo])

        inact_docc = np.array(inact_docc, dtype=int)
        inact_uocc = np.array(inact_uocc, dtype=int)
        act_docc = np.array(act_docc, dtype=int)
        act_uocc = np.array(act_uocc, dtype=int)

        logger.log_info1(
            f"\nSum of {sigma_type}values of selected orbitals:\t{s_act_sum:.6f}"
        )
        logger.log_info1(f"Sum of {sigma_type}values of all orbitals:\t\t{s_sum:.6f}")
        logger.log_info1("-" * 60)
        logger.log_info1(
            f"AVAS coverage of the subspace:\t\t\t{100*s_act_sum/s_sum:.2f}%\n"
        )

        logger.log_info1("AVAS has chosen the following orbitals:")
        logger.log_info1("=" * 25)
        logger.log_info1(f"{'# ^':<5} {sigma_type+' *':<12} {'occ':<6}")
        logger.log_info1("-" * 25)
        for i in act_docc:
            logger.log_info1(f"{i:<5d} {s_docc[i]:<12.6f} {'2':<6}")
        for i in range(ndocc, ndocc + nsocc):
            logger.log_info1(f"{i:<5d} {'-':<12} {'1 **':<6}")
        for i in act_uocc:
            logger.log_info1(f"{i:<5d} {s_uocc[i - ndocc - nsocc]:<12.6f} {'0':<6}")
        logger.log_info1("=" * 25)
        logger.log_info1(
            "^ These indices are internal to the re-sorted AVAS orbitals, and may not correspond to the original MOs."
        )
        if sigma_type == "eigen":
            logger.log_info1("* 'eigen': eigenvalue of the projected overlap matrix")
        if sigma_type == "diagonal ":
            logger.log_info1(
                "* 'diagonal': diagonal element of the projected overlap matrix"
            )
        if nsocc > 0:
            logger.log_info1(
                "** Singly occupied orbitals (occ = 1) are always selected!"
            )
        self.nactv = len(act_docc) + len(act_uocc) + nsocc
        self.ncore = len(inact_docc)

        # AVAS is a provider of MOSpace: avas.mo_space be automatically used downstream if not overridden
        self.mo_space = MOSpace(
            nmo=self.nmo,
            active_orbitals=list(range(self.ncore, self.ncore + self.nactv)),
            core_orbitals=list(range(self.ncore)),
        )

        logger.log_info1(f"\nNumber of core orbitals:      {self.ncore}")
        logger.log_info1(f"Number of active orbitals:    {self.nactv}")

        logger.log_info1("\n4. Canonicalizing the AVAS orbitals")
        # reminder that C_tilde will have zero SOCC coefficients, if ROHF
        C_tilde = self.C[0] @ U
        fock = self.parent_method.F[0]
        # separately canonicalize the Fock matrix blocks
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
        iu_sl = slice(au_sl.stop, self.nmo)

        self.C[0][:, id_sl] = C_inact_docc
        self.C[0][:, ad_sl] = C_act_docc
        self.C[0][:, au_sl] = C_act_uocc
        self.C[0][:, iu_sl] = C_inact_uocc

        logger.log_info1(
            "\nAO composition of final canonicalized active MOs prepared by AVAS:"
        )
        self.basis_info.print_ao_composition(
            self.C[0],
            list(range(ad_sl.start, au_sl.stop)),
            spinorbital=True,
        )

    def _canonicalize_block(self, F, C, mos):
        C_sub = C[:, mos]
        F_sub = C_sub.T.conj() @ F @ C_sub
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

        # test input
        if not isinstance(system, System):
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

        xyz = system.atomic_positions
        centroid = system.centroid
        atom_to_center = system.atom_to_center

        # parse planes
        atom_dirs = {}
        atom_regex = r"([A-Za-z]{1,2})\s*(\d*)\s*-?\s*(\d*)"

        for n, plane_atoms in enumerate(planes_expr):
            plane = []  # absolute index for atoms forming the plane
            plane_z = []  # pair of atomic number and relative index

            # parse each plane entry
            for atom_expr in plane_atoms:
                atom_expr = atom_expr.upper()

                m = re.match(atom_regex, atom_expr)
                if not m:
                    raise ValueError("Invalid expression of atoms!")

                atom, start_str, end_str = m.groups()
                Z = ATOM_SYMBOL_TO_Z[atom]
                if Z not in atom_to_center:
                    raise ValueError(f"Atom '{atom}' not in molecule!")

                start = 1
                end = int(end_str) if end_str else len(atom_to_center[Z])
                if start_str:
                    start = int(start_str)
                    end = int(end_str) if end_str else start

                for i in range(start - 1, end):
                    plane.append(atom_to_center[Z][i])
                    # relative index is 1-indexed (e.g. C1 is the first carbon atom)
                    plane_z.append((Z, i + 1))

            # compute the plane unit normal (smallest principal axis)
            plane_xyz = xyz[plane]
            plane_centroid = np.mean(plane_xyz, axis=0)
            plane_xyz = plane_xyz - plane_centroid

            # SVD the xyz coordinate
            u, s, vh = np.linalg.svd(plane_xyz)

            # fix phase
            p = plane_centroid - centroid
            plane_normal = vh[2] if np.inner(vh[2], p) >= 0.0 else vh[2] * -1.0

            # attach each atom to the unit normal
            for z_i in plane_z:
                if z_i in atom_dirs:
                    atom_dirs[z_i] = atom_dirs[z_i] + plane_normal
                else:
                    atom_dirs[z_i] = plane_normal

        # normalize the directions on each requested atom
        atom_dirs = {z_i: n / np.linalg.norm(n) for z_i, n in atom_dirs.items()}

        return atom_dirs
