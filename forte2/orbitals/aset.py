from dataclasses import dataclass, field
import numpy as np
import re

import forte2
from forte2.lib import ints
from forte2.state import MOSpace, EmbeddingMOSpace
from forte2.system.basis_utils import BasisInfo
from forte2.helpers import logger
from forte2.helpers.matrix_functions import block_diag_2x2, block_eigh
from forte2.base_classes import ActiveSpaceDriver, Method
from forte2.data import ATOM_SYMBOL_TO_Z
from forte2.orbitals.semicanonicalizer import Semicanonicalizer
from forte2.orbitals.final_orbitals import FinalOrbitals


@dataclass
class ASET(Method):
    """
    Zeroth-order Active Space Embedding Theory (ASET(0)) method for paritioning and projecting molecules.

    Parameters
    ----------
    fragment : list[str]
        List of atomic symbols defining the fragment.
    frozen_core_orbitals : int | list[int], optional, default=None
        Number or list of frozen core orbital indices.
    frozen_virtual_orbitals : int | list[int], optional, default=None
        Number or list of frozen virtual orbital indices.
    cutoff_method : str, optional, default="threshold"
        Method for choosing the embedding cutoff. Options include "threshold", "num_of_orbitals".
    cutoff : float, optional, default = 0.5
        Projector eigenvalue for simple threshold methods.
    num_A_occ : int | None, optional, default=0
        Number of occupied orbitals assigned to fragment A when ``cutoff_method`` is
        ``"num_of_orbitals"``. If ``None``, occupied orbitals fall back to ``cutoff``.
    num_A_vir : int | None, optional, default=0
        Number of virtual orbitals assigned to fragment A when ``cutoff_method`` is
        ``"num_of_orbitals"``. If ``None``, virtual orbitals fall back to ``cutoff``.

    Notes
    -----
    The allowed subspace specification is a list of strings, non-exhaustive examples::

    - ["C"]              # all carbon atoms
    - ["C","N"]          # all carbon and nitrogen atoms
    - ["C1"]             # carbon atom #1
    - ["C1-7"]           # carbon atoms #1 through #7
    - ["C1-3","N2"]      # carbon atoms #1, #2, #3 and nitrogen atom #2

    If the parent method is two-component, the partition is carried out in the
    spinor basis. The fragment projector is spin-independent, so it commutes with
    time reversal and never assigns the two members of a Kramers pair to different
    subspaces.

    References
    ----------
    .. [1] N. He and F. A. Evangelista, "A zeroth-order active-space frozen-orbital embedding scheme for multireference calculations",
            J. Chem. Phys. 2020, 152 (9), 094107
    """

    fragment: list
    frozen_core_orbitals: int | list[int] = None
    frozen_virtual_orbitals: int | list[int] = None
    cutoff_method: str = "threshold"
    cutoff: float = 0.5
    num_A_occ: int | None = 0
    num_A_vir: int | None = 0

    executed: bool = field(default=False, init=False)
    final_orbitals: FinalOrbitals = field(default="semicanonical", init=False)

    def __post_init__(self):
        self._regex = r"^([a-zA-Z]{1,2})([0-9]+)?-?([0-9]+)??$"
        self._check_parameters()
        self.requires = {"system", "mos"}
        self.provides = {"system", "mos", "mo_space"}

    def __call__(self, parent_method):
        assert isinstance(
            parent_method,
            (ActiveSpaceDriver, forte2.scf.RHF, forte2.scf.ROHF),
        ), f"Parent method must be MCOptimizer, CI, RHF, or ROHF, got {type(parent_method)}"
        self._register_parent_method(parent_method)
        return self

    def run(self):
        if not self.parent_method.executed:
            self.parent_method.run()
        self.system = self.parent_method.system
        self.mos = self.parent_method.mos.copy()
        nmo = self.mos.C[0].shape[1]
        if isinstance(self.parent_method, ActiveSpaceDriver):
            self.mo_space = self.parent_method.mo_space
            self._g1 = self.parent_method.ci_solver.make_average_rdm(1)
        elif isinstance(self.parent_method, forte2.scf.RHF):
            self.mo_space = MOSpace(
                nmo=nmo,
                core_orbitals=list(range(self.parent_method.ndocc)),
                active_orbitals=[],
            )
            self._g1 = np.zeros((0, 0))
        elif isinstance(self.parent_method, forte2.scf.ROHF):
            nd = self.parent_method.ndocc
            ns = self.parent_method.nsocc
            self.mo_space = MOSpace(
                nmo=nmo,
                core_orbitals=list(range(nd)),
                active_orbitals=list(range(nd, nd + ns)),
            )
            self._g1 = np.eye(ns)
        self.mo_space = self.mo_space.update_frozen_orbitals(
            self.frozen_core_orbitals, self.frozen_virtual_orbitals
        )
        self.ncore = self.mo_space.ncore
        self.nactv = self.mo_space.nactv

        self.nmo = self.mo_space.nmo
        self.nvirt = self.mo_space.nvirt

        ao_info = BasisInfo(self.system, self.system.basis)

        raw_map = ao_info.atom_to_aos

        self.atom_to_center = self.system.atom_to_center
        self.atom_to_aos = {
            int(Z): {int(rel) - 1: list(aos_list) for rel, aos_list in inner.items()}
            for Z, inner in raw_map.items()
        }

        logger.log_info1("\nRunning Active Space Embedding Theory (ASET)")
        logger.log_info1(f"Fragment: {self.fragment}")
        logger.log_info1("\nProjecting orbitals to fragment")
        logger.log_info1(
            f"Cutoff method: {self.cutoff_method} \nCutoff value: {self.cutoff}"
        )
        self.fragment_atoms = self._parse_fragment(self.fragment)
        self.P_ao_frag = self._make_fragment_projector()
        self.partition = self._make_embedding()
        self._print_embedding_info(**self.partition)
        logger.log_info1("\nMO space is updated.")
        self._update_mo_space_from_partition()
        logger.log_info1("\nASET procedure completed.")
        self.executed = True

        return self

    def _check_parameters(self):
        self.cutoff_method = self.cutoff_method.lower()
        assert self.cutoff_method in [
            "threshold",
            "num_of_orbitals",
        ], f"Invalid cutoff method: {self.cutoff_method}"
        if self.cutoff_method == "threshold":
            assert (
                self.cutoff > 0 and self.cutoff < 1
            ), f"threshold must be in (0, 1), got {self.cutoff}"
        elif self.cutoff_method == "num_of_orbitals":
            if self.num_A_occ is None or self.num_A_vir is None:
                assert (
                    self.cutoff > 0 and self.cutoff < 1
                ), f"threshold must be in (0, 1), got {self.cutoff}"
            if self.num_A_occ is not None:
                assert (
                    self.num_A_occ >= 0
                ), f"Number of occupied orbitals in Fragment A must be non-negative, got {self.num_A_occ}"
            if self.num_A_vir is not None:
                assert (
                    self.num_A_vir >= 0
                ), f"Number of virtual orbitals in Fragment A must be non-negative, got {self.num_A_vir}"

    def _parse_fragment(self, frag_str: str | list[str]) -> list[int]:
        """
        Parse a list of fragment specification strings into atom indices.

        Supported input formats (all 1-indexed for the user):
            ["C"]         → all carbon atoms
            ["C1"]        → carbon atom #1
            ["C1-3"]      → carbon atoms #1 through #3
            ["C", "N2"]   → all carbon atoms and nitrogen atom #2

        Parameters
        ----------
        frag_list : list[str]
            A list of fragment specifications like ["C1-3", "N"].

        Returns
        -------
        list[int]
            A sorted list of unique atom indices (0-based) matching the specification.
        """
        frag_list = [frag_str] if isinstance(frag_str, str) else frag_str

        atom_indices = []

        for token in frag_list:
            match = re.match(self._regex, token)
            if not match:
                raise ValueError(f"Invalid fragment specification: {token}")

            symbol = match.group(1)  # e.g., "C"
            start = match.group(2)  # e.g., "1"
            end = match.group(3)  # e.g., "3" (if it's a range)

            try:
                Z = ATOM_SYMBOL_TO_Z[symbol.upper()]
            except KeyError:
                raise ValueError(f"Unknown atom symbol: {symbol}")

            # ensure the element actually exists
            if Z not in self.system.atom_counts:
                raise ValueError(f"Element {symbol} is not present in the system.")

            element_atoms = self.atom_to_center[Z]

            # No index provided (e.g., "C") → select all atoms of this element
            if start is None:
                atom_indices.extend(element_atoms)
                continue

            else:
                # Convert 1-based user indices to 0-based Python indices
                start_idx = int(start) - 1
                end_idx = int(end) if end else start_idx + 1

                # Bounds check
                if end_idx > len(element_atoms):
                    raise IndexError(
                        f"Fragment index range {start}-{end or start} out of bounds "
                        f"for element {symbol} (only {len(element_atoms)} atoms available)."
                    )

                # Slice the appropriate atoms (relative to atoms of that element)
                atom_indices.extend(element_atoms[start_idx:end_idx])

        return sorted(set(atom_indices))

    def _make_fragment_projector(self):
        """
        Build an AO-space fragment projector P = S^T (S_A) S in the AO basis:
        1. Compute AO overlap S_ff
        2. Extract fragment block S_A
        3. Pseudo invert S_A and embed into full AO space
        4. Form P_ao_frag = S_ff @ S_A_nn @ S_ff
        5. For a two-component parent, repeat the projector on the diagonal so
           that it acts identically on the alpha and beta halves of a spinor

        Returns
        -------
        P_ao_frag : ndarray
            The fragment projector matrix S_ff @ S_A_mm @ S_ff. Its dimension is
            the number of basis functions for a one-component parent and twice
            that for a two-component one.
        """
        # 1. Compute AO overlap S_ff
        S_ff = ints.overlap(self.system.basis)
        nbf_f = S_ff.shape[0]

        # 2. Collect ao indices in fragment
        frag_ao_indices = []
        for atom_idx in self.fragment_atoms:
            # atomic number for this atom
            Z_i = int(self.system.atoms[atom_idx][0])

            # make sure that element is present
            if Z_i not in self.atom_to_center:
                raise ValueError(f"Element Z={Z_i} not in system.atom_to_center")

            # get the list of all atom‐indices of this element
            element_atoms = self.atom_to_center[Z_i]  # e.g. [0, 2] for two H’s

            # find which position in that list our atom_idx occupies
            rel_idx = element_atoms.index(atom_idx)

            # now fetch the AO indices for that element & rel_idx
            ao_list = self.atom_to_aos[Z_i][rel_idx]
            frag_ao_indices.extend(ao_list)

        frag_ao_indices = sorted(set(frag_ao_indices))

        # 3. Build fragment overlap block S_A and invert
        S_A = S_ff[np.ix_(frag_ao_indices, frag_ao_indices)]
        S_A_inv = np.linalg.pinv(S_A, rcond=1e-8)

        # 4. Embed S_A_inv into full space and form projector
        S_A_ff = np.zeros((nbf_f, nbf_f))
        for i, mu in enumerate(frag_ao_indices):
            for j, nu in enumerate(frag_ao_indices):
                S_A_ff[mu, nu] = S_A_inv[i, j]

        # Build fragment projector P_ao_frag = S_ff S_A_ff S_ff
        P_ao_frag = S_ff @ S_A_ff @ S_ff

        if self.two_component:
            # The projector is spin-independent, so it commutes with time
            # reversal: its eigenvalues come in Kramers-degenerate pairs and a
            # partition by eigenvalue never separates the members of a pair.
            P_ao_frag = block_diag_2x2(P_ao_frag)

        return P_ao_frag

    @staticmethod
    def _canonicalize_projector_nullspace(
        vals: np.ndarray,
        U: np.ndarray,
        h: np.ndarray,
        C_block: np.ndarray,
        F_ao: np.ndarray,
        tol: float,
        ascending: bool,
        null_tol: float = 1e-10,
    ) -> tuple[np.ndarray, np.ndarray, list[int]]:
        """
        Canonicalize the degenerate zero-eigenvalue null space of a projector block
        using the generalized Fock matrix.

        Returns the updated rotation matrix ``U``, irrep array ``h``, and priority
        index list (non-null orbitals ordered by descending projector eigenvalue,
        followed by null-space orbitals ordered by Fock energy).
        """
        nonnull_pos = [k for k, v in enumerate(vals) if v >= null_tol]
        null_pos = [k for k, v in enumerate(vals) if v < null_tol]
        nonnull_pos.sort(key=lambda k: vals[k], reverse=True)
        if not null_pos:
            return U, h, nonnull_pos

        C_null = C_block @ U[:, null_pos]
        F_null = C_null.conj().T @ F_ao @ C_null
        h_null = h[null_pos]
        f_vals, U_f, h_f = block_eigh(F_null, h_null, atol=0.0, rtol=tol, sort=True)
        if not ascending:
            U_f = U_f[:, ::-1]
            h_f = h_f[::-1]

        U_out = U.copy()
        h_out = h.copy()
        U_out[:, null_pos] = U_out[:, null_pos] @ U_f
        h_out[null_pos] = h_f
        return U_out, h_out, nonnull_pos + null_pos

    def _make_embedding(self):
        """
        Perform Orbital Partitioning for ASET.
        """
        # Copy the input orbitals
        C = self.parent_method.mos.C[0].copy()
        # Sort the orbitals into contiguous core, active, virtual blocks
        C_contig = C[:, self.mo_space.orig_to_contig]

        # Build the fragment projector
        P_ao_frag = self.P_ao_frag
        P_frag = C_contig.conj().T @ P_ao_frag @ C_contig

        # 6) Split P_frag into occupied and virtual blocks
        core = self.mo_space.core
        core_inds = self.mo_space.core_indices
        active_orbitals = [list(active) for active in self.mo_space.active_orbitals]
        virt = self.mo_space.virt
        virt_inds = self.mo_space.virtual_indices

        # Split P_frag into occupied and virtual blocks
        P_frag_oo = P_frag[core, core]
        P_frag_vv = P_frag[virt, virt]

        # irrep of each MO; all rotations below stay within irreps
        irreps = np.array(self.mos.irrep_indices[0])
        tol = getattr(self.system.symmetry_basis, "tol", 0.0)
        try:
            lo_vals, Uo, h_o = block_eigh(
                P_frag_oo, irreps[core_inds], atol=0.0, rtol=tol, sort=True
            )
            lv_vals, Uv, h_v = block_eigh(
                P_frag_vv, irreps[virt_inds], atol=0.0, rtol=tol, sort=True
            )
        except ValueError as e:
            raise ValueError(
                f"The ASET fragment breaks {self.system.point_group} symmetry. Use a "
                "fragment that the symmetry operations map onto itself, or run with "
                "symmetry=False."
            ) from e

        # resort the virtual eigenvalue/eigenvector pairs, and their irreps with them
        lv_vals = lv_vals[::-1]
        Uv = Uv[:, ::-1]
        h_v = h_v[::-1]

        # When cutoff_method="num_of_orbitals" requests more fragment orbitals than the
        # numerical rank of the fragment projector, canonicalize the degenerate zero-eigenvalue
        # null space with the generalized Fock matrix so the A/B cut is deterministic and
        # minimizes off-diagonal F_{A,B} couplings (crucial for smooth ASET(2) PESs).
        null_tol = 1e-10
        need_occ_null = (
            self.cutoff_method == "num_of_orbitals"
            and self.num_A_occ is not None
            and self.num_A_occ > int(np.sum(lo_vals >= null_tol))
            and np.any(lo_vals < null_tol)
        )
        need_vir_null = (
            self.cutoff_method == "num_of_orbitals"
            and self.num_A_vir is not None
            and self.num_A_vir > int(np.sum(lv_vals >= null_tol))
            and np.any(lv_vals < null_tol)
        )
        occ_order = sorted(range(len(core_inds)), key=lambda k: lo_vals[k], reverse=True)
        vir_order = sorted(range(len(virt_inds)), key=lambda k: lv_vals[k], reverse=True)
        if need_occ_null or need_vir_null:
            F_ao = self.system.fock_builder.build_generalized_fock(
                C_core=C_contig[:, self.mo_space.docc],
                C_act=C_contig[:, self.mo_space.actv],
                g1=self._g1,
            )
            if need_occ_null:
                Uo, h_o, occ_order = self._canonicalize_projector_nullspace(
                    lo_vals,
                    Uo,
                    h_o,
                    C[:, core_inds],
                    F_ao,
                    tol=tol,
                    ascending=False,
                    null_tol=null_tol,
                )
            if need_vir_null:
                Uv, h_v, vir_order = self._canonicalize_projector_nullspace(
                    lv_vals,
                    Uv,
                    h_v,
                    C[:, virt_inds],
                    F_ao,
                    tol=tol,
                    ascending=True,
                    null_tol=null_tol,
                )

        irreps[core_inds] = h_o
        irreps[virt_inds] = h_v

        # Ordered (global_mo_idx, projector_eigenvalue) lists
        occ_pairs = [(core_inds[k], lo_vals[k]) for k in occ_order]
        vir_pairs = [(virt_inds[k], lv_vals[k]) for k in vir_order]

        # Apply the unitary transformations to the occupied and virtual blocks of C
        C[:, core_inds] = C[:, core_inds] @ Uo
        C[:, virt_inds] = C[:, virt_inds] @ Uv

        # Partition by cutoff
        index_A_occ, index_B_occ = [], []
        index_A_vir, index_B_vir = [], []
        if self.cutoff_method == "threshold":
            # Select orbitals with eigenvalues above the threshold
            for i, v in occ_pairs:
                (index_A_occ if v > self.cutoff else index_B_occ).append(i)
            for i, v in vir_pairs:
                (index_A_vir if v > self.cutoff else index_B_vir).append(i)

        elif self.cutoff_method == "num_of_orbitals":
            if self.num_A_occ is None:
                for i, v in occ_pairs:
                    (index_A_occ if v > self.cutoff else index_B_occ).append(i)
            else:
                for rank, (glob_i, v) in enumerate(occ_pairs):
                    if rank < self.num_A_occ:
                        index_A_occ.append(glob_i)
                    else:
                        index_B_occ.append(glob_i)
                        if v > 0.5:
                            logger.log_info1(
                                f"Warning! Occupied orbital {glob_i} has eigenvalue {v:8.6f} "
                                "and is assigned to B."
                            )

            if self.num_A_vir is None:
                for i, v in vir_pairs:
                    (index_A_vir if v > self.cutoff else index_B_vir).append(i)
            else:
                for rank, (glob_i, v) in enumerate(vir_pairs):
                    if rank < self.num_A_vir:
                        index_A_vir.append(glob_i)
                    else:
                        index_B_vir.append(glob_i)
                        if v > 0.5:
                            logger.log_info1(
                                f"Warning! Virtual orbital {glob_i} has eigenvalue {v:8.6f} "
                                "and is assigned to B."
                            )
        index_A_occ.sort()
        index_B_occ.sort()
        index_A_vir.sort()
        index_B_vir.sort()

        # Semi-canonicalize the blocks
        frozen_core_inds = self.mo_space.frozen_core_indices
        frozen_virt_inds = self.mo_space.frozen_virtual_indices
        g1 = self._g1
        emb_space = EmbeddingMOSpace(
            nmo=self.nmo,
            frozen_core_orbitals=frozen_core_inds,
            B_core_orbitals=index_B_occ,
            A_core_orbitals=index_A_occ,
            active_orbitals=active_orbitals,
            A_virtual_orbitals=index_A_vir,
            B_virtual_orbitals=index_B_vir,
            frozen_virtual_orbitals=frozen_virt_inds,
        )
        self.emb_space = emb_space

        # Copy the orbitals into embedding contiguous order
        C_emb = C[:, emb_space.orig_to_contig].copy()

        # Semi-canonicalize according to the embedding space
        semican = Semicanonicalizer(
            system=self.system,
            mo_space=emb_space,
            irrep_indices=irreps[emb_space.orig_to_contig],
        )
        semican.semi_canonicalize(g1=g1, C_contig=C_emb)
        self.semican = semican

        # Replace the original orbitals with the semi-canonicalized ones in the original order
        self.mos.C[0] = semican.C_semican[:, emb_space.contig_to_orig].copy()
        # semicanonicalization keeps each orbital's irrep at its position
        names = dict(zip(self.mos.irrep_indices[0], self.mos.irrep_labels[0]))
        self.mos.irrep_indices[0] = irreps.tolist()
        self.mos.irrep_labels[0] = [names[h] for h in irreps]

        return {
            "index_A_occ": index_A_occ,
            "active_orbitals": active_orbitals,
            "index_A_vir": index_A_vir,
            "index_B_occ": index_B_occ,
            "index_B_vir": index_B_vir,
            "lo_vals": lo_vals,
            "lv_vals": lv_vals,
        }

    def _print_embedding_info(
        self, **info: dict[str, np.ndarray | list[int] | list[list[int]]]
    ) -> None:
        """
        Print the sizes and MO lists for fragment embedding
        """
        index_A_occ = info["index_A_occ"]
        index_A_vir = info["index_A_vir"]
        index_B_occ = info["index_B_occ"]
        index_B_vir = info["index_B_vir"]
        active_orbitals = info["active_orbitals"]
        lo_vals: np.ndarray = info["lo_vals"]
        lv_vals: np.ndarray = info["lv_vals"]

        core_inds = self.mo_space.core_indices
        virt_inds = self.mo_space.virtual_indices
        num_Fo = len(self.mo_space.frozen_core_indices)
        num_Ao = len(index_A_occ)
        num_Av = len(index_A_vir)
        num_Bo = len(index_B_occ)
        num_Bv = len(index_B_vir)
        num_Fv = len(self.mo_space.frozen_virtual_indices)
        num_actv = sum(len(gas) for gas in active_orbitals)

        # Fragment A
        logger.log_info1("\n    ASET Fragment MOs:")
        logger.log_info1("    ============================")
        logger.log_info1("      MO     Type    <phi|P|phi>")
        logger.log_info1("    ----------------------------")
        for i in index_A_occ:
            local = core_inds.index(i)
            val = lo_vals[local]
            logger.log_info1(f"      {i:4d}  occupied  {val: 8.6f}")

        if len(active_orbitals) == 1:
            for i in active_orbitals[0]:
                logger.log_info1(f"      {i:4d}  active         --")
        else:
            for gasn, gas in enumerate(active_orbitals):
                for i in gas:
                    logger.log_info1(f"      {i:4d}  GAS{gasn+1}           --")
        for i in index_A_vir:
            local = virt_inds.index(i)
            val = lv_vals[local]
            logger.log_info1(f"      {i:4d}  virtual   {val: 8.6f}")

        # Environment B
        total_env = num_Bo + num_Bv
        if total_env < 50:
            logger.log_info1("\n    ASET Environment MOs:")
            logger.log_info1("    ============================")
            logger.log_info1("      MO     Type    <phi|P|phi>")
            logger.log_info1("    ----------------------------")
            for i in index_B_occ:
                local = core_inds.index(i)
                val = lo_vals[local]
                logger.log_info1(f"      {i:4d}  occupied  {val: 8.6f}")
            for i in index_B_vir:
                local = virt_inds.index(i)
                val = lv_vals[local]
                logger.log_info1(f"      {i:4d}  virtual   {val: 8.6f}")
            logger.log_info1("    ============================")
        else:
            logger.log_info1(
                "\n    More than 50 ASET Embedding MOs (Environment B), no printing."
            )

        # Summary
        logger.log_info1("\n  Summary:")
        logger.log_info1(
            f"    Fragment (A): {num_Ao} Occupied MOs, {num_actv} Active MOs, {num_Av} Virtual MOs"
        )
        logger.log_info1(
            f"    Environment (B): {num_Bo} Occupied MOs, {num_Bv} Virtual MOs"
        )
        logger.log_info1(
            f"    Frozen Orbitals: {num_Fo} Core MOs, {num_Fv} Virtual MOs\n"
        )

    def _update_mo_space_from_partition(self):
        """
        Update the MO space based on the partitioning results.
        """

        self.mo_space = MOSpace(
            nmo=self.nmo,
            active_orbitals=self.partition["active_orbitals"],
            core_orbitals=self.partition["index_A_occ"],
            frozen_core_orbitals=sorted(
                set(self.mo_space.frozen_core_indices + self.partition["index_B_occ"])
            ),
            frozen_virtual_orbitals=sorted(
                set(
                    self.mo_space.frozen_virtual_indices + self.partition["index_B_vir"]
                )
            ),
        )
        return self.mo_space
