from dataclasses import dataclass, field
import numpy as np
import re

import forte2
from forte2 import ints
from forte2.state import MOSpace, EmbeddingMOSpace
from forte2.system.basis_utils import BasisInfo
from forte2.helpers import logger
from forte2.helpers.matrix_functions import invsqrt_matrix
from forte2.base_classes.mixins import MOsMixin, SystemMixin, MOSpaceMixin
from forte2.data import ATOM_SYMBOL_TO_Z
from forte2.orbitals.semicanonicalizer import Semicanonicalizer


@dataclass
class ASET(MOsMixin, SystemMixin, MOSpaceMixin):
    """
    Active Space Embedding Theory (ASET) method for paritioning and projecting molecules.

    Parameters
    ----------
    fragment : list[str]
        List of atomic symbols defining the fragment.
    cutoff_method : str, optional, default="threshold"
        Method for choosing the embedding cutoff. Options include "threshold", "num_of_orbitals".
    cutoff : float, optional, default = 0.5
        Projector eigenvalue for simple threshold methods.
    num_A_occ : int, optional, default=0
        Number of occupied orbitals fixed to this value in fragment A when cutoff method is "num_of_orbitals".
    num_A_vir : int, optional, default=0
        Number of virtual orbitals fixed to this value in fragment A when cutoff method is "num_of_orbitals".
    semicanonicalize_active : bool, optional, default=True
        Whether to semicanonicalize the active space orbitals.
    semicanonicalize_frozen : bool, optional, default=True
        Whether to semicanonicalize the frozen orbitals.

    Notes
    -----
    The allowed subspace specification is a list of strings, non-exhaustive examples::

    - ["C"]              # all carbon atoms
    - ["C","N"]          # all carbon and nitrogen atoms
    - ["C1"]             # carbon atom #1
    - ["C1-7"]           # carbon atoms #1 through #7
    - ["C1-3","N2"]      # carbon atoms #1, #2, #3 and nitrogen atom #2

    See J. Chem. Phys. 2020, 152 (9), 094107 <https://doi.org/10.1063/1.5142481>_ for details on the ASET(mf) method.
    """

    fragment: list
    cutoff_method: str = "threshold"
    cutoff: float = 0.5
    num_A_occ: int = 0
    num_A_vir: int = 0
    semicanonicalize_active: bool = True
    semicanonicalize_frozen: bool = True

    executed: bool = field(default=False, init=False)

    def __post_init__(self):
        self._regex = r"^([A-Z][a-z]?)(\d+)?(?:-(\d+))?$"
        self._check_parameters()

    def __call__(self, parent_method):
        assert isinstance(
            parent_method, forte2.mcopt.MCOptimizer
        ), f"Parent method must be MCSCF, got {type(parent_method)}"
        self.parent_method = parent_method
        return self

    def run(self):
        if not self.parent_method.executed:
            self.parent_method.run()
        SystemMixin.copy_from_upstream(self, self.parent_method)
        MOsMixin.copy_from_upstream(self, self.parent_method)
        self.mo_space = self.parent_method.mo_space
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
        self.fragment = self._parse_fragment(self.fragment)
        self.P_ao_frag = self._make_fragment_projector()
        self.executed = True
        self.partition = self._make_embedding()
        self._print_embedding_info(**self.partition)
        logger.log_info1("\nMO space is updated.")
        self._update_mo_space_from_partition()
        logger.log_info1("\nASET procedure completed.")

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
            ), f"threshold must be in [0, 1], got {self.cutoff}"
        elif self.cutoff_method == "num_of_orbitals":
            assert (
                self.num_A_occ >= 0 and self.num_A_vir >= 0
            ), f"Number of occupied and virtual orbitals in Fragment A must be non-negative, got {self.num_A_occ}, {self.num_A_vir}"

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
                Z = ATOM_SYMBOL_TO_Z[symbol]
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

        Returns
        ------
        P_ao_frag : ndarray
            The fragment projector matrix S_ff @ S_A_mm @ S_ff.
        """
        # 1. Compute AO overlap S_ff
        S_ff = ints.overlap(self.system.basis)
        nbf_f = S_ff.shape[0]

        # 2. Collect ao indices in fragment
        frag_ao_indices = []
        for atom_idx in self.fragment:
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

        return P_ao_frag

    def _make_embedding(self):
        """
        Perform Orbital Partitioning for ASET.
        """
        # Copy the input orbitals and sort them into blocks of frozen core, core, active, ...
        C = self.parent_method.C[0][:, self.mo_space.orig_to_contig]

        # Build the fragment projector
        P_ao_frag = self.P_ao_frag
        P_frag = C.T @ P_ao_frag @ C

        # 6) Split P_frag into occupied and virtual blocks
        core = self.mo_space.core
        core_inds = self.mo_space.core_indices
        actv_inds = self.mo_space.active_indices
        virt = self.mo_space.virt
        virt_inds = self.mo_space.virtual_indices

        # Split P_frag into occupied and virtual blocks
        P_frag_oo = P_frag[core, core]
        P_frag_vv = P_frag[virt, virt]
        lo_vals, Uo = np.linalg.eigh(P_frag_oo)
        lv_vals, Uv = np.linalg.eigh(P_frag_vv)
        # resort the virtual eigenvalue/eigenvector pairs
        lv_vals = lv_vals[::-1]
        Uv = Uv[:, ::-1]
        occ_pairs = sorted(zip(core_inds, lo_vals), key=lambda x: x[1], reverse=True)
        vir_pairs = sorted(zip(virt_inds, lv_vals), key=lambda x: x[1], reverse=True)

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
            for rank, (glob_i, v) in enumerate(occ_pairs):
                if rank < self.num_A_occ:
                    index_A_occ.append(glob_i)
                else:
                    index_B_occ.append(glob_i)
                    if v > 0.5:
                        logger.log_info1(
                            f"Warning! Occupied orbital {glob_i+1} has eigenvalue {v:8.6f} "
                            "and is assigned to B."
                        )

            for rank, (glob_i, v) in enumerate(vir_pairs):
                if rank < self.num_A_vir:
                    index_A_vir.append(glob_i)
                else:
                    index_B_vir.append(glob_i)
                    if v > 0.5:
                        logger.log_info1(
                            f"Warning! Virtual orbital {glob_i+1} has eigenvalue {v:8.6f} "
                            "and is assigned to B."
                        )
        index_A_occ.sort()
        index_B_occ.sort()
        index_A_vir.sort()
        index_B_vir.sort()

        # # Semi-canonicalize the blocks
        if not self.semicanonicalize_active:
            logger.log_info1(
                "\nSkipping semicanonicalization of active space orbitals."
            )
        if not self.semicanonicalize_frozen:
            logger.log_info1(
                "\nSkipping semicanonicalization of frozen core and frozen virtual orbitals."
            )

        frozen_core_inds = self.mo_space.frozen_core_indices
        frozen_virt_inds = self.mo_space.frozen_virtual_indices
        g1 = self.parent_method.ci_solver.make_average_1rdm()
        emb_space = EmbeddingMOSpace(
            nmo=self.nmo,
            frozen_core_orbitals=frozen_core_inds,
            B_core_orbitals=index_B_occ,
            A_core_orbitals=index_A_occ,
            active_orbitals=actv_inds,
            A_virtual_orbitals=index_A_vir,
            B_virtual_orbitals=index_B_vir,
            frozen_virtual_orbitals=frozen_virt_inds,
        )

        semican = Semicanonicalizer(
            system=self.system,
            mo_space=emb_space,
            do_frozen=self.semicanonicalize_frozen,
            do_active=self.semicanonicalize_active,
        )
        semican.semi_canonicalize(g1=g1, C_contig=C)
        self.C[0] = semican.C_semican.copy()

        return {
            "index_A_occ": index_A_occ,
            "index_actv": actv_inds,
            "index_A_vir": index_A_vir,
            "index_B_occ": index_B_occ,
            "index_B_vir": index_B_vir,
            "lo_vals": lo_vals,
            "lv_vals": lv_vals,
        }

    def _print_embedding_info(self, **info: dict[str, np.ndarray | list[int]]) -> None:
        """
        Print the sizes and MO lists for fragment embedding
        """
        index_A_occ = info["index_A_occ"]
        index_actv = info["index_actv"]
        index_A_vir = info["index_A_vir"]
        index_B_occ = info["index_B_occ"]
        index_B_vir = info["index_B_vir"]
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
        num_actv = len(index_actv)

        # Environment A
        logger.log_info1("\nFrozen-Orbital Embedding MOs (Fragment A):")
        logger.log_info1("    ============================")
        logger.log_info1("      MO     Type    <phi|P|phi>")
        logger.log_info1("    ----------------------------")
        for i in index_A_occ:
            local = core_inds.index(i)
            val = lo_vals[local]
            logger.log_info1(f"      {i+1:4d}  occupied  {val:8.6f}")

        for i in index_actv:
            logger.log_info1(f"      {i+1:4d}  active    --")
        for i in index_A_vir:
            local = virt_inds.index(i)
            val = lv_vals[local]
            logger.log_info1(f"      {i+1:4d}  virtual   {val:8.6f}")

        # Environment B
        total_env = num_Bo + num_Bv
        if total_env < 50:
            logger.log_info1("\n    Frozen‑orbital Embedding MOs (Environment B)")
            logger.log_info1("    ============================")
            logger.log_info1("      MO     Type    <phi|P|phi>")
            logger.log_info1("    ----------------------------")
            for i in index_B_occ:
                local = core_inds.index(i)
                val = lo_vals[local]
                logger.log_info1(f"    {i+1:4d}   Occupied   {val:8.6f}")
            for i in index_B_vir:
                local = virt_inds.index(i)
                val = lv_vals[local]
                logger.log_info1(f"    {i+1:4d}   Virtual    {val:8.6f}")
            logger.log_info1("    ============================")
        else:
            logger.log_info1(
                "\n    Frozen‑orbital Embedding MOs (Environment B) more than 50, no printing."
            )

        # Summary
        logger.log_info1("\n  Summary:")
        logger.log_info1(
            f"    System (A): {num_Ao} Occupied MOs, {num_actv} Active MOs, {num_Av} Virtual MOs"
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
            nmo=self.system.nmo,
            active_orbitals=self.partition["index_actv"],
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
