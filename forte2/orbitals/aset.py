from dataclasses import dataclass, field
import numpy as np
import re
import ast
import forte2
from forte2 import ints
from forte2.state import MOSpace
from forte2.system.basis_utils import BasisInfo
from forte2.system import System
from forte2.helpers import logger
from forte2.helpers.mixins import MOsMixin, SystemMixin
from forte2.system.atom_data import ATOM_SYMBOL_TO_Z

@dataclass
class ASET(MOsMixin, SystemMixin):
    """
    Active Space Embedding Theory (ASET) method for paritioning and projecting molecules. 

    Parameters
    ----------
    fragment : list[str]
        List of atomic symbols defining the fragment.
    cutoff_method : str, optional, default="THRESHOLD"
        Method for choosing the embedding cutoff. Options include "THRESHOLD", "CUMULATIVE_THRESHOLD", "NUM_OF_ORBITALS".
    cut_off : float, optional, default = 0.5
        Projector eigenvalue for both simple and cumulative threshold methods.
    num_a_docc : int, optional, default=0
        Number of occupied orbitals fixed to this value in fragment A when cutoff method is "NUM_OF_ORBITALS".
    num_a_uocc : int, optional, default=0
        Number of virtual orbitals fixed to this value in fragment A when cutoff method is "NUM_OF_ORBITALS".
    adjust_B_docc : int, optional, default=0
        Adjust this number of occupied orbitals between environment B and fragment A. If set to positive, move to B; if set to negative, move to A. 
    adjust_B_uocc : int, optional, default=0
        Adjust this number of virtual orbitals between environment B and fragment A. If set to positive, move to B; if set to negative, move to A.
    semicanonicalize_active : bool, optional, default=True
        Whether to semicanonicalize the active space orbitals.
    semicanonicalize_frozen : bool, optional, default=True
        Whether to semicanonicalize the frozen orbitals. 
    virtual_space : str, optional, default ="ASET"
        Virtual space scheme to use. Options include "ASET", "PAO", "IAO".
    pao_fix_virtual_num : bool, optional, default=False
        Enable this option to generate PAOs equivalent to ASET virtuals instead of using cutoff method = "THRESHOLD".
    pao_threshold : float, optiona, default = 1e-08. 
        Virtual space truncation threshold for PAO. 
    
    Notes
    -----
    The allow subspace specification is a list of strings, non-exhaustive examples::

    - ["C"]              # all carbon atoms
    - ["C","N"]          # all carbon and nitrogen atoms
    - ["C1"]             # carbon atom #1
    - ["C1-7"]           # carbon atoms #1 through #7
    - ["C1-3","N2"]       # carbon atoms #1, #2, #3 and nitrogen atom #2

    # add ASET citation 
    """

    fragment : list
    cutoff_method: str = "THRESHOLD"
    cut_off: float = 0.5
    num_a_docc: int = 0
    num_a_uocc: int = 0
    adjust_B_docc: int = 0
    adjust_B_uocc: int = 0
    semicanonicalize_active: bool = True
    semicanonicalize_frozen: bool = True
    virtual_space: str = "ASET"
    pao_fix_virtual_num: bool = False
    pao_threshold: float = 1e-8

    executed: bool = field(default=False, init=False)

    def __post_init__(self):
        self._regex = r"^([A-Z][a-z]?)(\d+)?(?:-(\d+))?$"
        self._check_parameters()

    def __call__(self, parent_method): 
        assert isinstance (
            parent_method, (forte2.orbopt.MCOptimizer)
        ), f"Parent method must be MCSCF, got {type(parent_method)}"
        self.parent_method = parent_method
        return self 
    
    def run(self) -> "ASET": 
        if not self.parent_method.executed:
            self.parent_method.run()
        SystemMixin.copy_from_upstream(self, self.parent_method)
        MOsMixin.copy_from_upstream(self, self.parent_method)
        self.mo_space = self.parent_method.mo_space
        self.ncore   = self.mo_space.ncore
        self.nactv   = self.mo_space.nactv

        self.Ca = self.parent_method.C[0]
        self.nmo = self.Ca.shape[1]
        self.nvirt = self.nmo - self.ncore - self.nactv

        minao_info = BasisInfo(self.system, self.system.minao_basis)

        raw_map = minao_info.atom_to_aos

        self.atom_to_center = self.system.atom_to_center
        self.atom_to_aos = {
            int(Z): {
                int(rel) - 1: list(aos_list)
                for rel, aos_list in inner.items()
            }
            for Z, inner in raw_map.items()
        }

        logger.log_info1("\nRunning Active Space Embedding Theory (ASET)")
        logger.log_info1(f"Fragment: {self.fragment}")
        logger.log_info1("\nProjecting orbitals to fragment")
        logger.log_info1(f"Cutoff method: {self.cutoff_method} \nCutoff value: {self.cut_off}")
        self.fragment = self._parse_fragment(self.fragment)
        self._S_mm = ints.overlap(self.system.minao_basis)
        self.P_frag, self.X_mm = self._make_fragment_projector()
        self.executed = True
        partition = self._make_embedding()
        self._print_embedding_info(**partition)
        return self
    
    def _check_parameters(self): 
        self.cutoff_method = self.cutoff_method.lower()
        assert self.cutoff_method in [
            "threshold",
            "cumulative_threshold",
            "num_of_orbitals",
        ], f"Invalid cutoff method: {self.cutoff_method}"
        if self.cutoff_method == "threshold": 
            assert (
                self.cut_off > 0 and self.cut_off < 1
            ), f"thresholdold must be in [0, 1], got {self.cut_off}"
        elif self.cutoff_method == "cumulative_threshold":
            assert (
                self.cut_off > 0
            ), f"Cumulative threshold must be positive, got {self.cut_off}"
        elif self.cutoff_method == "num_of_orbitals":
            assert (
                self.num_a_docc >= 0 or self.num_a_uocc >= 0
            ), f"Number of occupied and virtual orbitals in fragment A must be non-negative, got {self.num_a_docc}, {self.num_a_uocc}"
        
    def _parse_fragment(self, frag_str: str) -> list[int]:
        """
        Parse a fragment specification string or list into atom indices.

        Supported input formats (all 1-indexed for the user):
            ["C"]         → all carbon atoms
            ["C1"]        → carbon atom #1
            ["C1-3"]      → carbon atoms #1 through #3
            ["C", "N2"]   → all carbon atoms and nitrogen atom #2

        Parameters
        ----------
        frag_str : str or list[str]
            A string like '["C1-3", "N"]' or a list of such tokens.

        Returns
        -------
        list[int]
            A sorted list of unique atom indices (0-based) matching the specification.
        """

        if isinstance(frag_str, str):
            frag_list = ast.literal_eval(frag_str)  # e.g., turns '["C1-3", "N"]' into list
        else:
            frag_list = frag_str  # already a list

        atom_indices = []

        for token in frag_list:
            match = re.match(self._regex, token)
            if not match:
                raise ValueError(f"Invalid fragment specification: {token}")

            symbol = match.group(1)  # e.g., "C"
            start = match.group(2)   # e.g., "1"
            end = match.group(3)     # e.g., "3" (if it's a range)

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
        Build an AO-space fragment projector P = S^T (S_A) S in the minimal-AO basis:
        1. Compute AO overlap S_mm
        2. Extract fragment block S_A
        3. Pseudo invert S_A and embed into full minimal-AO space
        4. Form P_frag = S_mm @ S_A_nn @ S_mm
        
        Returns
        ------
        P_frag : NDArray
            The fragment projector matrix in the full minimal-AO space.
        """
        # 1. Compute minAO overlap S_mm
        S_mm = self._S_mm
        nbf_m = S_mm.shape[0]

        # 2. Collect ao indices in fragment 
        frag_ao_indices = []
        for atom_idx in self.fragment:
            # atomic number for this atom
            Z_i = int(self.system.atoms[atom_idx][0])

            # make sure that element is present
            if Z_i not in self.atom_to_center:
                raise ValueError(f"Element Z={Z_i} not in system.atom_to_center")

            # get the list of all atom‐indices of this element
            element_atoms = self.atom_to_center[Z_i]   # e.g. [0, 2] for two H’s

            # find which position in that list our atom_idx occupies
            rel_idx = element_atoms.index(atom_idx)

            # now fetch the AO indices for that element & rel_idx
            ao_list = self.atom_to_aos[Z_i][rel_idx]
            frag_ao_indices.extend(ao_list)

        frag_ao_indices = sorted(set(frag_ao_indices))

        # 3. Build fragment overlap block S_A and invert 
        S_A = S_mm[np.ix_(frag_ao_indices, frag_ao_indices)]
        S_A_inv = np.linalg.pinv(S_A, rcond=1e-8)

        # 4. Embed S_A_inv into full space and form projector
        S_A_mm = np.zeros((nbf_m, nbf_m))
        for i, mu in enumerate(frag_ao_indices):
            for j, nu in enumerate(frag_ao_indices):
                S_A_mm[mu, nu] = S_A_inv[i, j]
        
        evals, evecs = np.linalg.eigh(S_mm)
        X_mm = evecs @ np.diag((1.0 / np.sqrt(evals))) @ evecs.T
        P_ao = S_mm @ S_A_mm @ S_mm
        P_frag = X_mm @ P_ao @ X_mm

        # check for hermiticity
        if not np.allclose(P_frag, P_frag.conj().T, atol = 1e-8):
            raise RuntimeError("Fragment projector is not Hermitian.")
        # check for idempotency
        if not np.allclose(P_frag @ P_frag, P_frag, atol = 1e-8):
            raise RuntimeError("Fragment projector is not idempotent.")
        
        return P_frag, X_mm
    
    def _make_embedding(self): 
        """
        Perform Orbital Partitioning for ASET. 
        """
        # 1. Get MO coeff and dimensions
        Ca = self.Ca

        # 1) Compute the overlaps
        S_fm = ints.overlap(self.system.basis,
                            self.system.minao_basis)
        
        # 2) Orthonormalize the minimal basis
        X_mm = self.X_mm

        # 3) Build the projection operator from full AO into orthonormal min‐AO
        #    T[t,μ] = (X_mm @ S_fmᵀ)[t,μ]
        T = X_mm @ S_fm.T

        # 4) Project full‐AO MOs into the minimal basis:
        C_min = T @ Ca  # shape (n_minAO, nmo)

        # 5) Build the fragment projector in the minimal basis
        P_frag = self.P_frag
        F = C_min.T @ P_frag @ C_min

        # 6) Split F into occupied and virtual blocks
        core_inds = list(self.mo_space.core_orbitals)
        actv_inds = actv_inds = [i for sub in self.mo_space.active_orbitals for i in sub]
        virt_inds = list(range(self.ncore + self.nactv, self.nmo))

        # Split F 
        F_oo = F[np.ix_(core_inds, core_inds)]
        F_vv = F[np.ix_(virt_inds, virt_inds)]
        lo_vals, Uo = np.linalg.eigh(F_oo)
        lv_vals, Uv = np.linalg.eigh(F_vv)
        occ_pairs = zip(core_inds, lo_vals)
        vir_pairs = zip(virt_inds,  lv_vals)

        # Partition by cutoff 
        index_A_occ, index_B_occ = [], []
        index_A_vir, index_B_vir = [], []
        if self.cutoff_method == "threshold":
            # Select orbitals with eigenvalues above the threshold
            for i, v in occ_pairs:
                (index_A_occ if v > self.cut_off else index_B_occ).append(i)
            for (i,v) in vir_pairs:
                (index_A_vir if v > self.cut_off else index_B_vir).append(i)
        elif self.cutoff_method == "cumulative_threshold":
            # occupied cumulative‐threshold partitioning
            cum_lo = 0.0
            for i, v in occ_pairs:
                cum_lo += v
                if cum_lo < self.cut_off:
                    index_A_occ.append(i)
                else:
                    index_B_occ.append(i)
                    if v > 0.5:
                        logger.log_info1(
                            f"Warning! Occupied orbital {i+1} has eigenvalue {v:8.6f} "
                            "and is assigned to B."
                        )

            # virtual cumulative‐threshold partitioning
            cum_lv = cum_lo
            for i, v in vir_pairs:
                cum_lv += v
                if cum_lv < self.cut_off:
                    index_A_vir.append(i)
                else:
                    index_B_vir.append(i)
                    if v > 0.5:
                        logger.log_info1(
                            f"Warning! Virtual orbital {i+1} has eigenvalue {v:8.6f} "
                            "and is assigned to B."
                        )

        elif self.cutoff_method == "num_of_orbitals":
            for i, v in enumerate(lo_vals):
                glob_i = core_inds[i]
                if i < self.num_a_docc:
                    index_A_occ.append(glob_i)
                else:
                    index_B_occ.append(glob_i)
                    if v > 0.5:
                        logger.log_info1(
                            f"Warning! Occupied orbital {glob_i+1} has eigenvalue {v:8.6f} "
                            "and is assigned to B."
                        )
                        
            for i, v in enumerate(lv_vals):
                glob_i = virt_inds[i]
                if i < self.num_a_uocc:
                    index_A_vir.append(glob_i)
                else:
                    index_B_vir.append(glob_i)
                    if v > 0.5:
                        logger.log_info1(
                            f"Warning! Virtual orbital {glob_i+1} has eigenvalue {v:8.6f} "
                            "and is assigned to B."
                        )

        # For PAO:
        if self.virtual_space.lower() == "pao":
            print(f"\n****** Build PAOs for virtual space ******")
            
        # Adjust occupied and virtual indices based on user input
        #################################
        # Skipping this section for now #
        #################################

        # Semi-canonicalize the blocks by default
        #################################
        # Skipping this section for now #
        #################################
        # C_tilde = Ca.copy()
        # C_av = semicanonicalize_block() ... 

        return {
            "index_A_occ": index_A_occ,
            "index_actv": actv_inds,
            "index_A_vir": index_A_vir,
            "index_B_occ": index_B_occ,
            "index_B_vir": index_B_vir,
            "lo_vals": lo_vals,
            "lv_vals": lv_vals,
            #"C_tilde": C_tilde
        }

    def _print_embedding_info(
        self,
        index_A_occ: list[int],
        index_actv:  list[int],
        index_A_vir: list[int],
        index_B_occ: list[int],
        index_B_vir: list[int],
        lo_vals:     np.ndarray,
        lv_vals:     np.ndarray,
        ) -> None:
        """
        Print the sizes and MO lists for fragment embedding
        """
        core_inds = list(self.mo_space.core_orbitals)
        actv_inds = [i for sub in self.mo_space.active_orbitals for i in sub]
        virt_inds = sorted(set(range(self.nmo)) - set(core_inds) - set(actv_inds))
        num_Ao = len(index_A_occ)
        num_Av = len(index_A_vir)
        num_Bo = len(index_B_occ)
        num_Bv = len(index_B_vir)
        num_actv = len(index_actv)

        # Environment A
        logger.log_info1(
            "\nFrozen-Orbital Embedding MOs (fragment A):"
            )
        logger.log_info1("    ============================")
        logger.log_info1("      MO     Type    <phi|P|phi>")
        logger.log_info1("    ----------------------------")
        for i in index_A_occ:
            local = core_inds.index(i)
            val = lo_vals[local]
            logger.log_info1(f"      {i+1:4d}  occupied  {val:8.6f}")

        for i in index_actv:
            logger.log_info1(
                f"      {i+1:4d}  active    --"
            )
        for i in index_A_vir:
            local = virt_inds.index(i)
            val = lv_vals[local]
            logger.log_info1(f"      {i+1:4d}  virtual   {val:8.6f}")
 
        # Environment B
        total_env = num_Bo + num_Bv
        if total_env < 80:
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
                "\n    Frozen‑orbital Embedding MOs (Environment B) more than 80, no printing."
            )
        
        # Summary
        logger.log_info1("\n  Summary:")
        logger.log_info1(
            f"    System (A): {num_Ao} Occupied MOs, {num_actv} Active MOs, {num_Av} Virtual MOs"
        )
        logger.log_info1(
            f"    Environment (B): {num_Bo} Occupied MOs, {num_Bv} Virtual MOs"
        )
        
        # no frozen atm 
        # logger.log_info1(
        #     f"    Frozen Orbitals: {num_Fo} Core MOs, {num_Bv} Virtual MOs\n"
        # ) 