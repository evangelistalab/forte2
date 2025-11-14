from dataclasses import dataclass, field
from collections import OrderedDict

import numpy as np

from forte2 import (
    CIStrings,
    CISigmaBuilder,
    CISpinAdapter,
    cpp_helpers,
    RelCISigmaBuilder,
    SparseState,
    apply_op,
    sparse_operator_hamiltonian,
)
from forte2.state import State, MOSpace
from forte2.helpers.comparisons import approx
from forte2.helpers.davidsonliu import DavidsonLiuSolver
from forte2.base_classes.active_space_solver import (
    ActiveSpaceSolver,
    RelActiveSpaceSolver,
)
from forte2.helpers import logger
from forte2.jkbuilder import RestrictedMOIntegrals, SpinorbitalIntegrals
from forte2.props import get_1e_property
from forte2.orbitals import Semicanonicalizer
from forte2.scf.scf_utils import convert_coeff_spatial_to_spinor
from .ci_utils import (
    pretty_print_gas_info,
    pretty_print_ci_summary,
    pretty_print_ci_nat_occ_numbers,
    pretty_print_ci_dets,
    pretty_print_ci_transition_props,
    make_2cumulant_sf,
    make_2cumulant_so,
    make_2cumulant_sf,
    make_2cumulant_so,
    make_3cumulant_sf,
    make_3cumulant_so,
)


@dataclass
class _CIBase:
    """
    A general configuration interaction (CI) solver class for a single `State`.
    Although possible, is not recommended to instantiate this class directly.
    Consider using the `CI` class instead.

    Parameters
    ----------
    mo_space : MOSpace
        Specifies the GASes and core orbitals.
    state : State
        The electronic state for which the CI is solved.
    ints : RestrictedMOIntegrals
        The molecular orbital integrals for the system.
    nroot : int
        The number of roots to compute.
    do_test_rdms : bool, optional, default=False
        If True, compute and test the reduced density matrices (RDMs) after the CI calculation.
    log_level : int, optional
        The logging level for the CI solver. Defaults to the global logger's verbosity level.
    die_if_not_converged : bool, optional, default=False
        If True, raise an error if the CI solver does not converge.
    ci_algorithm : str, optional, default="hz"
        The algorithm used for the CI sigma builder.
        Non-relativistic options are:
            - "hz": Harrison-Zarrabian
            - "kh": Knowles-Handy
            - "exact": Exact diagonalization
        Two-component (relativistic) options are:
            - "hz": Harrison-Zarrabian
            - "exact": Exact diagonalization
            - "sparse": Sigma builder using sparse representation of the Hamiltonian and states.
                Recommended for debug use only.
    guess_per_root : int, optional, default=2
        The number of guess vectors for each root.
    ndets_per_guess : int, optional, default=10
        The number of determinants per guess vector.
    collapse_per_root : int, optional, default=2
        The number of determinants to collapse per root.
    basis_per_root : int, optional, default=4
        The maximum number of basis vectors per root.
    maxiter : int, optional, default=100
        The maximum number of iterations for the Davidson-Liu solver.
    econv : float, optional, default=1e-10
        The energy convergence threshold for the solver.
    rconv : float, optional, default=1e-5
        The residual convergence threshold for the solver.
    energy_shift : float, optional, default=None
        An energy shift to find roots around. If None, no shift is applied.

    Attributes
    ----------
    eigensolver : DavidsonLiuSolver
        The eigensolver used to find the roots of the CI problem.
    E (evals) : NDArray
        The eigenvalues (energies) of the CI problem.
    evecs : NDArray
        The eigenvectors (CI coefficients) of the CI problem.

    """

    mo_space: MOSpace
    state: State
    ints: RestrictedMOIntegrals
    nroot: int
    active_orbsym: list[int]
    two_component: bool = False
    do_test_rdms: bool = False
    log_level: int = field(default=logger.get_verbosity_level())
    die_if_not_converged: bool = False

    ### Sigma builder parameters
    ci_algorithm: str = "hz"

    ### Davidson-Liu parameters
    guess_per_root: int = 2
    ndets_per_guess: int = 10
    collapse_per_root: int = 2
    basis_per_root: int = 4
    maxiter: int = 100
    econv: float = 1e-10
    rconv: float = 1e-5
    energy_shift: float = None

    ### Non-init attributes
    ci_builder_memory: int = field(default=1024, init=False)  # in MB
    first_run: bool = field(default=True, init=False)
    executed: bool = field(default=False, init=False)

    def __post_init__(self):
        self.norb = self.mo_space.nactv
        self.ncore = self.mo_space.ncore + self.mo_space.nfrozen_core
        self.ngas = self.mo_space.ngas
        self.gas_min = self.state.gas_min
        self.gas_max = self.state.gas_max
        self.eigensolver = None

        self.dtype = complex if self.two_component else float

        if self.two_component:
            assert self.ci_algorithm.lower() in [
                "hz",
                "sparse",
                "exact",
            ], "Two-component CI only supports 'hz', 'sparse', or 'exact' algorithms."
        else:
            assert self.ci_algorithm.lower() in [
                "hz",
                "kh",
                "exact",
            ], "CI algorithm must be 'hz', 'kh', or 'exact'."

    def _ci_solver_startup(self):
        if self.two_component:
            self.ci_strings = CIStrings(
                self.state.nel - self.ncore,
                0,
                self.state.symmetry,
                self.active_orbsym,
                self.state.gas_min,
                self.state.gas_max,
            )
        else:
            self.ci_strings = CIStrings(
                self.state.na - self.ncore,
                self.state.nb - self.ncore,
                self.state.symmetry,
                self.active_orbsym,
                self.gas_min,
                self.gas_max,
            )

        pretty_print_gas_info(self.ci_strings)

        if self.two_component:
            logger.log(f"\nNumber of electrons: {self.ci_strings.na}", self.log_level)
            logger.log(f"Number of strings: {self.ci_strings.nas}", self.log_level)
        else:
            logger.log(f"\nNumber of α electrons: {self.ci_strings.na}", self.log_level)
            logger.log(f"Number of β electrons: {self.ci_strings.nb}", self.log_level)
            logger.log(f"Number of α strings: {self.ci_strings.nas}", self.log_level)
            logger.log(f"Number of β strings: {self.ci_strings.nbs}", self.log_level)
            self.ndet = self.ci_strings.ndet
        logger.log(f"Number of determinants: {self.ci_strings.ndet}", self.log_level)

        if self.ci_strings.ndet == 0:
            raise ValueError(
                "No determinants could be generated for the given state and orbitals."
            )
        if self.two_component:
            # no "spin-adaptation" for 2c, we use a basis of determiants directly
            self.ndet = self.ci_strings.ndet
            self.basis_size = self.ndet
            self.dets = self.ci_strings.make_determinants()
            self.sigma_det = np.zeros((self.ndet,), dtype=complex)
            self.b_det = np.zeros((self.ndet,), dtype=complex)
        else:
            self.spin_adapter = CISpinAdapter(
                self.state.multiplicity - 1, self.state.twice_ms, self.norb
            )
            self.spin_adapter.set_log_level(self.log_level)
            self.dets = self.ci_strings.make_determinants()

            self.spin_adapter.prepare_couplings(self.dets)
            logger.log(
                f"Number of configurations: {self.spin_adapter.nconf}", self.log_level
            )
            logger.log(f"Number of CSFs: {self.spin_adapter.ncsf}", self.log_level)

            # 1. Allocate memory for the CI vectors
            self.ndet = self.ci_strings.ndet
            self.basis_size = self.spin_adapter.ncsf

            # Create the CI vectors that will hold the results of the sigma builder in the
            # determinant basis
            self.b_det = np.zeros((self.ndet))
            self.sigma_det = np.zeros((self.ndet))

    def run(self, use_asym_ints=False):
        if not self.executed:
            self._ci_solver_startup()

        # Create the CISigmaBuilder from the CI strings and integrals
        # This object handles some temporary memory deallocated at destruction
        # and is used to compute the Hamiltonian matrix elements in the determinant basis
        if self.two_component:
            self.ci_sigma_builder = RelCISigmaBuilder(
                self.ci_strings,
                self.ints.E.real,
                self.ints.H,
                self.ints.V,
                self.log_level,
                use_asym_ints,
            )
        else:
            self.ci_sigma_builder = CISigmaBuilder(
                self.ci_strings,
                self.ints.E,
                self.ints.H,
                self.ints.V,
                self.log_level,
            )
        self.ci_sigma_builder.set_memory(self.ci_builder_memory)
        if self.ci_algorithm.lower() == "exact":
            self._do_exact_diagonalization()
        else:
            self._do_iterative_ci()

        self.E = self.evals
        for i, e in enumerate(self.evals):
            logger.log(f"Final CI Energy Root {i}: {e:20.12f} [Eh]", self.log_level)

        if self.do_test_rdms:
            self._test_rdms()

        self.executed = True

        return self

    def _do_iterative_ci(self):
        """
        Solve CI with an iterative Davidson-Liu solver, using either
        Harrison-Zarrabian or Knowles-Handy sigma builder algorithm.
        """
        if self.two_component:
            assert self.ci_algorithm.lower() in [
                "hz",
                "sparse",
            ], "For two-component CI, only the Harrison-Zarrabian (hz) algorithm is supported."
            self.ci_sigma_builder.set_algorithm("hz")
        else:
            self.ci_sigma_builder.set_algorithm(self.ci_algorithm)

        logger.log(
            f"Using CI algorithm: {self.ci_sigma_builder.get_algorithm()}",
            self.log_level,
        )

        if self.two_component:
            Hdiag = self.ci_sigma_builder.form_Hdiag(self.dets)
        else:
            Hdiag = self.ci_sigma_builder.form_Hdiag_csf(
                self.dets, self.spin_adapter, spin_adapt_full_preconditioner=False
            )

        # If there is only one determinant, we can skip calling the eigensolver
        if self.ndet == 1:
            self.evals = np.array([Hdiag[0]])
            self.evecs = np.ones((1, 1))
            logger.log(
                f"Final CI Energy Root {0}: {self.evals[0]:20.12f} [Eh]", self.log_level
            )
            self.executed = True
            return self

        # 3. Instantiate and configure solver
        if self.eigensolver is None:
            self.eigensolver = DavidsonLiuSolver(
                size=self.basis_size,  # size of the basis (number of CSF if we spin adapt)
                nroot=self.nroot,
                collapse_per_root=self.collapse_per_root,
                basis_per_root=self.basis_per_root,
                e_tol=self.econv,  # eigenvalue convergence
                r_tol=self.rconv,  # residual convergence
                maxiter=self.maxiter,
                eta=self.energy_shift,
                log_level=self.log_level,
                dtype=complex if self.two_component else float,
            )

        # 4. Compute diagonal of the Hamiltonian
        self.eigensolver.add_h_diag(Hdiag)

        # 5. Build the guess vectors if this is the first run
        if self.first_run:
            self._build_guess_vectors(Hdiag)
            self.first_run = False

        if self.two_component:
            if self.ci_algorithm.lower() == "sparse":
                ham = sparse_operator_hamiltonian(
                    self.ints.E.real,
                    self.ints.H,
                    self.ints.V,
                    1e-100,
                )

                def sigma_builder(basis_block, sigma_block):
                    nstate = basis_block.shape[1]
                    for istate in range(nstate):
                        psi = SparseState(
                            {d: c for d, c in zip(self.dets, basis_block[:, istate])}
                        )
                        Hpsi = apply_op(ham, psi, screen_thresh=1e-100)
                        for idet in range(self.ndet):
                            sigma_block[idet, istate] = Hpsi[self.dets[idet]]

            else:

                def sigma_builder(Bblock, Sblock):
                    # Compute the sigma block from the basis block
                    ncols = Bblock.shape[1]
                    for i in range(ncols):
                        # copies ensure continguous arrays are passed to C++
                        self.b_det = Bblock[:, i].copy()
                        self.ci_sigma_builder.Hamiltonian(self.b_det, self.sigma_det)
                        Sblock[:, i] = self.sigma_det.copy()

        else:

            def sigma_builder(Bblock, Sblock):
                # Compute the sigma block from the basis block
                ncols = Bblock.shape[1]
                for i in range(ncols):
                    self.spin_adapter.csf_C_to_det_C(Bblock[:, i], self.b_det)
                    self.ci_sigma_builder.Hamiltonian(self.b_det, self.sigma_det)
                    self.spin_adapter.det_C_to_csf_C(self.sigma_det, Sblock[:, i])

        self.eigensolver.add_sigma_builder(sigma_builder)

        # 6. Run Davidson
        self.evals, self.evecs = self.eigensolver.solve()

        if self.eigensolver.converged:
            logger.log("\nDavidson-Liu solver converged.\n", self.log_level)
        else:
            if self.die_if_not_converged:
                raise RuntimeError("Davidson-Liu solver did not converge.")
            else:
                logger.log(
                    f"\nDavidson-Liu solver did not converge in {self.eigensolver.maxiter} iterations.\n",
                    self.log_level,
                )

        if not self.two_component:
            h_tot, h_aabb, h_aaaa, h_bbbb = self.ci_sigma_builder.avg_build_time()
            logger.log("\nAverage CI Sigma Builder time summary:", self.log_level)
            logger.log(f"h_aabb time:    {h_aabb:.3f} s/build", self.log_level)
            logger.log(f"h_aaaa time:    {h_aaaa:.3f} s/build", self.log_level)
            logger.log(f"h_bbbb time:    {h_bbbb:.3f} s/build", self.log_level)
            logger.log(f"total time:     {h_tot:.3f} s/build\n", self.log_level)

    def _do_exact_diagonalization(self):
        logger.log("Using CI algorithm: Exact Diagonalization", self.log_level)

        if self.two_component:
            H = np.zeros((self.ndet,) * 2, dtype=complex)
            for i in range(self.ndet):
                for j in range(i + 1):
                    H[i, j] = self.ci_sigma_builder.slater_rules(self.dets, i, j)
                    H[j, i] = np.conj(H[i, j])
        else:
            H = self.ci_sigma_builder.form_H_csf(self.dets, self.spin_adapter)

        self.evals_full, self.evecs_full = np.linalg.eigh(H)
        if self.energy_shift is not None:
            argsort = np.argsort(np.abs(self.evals_full - self.energy_shift))
            self.evals_full = self.evals_full[argsort]
            self.evecs_full = self.evecs_full[:, argsort]

        self.evals = self.evals_full[: self.nroot]
        self.evecs = self.evecs_full[:, : self.nroot]

    def _test_rdms(self):
        # Compute the RDMs from the CI vectors
        # and verify the energy from the RDMs matches the CI energy
        logger.log("\nComputing RDMs from CI vectors.\n", self.log_level)
        if self.two_component:
            for root in range(self.nroot):
                rdm1 = self.make_1rdm(root)
                rdm2 = self.make_2rdm(root)

                rdms_energy = self.ints.E
                rdms_energy += np.einsum("ij,ij", rdm1, self.ints.H)
                rdms_energy += 0.5 * np.einsum("ijkl,ijkl", rdm2, self.ints.V)
                logger.log(
                    f"CI energy from RDMs: {rdms_energy:.12f} Eh", self.log_level
                )

                assert self.E[root] == approx(rdms_energy)

                logger.log(
                    f"RDMs for root {root} validated successfully.\n", self.log_level
                )
                return

        for root in range(self.nroot):
            root_rdms = {}
            root_rdms["rdm1"] = self.make_sf_1rdm(root)
            rdm2_aa, rdm2_ab, rdm2_bb = self.make_sd_2rdm(root)
            root_rdms["rdm2_aa"] = rdm2_aa
            root_rdms["rdm2_ab"] = rdm2_ab
            root_rdms["rdm2_bb"] = rdm2_bb

            rdm2_aa_full, _, rdm2_bb_full = self.make_sd_2rdm(root)
            # Convert to full-dimension RDMs
            root_rdms["rdm2_aa_full"] = cpp_helpers.packed_tensor4_to_tensor4(
                rdm2_aa_full
            )
            root_rdms["rdm2_bb_full"] = cpp_helpers.packed_tensor4_to_tensor4(
                rdm2_bb_full
            )

            root_rdms["rdm2_sf"] = self.make_sf_2rdm(root)

            # Compute the energy from the RDMs
            # from the numpy tensor V[i, j, k, l] = <ij|kl> make the np matrix with indices
            # V[i > j, k > l] = <ij|kl>
            i_idx, j_idx = np.tril_indices(self.norb, k=-1)
            # broadcast into a 2D matrix
            i_row = i_idx[:, None]
            j_row = j_idx[:, None]
            i_col = i_idx[None, :]
            j_col = j_idx[None, :]
            # Create the antisymmetrized two electron integrals matrix
            A = self.ints.V.copy()
            A -= np.einsum("ijkl->ijlk", self.ints.V)
            M = A[i_row, j_row, i_col, j_col]
            rdms_energy = (
                self.ints.E
                + np.einsum("ij,ij", root_rdms["rdm1"], self.ints.H)
                + np.einsum("ij,ij", root_rdms["rdm2_aa"], M)
                + np.einsum("ijkl,ijkl", root_rdms["rdm2_ab"], self.ints.V)
                + np.einsum("ij,ij", root_rdms["rdm2_bb"], M)
            )
            logger.log(
                f"CI energy from RDMs:           {rdms_energy:.12f} Eh", self.log_level
            )
            assert np.isclose(
                self.E[root], rdms_energy
            ), f"CI energy {self.E[root]} Eh does not match RDMs energy {rdms_energy} Eh"

            rdms_energy = (
                self.ints.E
                + np.einsum("ij,ij", root_rdms["rdm1"], self.ints.H)
                + np.einsum("ijkl,ijkl", root_rdms["rdm2_aa_full"], A) * 0.25
                + np.einsum("ijkl,ijkl", root_rdms["rdm2_ab"], self.ints.V)
                + np.einsum("ijkl,ijkl", root_rdms["rdm2_bb_full"], A) * 0.25
            )
            logger.log(
                f"CI energy from expanded RDMs:  {rdms_energy:.12f} Eh", self.log_level
            )

            assert self.E[root] == approx(rdms_energy)

            rdms_energy = (
                self.ints.E
                + np.einsum("ij,ij", root_rdms["rdm1"], self.ints.H)
                + np.einsum(
                    "ijkl,ijkl",
                    0.5 * root_rdms["rdm2_sf"],
                    self.ints.V,
                )
            )
            logger.log(
                f"CI energy from spin-free RDMs: {rdms_energy:.12f} Eh", self.log_level
            )

            assert self.E[root] == approx(rdms_energy)

            logger.log(
                f"RDMs for root {root} validated successfully.\n", self.log_level
            )

    def _build_guess_vectors(self, Hdiag):
        """Build the guess vectors for the CI calculation."""
        # determine the number of guess vectors
        self.num_guess_states = min(self.guess_per_root * self.nroot, self.basis_size)
        logger.log(f"Number of guess states: {self.num_guess_states}", self.log_level)
        nguess_dets = min(self.ndets_per_guess * self.num_guess_states, self.basis_size)
        logger.log(f"Number of guess basis: {nguess_dets}", self.log_level)

        # find the indices of the elements of Hdiag with the lowest values
        if self.energy_shift is not None:
            indices = np.argsort(np.abs(Hdiag - self.energy_shift))[:nguess_dets]
        else:
            indices = np.argsort(Hdiag)[:nguess_dets]

        if self.two_component:
            _slater_rules = lambda I, J: self.ci_sigma_builder.slater_rules(
                self.dets, I, J
            )
        else:
            _slater_rules = lambda I, J: self.ci_sigma_builder.slater_rules_csf(
                self.dets, self.spin_adapter, I, J
            )
        # create the Hamiltonian matrix in the basis of the guess CSFs
        Hguess = np.zeros((nguess_dets, nguess_dets), dtype=self.dtype)
        for i, I in enumerate(indices):
            for j, J in enumerate(indices):
                if i >= j:
                    Hij = _slater_rules(I, J)
                    Hguess[i, j] = Hij
                    Hguess[j, i] = np.conj(Hij)

        # Diagonalize the Hamiltonian to get the initial guess vectors
        evals_guess, evecs_guess = np.linalg.eigh(Hguess)

        # Select the lowest eigenvalues and their corresponding eigenvectors
        guess_mat = np.zeros((self.basis_size, self.num_guess_states), dtype=self.dtype)
        for i in range(self.num_guess_states):
            guess = evecs_guess[:, i]
            for j, d in enumerate(indices):
                guess_mat[d, i] = guess[j]

        self.eigensolver.add_guesses(guess_mat)

    def make_1rdm(self, left_root: int, right_root: int | None = None):
        """
        Make the one-particle RDM for two CI roots.
        Spin-free for non-relativistic CI (use make_sd_* for spin-dependent RDMs).
        Spin(or)-orbital for two-component CI.

        Parameters
        ----------
        left_root : int
            the CI root for the bra state.
        right_root : int | None, optional (default=left_root)
            the CI root for the ket state.

        Returns
        -------
        NDArray
            One-particle RDM.
        """
        if self.two_component:
            return self.make_so_1rdm(left_root, right_root)
        else:
            return self.make_sf_1rdm(left_root, right_root)

    def make_2rdm(self, left_root: int, right_root: int | None = None):
        """
        Make the two-particle RDM for two CI roots.
        Spin-free for non-relativistic CI (use make_sd_* for spin-dependent RDMs).
        Spin(or)-orbital for two-component CI.

        Parameters
        ----------
        left_root : int
            the CI root for the bra state.
        right_root : int | None, optional (default=left_root)
            the CI root for the ket state.

        Returns
        -------
        NDArray
            Two-particle RDM.
        """
        if self.two_component:
            return self.make_so_2rdm(left_root, right_root)
        else:
            return self.make_sf_2rdm(left_root, right_root)

    def make_2cumulant(self, left_root: int, right_root: int | None = None):
        """
        Make the two-particle cumulant for two CI roots.
        Spin-free for non-relativistic CI (use make_sd_* for spin-dependent RDMs).
        Spin(or)-orbital for two-component CI.

        Parameters
        ----------
        left_root : int
            the CI root for the bra state.
        right_root : int | None, optional (default=left_root)
            the CI root for the ket state.

        Returns
        -------
        NDArray
            Two-particle cumulant.
        """
        if self.two_component:
            l2 = self.make_so_2cumulant(left_root, right_root)
        else:
            l2 = self.make_sf_2cumulant(left_root, right_root)
        return l2

    def make_3rdm(self, left_root: int, right_root: int | None = None):
        """
        Make the three-particle RDM for two CI roots.
        Spin-free for non-relativistic CI (use make_sd_* for spin-dependent RDMs).
        Spin(or)-orbital for two-component CI.

        Parameters
        ----------
        left_root : int
            the CI root for the bra state.
        right_root : int | None, optional (default=left_root)
            the CI root for the ket state.

        Returns
        -------
        NDArray
            Three-particle RDM.
        """
        if self.two_component:
            return self.make_so_3rdm(left_root, right_root)
        else:
            return self.make_sf_3rdm(left_root, right_root)

    def make_3cumulant(self, left_root: int, right_root: int | None = None):
        """
        Make the three-particle cumulant for two CI roots.
        Spin-free for non-relativistic CI (use make_sd_* for spin-dependent RDMs).
        Spin(or)-orbital for two-component CI.

        Parameters
        ----------
        left_root : int
            the CI root for the bra state.
        right_root : int | None, optional (default=left_root)
            the CI root for the ket state.

        Returns
        -------
        NDArray
            Three-particle cumulant.
        """
        if self.two_component:
            l3 = self.make_so_3cumulant(left_root, right_root)
        else:
            l3 = self.make_sf_3cumulant(left_root, right_root)
        return l3

    def make_sd_1rdm(self, left_root: int, right_root: int | None = None):
        r"""
        Make the spin-dependent one-particle RDM for two CI roots.

        Parameters
        ----------
        left_root : int
            the CI root for the bra state.
        right_root : int | None, optional (default=left_root)
            the CI root for the ket state.

        Returns
        -------
        tuple[NDArray, NDArray]:
            Spin-dependent one-particle RDMs (a, b).
        """
        assert (
            not self.two_component
        ), "make_sd_1rdm is only available for non-relativistic CI."

        left_ci_vec_det = np.zeros((self.ndet))
        self.spin_adapter.csf_C_to_det_C(self.evecs[:, left_root], left_ci_vec_det)
        if right_root is None:
            right_ci_vec_det = left_ci_vec_det
        else:
            right_ci_vec_det = np.zeros((self.ndet))
            self.spin_adapter.csf_C_to_det_C(
                self.evecs[:, right_root], right_ci_vec_det
            )
        a = self.ci_sigma_builder.a_1rdm(left_ci_vec_det, right_ci_vec_det)
        b = self.ci_sigma_builder.b_1rdm(left_ci_vec_det, right_ci_vec_det)
        return a, b

    def make_sd_2rdm(self, left_root: int, right_root: int | None = None):
        """
        Make the spin-dependent two-particle RDMs (aa, ab, bb) for two CI roots.

        Parameters
        ----------
        left_root : int
            the CI root for the bra state.
        right_root : int | None, optional (default=left_root)
            the CI root for the ket state.

        Returns
        -------
        tuple[NDArray, NDArray, NDArray]:
            Spin-dependent two-particle RDMs (aa, ab, bb).
        """
        assert (
            not self.two_component
        ), "make_sd_2rdm is only available for non-relativistic CI."

        left_ci_vec_det = np.zeros((self.ndet))
        self.spin_adapter.csf_C_to_det_C(self.evecs[:, left_root], left_ci_vec_det)
        if right_root is None:
            right_ci_vec_det = left_ci_vec_det
        else:
            right_ci_vec_det = np.zeros((self.ndet))
            self.spin_adapter.csf_C_to_det_C(
                self.evecs[:, right_root], right_ci_vec_det
            )
        aa = self.ci_sigma_builder.aa_2rdm(left_ci_vec_det, right_ci_vec_det)
        ab = self.ci_sigma_builder.ab_2rdm(left_ci_vec_det, right_ci_vec_det)
        bb = self.ci_sigma_builder.bb_2rdm(left_ci_vec_det, right_ci_vec_det)
        return aa, ab, bb

    def make_sd_3rdm(self, left_root: int, right_root: int | None = None):
        """
        Make the spin-dependent three-particle RDMs (aaa, aab, abb, bbb) for two CI roots.

        Parameters
        ----------
        left_root : int
            the CI root for the bra state.
        right_root : int | None, optional (default=left_root)
            the CI root for the ket state.

        Returns
        -------
        tuple[NDArray, NDArray, NDArray, NDArray]:
            Spin-dependent three-particle RDMs (aaa, aab, abb, bbb).
        """
        assert (
            not self.two_component
        ), "make_sd_3rdm is only available for non-relativistic CI."

        left_ci_vec_det = np.zeros((self.ndet))
        self.spin_adapter.csf_C_to_det_C(self.evecs[:, left_root], left_ci_vec_det)
        if right_root is None:
            right_ci_vec_det = left_ci_vec_det
        else:
            right_ci_vec_det = np.zeros((self.ndet))
            self.spin_adapter.csf_C_to_det_C(
                self.evecs[:, right_root], right_ci_vec_det
            )

        aaa = self.ci_sigma_builder.aaa_3rdm(left_ci_vec_det, right_ci_vec_det)
        aab = self.ci_sigma_builder.aab_3rdm(left_ci_vec_det, right_ci_vec_det)
        abb = self.ci_sigma_builder.abb_3rdm(left_ci_vec_det, right_ci_vec_det)
        bbb = self.ci_sigma_builder.bbb_3rdm(left_ci_vec_det, right_ci_vec_det)
        return aaa, aab, abb, bbb

    def make_sf_1rdm(self, left_root: int, right_root: int | None = None):
        """
        Make the spin-free one-particle RDM for two CI roots.

        Parameters
        ----------
        left_root : int
            the CI root for the bra state.
        right_root : int | None, optional (default=left_root)
            the CI root for the ket state.

        Returns
        -------
        NDArray
            Spin-free one-particle RDM.
        """
        assert (
            not self.two_component
        ), "make_sf_1rdm is only available for non-relativistic CI."

        left_ci_vec_det = np.zeros((self.ndet))
        self.spin_adapter.csf_C_to_det_C(self.evecs[:, left_root], left_ci_vec_det)
        if right_root is None:
            right_ci_vec_det = left_ci_vec_det
        else:
            right_ci_vec_det = np.zeros((self.ndet))
            self.spin_adapter.csf_C_to_det_C(
                self.evecs[:, right_root], right_ci_vec_det
            )
        return self.ci_sigma_builder.sf_1rdm(left_ci_vec_det, right_ci_vec_det)

    def make_sf_2rdm(self, left_root: int, right_root: int | None = None):
        """
        Make the spin-free two-particle RDM for two CI roots.

        Parameters
        ----------
        left_root : int
            the CI root for the bra state.
        right_root : int | None, optional (default=left_root)
            the CI root for the ket state.

        Returns
        -------
        NDArray
            Spin-free two-particle RDM.
        """
        assert (
            not self.two_component
        ), "make_sf_2rdm is only available for non-relativistic CI."

        left_ci_vec_det = np.zeros((self.ndet))
        self.spin_adapter.csf_C_to_det_C(self.evecs[:, left_root], left_ci_vec_det)
        if right_root is None:
            right_ci_vec_det = left_ci_vec_det
        else:
            right_ci_vec_det = np.zeros((self.ndet))
            self.spin_adapter.csf_C_to_det_C(
                self.evecs[:, right_root], right_ci_vec_det
            )
        return self.ci_sigma_builder.sf_2rdm(left_ci_vec_det, right_ci_vec_det)

    def make_sf_3rdm(self, left_root: int, right_root: int | None = None):
        """
        Make the spin-free three-particle RDM for two CI roots.

        Parameters
        ----------
        left_root : int
            the CI root for the bra state.
        right_root : int | None, optional (default=left_root)
            the CI root for the ket state.

        Returns
        -------
        NDArray
            Spin-free three-particle RDM.
        """
        assert (
            not self.two_component
        ), "make_sf_3rdm is only available for non-relativistic CI."

        left_ci_vec_det = np.zeros((self.ndet))
        self.spin_adapter.csf_C_to_det_C(self.evecs[:, left_root], left_ci_vec_det)
        if right_root is None:
            right_ci_vec_det = left_ci_vec_det
        else:
            right_ci_vec_det = np.zeros((self.ndet))
            self.spin_adapter.csf_C_to_det_C(
                self.evecs[:, right_root], right_ci_vec_det
            )
        return self.ci_sigma_builder.sf_3rdm(left_ci_vec_det, right_ci_vec_det)

    def make_sf_2cumulant(self, left_root: int, right_root: int | None = None):
        """
        Make the spin-free cumulant of the two-particle RDM for two CI roots.

        Parameters
        ----------
        left_root : int
            the CI root for the bra state.
        right_root : int | None, optional (default=left_root)
            the CI root for the ket state.

        Returns
        -------
        NDArray
            Spin-free cumulant of the two-particle RDM.
        """
        assert (
            not self.two_component
        ), "make_sf_2cumulant is only available for non-relativistic CI."

        left_ci_vec_det = np.zeros((self.ndet))
        self.spin_adapter.csf_C_to_det_C(self.evecs[:, left_root], left_ci_vec_det)
        if right_root is None:
            right_ci_vec_det = left_ci_vec_det
        else:
            right_ci_vec_det = np.zeros((self.ndet))
            self.spin_adapter.csf_C_to_det_C(
                self.evecs[:, right_root], right_ci_vec_det
            )
        return self.ci_sigma_builder.sf_2cumulant(left_ci_vec_det, right_ci_vec_det)

    def make_sf_3cumulant(self, left_root: int, right_root: int | None = None):
        """
        Make the spin-free cumulant of the three-particle RDM for two CI roots.

        Parameters
        ----------
        left_root : int
            the CI root for the bra state.
        right_root : int | None, optional (default=left_root)
            the CI root for the ket state.

        Returns
        -------
        NDArray
            Spin-free cumulant of the three-particle RDM.
        """
        assert (
            not self.two_component
        ), "make_sf_3cumulant is only available for non-relativistic CI."

        left_ci_vec_det = np.zeros((self.ndet))
        self.spin_adapter.csf_C_to_det_C(self.evecs[:, left_root], left_ci_vec_det)
        if right_root is None:
            right_ci_vec_det = left_ci_vec_det
        else:
            right_ci_vec_det = np.zeros((self.ndet))
            self.spin_adapter.csf_C_to_det_C(
                self.evecs[:, right_root], right_ci_vec_det
            )
        return self.ci_sigma_builder.sf_3cumulant(left_ci_vec_det, right_ci_vec_det)

    def make_so_1rdm(self, left_root: int, right_root: int = None):
        """
        Make the one-particle RDM for two CI roots. For two-component CI only.

        Parameters
        ----------
        left_root : int
            the CI root for the bra state.
        right_root : int, optional (default=left_root)
            the CI root for the ket state.

        Returns
        -------
        NDArray
            One-particle RDM.
        """
        assert self.two_component, "make_1rdm is only available for two-component CI."

        if right_root is None:
            right_root = left_root
        # copy to ensure contiguous arrays are passed to the sigma builder
        rdm = self.ci_sigma_builder.so_1rdm(
            self.evecs[:, left_root].copy(),
            self.evecs[:, right_root].copy(),
        )

        return rdm

    def make_so_1rdm_debug(self, left_root: int, right_root: int = None):
        """
        Make the one-particle RDM for two CI roots. For two-component CI only.

        Parameters
        ----------
        left_root : int
            the CI root for the bra state.
        right_root : int, optional (default=left_root)
            the CI root for the ket state.

        Returns
        -------
        NDArray
            One-particle RDM.
        """
        assert (
            self.two_component
        ), "make_1rdm_debug is only available for two-component CI."

        if right_root is None:
            right_root = left_root
        # copy to ensure contiguous arrays are passed to the sigma builder
        rdm = self.ci_sigma_builder.so_1rdm_debug(
            self.evecs[:, left_root].copy(),
            self.evecs[:, right_root].copy(),
        )

        return rdm

    def make_so_2rdm_debug(self, left_root: int, right_root: int = None):
        """
        Make the two-particle RDM for two CI roots. For two-component CI only.
        Parameters
        ----------
        left_root : int
            the CI root for the bra state.
        right_root : int, optional (default=left_root)
            the CI root for the ket state.

        Returns
        -------
        NDArray
            Two-particle RDM.
        """
        assert (
            self.two_component
        ), "make_2rdm_debug is only available for two-component CI."

        if right_root is None:
            right_root = left_root
        # copy to ensure contiguous arrays are passed to the sigma builder
        rdm = self.ci_sigma_builder.so_2rdm_debug(
            self.evecs[:, left_root].copy(),
            self.evecs[:, right_root].copy(),
        )

        return rdm

    def make_so_2cumulant(self, left_root: int, right_root: int = None):
        """
        Make the cumulant of the two-particle RDM for two CI roots. For two-component CI only.

        Parameters
        ----------
        left_root : int
            the CI root for the bra state.
        right_root : int, optional (default=left_root)
            the CI root for the ket state.

        Returns
        -------
        NDArray
            Cumulant of the two-particle RDM.
        """
        assert (
            self.two_component
        ), "make_2cumulant is only available for two-component CI."

        if right_root is None:
            right_root = left_root
        lambda2 = self.ci_sigma_builder.so_2cumulant(
            self.evecs[:, left_root].copy(),
            self.evecs[:, right_root].copy(),
        )
        return lambda2

    def make_so_2rdm(self, left_root: int, right_root: int = None):
        """
        Make the two-particle RDM for two CI roots. For two-component CI only.

        Parameters
        ----------
        left_root : int
            the CI root for the bra state.
        right_root : int, optional (default=left_root)
            the CI root for the ket state.

        Returns
        -------
        NDArray
            Two-particle RDM.
        """
        assert self.two_component, "make_2rdm is only available for two-component CI."

        if right_root is None:
            right_root = left_root
        # copy to ensure contiguous arrays are passed to the sigma builder
        rdm = self.ci_sigma_builder.so_2rdm(
            self.evecs[:, left_root].copy(),
            self.evecs[:, right_root].copy(),
        )

        return rdm

    def make_so_2cumulant_debug(self, left_root: int, right_root: int = None):
        """
        Make the cumulant of the two-particle RDM for two CI roots. For two-component CI only.

        Parameters
        ----------
        left_root : int
            the CI root for the bra state.
        right_root : int, optional (default=left_root)
            the CI root for the ket state.

        Returns
        -------
        NDArray
            Cumulant of the two-particle RDM.
        """
        assert (
            self.two_component
        ), "make_2cumulant_debug is only available for two-component CI."
        if right_root is None:
            right_root = left_root
        rdm1 = self.make_so_1rdm_debug(left_root, right_root)
        rdm2 = self.make_so_2rdm_debug(left_root, right_root)
        lambda2 = (
            rdm2
            - np.einsum("pr,qs->pqrs", rdm1, rdm1, optimize=True)
            + np.einsum("ps,qr->pqrs", rdm1, rdm1, optimize=True)
        )
        return lambda2

    def make_so_3rdm_debug(self, left_root: int, right_root: int = None):
        assert (
            self.two_component
        ), "make_3rdm_debug is only available for two-component CI."
        if right_root is None:
            right_root = left_root
        # copy to ensure contiguous arrays are passed to the sigma builder
        rdm = self.ci_sigma_builder.so_3rdm_debug(
            self.evecs[:, left_root].copy(),
            self.evecs[:, right_root].copy(),
        )

        return rdm

    def make_so_3rdm(self, left_root: int, right_root: int = None):
        assert self.two_component, "make_3rdm is only available for two-component CI."
        if right_root is None:
            right_root = left_root
        # copy to ensure contiguous arrays are passed to the sigma builder
        rdm = self.ci_sigma_builder.so_3rdm(
            self.evecs[:, left_root].copy(),
            self.evecs[:, right_root].copy(),
        )

        return rdm

    def make_so_3cumulant(self, left_root: int, right_root: int = None):
        assert (
            self.two_component
        ), "make_3cumulant is only available for two-component CI."
        if right_root is None:
            right_root = left_root
        lambda3 = self.ci_sigma_builder.so_3cumulant(
            self.evecs[:, left_root].copy(),
            self.evecs[:, right_root].copy(),
        )
        return lambda3

    def compute_natural_occupation_numbers(self):
        """
        Compute the natural occupation numbers from the spin-free 1-RDMs.

        Returns
        -------
        (norb, nroot) NDArray
            The natural occupation numbers for each root.
        """
        if not self.executed:
            raise RuntimeError("CI solver has not been executed yet.")
        no = np.zeros((self.norb, self.nroot))
        if self.two_component:
            _make_1rdm = lambda i: self.make_1rdm(i)
        else:
            _make_1rdm = lambda i: self.make_sf_1rdm(i)
        for i in range(self.nroot):
            g1 = _make_1rdm(i)
            no[:, i] = np.linalg.eigvalsh(g1)[::-1]

        return no

    def set_ints(self, scalar, oei, tei):
        """
        Set the active-space integrals for the CI solver.

        Parameters
        ----------
        scalar : float
            The scalar energy term.
        oei : NDArray
            One-electron active-space integrals in the MO basis.
        tei : NDArray
            Two-electron active-space integrals in the MO basis.
        """
        self.ints.E = scalar
        self.ints.H = oei
        self.ints.V = tei

    def set_maxiter(self, maxiter):
        """
        Set the maximum number of iterations for the CI solver.

        Parameters
        ----------
        maxiter : int
            The maximum number of iterations to set.
        """
        self.maxiter = maxiter
        if self.eigensolver is not None:
            self.eigensolver.maxiter = maxiter

    def get_maxiter(self):
        """
        Get the maximum number of iterations for the CI solver.

        Returns
        -------
        int
            The maximum number of iterations.
        """
        return self.maxiter

    def get_top_determinants(self, n=5):
        """
        Get the top `n` determinants for each root based on their coefficients in the CI vector.

        Parameters
        ----------
        n : int, optional, default=5
            The number of top determinants to return.

        Returns
        -------
        list[list[tuple[Determinant, float]]]
            A list of lists, where each inner list contains tuples of the top determinants
            and their coefficients for each root.
        """
        if not self.executed:
            raise RuntimeError("CI solver has not been executed yet.")

        top_dets_per_root = []
        if self.two_component:
            for i in range(self.nroot):
                top_dets = []
                ci_det = self.evecs[:, i]
                argsort = np.argsort(np.abs(ci_det))[
                    ::-1
                ]  # descending in absolute coeff
                for j in range(n):
                    if j < len(argsort):
                        top_dets.append((self.dets[argsort[j]], ci_det[argsort[j]]))
                top_dets_per_root.append(top_dets)
            return top_dets_per_root

        for i in range(self.nroot):
            top_dets = []
            ci_det = np.zeros((self.ndet))
            self.spin_adapter.csf_C_to_det_C(self.evecs[:, i], ci_det)
            argsort = np.argsort(np.abs(ci_det))[::-1]  # descending in absolute coeff
            for j in range(n):
                if j < len(argsort):
                    top_dets.append((self.dets[argsort[j]], ci_det[argsort[j]]))
            top_dets_per_root.append(top_dets)

        return top_dets_per_root


@dataclass
class CISolver(ActiveSpaceSolver):
    """
    A general configuration interaction (CI) solver class.
    This solver is can be called iteratively, e.g., in a MCSCF loop or a DSRG reference relaxation loop.

    Parameters
    ----------
    states : State | list[State]
        The electronic states for which the CI is solved. Can be a single state or a list of states.
        A state-averaged CI is performed if multiple states are provided.
    nroots : int | list[int], optional, default=1
        The number of roots to compute.
        If a list is provided, each element corresponds to the number of roots for each state.
        If a single integer is provided, `states` must be a single `State` object.
    weights : list[float] | list[list[float]], optional
        The weights for state averaging.
        If a list of lists is provided, each sublist corresponds to the weights for each state.
        The number of weights must match the number of roots for each state.
        If not provided, equal weights are assumed for all states.
        If a single list is provided, `states` must be a single `State` object.
    mo_space : MOSpace, optional
        A `MOSpace` object defining the partitioning of the molecular orbitals.
        If not provided, CISolver must be called with a parent method that has MOSpaceMixin (e.g., AVAS).
        If provided, it overrides the one from the parent method.
    guess_per_root : int, optional, default=2
        The number of guess vectors for each root.
    ndets_per_guess : int, optional, default=10
        The number of determinants per guess vector.
    collapse_per_root : int, optional, default=2
        The number of determinants to collapse per root.
    basis_per_root : int, optional, default=4
        The maximum number of basis vectors per root.
    maxiter : int, optional, default=100
        The maximum number of iterations for the Davidson-Liu solver.
    econv : float, optional, default=1e-10
        The energy convergence threshold for the solver.
    rconv : float, optional, default=1e-5
        The residual convergence threshold for the solver.
    energy_shift : float, optional, default=None
        An energy shift to find roots around. If None, no shift is applied.
    do_test_rdms : bool, optional, default=False
        If True, compute and test the reduced density matrices (RDMs) after the CI calculation.
    log_level : int, optional
        The logging level for the CI solver. Defaults to the global logger's verbosity level.
    ci_algorithm : str, optional, valid choices=["hz", "kh"], default="hz"
        The algorithm used for the CI sigma builder.

    Attributes
    ----------
    sub_solvers : list[_CIBase]
        A list of CI solvers for each state in the state-averaged CI.
    evals_per_solver : list[NDArray]
        The eigenvalues (energies) computed by each sub-solver.
    evals_flat, E : NDArray
        The flattened array of eigenvalues from all sub-solvers.
    E_avg : float
        The average energy computed from the state-averaged CI roots.
    """

    ### Davidson-Liu parameters
    guess_per_root: int = 2
    ndets_per_guess: int = 10
    collapse_per_root: int = 2
    basis_per_root: int = 4
    maxiter: int = 100
    econv: float = 1e-10
    rconv: float = 1e-5
    energy_shift: float = None

    do_test_rdms: bool = False
    log_level: int = field(default=logger.get_verbosity_level())

    ### Non-init attributes
    ci_builder_memory: int = field(default=1024, init=False)  # in MB
    first_run: bool = field(default=True, init=False)
    executed: bool = field(default=False, init=False)

    def __call__(self, method):
        self.parent_method = method
        return self

    def _startup(self):
        super()._startup()
        self.norb = self.mo_space.nactv
        # no distinction between core and frozen core in the CI solver
        self.core_indices = (
            self.mo_space.frozen_core_indices + self.mo_space.core_indices
        )
        self.active_indices = self.mo_space.active_indices

        ints = RestrictedMOIntegrals(
            self.system,
            self.C[0],
            self.active_indices,
            self.core_indices,
            use_aux_corr=True,
        )

        self.sub_solvers = []
        active_orbsym = [
            [self.irrep_indices[i] for i in active_space]
            for active_space in self.mo_space.active_orbitals
        ]
        for i, state in enumerate(self.sa_info.states):
            # Create a CI solver for each state and MOSpace
            self.sub_solvers.append(
                _CIBase(
                    mo_space=self.mo_space,
                    ints=ints,
                    state=state,
                    nroot=self.sa_info.nroots[i],
                    active_orbsym=active_orbsym,
                    do_test_rdms=self.do_test_rdms,
                    die_if_not_converged=self.die_if_not_converged,
                    ci_algorithm=self.ci_algorithm,
                    guess_per_root=self.guess_per_root,
                    ndets_per_guess=self.ndets_per_guess,
                    collapse_per_root=self.collapse_per_root,
                    basis_per_root=self.basis_per_root,
                    maxiter=self.maxiter,
                    econv=self.econv,
                    rconv=self.rconv,
                    energy_shift=self.energy_shift,
                    log_level=self.log_level,
                )
            )

    def run(self):
        if self.first_run:
            self._startup()
            self.first_run = False

        self.evals_per_solver = []
        for ci_solver in self.sub_solvers:
            ci_solver.run()
            self.evals_per_solver.append(ci_solver.evals)

        self.evals_flat = np.concatenate(self.evals_per_solver)
        self.E_avg = self.compute_average_energy()

        self.E = self.evals_flat

        self.executed = True
        return self

    def compute_average_energy(self):
        """
        Compute the average energy from the CI roots using the weights.

        Returns
        -------
        float
            Average energy of the CI roots.
        """
        return np.dot(self.weights_flat, self.evals_flat)

    def make_average_1rdm(self):
        """
        Make the average spin-free one-particle RDM from the CI vectors.

        Returns
        -------
        NDArray
            Average spin-free one-particle RDM.
        """
        rdm1 = np.zeros((self.norb,) * 2, dtype=self.dtype)
        for i, ci_solver in enumerate(self.sub_solvers):
            for j in range(ci_solver.nroot):
                rdm1 += ci_solver.make_1rdm(j) * self.weights[i][j]
        return rdm1

    def make_average_2rdm(self):
        """
        Make the average spin-free two-particle RDM from the CI vectors.

        Returns
        -------
        NDArray
            Average spin-free two-particle RDM.
        """
        rdm2 = np.zeros((self.norb,) * 4, dtype=self.dtype)
        for i, ci_solver in enumerate(self.sub_solvers):
            for j in range(ci_solver.nroot):
                rdm2 += ci_solver.make_2rdm(j) * self.weights[i][j]

        return rdm2

    def make_average_3rdm(self):
        """
        Make the average spin-free three-particle RDM from the CI vectors.

        Returns
        -------
        NDArray
            Average spin-free three-particle RDM.
        """
        rdm3 = np.zeros((self.norb,) * 6, dtype=self.dtype)
        for i, ci_solver in enumerate(self.sub_solvers):
            for j in range(ci_solver.nroot):
                rdm3 += ci_solver.make_3rdm(j) * self.weights[i][j]

        return rdm3
    
    def make_average_2cumulant(self):
        dm1 = self.make_average_1rdm()
        dm2 = self.make_average_2rdm()
        if self.two_component:
            return make_2cumulant_so(dm1, dm2)
        else:
            return make_2cumulant_sf(dm1, dm2)
        
    def make_average_3cumulant(self):
        dm1 = self.make_average_1rdm()
        dm2 = self.make_average_2rdm()
        dm3 = self.make_average_3rdm()
        if self.two_component:
            return self.make_3cumulant_so(dm1, dm2, dm3)
        else:
            return self.make_3cumulant_sf(dm1, dm2, dm3)
        
    def make_average_cumulants(self):
        dm1 = self.make_average_1rdm()
        dm2 = self.make_average_2rdm()
        dm3 = self.make_average_3rdm()
        if self.two_component:
            lambda2 = make_2cumulant_so(dm1, dm2)
            lambda3 = make_3cumulant_so(dm1, dm2, dm3)
        else:
            lambda2 = make_2cumulant_sf(dm1, dm2)
            lambda3 = make_3cumulant_sf(dm1, dm2, dm3)
        return dm1, dm2, lambda2, lambda3

    def set_ints(self, scalar, oei, tei):
        """
        Set the active-space integrals for the CI solver.

        Parameters
        ----------
        scalar : float
            The scalar energy term.
        oei : NDArray
            One-electron active-space integrals in the MO basis.
        tei : NDArray
            Two-electron active-space integrals in the MO basis.
        """
        for ci_solver in self.sub_solvers:
            ci_solver.set_ints(scalar, oei, tei)

    def compute_natural_occupation_numbers(self):
        """
        Compute the natural occupation numbers for the CI states.

        Returns
        -------
        (norb, nroot) NDArray
            The natural occupation numbers for each root.
        """
        nos = []
        for ci_solver in self.sub_solvers:
            nos.append(ci_solver.compute_natural_occupation_numbers())
        self.nat_occs = np.concatenate(nos, axis=1)

    def get_top_determinants(self, n=5):
        """
        Get the top `n` determinants for each root based on their coefficients in the CI vector.

        Parameters
        ----------
        n : int, optional, default=5
            The number of top determinants to return.

        Returns
        -------
        top_dets : list[list[tuple[Determinant, float]]]]
            top_dets[i] contains a list of tuples (Determinant, coefficient) for the `i`-th root.
        """
        top_dets = []
        for ci_solver in self.sub_solvers:
            top_dets += ci_solver.get_top_determinants(n)
        return top_dets

    def compute_transition_properties(self, C=None):
        """
        Compute the transition dipole moments and oscillator strengths from the spin-free 1-TDMs.
        The results are stored in `self.tdm_per_solver` and `self.fosc_per_solver`.
        """
        if not self.executed:
            raise RuntimeError("CI solver has not been executed yet.")

        if C is None:
            C = self.C[0]

        Cact = C[:, self.active_indices]
        Ccore = C[:, self.core_indices]
        factor = 1.0 if self.two_component else 2.0
        rdm_core = factor * np.einsum("pi,qi->pq", Ccore, Ccore.conj(), optimize=True)
        # this includes nuclear dipole contribution
        core_dip = get_1e_property(
            self.system, rdm_core, property_name="dipole", unit="au"
        )
        self.tdm_per_solver = []
        self.fosc_per_solver = []

        for ici, ci_solver in enumerate(self.sub_solvers):
            tdmdict = OrderedDict()
            foscdict = OrderedDict()
            for i in range(ci_solver.nroot):
                rdm = ci_solver.make_1rdm(i)
                # Different (back-)transformation rules for RDMs:
                # O_{mu}^{nu} = C_{mu}^p <phi_p|O|phi^q> C^q_{nu} = C^H O[mo] C
                # rdm^{mu}_{nu} = C^{mu}_p <a^p a_q> C^q_{nu} = C^* rdm[mo] C^T
                rdm = np.einsum("ij,pi,qj->pq", rdm, Cact.conj(), Cact, optimize=True)
                dip = get_1e_property(
                    self.system, rdm, property_name="electric_dipole", unit="au"
                )
                tdmdict[(i, i)] = dip + core_dip
                foscdict[(i, i)] = 0.0  # No oscillator strength for i->i transitions
                for j in range(i + 1, ci_solver.nroot):
                    tdm = ci_solver.make_1rdm(i, j)
                    tdm = np.einsum(
                        "ij,pi,qj->pq", tdm, Cact.conj(), Cact, optimize=True
                    )
                    tdip = get_1e_property(
                        self.system, tdm, property_name="electric_dipole", unit="au"
                    )
                    tdmdict[(i, j)] = tdip
                    vte = self.evals_per_solver[ici][j] - self.evals_per_solver[ici][i]
                    foscdict[(i, j)] = (2 / 3) * vte * np.linalg.norm(tdip) ** 2
            self.fosc_per_solver.append(foscdict)
            self.tdm_per_solver.append(tdmdict)


@dataclass
class CI(CISolver):
    """
    CI solver specialized for a single CI calculation. (i.e., not used in a loop).
    See `CISolver` for all parameters and attributes.
    """

    die_if_not_converged: bool = True
    final_orbital: str = "original"
    do_transition_dipole: bool = False

    def run(self):
        super().run()
        self._post_process()
        if self.final_orbital == "semicanonical":
            semi = Semicanonicalizer(
                mo_space=self.mo_space,
                system=self.system,
            )
            C_contig = self.C[0][:, self.mo_space.orig_to_contig].copy()
            semi.semi_canonicalize(g1=self.make_average_1rdm(), C_contig=C_contig)
            self.C[0] = semi.C_semican[self.mo_space.contig_to_orig].copy()

            # recompute the CI vectors in the semicanonical basis
            ints = RestrictedMOIntegrals(
                self.system,
                self.C[0],
                self.active_indices,
                self.core_indices,
                use_aux_corr=True,
            )
            self.set_ints(ints.E, ints.H, ints.V)
            super().run()

        return self

    def _post_process(self):
        pretty_print_ci_summary(self.sa_info, self.evals_per_solver)
        self.compute_natural_occupation_numbers()
        pretty_print_ci_nat_occ_numbers(self.sa_info, self.mo_space, self.nat_occs)
        top_dets = self.get_top_determinants()
        pretty_print_ci_dets(self.sa_info, self.mo_space, top_dets)

        if self.do_transition_dipole:
            self.compute_transition_properties()
            pretty_print_ci_transition_props(
                self.sa_info,
                self.tdm_per_solver,
                self.fosc_per_solver,
                self.evals_per_solver,
            )


@dataclass
class RelCISolver(RelActiveSpaceSolver):
    """
    Relativistic Configuration Interaction
    """

    ### Davidson-Liu parameters
    guess_per_root: int = 2
    ndets_per_guess: int = 10
    collapse_per_root: int = 2
    basis_per_root: int = 4
    maxiter: int = 100
    econv: float = 1e-10
    rconv: float = 1e-5
    energy_shift: float = None

    do_test_rdms: bool = False
    log_level: int = field(default=logger.get_verbosity_level())

    ### Non-init attributes
    ci_builder_memory: int = field(default=1024, init=False)  # in MB
    first_run: bool = field(default=True, init=False)
    executed: bool = field(default=False, init=False)

    compute_average_energy = CISolver.compute_average_energy
    make_average_1rdm = CISolver.make_average_1rdm
    make_average_2rdm = CISolver.make_average_2rdm
    make_average_3rdm = CISolver.make_average_3rdm
    make_average_2cumulant = CISolver.make_average_2cumulant
    make_average_3cumulant = CISolver.make_average_3cumulant
    make_average_cumulants = CISolver.make_average_cumulants
    compute_natural_occupation_numbers = CISolver.compute_natural_occupation_numbers
    get_top_determinants = CISolver.get_top_determinants
    set_ints = CISolver.set_ints
    compute_transition_properties = CISolver.compute_transition_properties

    def __call__(self, parent_method):
        self.parent_method = parent_method
        return self

    def _startup(self):
        super()._startup(two_component=True)
        if not self.system.two_component:
            self.C = convert_coeff_spatial_to_spinor(self.system, self.C)
            self.system.two_component = True

        self.norb = self.mo_space.nactv
        # no distinction between core and frozen core in the CI solver
        self.core_indices = (
            self.mo_space.frozen_core_indices + self.mo_space.core_indices
        )
        self.active_indices = self.mo_space.active_indices

        ints = SpinorbitalIntegrals(
            self.system,
            self.C[0],
            self.active_indices,
            self.core_indices,
            use_aux_corr=True,
        )

        self.sub_solvers = []
        active_orbsym = [
            [self.irrep_indices[i] for i in active_space]
            for active_space in self.mo_space.active_orbitals
        ]

        for i, state in enumerate(self.sa_info.states):
            # Create a CI solver for each state and MOSpace
            self.sub_solvers.append(
                _CIBase(
                    mo_space=self.mo_space,
                    ints=ints,
                    state=state,
                    nroot=self.sa_info.nroots[i],
                    active_orbsym=active_orbsym,
                    do_test_rdms=self.do_test_rdms,
                    ci_algorithm=self.ci_algorithm,
                    guess_per_root=self.guess_per_root,
                    ndets_per_guess=self.ndets_per_guess,
                    collapse_per_root=self.collapse_per_root,
                    basis_per_root=self.basis_per_root,
                    maxiter=self.maxiter,
                    econv=self.econv,
                    rconv=self.rconv,
                    energy_shift=self.energy_shift,
                    log_level=self.log_level,
                    two_component=True,
                )
            )

    def run(self, use_asym_ints=False):
        if self.first_run:
            self._startup()
            self.first_run = False

        self.evals_per_solver = []
        for ci_solver in self.sub_solvers:
            ci_solver.run(use_asym_ints=use_asym_ints)
            self.evals_per_solver.append(ci_solver.evals)

        self.evals_flat = np.concatenate(self.evals_per_solver)
        self.E_avg = self.compute_average_energy()

        self.E = self.evals_flat

        self.executed = True
        return self


@dataclass
class RelCI(RelCISolver):
    final_orbital: str = "original"
    do_transition_dipole: bool = False

    def run(self):
        super().run()
        self._post_process()
        if self.final_orbital == "semicanonical":
            semi = Semicanonicalizer(
                mo_space=self.mo_space,
                system=self.system,
            )
            C_contig = self.C[0][:, self.mo_space.orig_to_contig].copy()
            semi.semi_canonicalize(g1=self.make_average_1rdm(), C_contig=C_contig)
            self.C[0] = semi.C_semican[self.mo_space.contig_to_orig].copy()

            # recompute the CI vectors in the semicanonical basis
            ints = SpinorbitalIntegrals(
                self.system,
                self.C[0],
                self.active_indices,
                self.core_indices,
                use_aux_corr=True,
            )
            self.set_ints(ints.E, ints.H, ints.V)
            super().run()

        return self

    def _post_process(self):
        pretty_print_ci_summary(self.sa_info, self.evals_per_solver)
        self.compute_natural_occupation_numbers()
        pretty_print_ci_nat_occ_numbers(self.sa_info, self.mo_space, self.nat_occs)
        top_dets = self.get_top_determinants()
        pretty_print_ci_dets(self.sa_info, self.mo_space, top_dets)

        if self.do_transition_dipole:
            self.compute_transition_properties()
            pretty_print_ci_transition_props(
                self.sa_info,
                self.tdm_per_solver,
                self.fosc_per_solver,
                self.evals_per_solver,
            )
