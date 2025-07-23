from dataclasses import dataclass, field
from abc import ABC, abstractmethod

import numpy as np
import forte2

from forte2.state.state import State
from forte2.helpers.mixins import MOsMixin, SystemMixin
from forte2.helpers.davidsonliu import DavidsonLiuSolver
from forte2.helpers import logger
from forte2.orbitals import MOSpace, AVAS
from forte2.jkbuilder import RestrictedMOIntegrals
from forte2.system.system import System


@dataclass
class CIBase:
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
    ci_algorithm : str, optional, default="hz"
        The algorithm used for the CI sigma builder.
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
    gas_min : list[int], optional, default=[]
        The minimum number of orbitals in each general orbital space (GAS).
    gas_max : list[int], optional, default=[]
        The maximum number of orbitals in each general orbital space (GAS).
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
    do_test_rdms: bool = False
    log_level: int = field(default=logger.get_verbosity_level())

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
        self.ncore = self.mo_space.ncore
        self.ngas = self.mo_space.ngas
        self.gas_min = self.state.gas_min
        self.gas_max = self.state.gas_max
        self.eigensolver = None

    def _ci_solver_startup(self):
        self.orbital_symmetry = [
            [0] * len(self.mo_space.active_spaces[x]) for x in range(self.ngas)
        ]

        self.ci_strings = forte2.CIStrings(
            self.state.na - self.ncore,
            self.state.nb - self.ncore,
            self.state.symmetry,
            self.orbital_symmetry,
            self.gas_min,
            self.gas_max,
            log_level=self.log_level,
        )

        logger.log(f"\nNumber of α electrons: {self.ci_strings.na}", self.log_level)
        logger.log(f"Number of β electrons: {self.ci_strings.nb}", self.log_level)
        logger.log(f"Number of α strings: {self.ci_strings.nas}", self.log_level)
        logger.log(f"Number of β strings: {self.ci_strings.nbs}", self.log_level)
        logger.log(f"Number of determinants: {self.ci_strings.ndet}", self.log_level)

        self.spin_adapter = forte2.CISpinAdapter(
            self.state.multiplicity - 1, self.state.twice_ms, self.norb
        )
        self.spin_adapter.set_log_level(self.log_level)
        self.dets = self.ci_strings.make_determinants()

        self.spin_adapter.prepare_couplings(self.dets)
        logger.log(f"Number of CSFs: {self.spin_adapter.ncsf()}", self.log_level)

        # 1. Allocate memory for the CI vectors
        self.ndet = self.ci_strings.ndet
        self.basis_size = self.spin_adapter.ncsf()

        # Create the CI vectors that will hold the results of the sigma builder in the
        # determinant basis
        self.b_det = np.zeros((self.ndet))
        self.sigma_det = np.zeros((self.ndet))

    def run(self):
        if not self.executed:
            self._ci_solver_startup()

        # Create the CISigmaBuilder from the CI strings and integrals
        # This object handles some temporary memory deallocated at destruction
        # and is used to compute the Hamiltonian matrix elements in the determinant basis
        self.ci_sigma_builder = forte2.CISigmaBuilder(
            self.ci_strings, self.ints.E, self.ints.H, self.ints.V, self.log_level
        )
        self.ci_sigma_builder.set_memory(self.ci_builder_memory)
        self.ci_sigma_builder.set_algorithm(self.ci_algorithm)

        Hdiag = self.ci_sigma_builder.form_Hdiag_csf(
            self.dets, self.spin_adapter, spin_adapt_full_preconditioner=False
        )

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
            )

        # 4. Compute diagonal of the Hamiltonian
        self.eigensolver.add_h_diag(Hdiag)

        # 5. Build the guess vectors if this is the first run
        if self.first_run:
            self._build_guess_vectors(Hdiag)
            self.first_run = False

        def sigma_builder(Bblock, Sblock):
            # Compute the sigma block from the basis block
            ncols = Bblock.shape[1]
            for i in range(ncols):
                self.spin_adapter.csf_C_to_det_C(Bblock[:, i], self.b_det)
                self.ci_sigma_builder.Hamiltonian(self.b_det, self.sigma_det)
                # self.sigma_det = np.dot(Hex, self.b_det)
                self.spin_adapter.det_C_to_csf_C(self.sigma_det, Sblock[:, i])

        self.eigensolver.add_sigma_builder(sigma_builder)

        # 6. Run Davidson
        self.evals, self.evecs = self.eigensolver.solve()

        logger.log(f"\nDavidson-Liu solver converged.\n", self.log_level)

        # 7. Store the final energy and properties
        self.E = self.evals
        for i, e in enumerate(self.evals):
            logger.log(f"Final CI Energy Root {i}: {e:20.12f} [Eh]", self.log_level)

        h_tot, h_aabb, h_aaaa, h_bbbb = self.ci_sigma_builder.avg_build_time()
        logger.log("\nAverage CI Sigma Builder time summary:", self.log_level)
        logger.log(f"h_aabb time:    {h_aabb:.3f} s/build", self.log_level)
        logger.log(f"h_aaaa time:    {h_aaaa:.3f} s/build", self.log_level)
        logger.log(f"h_bbbb time:    {h_bbbb:.3f} s/build", self.log_level)
        logger.log(f"total time:     {h_tot:.3f} s/build", self.log_level)

        if self.do_test_rdms:
            self._test_rdms()

        self.executed = True

        return self

    def _test_rdms(self):
        # Compute the RDMs from the CI vectors
        # and verify the energy from the RDMs matches the CI energy
        logger.log("\nComputing RDMs from CI vectors.\n", self.log_level)
        for root in range(self.nroot):
            root_rdms = {}
            root_rdms["rdm1"] = self.make_rdm1_sf(self.evecs[:, root])
            rdm2_aa, rdm2_ab, rdm2_bb = self.make_rdm2_sd(
                self.evecs[:, root], full=False
            )
            root_rdms["rdm2_aa"] = rdm2_aa
            root_rdms["rdm2_ab"] = rdm2_ab
            root_rdms["rdm2_bb"] = rdm2_bb

            rdm2_aa_full, _, rdm2_bb_full = self.make_rdm2_sd(
                self.evecs[:, root], full=True
            )
            root_rdms["rdm2_aa_full"] = rdm2_aa_full
            root_rdms["rdm2_bb_full"] = rdm2_bb_full

            root_rdms["rdm2_sf"] = self.make_rdm2_sf(self.evecs[:, root])

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

            from forte2.helpers.comparisons import approx

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
        """
        Build the guess vectors for the CI calculation.
        This method is a placeholder and should be implemented in subclasses.
        """

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

        # create the Hamiltonian matrix in the basis of the guess CSFs
        Hguess = np.zeros((nguess_dets, nguess_dets))
        for i, I in enumerate(indices):
            for j, J in enumerate(indices):
                if i >= j:
                    Hij = self.ci_sigma_builder.slater_rules_csf(
                        self.dets, self.spin_adapter, I, J
                    )
                    Hguess[i, j] = Hij
                    Hguess[j, i] = Hij

        # Diagonalize the Hamiltonian to get the initial guess vectors
        evals_guess, evecs_guess = np.linalg.eigh(Hguess)

        # Select the lowest eigenvalues and their corresponding eigenvectors
        guess_mat = np.zeros((self.basis_size, self.num_guess_states))
        for i in range(self.num_guess_states):
            guess = evecs_guess[:, i]
            for j, d in enumerate(indices):
                guess_mat[d, i] = guess[j]

        self.eigensolver.add_guesses(guess_mat)

    def make_rdm1_sf(self, ci_vec):
        """
        Make the spin-free one-particle RDM from a CI vector.

        Parameters
        ----------
            ci_vec : NDArray
                CI vector in the CSF basis.

        Returns
        -------
            NDArray
                Spin-free one-particle RDM."""
        ci_vec_det = np.zeros((self.ndet))
        self.spin_adapter.csf_C_to_det_C(ci_vec, ci_vec_det)
        return self.ci_sigma_builder.rdm1_sf(ci_vec_det, ci_vec_det)

    def make_rdm1_a(self, ci_vec, spin):
        """
        Make the spin-free one-particle RDM from a CI vector.
        Args:
            ci_vec (ndarray): CI vector in the CSF basis.
        Returns:
            ndarray: Spin-free one-particle RDM."""
        ci_vec_det = np.zeros((self.ndet))
        self.spin_adapter.csf_C_to_det_C(ci_vec, ci_vec_det)
        return self.ci_sigma_builder.rdm1_a(ci_vec_det, ci_vec_det, spin)

    def make_tdm1_sf(self, ci_l, ci_r):
        """
        Make the spin-free one-particle transition density matrix from two CI vectors.

        Parameters
        ----------
            ci_l : NDArray
                Left CI vector in the CSF basis.
            ci_r : NDArray
                Right CI vector in the CSF basis.

        Returns
        -------
            NDArray
                Spin-free one-particle transition density matrix.
        """
        ci_l_det = np.zeros((self.ndet))
        ci_r_det = np.zeros((self.ndet))
        self.spin_adapter.csf_C_to_det_C(ci_l, ci_l_det)
        self.spin_adapter.csf_C_to_det_C(ci_r, ci_r_det)
        return self.ci_sigma_builder.rdm1_sf(ci_l_det, ci_r_det)

    def make_rdm2_sd(self, ci_vec, full=True):
        """
        Make the spin-dependent two-particle RDMs (aa, ab, bb) from a CI vector in the CSF basis.

        Parameters
        ----------
            ci_vec : ndarray
                CI vector in the CSF basis.
            full : bool, optional, default=True
                If True, compute the full-dimension RDMs, otherwise compute compact aa and bb RDMs.

        Returns
        -------
            tuple :
                Spin-dependent two-particle RDMs (aa, ab, bb).
        """
        ci_vec_det = np.zeros((self.ndet))
        self.spin_adapter.csf_C_to_det_C(ci_vec, ci_vec_det)
        aa_build = (
            self.ci_sigma_builder.rdm2_aa_full
            if full
            else self.ci_sigma_builder.rdm2_aa
        )
        aa = aa_build(ci_vec_det, ci_vec_det, True)
        bb = aa_build(ci_vec_det, ci_vec_det, False)
        ab = self.ci_sigma_builder.rdm2_ab(ci_vec_det, ci_vec_det)
        return aa, ab, bb

    def make_tdm2_sd(self, ci_l, ci_r, full=True):
        """
        Make the spin-dependent two-particle transition density matrices (aa, ab, bb)
        from two CI vectors in the CSF basis.

        Parameters
        ----------
            ci_l : NDArray
                Left CI vector in the CSF basis.
            ci_r : NDArray
                Right CI vector in the CSF basis.
            full : bool, optional, default=True
                If True, compute the full-dimension RDMs, otherwise compute compact aa and bb RDMs.

        Returns
        -------
            tuple
                Spin-dependent two-particle transition density matrices (aa, ab, bb).
        """
        ci_l_det = np.zeros((self.ndet))
        ci_r_det = np.zeros((self.ndet))
        self.spin_adapter.csf_C_to_det_C(ci_l, ci_l_det)
        self.spin_adapter.csf_C_to_det_C(ci_r, ci_r_det)
        aa_build = (
            self.ci_sigma_builder.rdm2_aa_full
            if full
            else self.ci_sigma_builder.rdm2_aa
        )
        aa = aa_build(ci_l_det, ci_r_det, True)
        bb = aa_build(ci_l_det, ci_r_det, False)
        ab = self.ci_sigma_builder.rdm2_ab(ci_l_det, ci_r_det)
        return aa, ab, bb

    def make_rdm2_sf(self, ci_vec):
        """
        Make the spin-free two-particle RDM from a CI vector in the CSF basis.

        Parameters
        ----------
        ci_vec : NDArray
            CI vector in the CSF basis.

        Returns
        -------
        NDArray
            Spin-free two-particle RDM.
        """
        ci_vec_det = np.zeros((self.ndet))
        self.spin_adapter.csf_C_to_det_C(ci_vec, ci_vec_det)
        return self.ci_sigma_builder.rdm2_sf(ci_vec_det, ci_vec_det)

    def make_tdm2_sf(self, ci_l, ci_r):
        """
        Make the spin-free two-particle transition density matrix from two CI vectors in the CSF basis.

        Parameters
        ----------
            ci_l : NDArray
                Left CI vector in the CSF basis.
            ci_r : NDArray
                Right CI vector in the CSF basis.

        Returns
        -------
            NDArray
                Spin-free two-particle transition density matrix.
        """
        ci_l_det = np.zeros((self.ndet))
        ci_r_det = np.zeros((self.ndet))
        self.spin_adapter.csf_C_to_det_C(ci_l, ci_l_det)
        self.spin_adapter.csf_C_to_det_C(ci_r, ci_r_det)
        return self.ci_sigma_builder.rdm2_sf(ci_l_det, ci_r_det)

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


@dataclass
class CIStates:
    """
    A class to hold information about state averaging in multireference calculations.

    Parameters
    ----------
    states : list[State] | State
        A list of `State` objects or a single `State` object representing the electronic states.
        This also includes the gas_min and gas_max attributes.
    nroots : list[int] | int, optional, default=1
        A list of integers specifying the number of roots for each state.
        If only one state is provided, this can be a single integer.
    weights : list[list[float]], optional
        A list of lists of floats specifying the weights for each root in each state.
        These do not have to be normalized, but must be non-negative.
        If not provided, equal weights are assigned to each root.
    mo_space : MOSpace | AVAS, optional
        The molecular orbital space defining the active spaces and core orbitals.
        This is used with each `State` to define a `CIBase` solver.
        If not provided, it will be constructed from `core_orbitals` and `active_spaces`.
        An `AVAS` object can provided here instead, it does not need to be run first.
    core_orbitals : list[int], optional
        A list of integers specifying the core orbitals.
        If `AVAS` is provided, this field will be fetched from it after its execution.
    active_spaces : list[list[int]], optional
        A list of lists of integers specifying the orbital indices for each GAS.
        If `AVAS` is provided, this field will be fetched from it after its execution.

    Attributes
    ----------
    ncis : int
        The number of CI states, which is the length of the `states` list.
    nroots_sum : int
        The total number of roots across all states.
    weights_flat : NDArray
        A flattened array of weights for all roots across all states.
    norb : int
        The number of active orbitals in the molecular orbital space.
    ncore : int
        The number of core orbitals in the molecular orbital space.
        If `AVAS` is provided, this is only available after its execution.
    """

    states: list[State] | State
    nroots: list[int] | int = 1
    weights: list[list[float]] = None
    mo_space: MOSpace | AVAS = None
    core_orbitals: list[int] = None
    active_spaces: list[list[int]] = None

    def __post_init__(self):
        # 1. Validate states
        if isinstance(self.states, State):
            self.states = [self.states]
        assert isinstance(self.states, list), "states_and_mo_spaces must be a list"
        assert all(
            isinstance(state, State) for state in self.states
        ), "All elements in states_and_mo_spaces must be State instances"
        assert len(self.states) > 0, "states_and_mo_spaces cannot be empty"
        self.ncis = len(self.states)

        # 2. Make mo_space from core_orbitals and active_spaces if mo_space is not provided
        if self.mo_space is None:
            assert (
                self.active_spaces is not None
            ), "If mo_space is not provided, active_spaces must be provided"
            if self.core_orbitals is None:
                self.core_orbitals = []
            self.mo_space = MOSpace(
                core_orbitals=self.core_orbitals,
                active_spaces=self.active_spaces,
            )

        # 3. Validate nroots
        if isinstance(self.nroots, int):
            assert (
                self.ncis == 1
            ), "If nroots is an integer, there must be exactly one state."
            self.nroots = [self.nroots]
        assert isinstance(self.nroots, list), "nroots must be a list"
        assert all(
            isinstance(n, int) and n > 0 for n in self.nroots
        ), "nroots must be a list of positive integers"
        self.nroots_sum = sum(self.nroots)

        # 4. Validate weights
        if self.weights is None:
            self.weights = [[1.0 / self.nroots_sum] * n for n in self.nroots]
            self.weights_flat = np.concatenate(self.weights)
        else:
            assert (
                sum(len(w) for w in self.weights) == self.nroots_sum
            ), "Weights must match the total number of roots across all states"
            self.weights_flat = np.array(
                [w for sublist in self.weights for w in sublist], dtype=float
            )
            n = self.weights_flat.sum()
            self.weights = [[w / n for w in sublist] for sublist in self.weights]
            self.weights_flat /= n
            assert np.all(self.weights_flat >= 0), "Weights must be non-negative"

    def fetch_mo_space(self):
        if isinstance(self.mo_space, AVAS):
            assert (
                self.mo_space.executed
            ), "AVAS must be executed before fetching MOSpace"
        self.norb = self.mo_space.nactv
        self.ncore = self.mo_space.ncore
        self.active_orbitals = self.mo_space.active_orbitals
        self.core_orbitals = self.mo_space.core_orbitals

    def pretty_print_ci_states(self):
        """
        Pretty print the CI states
        """
        width = 33
        logger.log_info1("\nRequested CI states:")
        logger.log_info1("=" * width)
        logger.log_info1(
            f"{'Root':>4} {'Nel':>5} {'Mult.':>6} {'Ms':>4} {'Weight':>10}"
        )
        logger.log_info1("-" * width)
        iroot = 0
        for i, state in enumerate(self.states):
            for j in range(self.nroots[i]):
                logger.log_info1(
                    f"{iroot:>4} {state.nel:>5} {state.multiplicity:>6d} {state.ms:>4.1f} {self.weights[i][j]:>10.6f}"
                )
                iroot += 1
            if i < len(self.states) - 1:
                logger.log_info1("-" * width)
        logger.log_info1("=" * width + "\n")


@dataclass
class CISolver:
    """
    A general configuration interaction (CI) solver class.
    This solver is can be called iteratively, e.g., in a MCSCF loop or a DSRG reference relaxation loop.

    Parameters
    ----------
    ci_states : CIStates
        An instance of `CIStates` that holds information about the states to be solved.
        This enables arbitrary state averaging in multireference calculations.
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
    ci_algorithm : str, optional, default="hz"
        The algorithm used for the CI sigma builder.
    """

    ci_states: CIStates

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

    ### Sigma builder parameters
    ci_algorithm: str = "hz"

    ### Non-init attributes
    ci_builder_memory: int = field(default=1024, init=False)  # in MB
    first_run: bool = field(default=True, init=False)
    executed: bool = field(default=False, init=False)

    def __call__(self, method):
        self.parent_method = method
        return self

    def _startup(self):
        if not self.parent_method.executed:
            self.parent_method.run()

        self.ci_states.fetch_mo_space()
        self.ncis = self.ci_states.ncis
        self.core_orbitals = self.ci_states.core_orbitals
        self.active_orbitals = self.ci_states.active_orbitals
        self.norb = self.ci_states.norb
        self.weights = self.ci_states.weights
        self.weights_flat = self.ci_states.weights_flat

        SystemMixin.copy_from_upstream(self, self.parent_method)
        MOsMixin.copy_from_upstream(self, self.parent_method)

        ints = RestrictedMOIntegrals(
            self.system,
            self.C[0],
            self.active_orbitals,
            self.core_orbitals,
            use_aux_corr=True,
        )

        self.ci_solvers = []
        for i, state in enumerate(self.ci_states.states):
            # Create a CI solver for each state and MOSpace
            self.ci_solvers.append(
                CIBase(
                    mo_space=self.ci_states.mo_space,
                    ints=ints,
                    state=state,
                    nroot=self.ci_states.nroots[i],
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
                )
            )

    def run(self):
        if self.first_run:
            self._startup()
            self.first_run = False

        self.evals_per_solver = []
        for ci_solver in self.ci_solvers:
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

    def make_average_rdm1_sf(self):
        """
        Make the average spin-free one-particle RDM from the CI vectors.

        Returns
        -------
        NDArray
            Average spin-free one-particle RDM.
        """
        rdm1 = np.zeros((self.norb,) * 2)
        for i, ci_solver in enumerate(self.ci_solvers):
            for j in range(ci_solver.nroot):
                rdm1 += (
                    ci_solver.make_rdm1_sf(ci_solver.evecs[:, j]) * self.weights[i][j]
                )
        return rdm1

    def make_average_rdm2_sf(self):
        """
        Make the average spin-free two-particle RDM from the CI vectors.

        Returns
        -------
        NDArray
            Average spin-free two-particle RDM.
        """
        rdm2 = np.zeros((self.norb,) * 4)
        for i, ci_solver in enumerate(self.ci_solvers):
            for j in range(ci_solver.nroot):
                rdm2 += (
                    ci_solver.make_rdm2_sf(ci_solver.evecs[:, j]) * self.weights[i][j]
                )
        return rdm2

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
        for ci_solver in self.ci_solvers:
            ci_solver.ints.E = scalar
            ci_solver.ints.H = oei
            ci_solver.ints.V = tei


@dataclass
class CI(CISolver):
    """
    CI solver specialized for a single CI calculation. (i.e., not used in a loop).
    See `CISolver` for all parameters and attributes.
    """

    def run(self):
        super().run()
        self._post_process()

    def _post_process(self):
        pretty_print_ci_summary(self.ci_states, self.evals_per_solver)


def pretty_print_ci_summary(cistates: CIStates, eigvals_per_solver: list[list[float]]):
    """
    Pretty print the CI energy summary for the given CI states and eigenvalues.

    Parameters
    ----------
    cistates : CIStates
        An instance of `CIStates` that holds information about the states and their properties.
    eigvals_per_solver : list[list[float]]
        A list of lists containing the eigenvalues (energies) for each CI solver.
    """
    ncis = cistates.ncis
    mult = [state.multiplicity for state in cistates.states]
    ms = [state.ms for state in cistates.states]
    irrep = [state.symmetry for state in cistates.states]
    weights = cistates.weights
    nroots = cistates.nroots

    logger.log_info1("CI energy summary:")
    width = 64
    logger.log_info1("=" * width)
    logger.log_info1(
        f"{'Root':>6} {'Mult.':>6} {'Ms':>6} {'Irrep':>6} {'Energy':>20} {'Weight':>15}"
    )
    logger.log_info1("-" * width)
    iroot = 0
    for i in range(ncis):
        for j in range(nroots[i]):
            logger.log_info1(
                f"{iroot:>6d} {mult[i]:>6d} {ms[i]:>6.1f} {irrep[i]:>6d} {eigvals_per_solver[i][j]:>20.10f} {weights[i][j]:>15.5f}"
            )
            iroot += 1
        sep = "-" if i < ncis - 1 else "="
        logger.log_info1(sep * width)
