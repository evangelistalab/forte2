from dataclasses import dataclass, field
import numpy as np

from forte2 import (
    RelSlaterRules,
    SparseState,
    CIStrings,
    apply_op,
    sparse_operator_hamiltonian,
)
from forte2.state import State, MOSpace
from forte2.base_classes import ActiveSpaceSolver
from forte2.scf.scf_utils import convert_coeff_spatial_to_spinor
from forte2.jkbuilder import SpinorbitalIntegrals
from forte2.helpers.davidsonliu import DavidsonLiuSolver
from forte2.helpers import logger
from forte2.ci.ci_utils import pretty_print_gas_info


@dataclass
class _RelCIBase:
    mo_space: MOSpace
    state: State
    ints: SpinorbitalIntegrals
    nroot: int
    active_orbsym: list[int]
    do_test_rdms: bool = False
    log_level: int = field(default=logger.get_verbosity_level())

    ### Sigma builder parameters
    ci_algorithm: str = "sparse"

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

    def _ci_solver_startup(self):
        self.ci_strings = CIStrings(
            self.state.nel - self.ncore,
            0,
            self.state.symmetry,
            self.active_orbsym,
            self.state.gas_min,
            self.state.gas_max,
        )

        pretty_print_gas_info(self.ci_strings)

        logger.log(f"\nNumber of electrons: {self.ci_strings.na}", self.log_level)
        logger.log(f"Number of strings: {self.ci_strings.nas}", self.log_level)
        logger.log(f"Number of determinants: {self.ci_strings.ndet}", self.log_level)

        self.ndet = self.ci_strings.nas
        if self.ndet == 0:
            raise ValueError(
                "No determinants could be generated for the given state and orbitals."
            )

        self.dets = self.ci_strings.make_determinants()

    def run(self):
        if self.first_run:
            # don't set first_run to false before _build_guess_vectors
            self._ci_solver_startup()

        self.slater_rules = RelSlaterRules(
            nspinor=self.norb,
            scalar_energy=self.ints.E.real,
            one_electron_integrals=self.ints.H,
            two_electron_integrals=self.ints.V,
        )

        if self.ci_algorithm == "exact":
            self._do_exact_diagonalization()
        elif self.ci_algorithm == "sparse":
            self._do_iterative_ci(self.ints)

        self.E = self.evals
        for i, e in enumerate(self.evals):
            logger.log(f"Final CI Energy Root {i}: {e:20.12f} [Eh]", self.log_level)

        # if self.do_test_rdms:
        #     self._test_rdms()

        self.executed = True

        return self

    def _do_iterative_ci(self, ints):
        self.eigensolver = DavidsonLiuSolver(
            size=self.ndet,
            nroot=self.nroot,
            collapse_per_root=self.collapse_per_root,
            basis_per_root=self.basis_per_root,
            e_tol=self.econv,
            r_tol=self.rconv,
            maxiter=self.maxiter,
            eta=self.energy_shift,
            log_level=self.log_level,
            dtype=complex,
        )

        Hdiag = []
        for i in self.dets:
            Hdiag.append(self.slater_rules.energy(i))

        if self.ndet == 1:
            self.evals = np.array([Hdiag[0]])
            self.evecs = np.ones((1, 1))
            logger.log(
                f"Final CI Energy Root {0}: {self.evals[0]:20.12f} [Eh]", self.log_level
            )
            self.executed = True
            return self

        self.eigensolver.add_h_diag(Hdiag)

        if self.first_run:
            self._build_guess_vectors(Hdiag)
            self.first_run = False

        ham = sparse_operator_hamiltonian(ints.E.real, ints.H, ints.V, 1e-12)

        def sigma_builder(basis_block, sigma_block):
            nstate = basis_block.shape[1]
            for istate in range(nstate):
                psi = SparseState(
                    {d: c for d, c in zip(self.dets, basis_block[:, istate])}
                )
                Hpsi = apply_op(ham, psi)
                for idet in range(self.ndet):
                    sigma_block[idet, istate] = Hpsi[self.dets[idet]]

        self.eigensolver.add_sigma_builder(sigma_builder)

        self.evals, self.evecs = self.eigensolver.solve()

        logger.log("\nDavidson-Liu solver converged.\n", self.log_level)

    def _do_exact_diagonalization(self):
        H = np.zeros((self.ndet,) * 2, dtype=complex)
        for i in range(self.ndet):
            for j in range(i + 1):
                H[i, j] = self.slater_rules.slater_rules(self.dets[i], self.dets[j])
                H[j, i] = np.conj(H[i, j])

        self.evals, self.evecs = np.linalg.eigh(H)

    def _build_guess_vectors(self, Hdiag):
        """Build the guess vectors for the CI calculation."""

        # determine the number of guess vectors
        self.num_guess_states = min(self.guess_per_root * self.nroot, self.ndet)
        logger.log(f"Number of guess states: {self.num_guess_states}", self.log_level)
        nguess_dets = min(self.ndets_per_guess * self.num_guess_states, self.ndet)
        logger.log(f"Number of guess basis: {nguess_dets}", self.log_level)

        # find the indices of the elements of Hdiag with the lowest values
        if self.energy_shift is not None:
            indices = np.argsort(np.abs(Hdiag - self.energy_shift))[:nguess_dets]
        else:
            indices = np.argsort(Hdiag)[:nguess_dets]

        # create the Hamiltonian matrix in the basis of the guess determinants
        Hguess = np.zeros((nguess_dets, nguess_dets), dtype=complex)
        for i, I in enumerate(indices):
            for j, J in enumerate(indices):
                if i >= j:
                    Hij = self.slater_rules.slater_rules(self.dets[I], self.dets[J])
                    Hguess[i, j] = Hij
                    Hguess[j, i] = np.conj(Hij)

        # Diagonalize the Hamiltonian to get the initial guess vectors
        evals_guess, evecs_guess = np.linalg.eigh(Hguess)

        # Select the lowest eigenvalues and their corresponding eigenvectors
        guess_mat = np.zeros((self.ndet, self.num_guess_states), dtype=complex)
        for i in range(self.num_guess_states):
            guess = evecs_guess[:, i]
            for j, d in enumerate(indices):
                guess_mat[d, i] = guess[j]

        self.eigensolver.add_guesses(guess_mat)


@dataclass
class RelCI(ActiveSpaceSolver):
    """
    Relativistic Configuration Interaction
    """

    ci_algorithm: str = "sparse"

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

        self.slater_rules = RelSlaterRules(
            nspinor=self.norb,
            scalar_energy=ints.E.real,
            one_electron_integrals=ints.H,
            two_electron_integrals=ints.V,
        )

        self.sub_solvers = []
        active_orbsym = [
            [0 for _ in active_space] for active_space in self.mo_space.active_orbitals
        ]

        for i, state in enumerate(self.sa_info.states):
            # Create a CI solver for each state and MOSpace
            self.sub_solvers.append(
                _RelCIBase(
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
