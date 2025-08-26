from dataclasses import dataclass, field
import numpy as np

from forte2 import (
    RelSlaterRules,
    SparseState,
    CIStrings,
    overlap,
    apply_op,
    hilbert_space,
    sparse_operator_hamiltonian,
)
from forte2.state import State
from forte2.base_classes import SystemMixin, MOsMixin
from forte2.scf.scf_utils import convert_coeff_spatial_to_spinor
from forte2.jkbuilder import SpinorbitalIntegrals
from forte2.helpers.davidsonliu import DavidsonLiuSolver
from forte2.helpers import logger


@dataclass
class RelCI(SystemMixin, MOsMixin):
    state: State
    active_spinorbitals: list[int]
    core_spinorbitals: list[int] = field(default_factory=list)
    nroot: int = 1
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

    def __call__(self, parent_method):
        self.parent_method = parent_method
        return self

    def run(self):
        if not self.parent_method.executed:
            self.parent_method.run()

        SystemMixin.copy_from_upstream(self, self.parent_method)
        MOsMixin.copy_from_upstream(self, self.parent_method)
        if not self.system.two_component:
            self.C = convert_coeff_spatial_to_spinor(self.system, self.C)
            self.system.two_component = True

        ints = SpinorbitalIntegrals(
            self.system,
            self.C[0],
            self.active_spinorbitals,
            self.core_spinorbitals,
            use_aux_corr=True,
        )

        self.nspinor = len(self.active_spinorbitals)
        self.ncore = len(self.core_spinorbitals)

        self.slater_rules = RelSlaterRules(
            nspinor=self.nspinor,
            scalar_energy=ints.E.real,
            one_electron_integrals=ints.H,
            two_electron_integrals=ints.V,
        )

        self.ci_strings = CIStrings(
            self.state.nel - self.ncore,
            0,
            self.state.symmetry,
            [[0] * self.nspinor],
            self.state.gas_min,
            self.state.gas_max,
        )
        self.ndet = self.ci_strings.nas
        if self.ndet == 0:
            raise ValueError(
                "No determinants could be generated for the given state and orbitals."
            )
        self.dets = self.ci_strings.make_determinants()

        if self.ci_algorithm == "exact":
            self._do_exact_diagonalization()
        elif self.ci_algorithm == "sparse":
            self._do_iterative_ci(ints)

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
