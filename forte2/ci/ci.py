from dataclasses import dataclass, field

import numpy as np
import forte2

from forte2.state.state import State
from forte2.helpers.mixins import MOsMixin, SystemMixin
from forte2.helpers.davidsonliu import DavidsonLiuSolver

from forte2.jkbuilder import RestrictedMOIntegrals
from forte2.system.system import System


@dataclass
class CI(MOsMixin, SystemMixin):
    orbitals: list[int] | list[list[int]]
    state: State
    nroot: int
    core_orbitals: list[int] = field(default_factory=list)

    # The number of guess vectors for each root
    guess_per_root: int = 2
    # The number of determinants per guess vector
    ndets_per_guess: int = 10
    # The number of roots to collapse per root
    collapse_per_root: int = 2
    # The maximum number of basis vectors per root
    basis_per_root: int = 4
    # The number of iterations for the Davidson-Liu solver
    maxiter: int = 100
    # The energy convergence threshold
    econv: float = 1e-10
    # The residual convergence threshold
    rconv: float = 1e-5

    def __call__(self, method):
        if not method.executed:
            method.run()

        SystemMixin.copy_from_upstream(self, method)
        MOsMixin.copy_from_upstream(self, method)

        return self

    def __post_init__(self):
        # handle multiple orbitals formats
        if isinstance(self.orbitals, list) and all(
            isinstance(x, int) for x in self.orbitals
        ):
            self.orbitals = [self.orbitals]
        elif not all(isinstance(x, list) for x in self.orbitals):
            raise ValueError("Invalid orbitals format")

        self.norb = sum(len(x) for x in self.orbitals)
        self.solver = None

    def run(self):
        print(
            f"\nRunning CI with orbitals: {self.orbitals}, state: {self.state}, nroot: {self.nroot}"
        )
        # Generate the integrals with all the orbital spaces flattened
        flattened_orbitals = [orb for sublist in self.orbitals for orb in sublist]

        self.ints = RestrictedMOIntegrals(
            self.system, self.C[0], flattened_orbitals, self.core_orbitals
        )

        # 1. Create the string lists
        orbital_symmetry = [[0] * len(x) for x in self.orbitals]
        gas_min = []
        gas_max = []

        ci_strings = forte2.CIStrings(
            self.state.na - len(self.core_orbitals),
            self.state.nb - len(self.core_orbitals),
            self.state.symmetry,
            orbital_symmetry,
            gas_min,
            gas_max,
        )

        print(f"\nNumber of α electrons: {ci_strings.na}")
        print(f"Number of β electrons: {ci_strings.nb}")
        print(f"Number of α strings: {ci_strings.nas}")
        print(f"Number of β strings: {ci_strings.nbs}")
        print(f"Number of determinants: {ci_strings.ndet}")

        self.spin_adapter = forte2.CISpinAdapter(
            self.state.multiplicity - 1, self.state.twice_ms, self.norb
        )
        self.dets = ci_strings.make_determinants()
        self.spin_adapter.prepare_couplings(self.dets)
        print(f"Number of CSFs: {self.spin_adapter.ncsf()}")

        # Allocate temporary space for the CISigmaBuilder
        # (this must be done before creating the CISigmaBuilder)
        forte2.CISigmaBuilder.allocate_temp_space(ci_strings)

        # Create the CISigmaBuilder from the CI strings and integrals
        ci_sigma_builder = forte2.CISigmaBuilder(
            ci_strings, self.ints.E, self.ints.H, self.ints.V
        )

        # 2. Allocate memory for the CI vectors
        det_size = ci_strings.ndet
        basis_size = self.spin_adapter.ncsf()

        # Create the CI vectors that will hold the results of the sigma builder in the
        # determinant basis
        b_det = np.zeros((det_size))
        sigma_det = np.zeros((det_size))

        Hdiag = ci_sigma_builder.form_Hdiag_csf(self.dets, self.spin_adapter, True)

        # 3. Instantiate and configure solver
        self.first_run = False
        if self.solver is None:
            self.solver = DavidsonLiuSolver(
                size=basis_size,  # size of the basis (number of CSF if we spin adapt)
                nroot=self.nroot,
                collapse_per_root=self.collapse_per_root,
                basis_per_root=self.basis_per_root,
                e_tol=self.econv,  # eigenvalue convergence
                r_tol=self.rconv,  # residual convergence
                maxiter=self.maxiter,
            )
            self.first_run = True

        # 3. Build initial guess vectors
        self.solver.add_h_diag(Hdiag)

        self.num_guess_states = min(self.guess_per_root * self.nroot, basis_size)
        print(f"Number of guess states: {self.num_guess_states}")
        nguess_dets = min(self.ndets_per_guess * self.num_guess_states, basis_size)
        print(f"Number of guess basis: {nguess_dets}")

        # find the indices of the elements of Hdiag with the lowest values
        indices = np.argsort(Hdiag)[:nguess_dets]

        # create the Hamiltonian matrix in the basis of the guess CSFs
        Hguess = np.zeros((nguess_dets, nguess_dets))
        for i, I in enumerate(indices):
            for j, J in enumerate(indices):
                if i >= j:
                    Hij = ci_sigma_builder.slater_rules_csf(
                        self.dets, self.spin_adapter, I, J
                    )
                    Hguess[i, j] = Hij
                    Hguess[j, i] = Hij

        # Diagonalize the Hamiltonian to get the initial guess vectors
        evals_guess, evecs_guess = np.linalg.eigh(Hguess)

        # Select the lowest eigenvalues and their corresponding eigenvectors
        guess_mat = np.zeros((basis_size, self.num_guess_states))
        for i in range(self.num_guess_states):
            guess = evecs_guess[:, i]
            for j, d in enumerate(indices):
                guess_mat[d, i] = guess[j]

        self.solver.add_guesses(guess_mat)

        def sigma_builder(Bblock, Sblock):
            # Compute the sigma block from the basis block
            ncols = Bblock.shape[1]
            for i in range(ncols):
                self.spin_adapter.csf_C_to_det_C(Bblock[:, i], b_det)
                ci_sigma_builder.Hamiltonian(b_det, sigma_det)
                self.spin_adapter.det_C_to_csf_C(sigma_det, Sblock[:, i])

        self.solver.add_sigma_builder(sigma_builder)

        # 4. Run Davidson
        evals, evecs = self.solver.solve()
        self.E = evals
        print(f"Eigenvalues: {evals}")

        # 5. Compute the final CI vectors

        # 6. Compute the final energy and properties

        # Deallocate temporary space for the CI strings
        # (this must be done after running the CISigmaBuilder to avoid memory leaks)
        forte2.CISigmaBuilder.release_temp_space()

        print(
            f"Average CI Sigma Builder build time: {ci_sigma_builder.avg_build_time():.3f} s/build"
        )

    # def _build_initial_guess(self):
    #     print("Building initial guess vectors...")
    #     # determine the number of guess vectors
    #     self.num_guess_states = min(self.guess_per_root * self.nroot, self.C.size)

    # auto Hdiag_vec =
    #     spin_adapt_ ? form_Hdiag_csf(as_ints_, spin_adapter_) : form_Hdiag_det(as_ints_);
    # dl_solver_->add_h_diag(Hdiag_vec);

    # // The first time we run Form the diagonal of the Hamiltonian and the initial guess
    # if (spin_adapt_) {
    #     if (first_run) {
    #         auto guesses = initial_guess_csf(Hdiag_vec, num_guess_states);
    #         dl_solver_->add_guesses(guesses);
    #     }
    # } else {
    #     bool use_initial_guess = (num_guess_states * ndets_per_guess_ >= det_size);
    #     if (first_run or use_initial_guess) {
    #         dl_solver_->reset();
    #         auto [guesses, bad_roots] = initial_guess_det(Hdiag_vec, num_guess_states, as_ints_);
    #         dl_solver_->add_guesses(guesses);
    #         dl_solver_->add_project_out_vectors(bad_roots);
    #     }
    # }

    # auto converged = dl_solver_->solve();
    # if (not converged) {
    #     throw std::runtime_error(
    #         "Davidson-Liu solver did not converge.\nPlease try to increase the number of "
    #         "Davidson-Liu iterations (DL_MAXITER). You can also try to increase:\n - the "
    #         "maximum "
    #         "size of the subspace (DL_SUBSPACE_PER_ROOT)"
    #         "\n - the number of guess states (DL_GUESS_PER_ROOT)");
    #     return false;
    # }

    # // Copy eigenvalues and eigenvectors from the Davidson-Liu solver
    # evals_ = dl_solver_->eigenvalues();
    # energies_ = std::vector<double>(nroot_, 0.0);
    # spin2_ = std::vector<double>(nroot_, 0.0);
    # for (size_t r = 0; r < nroot_; r++) {
    #     energies_[r] = evals_->get(r);
    #     b_basis = dl_solver_->eigenvector(r);
    #     if (spin_adapt_) {
    #         spin_adapter_->csf_C_to_det_C(b_basis, b);
    #     } else {
    #         b = b_basis;
    #     }
    #     C_->copy(b);
    #     spin2_[r] = C_->compute_spin2();
    # }
    # eigen_vecs_ = dl_solver_->eigenvectors();

    # if (print_ >= PrintLevel::Default) {
    #     print_timing("CI", t.get());
    # }

    # // Print determinants
    # if (print_ >= PrintLevel::Default) {
    #     print_solutions(100, b, b_basis, dl_solver_);
    # }

    # // Optionally, test the RDMs
    # if (test_rdms_) {
    #     test_rdms(b, b_basis, dl_solver_);
    # }

    # energy_ = dl_solver_->eigenvalues()->get(root_);
    # psi::Process::environment.globals["CURRENT ENERGY"] = energy_;
    # psi::Process::environment.globals["CI ENERGY"] = energy_;

    # // GenCIVector::release_temp_space();
    # return energy_;
