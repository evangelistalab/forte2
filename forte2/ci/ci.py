from dataclasses import dataclass, field

import forte2

from forte2.state.state import State
from forte2.helpers.mixins import MOsMixin, SystemMixin

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
    # The number of subspace vectors per root
    max_subspace_per_root: int = 4

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
            self.state.na,
            self.state.nb,
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

        # TODO: optionally create the spin adapter
        # local_timer t;
        # startup();

        # 2. Allocate memory for the CI vectors
        forte2.CIVector.allocate_temp_space(ci_strings)

        C = forte2.CIVector(ci_strings)
        T = forte2.CIVector(ci_strings)

        det_size = ci_strings.ndet
        # basis_size = spin_adapt_ ? spin_adapter_->ncsf() : det_size; TODO:spin_adapter_
        basis_size = det_size

        # // Create the vectors that stores the b and sigma vectors in the determinant basis
        # auto b = std::make_shared<psi::Vector>("b", det_size);
        # auto sigma = std::make_shared<psi::Vector>("sigma", det_size);

        # // Optionally create the vectors that stores the b and sigma vectors in the CSF basis
        # auto b_basis = b;
        # auto sigma_basis = sigma;

        # if (spin_adapt_) {
        #     b_basis = std::make_shared<psi::Vector>("b", basis_size);
        #     sigma_basis = std::make_shared<psi::Vector>("sigma", basis_size);
        # }

        # 3. Instantiate and configure solver
        self.solver = DavidsonLiuSolver(
            size=basis_size,
            nroot=self.nroot,
            collapse_per_root=self.collapse_per_root,
            subspace_per_root=self.max_subspace_per_root,
        )

        # Hdiag = form Hamiltonian_diagonal()

        # solver.add_h_diag(np.diag(H))
        # solver.add_sigma_builder(sigma_builder)

        # 4. Run Davidson
        # evals, evecs = solver.solve()

        # // if not allocate, create the DL solver
        # bool first_run = false;
        # if (dl_solver_ == nullptr) {
        #     dl_solver_ = std::make_shared<DavidsonLiuSolver>(basis_size, nroot_, collapse_per_root_,
        #                                                      subspace_per_root_);
        #     dl_solver_->set_e_convergence(e_convergence_);
        #     dl_solver_->set_r_convergence(r_convergence_);
        #     dl_solver_->set_print_level(print_);
        #     dl_solver_->set_maxiter(maxiter_davidson_);
        #     first_run = true;
        # }

        # 3. Build initial guess vectors
        # self._build_initial_guess()

        # 4. Run the iterative solver

        # 5. Compute the final CI vectors

        # 6. Compute the final energy and properties

        forte2.CIVector.release_temp_space()

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

    # // Print the initial guess
    # auto sigma_builder = [this, &b_basis, &b, &sigma, &sigma_basis](std::span<double> b_span,
    #                                                                 std::span<double> sigma_span) {
    #     // copy the b vector
    #     size_t basis_size = b_span.size();
    #     for (size_t I = 0; I < basis_size; ++I) {
    #         b_basis->set(I, b_span[I]);
    #     }
    #     if (spin_adapt_) {
    #         // Compute sigma in the CSF basis and convert it to the determinant basis
    #         spin_adapter_->csf_C_to_det_C(b_basis, b);
    #         C_->copy(b);
    #         C_->Hamiltonian(*T_, as_ints_);
    #         T_->copy_to(sigma);
    #         spin_adapter_->det_C_to_csf_C(sigma, sigma_basis);
    #     } else {
    #         // Compute sigma in the determinant basis
    #         C_->copy(b_basis);
    #         C_->Hamiltonian(*T_, as_ints_);
    #         T_->copy_to(sigma_basis);
    #     }
    #     for (size_t I = 0; I < basis_size; ++I) {
    #         sigma_span[I] = sigma_basis->get(I);
    #     }
    # };

    # // Run the Davidson-Liu solver
    # dl_solver_->add_sigma_builder(sigma_builder);

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
