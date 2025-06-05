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
    # The minimum number of orbitals in each general orbital space
    gas_min: list[int] = field(default_factory=list)
    # The maximum number of orbitals in each general orbital space
    gas_max: list[int] = field(default_factory=list)
    # First run flag
    first_run: bool = field(default=True, init=False)
    # The number of determinants
    ndef: int = field(default=None, init=False)

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

        # 1. Transform the orbitals to the MO basis
        self.ints = RestrictedMOIntegrals(
            self.system, self.C[0], flattened_orbitals, self.core_orbitals
        )

        # 2. Create the string lists
        orbital_symmetry = [[0] * len(x) for x in self.orbitals]

        ci_strings = forte2.CIStrings(
            self.state.na - len(self.core_orbitals),
            self.state.nb - len(self.core_orbitals),
            self.state.symmetry,
            orbital_symmetry,
            self.gas_min,
            self.gas_max,
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

        # Create the CISigmaBuilder from the CI strings and integrals
        # This object handles some temporary memory deallocated at destruction
        # and is used to compute the Hamiltonian matrix elements in the determinant basis
        self.ci_sigma_builder = forte2.CISigmaBuilder(
            ci_strings, self.ints.E, self.ints.H, self.ints.V
        )

        # 2. Allocate memory for the CI vectors
        self.ndet = ci_strings.ndet
        self.basis_size = self.spin_adapter.ncsf()

        # Create the CI vectors that will hold the results of the sigma builder in the
        # determinant basis
        b_det = np.zeros((self.ndet))
        sigma_det = np.zeros((self.ndet))

        Hdiag = self.ci_sigma_builder.form_Hdiag_csf(
            self.dets, self.spin_adapter, spin_adapt_full_preconditioner=True
        )

        # 3. Instantiate and configure solver
        if self.solver is None:
            self.solver = DavidsonLiuSolver(
                size=self.basis_size,  # size of the basis (number of CSF if we spin adapt)
                nroot=self.nroot,
                collapse_per_root=self.collapse_per_root,
                basis_per_root=self.basis_per_root,
                e_tol=self.econv,  # eigenvalue convergence
                r_tol=self.rconv,  # residual convergence
                maxiter=self.maxiter,
            )

        # 4. Compute diagonal of the Hamiltonian
        self.solver.add_h_diag(Hdiag)

        # 5. Build the guess vectors if this is the first run
        if self.first_run:
            self._build_guess_vectors(Hdiag)
            self.first_run = False

        def sigma_builder(Bblock, Sblock):
            # Compute the sigma block from the basis block
            ncols = Bblock.shape[1]
            for i in range(ncols):
                self.spin_adapter.csf_C_to_det_C(Bblock[:, i], b_det)
                self.ci_sigma_builder.Hamiltonian(b_det, sigma_det)
                self.spin_adapter.det_C_to_csf_C(sigma_det, Sblock[:, i])

        self.solver.add_sigma_builder(sigma_builder)

        # 6. Run Davidson
        self.evals, self.evecs = self.solver.solve()

        print(f"\nDavidson-Liu solver converged.\n")

        # 7. Store the final energy and properties
        self.E = self.evals
        for i, e in enumerate(self.evals):
            print(f"Final CI Energy Root {i}: {e:20.12f} [Eh]")

        h_tot, h_aabb, h_aaaa, h_bbbb = self.ci_sigma_builder.avg_build_time()
        print("\nAverage CI Sigma Builder time summary:")
        print(f"h_aabb time:    {h_aabb:.3f} s/build")
        print(f"h_aaaa time:    {h_aaaa:.3f} s/build")
        print(f"h_bbbb time:    {h_bbbb:.3f} s/build")
        print(f"total time:     {h_tot:.3f} s/build")

        # 8. Compute the RDMs from the CI vectors
        # and verify the energy from the RDMs matches the CI energy
        print("\nComputing RDMs from CI vectors.\n")
        rdms = {}
        for root in range(self.nroot):
            root_rdms = {}
            root_det = np.zeros((self.ndet))
            self.spin_adapter.csf_C_to_det_C(self.evecs[:, root], root_det)
            root_rdms["rdm1"] = self.ci_sigma_builder.rdm1_sf(root_det, root_det)
            root_rdms["rdm2_aa"] = self.ci_sigma_builder.rdm2_aa(
                root_det, root_det, True
            )
            root_rdms["rdm2_ab"] = self.ci_sigma_builder.rdm2_ab(root_det, root_det)
            root_rdms["rdm2_bb"] = self.ci_sigma_builder.rdm2_aa(
                root_det, root_det, False
            )

            rdms[root] = root_rdms
            # tr_rdm1 = np.einsum("ii", self.rdm1)
            # print(f"Trace of RDM1: {tr_rdm1:.6f}")

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
            print(f"CI energy from RDMs: {rdms_energy:.6f} Eh")
            assert np.isclose(
                self.E[root], rdms_energy
            ), f"CI energy {self.E[root]} Eh does not match RDMs energy {rdms_energy} Eh"

        self.rdms = rdms

        return self.rdms

    def _build_guess_vectors(self, Hdiag):
        """
        Build the guess vectors for the CI calculation.
        This method is a placeholder and should be implemented in subclasses.
        """

        # determine the number of guess vectors
        self.num_guess_states = min(self.guess_per_root * self.nroot, self.basis_size)
        print(f"Number of guess states: {self.num_guess_states}")
        nguess_dets = min(self.ndets_per_guess * self.num_guess_states, self.basis_size)
        print(f"Number of guess basis: {nguess_dets}")

        # find the indices of the elements of Hdiag with the lowest values
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

        self.solver.add_guesses(guess_mat)
