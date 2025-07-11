from dataclasses import dataclass, field

import numpy as np
import forte2

from forte2.state.state import State
from forte2.helpers.mixins import MOsMixin, SystemMixin
from forte2.helpers.davidsonliu import DavidsonLiuSolver
from forte2.helpers import logger

from forte2.jkbuilder import RestrictedMOIntegrals
from forte2.system.system import System


@dataclass
class CI(MOsMixin, SystemMixin):
    orbitals: list[int] | list[list[int]]
    state: State
    nroot: int
    weights: list[float] = None
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
    # logging level, default is 2 (INFO)
    log_level: int = field(default=logger.get_verbosity_level(), init=False)
    # find roots around the energy shift
    energy_shift: float = None
    # whether to test the rdms
    do_test_rdms: bool = False

    ## Options that control the CI calculation
    ci_builder_memory: int = field(default=1024, init=False)  # in MB
    ci_algorithm: str = "hz"

    # Flag for whether the method has been executed
    executed: bool = field(default=False, init=False)

    def __call__(self, method):
        self.parent_method = method
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
        if self.weights is None:
            self.weights = np.ones(self.nroot) / self.nroot
        else:
            assert isinstance(self.weights, list), "Weights must be a list"
            self.weights = np.array(self.weights, dtype=float)
            assert (
                len(self.weights) == self.nroot
            ), "Weights must match the number of roots"
            assert np.all(self.weights >= 0), "Weights must be non-negative"
            assert np.isclose(self.weights.sum(), 1), "Weights must sum to 1"
            self.weights = np.array(self.weights, dtype=float)

    def _ci_solver_startup(self):
        """
        Initialize the CI solver with the necessary parameters and data structures.
        If the CI solver is used in an iterative context, this method is only called
        ocne at the beginning of the iteration."""
        if not self.parent_method.executed:
            self.parent_method.run()

        SystemMixin.copy_from_upstream(self, self.parent_method)
        MOsMixin.copy_from_upstream(self, self.parent_method)

        logger.log(
            f"\nRunning CI with orbitals: {self.orbitals}, state: {self.state}, nroot: {self.nroot}",
            self.log_level,
        )
        # Generate the integrals with all the orbital spaces flattened
        self.flattened_orbitals = [orb for sublist in self.orbitals for orb in sublist]

        # 1. Transform the orbitals to the MO basis
        self.ints = RestrictedMOIntegrals(
            self.system,
            self.C[0],
            self.flattened_orbitals,
            self.core_orbitals,
            use_aux_corr=True,
        )

        # 2. Create the string lists
        self.orbital_symmetry = [[0] * len(x) for x in self.orbitals]

        self.ci_strings = forte2.CIStrings(
            self.state.na - len(self.core_orbitals),
            self.state.nb - len(self.core_orbitals),
            self.state.symmetry,
            self.orbital_symmetry,
            self.gas_min,
            self.gas_max,
        )
        self.ci_strings.set_log_level(self.log_level)

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
        if self.solver is None:
            self.solver = DavidsonLiuSolver(
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
        self.solver.add_h_diag(Hdiag)

        # 5. Build the guess vectors if this is the first run
        if self.first_run:
            self._build_guess_vectors(Hdiag)
            self.first_run = False

        # TODO: remove this once the CIStrings are fully implemented

        # V0 = np.zeros((self.norb, self.norb, self.norb, self.norb))
        # slater_rules = forte2.SlaterRules(self.norb, self.ints.E, self.ints.H, V0)
        # Hex = np.zeros((self.ndet, self.ndet))
        # for i in range(self.ndet):
        #     for j in range(self.ndet):
        #         if i >= j:
        #             Hij = slater_rules.slater_rules(self.dets[i], self.dets[j])
        #             Hex[i, j] = Hij
        #             Hex[j, i] = Hij

        # e = np.linalg.eigvalsh(Hex)
        # print(f"\nLowest 5 eigenvalues of the Hamiltonian:")
        # for i in range(5):
        #     print(f"Eigenvalue {i}: {e[i]:20.12f} [Eh]")

        # one body only -152.695278969574
        #               -152.695278969574
        #               -152.695278969574

        # Hex = np.zeros((self.basis_size, self.basis_size))
        # for i in range(self.basis_size):
        #     for j in range(self.basis_size):
        #         Hij = self.ci_sigma_builder.slater_rules_csf(
        #             self.dets, self.spin_adapter, i, j
        #         )
        #         Hex[i, j] = Hij
        # # find the lowest 5 eigenvalues of the Hamiltonian
        # e, _ = np.linalg.eigh(Hex)

        def sigma_builder(Bblock, Sblock):
            # Compute the sigma block from the basis block
            ncols = Bblock.shape[1]
            for i in range(ncols):
                self.spin_adapter.csf_C_to_det_C(Bblock[:, i], self.b_det)
                self.ci_sigma_builder.Hamiltonian(self.b_det, self.sigma_det)
                # self.sigma_det = np.dot(Hex, self.b_det)
                self.spin_adapter.det_C_to_csf_C(self.sigma_det, Sblock[:, i])

        self.solver.add_sigma_builder(sigma_builder)

        # 6. Run Davidson
        self.evals, self.evecs = self.solver.solve()

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
            logger.log(f"CI energy from RDMs: {rdms_energy:.6f} Eh", self.log_level)
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
                f"CI energy from expanded RDMs: {rdms_energy:.6f} Eh", self.log_level
            )

            assert np.isclose(
                self.E[root], rdms_energy
            ), f"CI energy {self.E[root]} Eh does not match RDMs energy {rdms_energy} Eh"

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
                f"CI energy from spin-free RDMs: {rdms_energy:.6f} Eh", self.log_level
            )

            assert np.isclose(
                self.E[root], rdms_energy
            ), f"CI energy {self.E[root]} Eh does not match RDMs energy {rdms_energy} Eh"

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

        self.solver.add_guesses(guess_mat)

    def compute_average_energy(self):
        return np.dot(self.weights, self.E)

    def make_average_rdm1_sf(self):
        """
        Make the average spin-free one-particle RDM from the CI vectors.
        Returns:
            ndarray: Average spin-free one-particle RDM."""
        rdm1 = np.zeros((self.norb,) * 2)
        for i in range(self.nroot):
            rdm1 += self.make_rdm1_sf(self.evecs[:, i]) * self.weights[i]
        return rdm1

    def make_average_rdm2_sf(self):
        """
        Make the average spin-free two-particle RDM from the CI vectors.
        Returns:
            ndarray: Average spin-free two-particle RDM."""
        rdm2 = np.zeros((self.norb,) * 4)
        for i in range(self.nroot):
            rdm2 += self.make_rdm2_sf(self.evecs[:, i]) * self.weights[i]
        return rdm2

    def make_rdm1_sf(self, ci_vec):
        """
        Make the spin-free one-particle RDM from a CI vector.
        Args:
            ci_vec (ndarray): CI vector in the CSF basis.
        Returns:
            ndarray: Spin-free one-particle RDM."""
        ci_vec_det = np.zeros((self.ndet))
        self.spin_adapter.csf_C_to_det_C(ci_vec, ci_vec_det)
        return self.ci_sigma_builder.rdm1_sf(ci_vec_det, ci_vec_det)

    def make_tdm1_sf(self, ci_l, ci_r):
        """
        Make the spin-free one-particle transition density matrix from two CI vectors.
        Args:
            ci_l (ndarray): Left CI vector in the CSF basis.
            ci_r (ndarray): Right CI vector in the CSF basis.
        Returns:
            ndarray: Spin-free one-particle transition density matrix."""
        ci_l_det = np.zeros((self.ndet))
        ci_r_det = np.zeros((self.ndet))
        self.spin_adapter.csf_C_to_det_C(ci_l, ci_l_det)
        self.spin_adapter.csf_C_to_det_C(ci_r, ci_r_det)
        return self.ci_sigma_builder.rdm1_sf(ci_l_det, ci_r_det)

    def make_rdm2_sd(self, ci_vec, full=True):
        """
        Make the spin-dependent two-particle RDMs (aa, ab, bb) from a CI vector in the CSF basis.
        Args:
            ci_vec (ndarray): CI vector in the CSF basis.
            full (bool): If True, compute the full-dimension RDMs,
                otherwise compute compact aa and bb RDMs. Defaults to True.
        Returns:
            tuple: Spin-dependent two-particle RDMs (aa, ab, bb).
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
        Args:
            ci_l (ndarray): Left CI vector in the CSF basis.
            ci_r (ndarray): Right CI vector in the CSF basis.
            full (bool): If True, compute the full-dimension RDMs,
                otherwise compute compact aa and bb RDMs. Defaults to True.
        Returns:
            tuple: Spin-dependent two-particle transition density matrices (aa, ab, bb).
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
        Args:
            ci_vec (ndarray): CI vector in the CSF basis.
        Returns:
            ndarray: Spin-free two-particle RDM."""
        ci_vec_det = np.zeros((self.ndet))
        self.spin_adapter.csf_C_to_det_C(ci_vec, ci_vec_det)
        return self.ci_sigma_builder.rdm2_sf(ci_vec_det, ci_vec_det)

    def make_tdm2_sf(self, ci_l, ci_r):
        """
        Make the spin-free two-particle transition density matrix from two CI vectors in the CSF basis.
        Args:
            ci_l (ndarray): Left CI vector in the CSF basis.
            ci_r (ndarray): Right CI vector in the CSF basis.
        Returns:
            ndarray: Spin-free two-particle transition density matrix."""
        ci_l_det = np.zeros((self.ndet))
        ci_r_det = np.zeros((self.ndet))
        self.spin_adapter.csf_C_to_det_C(ci_l, ci_l_det)
        self.spin_adapter.csf_C_to_det_C(ci_r, ci_r_det)
        return self.ci_sigma_builder.rdm2_sf(ci_l_det, ci_r_det)


@dataclass
class MultiCI(MOsMixin, SystemMixin):
    CIs: list[CI]
    weights: list[float] = None

    executed: bool = field(default=False, init=False)

    def __call__(self, method):
        self.parent_method = method
        return self

    def __post_init__(self):
        assert all(
            isinstance(ci, CI) for ci in self.CIs
        ), "All elements of CIs must be CI instances"
        assert len(self.CIs) > 0, "CIs list cannot be empty"
        self.ncis = len(self.CIs)

        assert (
            len(set(ci.norb for ci in self.CIs)) == 1
        ), "All CIs must have the same number of active orbitals"

        self.norb = self.CIs[0].norb

        if self.weights is None:
            self.weights = np.ones(self.ncis) / self.ncis
        else:
            assert isinstance(self.weights, list)
            assert (
                len(self.weights) == self.ncis
            ), "Weights must match the number of CIs"
            self.weights = np.array(self.weights, dtype=float)
            assert np.all(self.weights >= 0), "Weights must be non-negative"
            assert np.isclose(self.weights.sum(), 1), "Weights must sum to 1"

    def _ci_solver_startup(self):
        if not self.parent_method.executed:
            self.parent_method.run()
        self.CIs = [ci(self.parent_method) for ci in self.CIs]
        SystemMixin.copy_from_upstream(self, self.parent_method)
        MOsMixin.copy_from_upstream(self, self.parent_method)

    def run(self):
        if not self.executed:
            self._ci_solver_startup()

        self.E = []
        self.E_avg = []
        for ci in self.CIs:
            # Run each CI and collect results
            ci.run()
            self.E.append(ci.E)
            self.E_avg.append(ci.compute_average_energy())

        self.E = np.array([_ for sublist in self.E for _ in sublist])
        self.executed = True
        return self

    def compute_average_energy(self):
        return np.dot(self.weights, self.E_avg)

    def make_average_rdm1_sf(self):
        rdm1 = np.zeros((self.norb,) * 2)
        for i, ci in enumerate(self.CIs):
            rdm1 += ci.make_average_rdm1_sf() * self.weights[i]
        return rdm1

    def make_average_rdm2_sf(self):
        rdm2 = np.zeros((self.norb,) * 4)
        for i, ci in enumerate(self.CIs):
            rdm2 += ci.make_average_rdm2_sf() * self.weights[i]
        return rdm2


class CASCI(CI):
    def __init__(self, ncasorb, ncaselec, charge=0, multiplicity=1, ms=0.0, nroot=1):
        """
        Initialize a CASCI object with the given CAS orbitals and number of electrons.

        Args:
            norb (int): Number of orbitals in the CAS.
            nelec (int): Number of electrons in the CAS.
            charge (int, optional): Charge of the system. Defaults to 0.
        """
        self.ncasorb = ncasorb
        self.ncaselec = ncaselec
        self.charge = charge
        self.multiplicity = multiplicity
        self.ms = ms
        self.nroot = nroot

    def __call__(self, method):
        nel = method.system.Zsum - self.charge
        nelec_core = nel - self.ncaselec
        assert nelec_core % 2 == 0, "Number of core electrons must be even."
        ncore = nelec_core // 2
        core_orbitals = list(range(ncore))
        actv_orbitals = list(range(ncore, ncore + self.ncasorb))
        super().__init__(
            orbitals=actv_orbitals,
            core_orbitals=core_orbitals,
            state=State(nel=nel, multiplicity=self.multiplicity, ms=self.ms),
            nroot=self.nroot,
        )
        self = super().__call__(method)
        return self


class CISD(CI):
    def __init__(self, charge=0, multiplicity=1, ms=0.0, nroot=1, frozen_core=0):
        """
        Initialize a CISD object with the given charge, multiplicity, and number of roots.
        Args:
            charge (int, optional): Charge of the system. Defaults to 0.
            multiplicity (int, optional): Multiplicity of the system. Defaults to 1.
            ms (float, optional): Spin quantum number. Defaults to 0.0.
            nroot (int, optional): Number of roots to compute. Defaults to 1.
        """
        self.charge = charge
        self.multiplicity = multiplicity
        self.ms = ms
        self.nroot = nroot
        self.frozen_core = frozen_core

    def __call__(self, method):
        nel = method.system.Zsum - self.charge
        # TODO: Lift this restriction (for ROHF, etc.)
        assert nel % 2 == 0, "Number of electrons must be even."
        orbitals = [
            list(range(self.frozen_core, nel // 2)),
            list(range(nel // 2, method.system.nbf)),
        ]
        core_orbitals = list(range(self.frozen_core))
        nel_corr = nel - 2 * self.frozen_core
        super().__init__(
            orbitals=orbitals,
            core_orbitals=core_orbitals,
            state=State(nel=nel, multiplicity=self.multiplicity, ms=self.ms),
            nroot=self.nroot,
            gas_min=[nel_corr - 2, 0],
            gas_max=[nel_corr, 2],
        )
        self = super().__call__(method)
        return self
