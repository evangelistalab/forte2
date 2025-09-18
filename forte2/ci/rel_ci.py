from dataclasses import dataclass, field
from collections import OrderedDict
import numpy as np

from forte2 import (
    RelCISigmaBuilder,
    SparseState,
    CIStrings,
    apply_op,
    sparse_operator_hamiltonian,
)
from forte2.state import State, MOSpace
from forte2.base_classes import RelActiveSpaceSolver
from forte2.scf.scf_utils import convert_coeff_spatial_to_spinor
from forte2.jkbuilder import SpinorbitalIntegrals
from forte2.orbitals import Semicanonicalizer
from forte2.helpers.davidsonliu import DavidsonLiuSolver
from forte2.helpers import logger
from forte2.helpers.comparisons import approx
from forte2.props import get_1e_property
from .ci_utils import (
    pretty_print_gas_info,
    pretty_print_ci_summary,
    pretty_print_ci_nat_occ_numbers,
    pretty_print_ci_dets,
    pretty_print_ci_transition_props,
)


@dataclass
class _RelCIBase:
    mo_space: MOSpace
    state: State
    ints: SpinorbitalIntegrals
    nroot: int
    active_orbsym: list[int]
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

        if self.ci_algorithm == "hz":
            self.sigma_det = np.zeros((self.ndet,), dtype=complex)
            self.b_det = np.zeros((self.ndet,), dtype=complex)

    def run(self):
        if self.first_run:
            self._ci_solver_startup()

        self.ci_sigma_builder = RelCISigmaBuilder(
            self.ci_strings, self.ints.E.real, self.ints.H, self.ints.V, self.log_level
        )

        if self.ci_algorithm == "exact":
            self._do_exact_diagonalization()
        elif self.ci_algorithm == "sparse":
            self._do_sparse_ci()
        elif self.ci_algorithm == "hz":
            self._do_hz_ci()

        self.E = self.evals
        for i, e in enumerate(self.evals):
            logger.log(f"Final CI Energy Root {i}: {e:20.12f} [Eh]", self.log_level)

        if self.do_test_rdms:
            self._test_rdms()

        self.executed = True
        self.first_run = False

        return self

    def _do_hz_ci(self):
        self.ci_sigma_builder.set_memory(self.ci_builder_memory)
        self.ci_sigma_builder.set_algorithm("hz")
        Hdiag = self.ci_sigma_builder.form_Hdiag(self.dets)

        if self.ndet == 1:
            self.evals = np.array([Hdiag[0]])
            self.evecs = np.ones((1, 1))
            logger.log(
                f"Final CI Energy Root {0}: {self.evals[0]:20.12f} [Eh]", self.log_level
            )
            self.executed = True
            return self

        if self.eigensolver is None:
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
        self.eigensolver.add_h_diag(Hdiag)

        if self.first_run:
            self._build_guess_vectors(Hdiag)
            self.first_run = False

        def sigma_builder(Bblock, Sblock):
            # Compute the sigma block from the basis block
            ncols = Bblock.shape[1]
            for i in range(ncols):
                self.b_det = Bblock[:, i].copy()
                self.ci_sigma_builder.Hamiltonian(self.b_det, self.sigma_det)
                Sblock[:, i] = self.sigma_det.copy()

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

    def _do_sparse_ci(self):
        if self.eigensolver is None:
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

        Hdiag = self.ci_sigma_builder.form_Hdiag(self.dets)

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

        ham = sparse_operator_hamiltonian(
            self.ints.E.real, self.ints.H, self.ints.V, 1e-12
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

        self.eigensolver.add_sigma_builder(sigma_builder)

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

    def _do_exact_diagonalization(self):
        H = np.zeros((self.ndet,) * 2, dtype=complex)
        for i in range(self.ndet):
            for j in range(i + 1):
                H[i, j] = self.ci_sigma_builder.slater_rules(self.dets, i, j)
                H[j, i] = np.conj(H[i, j])

        self.evals, self.evecs = np.linalg.eigh(H)
        self.evals = self.evals[: self.nroot]
        self.evecs = self.evecs[:, : self.nroot]

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
                    Hij = self.ci_sigma_builder.slater_rules(self.dets, I, J)
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

    def _test_rdms(self):
        # Compute the RDMs from the CI vectors
        # and verify the energy from the RDMs matches the CI energy
        logger.log("\nComputing RDMs from CI vectors.\n", self.log_level)
        for root in range(self.nroot):
            rdm1 = self.make_1rdm(root)
            rdm2 = self.make_2rdm(root)

            rdms_energy = self.ints.E
            rdms_energy += np.einsum("ij,ij", rdm1, self.ints.H)
            rdms_energy += 0.5 * np.einsum("ijkl,ijkl", rdm2, self.ints.V)
            logger.log(f"CI energy from RDMs: {rdms_energy:.12f} Eh", self.log_level)

            assert self.E[root] == approx(rdms_energy)

            logger.log(
                f"RDMs for root {root} validated successfully.\n", self.log_level
            )

    def compute_natural_occupation_numbers(self):
        """
        Compute the natural occupation numbers from the 1-RDMs.

        Returns
        -------
        (norb, nroot) NDArray
            The natural occupation numbers for each root.
        """
        if not self.executed:
            raise RuntimeError("CI solver has not been executed yet.")
        no = np.zeros((self.norb, self.nroot))
        for i in range(self.nroot):
            g1 = self.make_1rdm(i)
            no[:, i] = np.linalg.eigvalsh(g1)[::-1]

        return no

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
        for i in range(self.nroot):
            top_dets = []
            ci_det = self.evecs[:, i]
            argsort = np.argsort(np.abs(ci_det))[::-1]  # descending in absolute coeff
            for j in range(n):
                if j < len(argsort):
                    top_dets.append((self.dets[argsort[j]], ci_det[argsort[j]]))
            top_dets_per_root.append(top_dets)

        return top_dets_per_root

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

    def make_1rdm(self, left_root: int, right_root: int = None):
        if right_root is None:
            right_root = left_root
        # copy to ensure contiguous arrays are passed to the sigma builder
        rdm = self.ci_sigma_builder.so_1rdm(
            self.evecs[:, left_root].copy(),
            self.evecs[:, right_root].copy(),
        )

        return rdm

    def make_1rdm_debug(self, left_root: int, right_root: int = None):
        if right_root is None:
            right_root = left_root
        # copy to ensure contiguous arrays are passed to the sigma builder
        rdm = self.ci_sigma_builder.so_1rdm_debug(
            self.evecs[:, left_root].copy(),
            self.evecs[:, right_root].copy(),
        )

        return rdm

    def make_2rdm_debug(self, left_root: int, right_root: int = None):
        if right_root is None:
            right_root = left_root
        # copy to ensure contiguous arrays are passed to the sigma builder
        rdm = self.ci_sigma_builder.so_2rdm_debug(
            self.evecs[:, left_root].copy(),
            self.evecs[:, right_root].copy(),
        )

        return rdm

    def make_2cumulant(self, left_root: int, right_root: int = None):
        if right_root is None:
            right_root = left_root
        lambda2 = self.ci_sigma_builder.so_2cumulant(
            self.evecs[:, left_root].copy(),
            self.evecs[:, right_root].copy(),
        )
        return lambda2

    def make_2rdm(self, left_root: int, right_root: int = None):
        if right_root is None:
            right_root = left_root
        # copy to ensure contiguous arrays are passed to the sigma builder
        rdm = self.ci_sigma_builder.so_2rdm(
            self.evecs[:, left_root].copy(),
            self.evecs[:, right_root].copy(),
        )

        return rdm

    def make_2cumulant_debug(self, left_root: int, right_root: int = None):
        if right_root is None:
            right_root = left_root
        rdm1 = self.make_1rdm(left_root, right_root)
        rdm2 = self.make_2rdm(left_root, right_root)
        lambda2 = (
            rdm2
            - np.einsum("pr,qs->pqrs", rdm1, rdm1, optimize=True)
            + np.einsum("ps,qr->pqrs", rdm1, rdm1, optimize=True)
        )
        return lambda2

    def make_3rdm_debug(self, left_root: int, right_root: int = None):
        if right_root is None:
            right_root = left_root
        # copy to ensure contiguous arrays are passed to the sigma builder
        rdm = self.ci_sigma_builder.so_3rdm_debug(
            self.evecs[:, left_root].copy(),
            self.evecs[:, right_root].copy(),
        )

        return rdm

    def make_3rdm(self, left_root: int, right_root: int = None):
        if right_root is None:
            right_root = left_root
        # copy to ensure contiguous arrays are passed to the sigma builder
        rdm = self.ci_sigma_builder.so_3rdm(
            self.evecs[:, left_root].copy(),
            self.evecs[:, right_root].copy(),
        )

        return rdm

    def make_3cumulant(self, left_root: int, right_root: int = None):
        if right_root is None:
            right_root = left_root
        lambda3 = self.ci_sigma_builder.so_3cumulant(
            self.evecs[:, left_root].copy(),
            self.evecs[:, right_root].copy(),
        )
        return lambda3


@dataclass
class RelCISolver(RelActiveSpaceSolver):
    """
    Relativistic Configuration Interaction
    """

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
        rdm_core = np.einsum("pi,qi->pq", Ccore, Ccore.conj(), optimize=True)
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
        Make the average one-particle RDM from the CI vectors.

        Returns
        -------
        NDArray
            Average one-particle RDM.
        """
        rdm1 = np.zeros((self.norb,) * 2, dtype=np.complex128)
        for i, ci_solver in enumerate(self.sub_solvers):
            for j in range(ci_solver.nroot):
                rdm1 += ci_solver.make_1rdm(j) * self.weights[i][j]
        return rdm1

    def make_average_2rdm(self):
        """
        Make the average two-particle RDM from the CI vectors.

        Returns
        -------
        NDArray
            Average two-particle RDM.
        """
        rdm2 = np.zeros((self.norb,) * 4, dtype=np.complex128)
        for i, ci_solver in enumerate(self.sub_solvers):
            for j in range(ci_solver.nroot):
                rdm2 += ci_solver.make_2rdm(j) * self.weights[i][j]

        return rdm2


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
                g1=self.make_average_1rdm(),
                C=self.C[0],
                system=self.system,
            )
            self.C[0] = semi.C_semican.copy()

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
