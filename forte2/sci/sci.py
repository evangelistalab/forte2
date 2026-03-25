import time
from dataclasses import dataclass, field
from collections import OrderedDict
from itertools import combinations

import numpy as np

from forte2 import (
    cpp_helpers,
    apply_op,
    sparse_operator_hamiltonian,
    spin2,
    Determinant,
    Configuration,
    SparseState,
    SlaterRules,
    SelectedCIHelper,
)
from forte2.helpers.table import AsciiTable
from forte2.state import State, MOSpace
from forte2.helpers.comparisons import approx
from forte2.helpers.davidsonliu import DavidsonLiuSolver
from forte2.base_classes.active_space_solver import (
    ActiveSpaceSolver,
    RelActiveSpaceSolver,
)
from forte2.base_classes.params import SelectedCIParams, DavidsonLiuParams
from forte2.helpers import logger
from forte2.jkbuilder import RestrictedMOIntegrals, SpinorbitalIntegrals
from forte2.props import get_1e_property
from forte2.orbitals import Semicanonicalizer
from forte2.scf.scf_utils import convert_coeff_spatial_to_spinor
from forte2.ci.ci_utils import (
    pretty_print_gas_info,
    pretty_print_ci_summary,
    pretty_print_ci_nat_occ_numbers,
    pretty_print_ci_dets,
    pretty_print_ci_transition_props,
)

@dataclass
class _SelectedCIBase:
    """
    A general selected configuration interaction (CI) solver class for a single `State`.
    Although possible, is not recommended to instantiate this class directly.
    Consider using the `SelectedCI` class instead.

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
    ci_algorithm : str, optional, default="sparse"
        The algorithm used for the CI sigma builder.
        The options are:
            - "sparse": A sparse string-based algorithm
            - "exact": Exact diagonalization
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
        The eigenvalues (energielapsed:.3fes) of the CI problem.
    evecs : NDArray
        The eigenvectors (CI coefficients) of the CI problem.

    """

    ### Init attributes
    mo_space: MOSpace = field(default=None)
    state: State = field(default=None)
    ints: RestrictedMOIntegrals = field(default=None)
    nroot: int = field(default=1)

    active_orbsym: list[int] = field(default_factory=list)
    two_component: bool = False
    do_test_rdms: bool = False
    log_level: int = field(default=logger.get_verbosity_level())
    die_if_not_converged: bool = False
    slater_rules: SlaterRules = field(default=None, init=False)

    sci_params: SelectedCIParams = field(default_factory=SelectedCIParams)
    davidson_liu_params: DavidsonLiuParams = field(default_factory=DavidsonLiuParams)

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
            assert self.sci_params.ci_algorithm.lower() in [
                "sparse",
                "exact",
            ], f"Two-component CI algorithm must be 'sparse' or 'exact'. Got '{self.sci_params.ci_algorithm}'."
        else:
            assert self.sci_params.ci_algorithm.lower() in [
                "sparse",
                "exact",
            ], f"CI algorithm must be 'sparse' or 'exact'. Got '{self.sci_params.ci_algorithm}'."

    def _sci_solver_startup(self):
        # Create the Slater rules object
        self.slater_rules = SlaterRules(
            self.norb, self.ints.E, self.ints.H, self.ints.V
        )

        # Create an initial guess
        (
            self.guess_determinants,
            self.guess_c,
            self.guess_energies,
            self.project_out,
        ) = self._initial_guess(
            self.sci_params.guess_occ_window, self.sci_params.guess_vir_window
        )
        self.evecs = self.guess_c.copy()

        self.ndet = len(self.guess_determinants)
        logger.log(f"Number of determinants: {self.ndet}", self.log_level)

    def run(self):
        if not self.executed:
            self._sci_solver_startup()

        # Create the SelectedCIHelper to manage the selected CI procedure
        self.sci_helper = SelectedCIHelper(
            self.norb,
            self.guess_determinants,
            self.guess_c,
            self.ints.E,
            self.ints.H,
            self.ints.V,
            self.log_level,
        )

        self.sci_helper.set_c(self.guess_c)
        self.sci_helper.set_energies(self.guess_energies)
        self.sci_helper.set_num_threads(self.sci_params.num_threads)
        self.sci_helper.set_num_batches_per_thread(
            self.sci_params.num_batches_per_thread
        )
        self.sci_helper.set_screening_criterion(self.sci_params.screening_criterion)
        self.sci_helper.set_energy_correction(self.sci_params.energy_correction)
        if self.sci_params.frozen_creation:
            self.sci_helper.set_frozen_creation(self.sci_params.frozen_creation)

        print()
        old_energy = 0.0
        for cycle in range(self.sci_params.maxcycle):
            print(f"{'=' * 67}")
            print(f"Selected CI Cycle {cycle + 1}")
            print(f"{'=' * 67}")

            print(f"Algorithm: {self.sci_params.selection_algorithm}")
            print(f"  var_threshold = {self.sci_params.var_threshold}")
            print(f"  pt2_threshold = {self.sci_params.pt2_threshold}")

            if self.sci_params.selection_algorithm.lower() == "hbci_ref":
                self.sci_helper.select_hbci_ref(
                    var_threshold=self.sci_params.var_threshold,
                    pt2_threshold=self.sci_params.pt2_threshold,
                )
            elif self.sci_params.selection_algorithm.lower() == "hbci":
                self.sci_helper.select_hbci(
                    var_threshold=self.sci_params.var_threshold,
                    pt2_threshold=self.sci_params.pt2_threshold,
                )
            else:
                raise ValueError(
                    f"Unknown selection algorithm: {self.sci_params.selection_algorithm}"
                )

            e_var = self.sci_helper.energies()
            ept2_var = self.sci_helper.ept2_var()
            ept2_pt = self.sci_helper.ept2_pt()
            spin2_var = self.sci_helper.compute_spin2()

            summary = "\nSummary of selection:"
            summary += (
                f"\n  Variational added:     {self.sci_helper.num_new_dets_var()}"
            )
            summary += (
                f"\n  Perturbative included: {self.sci_helper.num_new_dets_pt2()}"
            )
            summary += f"\n  Total determinants:    {self.sci_helper.ndets()}"
            summary += (
                f"\n  Selection time:        {self.sci_helper.selection_time():.3f} s\n"
            )
            logger.log(summary, self.log_level)

            table = AsciiTable(
                columns=[
                    "Root",
                    "E (var) [Eh]",
                    "S^2 (var)",
                    "E (var') [Eh]",
                    "E (var'+PT2) [Eh]",
                ],
                formats=["{:>4}", "{:>20.12f}", "{:>6.3f}", "{:>20.12f}", "{:>20.12f}"],
            )

            logger.log(table.header(), self.log_level)
            for r in range(self.nroot):
                logger.log(
                    table.row(
                        r,
                        e_var[r],
                        spin2_var[r],
                        e_var[r] + ept2_var[r],
                        e_var[r] + ept2_var[r] + ept2_pt[r],
                    )
                    # f"{r:>4} {e_var[r]:20.12f} {e_var[r] + ept2_var[r]:20.12f} {e_var[r] + ept2_var[r] + ept2_pt[r]:20.12f}",
                    ,
                    self.log_level,
                )
            # logger.log("=" * 67, self.log_level)
            logger.log(table.footer(), self.log_level)

            self.ndet = self.sci_helper.ndets()
            self.dets = self.sci_helper.dets()
            # Keep determinant guesses aligned with the current CI basis for subsequent reruns.
            self.guess_determinants = list(self.dets)

            self.b_det = np.zeros((self.ndet))
            self.sigma_det = np.zeros((self.ndet))

            # Save the current CI vectors as the guess for the next iteration
            # Here we assume that new determinants are added to the end of the list
            # so we can just take the first part of the CI vectors as the guess for the next iteration
            num_guess = min(self.evecs.shape[1], self.nroot)
            self.guess_c = np.zeros((self.ndet, num_guess), dtype=self.dtype)
            self.guess_c[0 : self.evecs.shape[0], 0:num_guess] = self.evecs[
                :, 0:num_guess
            ]

            if self.sci_params.ci_algorithm.lower() == "exact":
                self._do_exact_diagonalization()
            elif self.sci_params.ci_algorithm.lower() == "sparse":
                self._do_iterative_ci()
            else:
                raise ValueError(
                    f"Unknown CI algorithm: {self.sci_params.ci_algorithm}. Must be 'exact' or 'sparse'."
                )

            # logger.log(f"CI Energy Roots: {self.evals}", self.log_level)

            self.sci_helper.set_c(self.evecs)
            self.sci_helper.set_energies(self.evals)

            delta_energy = np.average(self.evals) - old_energy
            old_energy = np.average(self.evals)

            if abs(delta_energy) < self.davidson_liu_params.e_tol:
                logger.log(
                    f"Selected CI converged in {cycle + 1} cycles.", self.log_level
                )
                break

        self.executed = True

        return self

    def _initial_guess(self, window_occ=0, window_vir=0):
        # If there are no guess determinants, generate some based on occupation windows
        if len(self.sci_params.guess_dets) == 0:
            self.sci_params.guess_dets = self._generate_initial_guess_dets(window_occ, window_vir)
        else:
            self._check_guess_dets(self.sci_params.guess_dets)

        # Check that we have all spin complement pairs
        self.sci_params.guess_dets = self._generate_spin_complement_pairs(self.sci_params.guess_dets)

        print("Initial guess determinants (by energy):")
        for d in self.sci_params.guess_dets:
            print(f"  {d.str(self.norb)}: {self.slater_rules.energy(d):20.12f}")

        ndet = len(self.sci_params.guess_dets)
        S2guess = np.zeros((ndet, ndet), dtype=self.dtype)
        Hguess = np.zeros((ndet, ndet), dtype=self.dtype)
        for i in range(ndet):
            for j in range(i + 1):
                Hguess[i, j] = self.slater_rules.slater_rules(
                    self.sci_params.guess_dets[i], self.sci_params.guess_dets[j]
                )
                Hguess[j, i] = np.conj(Hguess[i, j])
                S2guess[i, j] = spin2(self.sci_params.guess_dets[i], self.sci_params.guess_dets[j])
                S2guess[j, i] = np.conj(S2guess[i, j])

        svals, svecs = np.linalg.eigh(S2guess)
        print(f"S^2 values of the guess determinants: {svals}")
        # find the multiplicity of the eigenvalue closest to S(S+1)
        S = (self.state.multiplicity - 1) / 2
        target_s2 = S * (S + 1)
        close_idx = [
            i for i, v in enumerate(svals) if np.isclose(v, target_s2, atol=1e-4)
        ]
        # find the indices of the eigenvalues that are not close to target_s2
        not_close_idx = [i for i in range(len(svals)) if i not in close_idx]

        print(
            f"Target S(S+1) = {target_s2}, found {len(close_idx)} eigenvalues close to it."
        )

        # project the guess determinants into the S^2 subspace
        S2sub = svecs[:, close_idx]
        S2project_out = (
            [svecs[:, i].copy() for i in not_close_idx]
            if self.sci_params.do_spin_penalty
            else []
        )

        Hguess = (
            S2sub.conj().T @ Hguess @ S2sub
            if self.sci_params.do_spin_penalty
            else Hguess
        )
        # Diagonalize the Hamiltonian to get the initial guess coefficients
        evals, evecs = np.linalg.eigh(Hguess)
        c = (
            S2sub @ evecs[:, : self.nroot].copy()
            if self.sci_params.do_spin_penalty
            else evecs[:, : self.nroot].copy()
        )
        energies = evals[: self.nroot].copy()
        print(f"Initial guess energies: {energies}")
        print(f"Initial guess states:")
        for r in range(c.shape[1]):
            print(f"  Root {r}:")
            for i in range(c.shape[0]):
                if abs(c[i, r]) > 1e-4:
                    print(f"    {self.sci_params.guess_dets[i].str(self.norb)}: {c[i, r]:20.12f}")
        return self.sci_params.guess_dets, c, energies, S2project_out

    def _generate_initial_guess_dets(self, window_occ, window_vir):
        logger.log("Generating initial determinant guess")
        # create the initial guess determinant
        d0 = Determinant.zero()
        if self.two_component:
            na = self.state.nel - self.ncore
            nb = 0
        else:
            na = self.state.na - self.ncore
            nb = self.state.nb - self.ncore
        for i in range(na):
            d0.set_na(i, True)
        for i in range(nb):
            d0.set_nb(i, True)

        n_guess_dets = max(8, 2 * self.nroot + 4)

        # define a window around HOMO and LUMO to generate excitations
        occ_a = range(max(0, na - window_occ), na)
        occ_b = range(max(0, nb - window_occ), nb)
        vir_a = range(na, min(self.norb, na + window_vir))
        vir_b = range(nb, min(self.norb, nb + window_vir))

        det_energy = {d0: self.slater_rules.energy(d0)}

        # Alpha excitations
        for i in occ_a:
            for a in vir_a:
                d1 = Determinant(d0)
                d1.set_na(i, False)
                d1.set_na(a, True)
                det_energy[d1] = self.slater_rules.energy(d1)

        # Beta excitations
        for i in occ_b:
            for a in vir_b:
                d1 = Determinant(d0)
                d1.set_nb(i, False)
                d1.set_nb(a, True)
                det_energy[d1] = self.slater_rules.energy(d1)

        # Pair excitations
        n_pairs = min(na, nb)
        n_occ = max(na, nb)
        occ_pair = range(max(0, n_pairs - window_occ), n_pairs)
        vir_pair = range(n_occ, min(self.norb, n_occ + window_vir))
        for i in occ_pair:
            for a in vir_pair:
                d1 = Determinant(d0)
                d1.set_na(i, False)
                d1.set_nb(i, False)
                d1.set_na(a, True)
                d1.set_nb(a, True)
                det_energy[d1] = self.slater_rules.energy(d1)

        # Sort the determinants by energy
        sorted_dets = sorted(det_energy.items(), key=lambda x: x[1])

        # Form the Hamiltonian matrix in this basis
        guess_dets = [d for d, e in sorted_dets[: min(n_guess_dets, len(sorted_dets))]]

        return guess_dets

    def _generate_spin_complement_pairs(self, guess_dets):
        # find all the unique electronic configurations
        configurations = {Configuration(d) for d in guess_dets}
        spin_complete_guess_dets = []
        for conf in configurations:
            docc = conf.get_docc_vec()
            socc = conf.get_socc_vec()
            # generate all combinations of spin complements that satisfy the same ms constraint
            nopen = len(socc)
            na = (nopen + self.state.twice_ms) // 2
            nb = nopen - na

            for alpha_indices in combinations(range(nopen), na):
                beta_indices = set(range(nopen)) - set(alpha_indices)
                dcomp = Determinant.zero()
                for i in docc:
                    dcomp.set_na(i, True)
                    dcomp.set_nb(i, True)
                for ia in alpha_indices:
                    orb = socc[ia]
                    dcomp.set_na(orb, True)
                for ib in beta_indices:
                    orb = socc[ib]
                    dcomp.set_nb(orb, True)
                spin_complete_guess_dets.append(dcomp)
        return spin_complete_guess_dets

    def _check_guess_dets(self, guess_dets):
        for d in guess_dets:
            na = d.count_a()
            nb = d.count_b()
            if na + self.ncore != self.state.na:
                raise ValueError(
                    f"Guess determinant {d.str(self.norb)} has {na} alpha electrons, expected {self.state.na - self.ncore}."
                )
            if nb + self.ncore != self.state.nb:
                raise ValueError(
                    f"Guess determinant {d.str(self.norb)} has {nb} beta electrons, expected {self.state.nb - self.ncore}."
                )

    def _do_iterative_ci(self):
        """
        Solve CI with an iterative Davidson-Liu solver, using either
        Harrison-Zarrabian or Knowles-Handy sigma builder algorithm.
        """
        if self.two_component:
            raise NotImplementedError(
                "Two-component Selected CI is not yet implemented."
            )

        Hdiag = self.sci_helper.Hdiag()

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
        # if self.eigensolver is None:
        self.eigensolver = DavidsonLiuSolver(
            size=self.ndet,  # size of the basis (number of CSF if we spin adapt)
            nroot=self.nroot,
            davidson_liu_params=self.davidson_liu_params,
            energy_shift=self.sci_params.energy_shift,
            log_level=self.log_level,
            dtype=complex if self.two_component else float,
        )

        # 4. Compute diagonal of the Hamiltonian
        self.eigensolver.add_h_diag(Hdiag)

        # # 5. Build the guess vectors
        self.eigensolver.add_guesses(self.guess_c)

        # Project out any states as needed
        if len(self.project_out) > 0:
            project_out = []
            for vec in self.project_out:
                resized_vec = vec.copy()
                resized_vec.resize(self.ndet)
                project_out.append(resized_vec)
            self.eigensolver.add_project_out(project_out)

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
                    self.b_det = Bblock[:, i].copy()
                    self.sci_helper.Hamiltonian(self.b_det, self.sigma_det)
                    Sblock[:, i] = self.sigma_det.copy()

        self.eigensolver.add_sigma_builder(sigma_builder)

        # 6. Run Davidson
        start = time.monotonic()
        self.evals, self.evecs = self.eigensolver.solve()
        end = time.monotonic()
        elapsed = end - start

        if self.eigensolver.converged:
            logger.log(
                f"\nDavidson-Liu solver converged in {elapsed:.3f} seconds.\n",
                self.log_level,
            )
        else:
            if self.die_if_not_converged:
                raise RuntimeError("Davidson-Liu solver did not converge.")
            else:
                logger.log(
                    f"\nDavidson-Liu solver did not converge in {self.eigensolver.maxiter} iterations.\n",
                    self.log_level,
                )

        # if not self.two_component:
        #     h_tot, h_aabb, h_aaaa, h_bbbb = self.ci_sigma_builder.avg_build_time()
        #     logger.log("\nAverage CI Sigma Builder time summary:", self.log_level)
        #     logger.log(f"h_aabb time:    {h_aabb:.3f} s/build", self.log_level)
        #     logger.log(f"h_aaaa time:    {h_aaaa:.3f} s/build", self.log_level)
        #     logger.log(f"h_bbbb time:    {h_bbbb:.3f} s/build", self.log_level)
        #     logger.log(f"total time:     {h_tot:.3f} s/build\n", self.log_level)

    def _do_exact_diagonalization(self):
        logger.log("Using CI algorithm: Exact Diagonalization", self.log_level)

        dets = self.sci_helper.dets()

        H = np.zeros((self.ndet,) * 2, dtype=self.dtype)
        for i in range(self.ndet):
            for j in range(i + 1):
                H[i, j] = self.slater_rules.slater_rules(dets[i], dets[j])
                H[j, i] = np.conj(H[i, j])

        self.evals_full, self.evecs_full = np.linalg.eigh(H)
        if self.sci_params.energy_shift is not None:
            argsort = np.argsort(np.abs(self.evals_full - self.sci_params.energy_shift))
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
        if right_root is None:
            right_root = left_root
        a = self.sci_helper.a_1rdm(left_root, right_root)
        b = self.sci_helper.b_1rdm(left_root, right_root)
        return a, b

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
        if right_root is None:
            right_root = left_root
        return self.sci_helper.sf_1rdm(left_root, right_root)

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
        if self.two_component:
            raise NotImplementedError(
                "Natural occupation numbers are only implemented for non-relativistic SelectedCI."
            )
        no = np.zeros((self.norb, self.nroot))
        for i in range(self.nroot):
            g1 = self.make_sf_1rdm(i)
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
            ci_det = self.evecs[:, i]
            argsort = np.argsort(np.abs(ci_det))[::-1]  # descending in absolute coeff
            for j in range(n):
                if j < len(argsort):
                    top_dets.append((self.dets[argsort[j]], ci_det[argsort[j]]))
            top_dets_per_root.append(top_dets)

        return top_dets_per_root


@dataclass
class SelectedCISolver(ActiveSpaceSolver):
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

    do_test_rdms: bool = False
    log_level: int = field(default=logger.get_verbosity_level())
    sci_params: SelectedCIParams = field(default_factory=SelectedCIParams)
    davidson_liu_params: DavidsonLiuParams = field(default_factory=DavidsonLiuParams)

    ### Non-init attributes
    ci_builder_memory: int = field(default=1024, init=False)  # in MB
    first_run: bool = field(default=True, init=False)
    executed: bool = field(default=False, init=False)

    def __call__(self, method):
        self.parent_method = method
        return self

    def _collect_child_kwargs(self, target_cls):
        """Collect keyword arguments for child solvers."""
        # Defer import to avoid polluting top-level namespace
        from dataclasses import fields as _dc_fields

        # Take all init fields of the target dataclass and copy values from `self` if present
        names = {f.name for f in _dc_fields(target_cls) if f.init}
        return {n: getattr(self, n) for n in names if hasattr(self, n)}

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

            kwargs = self._collect_child_kwargs(_SelectedCIBase)
            kwargs.update(
                {
                    "mo_space": self.mo_space,
                    "ints": ints,
                    "state": state,
                    "nroot": self.sa_info.nroots[i],
                    "active_orbsym": active_orbsym,
                }
            )
            self.sub_solvers.append(_SelectedCIBase(**kwargs))

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

    def make_average_sf_1rdm(self):
        """
        Make the average spin-free one-particle RDM from the CI vectors.

        Returns
        -------
        NDArray
            Average spin-free one-particle RDM.
        """
        rdm1 = np.zeros((self.norb,) * 2)
        for i, ci_solver in enumerate(self.sub_solvers):
            for j in range(ci_solver.nroot):
                rdm1 += ci_solver.make_sf_1rdm(j) * self.weights[i][j]
        return rdm1

    def make_average_sf_2rdm(self):
        """
        Make the average spin-free two-particle RDM from the CI vectors.

        Returns
        -------
        NDArray
            Average spin-free two-particle RDM.
        """
        raise NotImplementedError(
            "Average spin-free 2-RDM is not implemented for SelectedCI."
        )

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
        # factor of 2 for spin-summed 1-RDM
        rdm_core = 2 * np.einsum("pi,qi->pq", Ccore, Ccore.conj(), optimize=True)
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
                rdm = ci_solver.make_sf_1rdm(i)
                rdm = np.einsum("ij,pi,qj->pq", rdm, Cact, Cact.conj(), optimize=True)
                dip = get_1e_property(
                    self.system, rdm, property_name="electric_dipole", unit="au"
                )
                tdmdict[(i, i)] = dip + core_dip
                foscdict[(i, i)] = 0.0  # No oscillator strength for i->i transitions
                for j in range(i + 1, ci_solver.nroot):
                    tdm = ci_solver.make_sf_1rdm(i, j)
                    tdm = np.einsum(
                        "ij,pi,qj->pq", tdm, Cact, Cact.conj(), optimize=True
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
class SelectedCI(SelectedCISolver):
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
                system=self.system,
                mo_space=self.mo_space,
            )
            semi.semi_canonicalize(g1=self.make_average_sf_1rdm(), C_contig=self.C[0])
            self.C[0] = semi.C_semican.copy()

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
        # self.compute_natural_occupation_numbers()
        # pretty_print_ci_nat_occ_numbers(self.sa_info, self.mo_space, self.nat_occs)
        top_dets = self.get_top_determinants()
        pretty_print_ci_dets(self.sa_info, self.mo_space, top_dets)

        # if self.do_transition_dipole:
        #     self.compute_transition_properties()
        #     pretty_print_ci_transition_props(
        #         self.sa_info,
        #         self.tdm_per_solver,
        #         self.fosc_per_solver,
        #         self.evals_per_solver,
        #     )
