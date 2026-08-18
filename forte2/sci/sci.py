import time
from dataclasses import dataclass, field
from collections import OrderedDict
from itertools import combinations
from typing import ClassVar

import numpy as np

from forte2.lib import sparse_ops, det, cpp_helpers, ci_helpers, rdms
from forte2.lib.sparse_ops import SparseState
from forte2.lib.det import Determinant, Configuration, SlaterRules
from forte2.lib.ci_helpers import SelectedCIHelper, CIStrings
from forte2.helpers.table import AsciiTable
from forte2.state import State, RelState, MOSpace
from forte2.helpers.comparisons import approx
from forte2.helpers.davidsonliu import DavidsonLiuSolver
from forte2.base_classes import CIBase, RelCIBase
from forte2.base_classes.params import SelectedCIParams, DavidsonLiuParams
from forte2.helpers import logger
from forte2.jkbuilder import RestrictedMOIntegrals, SpinorbitalIntegrals
from forte2.props import get_1e_property
from forte2.orbitals import Semicanonicalizer
from forte2.ci.ci_utils import (
    pretty_print_ci_summary,
    pretty_print_ci_dets,
    pretty_print_ci_transition_props,
    pretty_print_ci_nat_occ_numbers,
)


def _selected_ci_runtime_params(sci_params, davidson_liu_params, nstates):
    """Return independent per-state runtime parameters without mutating inputs."""

    def expand(params, name):
        if isinstance(params, list):
            if len(params) != nstates:
                raise ValueError(
                    f"Number of {name} provided ({len(params)}) does not match "
                    f"the number of states ({nstates})."
                )
            return params
        if nstates > 1:
            logger.log_warning(
                f"Multiple states specified but only one set of {name} provided. "
                "Using the same parameters for all states."
            )
        return [params] * nstates

    runtime_sci = [
        params.copy(
            guess_dets=list(params.guess_dets),
            pinned_guess_dets=list(params.pinned_guess_dets),
            frozen_creation=list(params.frozen_creation),
            frozen_annihilation=list(params.frozen_annihilation),
        )
        for params in expand(sci_params, "SelectedCIParams")
    ]
    runtime_davidson = [
        params.copy() for params in expand(davidson_liu_params, "DavidsonLiuParams")
    ]
    return runtime_sci, runtime_davidson


@dataclass
class _SelectedCISingleStateSolver:
    """
    A general selected configuration interaction (CI) solver class for a single `State`.
    Although possible, is not recommended to instantiate this class directly.
    Consider using the `SelectedCI` class instead.

    Parameters
    ----------
    sci_params : SelectedCIParams, optional
        Parameters for the selected CI solver.
    davidson_liu_params : DavidsonLiuParams, optional
        Parameters for the Davidson-Liu eigensolver.
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

    Attributes
    ----------
    eigensolver : DavidsonLiuSolver
        The eigensolver used to find the roots of the CI problem.
    E (evals) : NDArray
        The eigenvalues (energies) of the CI problem.
    evecs : NDArray
        The eigenvectors (CI coefficients) of the CI problem.

    """

    ### Init attributes
    sci_params: SelectedCIParams = field(default_factory=SelectedCIParams)
    davidson_liu_params: DavidsonLiuParams = field(default_factory=DavidsonLiuParams)
    mo_space: MOSpace = field(default=None)
    state: State = field(default=None)
    ints: RestrictedMOIntegrals = field(default=None)
    nroot: int = field(default=1)
    active_orbsym: list[int] = field(default_factory=list)
    do_test_rdms: bool = False
    log_level: int = field(default=logger.get_verbosity_level() + 1)
    die_if_not_converged: bool = False

    ### Non-init attributes
    slater_rules: SlaterRules = field(default=None, init=False)
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

        self.dtype = float

        assert self.sci_params.ci_algorithm.lower() in [
            "sparse",
            "exact",
        ], f"CI algorithm must be 'sparse' or 'exact'. Got '{self.sci_params.ci_algorithm}'."

    def _make_slater_rules(self):
        """Build the Slater rules object used for guesses, diagonals, and exact diag."""
        return SlaterRules(self.norb, self.ints.E, self.ints.H, self.ints.V)

    def _make_sci_helper(self):
        """Build the C++ selected CI helper that owns selection and the sigma build."""
        return SelectedCIHelper(
            self.norb,
            self.guess_determinants,
            self.guess_c,
            self.ints.E,
            self.ints.H,
            self.ints.V,
            self.log_level,
            self.sci_params.screening_criterion,
            self.sci_params.frozen_creation,
            self.sci_params.frozen_annihilation,
        )

    def _compute_spin2(self):
        """Return <S^2> for each root. Overridden in the two-component solver, where spin is
        not a good quantum number and the helper does not expose compute_spin2."""
        return self.sci_helper.compute_spin2()

    def _sci_solver_startup(self):
        # Create the Slater rules object
        self.slater_rules = self._make_slater_rules()

        # Create an initial guess
        (
            self.guess_determinants,
            self.guess_c,
            self.guess_energies,
            self.project_out,
        ) = self._initial_guess()
        self.evecs = self.guess_c.copy()

        self.ndet = len(self.guess_determinants)
        logger.log(f"Number of determinants: {self.ndet}", self.log_level)

    def run(self):
        if not self.executed:
            self._sci_solver_startup()

        # Create the selected CI helper to manage the selected CI procedure
        self.sci_helper = self._make_sci_helper()

        self.sci_helper.set_energies(self.guess_energies)
        self.sci_helper.set_num_threads(self.sci_params.num_threads)
        self.sci_helper.set_num_batches_per_thread(
            self.sci_params.num_batches_per_thread
        )
        self.sci_helper.set_energy_correction(self.sci_params.energy_correction)
        self.sci_helper.set_pt2_regularizer(
            self.sci_params.pt2_regularizer.lower(),
            self.sci_params.pt2_regularizer_strength,
        )

        old_energy = 0.0
        for cycle in range(self.sci_params.maxcycle):
            logger.log(f"\n{'=' * 67}", self.log_level)
            logger.log(f"Selected CI Cycle {cycle + 1}", self.log_level)
            logger.log(f"{'=' * 67}", self.log_level)

            logger.log(
                f"Algorithm: {self.sci_params.selection_algorithm}", self.log_level
            )
            logger.log(
                f"  var_threshold = {self.sci_params.var_threshold}", self.log_level
            )
            logger.log(
                f"  pt2_threshold = {self.sci_params.pt2_threshold}", self.log_level
            )

            old_ndets = self.sci_helper.ndets()

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

            # These are the CI energies of each root
            self.e_var = np.array(self.sci_helper.energies())
            # These are the expectation values of S^2 for each root computed from the CI vectors
            self.spin2_var = np.array(self._compute_spin2())
            # These are the PT2 corrections due to the new variational determinants added in this cycle
            self.ept2_var = np.array(self.sci_helper.ept2_var())
            # These are the PT2 corrections due to the new perturbative determinants added in this cycle
            self.ept2_pt = np.array(self.sci_helper.ept2_pt())
            # These are the total energies of each root including the new variational and perturbative contributions
            self.e_tot = self.e_var + self.ept2_var + self.ept2_pt

            summary = "\nSummary of selection:"
            summary += f"\n  {'Initial # of variational determinants:':<40}{old_ndets}"
            summary += (
                f"\n  {'Variational added:':<40}{self.sci_helper.num_new_dets_var()}"
            )
            summary += (
                f"\n  {'Total variational determinants:':<40}{self.sci_helper.ndets()}"
            )
            summary += f"\n  {'Perturbatively included:':<40}{self.sci_helper.num_new_dets_pt2()}"
            summary += (
                f"\n  {'Selection time:':<40}{self.sci_helper.selection_time():.3f} s\n"
            )
            logger.log(summary, self.log_level)

            table = AsciiTable(
                columns=[
                    "Root",
                    "E (CI) [Eh]",
                    "S^2 (CI)",
                    "E (CI) + E (PT2) [Eh]",
                ],
                formats=["{:>4}", "{:>20.12f}", "{:>6.3f}", "{:>20.12f}"],
            )

            logger.log(table.header(), self.log_level)
            for r in range(self.nroot):
                logger.log(
                    table.row(
                        r,
                        self.e_var[r],
                        self.spin2_var[r],
                        self.e_tot[r],
                    ),
                    self.log_level,
                )
            logger.log(table.footer(), self.log_level)

            self.ndet = self.sci_helper.ndets()
            self.dets = self.sci_helper.dets()
            # Keep determinant guesses aligned with the current CI basis for subsequent reruns.
            self.guess_determinants = list(self.dets)

            self.b_det = np.zeros((self.ndet), dtype=self.dtype)
            self.sigma_det = np.zeros((self.ndet), dtype=self.dtype)

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

            self.sci_helper.set_c(self.evecs)
            self.sci_helper.set_energies(np.ascontiguousarray(self.evals.real))

            delta_energy = np.average(self.evals) - old_energy
            old_energy = np.average(self.evals)

            if abs(delta_energy) < self.sci_params.e_tol:
                logger.log(
                    f"Selected CI converged in {cycle + 1} cycles.", self.log_level
                )
                break
        else:
            logger.log(
                f"Selected CI did not converge in {self.sci_params.maxcycle} cycles.",
                self.log_level,
            )

        # final selection to update var' and pt2 contributions with the final CI coefficients
        logger.log(f"\n{'=' * 67}", self.log_level)
        logger.log(f"Final Selected CI Cycle", self.log_level)
        logger.log(f"{'=' * 67}", self.log_level)

        logger.log(f"Algorithm: {self.sci_params.selection_algorithm}", self.log_level)
        logger.log(f"  var_threshold = {self.sci_params.var_threshold}", self.log_level)
        logger.log(f"  pt2_threshold = {self.sci_params.pt2_threshold}", self.log_level)

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

        self.e_var = np.array(self.sci_helper.energies())
        self.ept2_var = np.array(self.sci_helper.ept2_var())
        self.ept2_pt = np.array(self.sci_helper.ept2_pt())
        self.spin2_var = np.array(self._compute_spin2())
        self.e_tot = self.e_var + self.ept2_var + self.ept2_pt

        summary = "\nSummary of selection:"
        summary += f"\n  Variational added:     {self.sci_helper.num_new_dets_var()}"
        summary += f"\n  Perturbative included: {self.sci_helper.num_new_dets_pt2()}"
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
                    self.e_var[r],
                    self.spin2_var[r],
                    self.e_var[r] + self.ept2_var[r],
                    self.e_tot[r],
                ),
                self.log_level,
            )
        logger.log(table.footer(), self.log_level)

        if self.do_test_rdms:
            self._test_rdms()

        self.executed = True

        return self

    def _initial_guess(self):
        window_occ = self.sci_params.guess_occ_window
        window_vir = self.sci_params.guess_vir_window
        # If there are no guess determinants, generate some based on occupation windows
        if (
            len(self.sci_params.guess_dets) + len(self.sci_params.pinned_guess_dets)
            == 0
        ):
            self.sci_params.guess_dets = self._generate_initial_guess_dets(
                window_occ, window_vir
            )
        else:
            self._check_guess_dets(self.sci_params.guess_dets)
            self._check_guess_dets(self.sci_params.pinned_guess_dets)

        # use the determinantal energies to refine the guess determinants
        # if there are more than needed for the initial guess
        # this can be controlled by DavidsonLiuParams
        if len(self.sci_params.guess_dets) > 0:
            guess_hdiag = self.slater_rules.energies(self.sci_params.guess_dets)
            nguess_dets = len(self.sci_params.guess_dets)
            num_guess_states = min(
                self.davidson_liu_params.guess_per_root * self.nroot, nguess_dets
            )
            nguess_dets = min(
                self.davidson_liu_params.ndets_per_guess * num_guess_states,
                nguess_dets,
            )
        else:
            # no guess dets and only pinned guess dets
            guess_hdiag = np.empty(0)
            nguess_dets = 0

        # find the indices of the elements of Hdiag with the lowest values
        # subject to an optional energy shift, which can be used to target specific states (e.g. excited states)
        if self.sci_params.energy_shift is not None:
            indices = np.argsort(np.abs(guess_hdiag - self.sci_params.energy_shift))[
                :nguess_dets
            ]
        else:
            indices = np.argsort(guess_hdiag)[:nguess_dets]

        self.sci_params.guess_dets = [self.sci_params.guess_dets[i] for i in indices]
        self.sci_params.guess_dets += self.sci_params.pinned_guess_dets

        # Check that we have all spin complement pairs
        self.sci_params.guess_dets = self._generate_spin_complement_pairs(
            self.sci_params.guess_dets
        )
        logger.log(
            f"Number of guess determinants: {len(self.sci_params.guess_dets)}",
            self.log_level,
        )

        ndet = len(self.sci_params.guess_dets)
        S2guess = np.zeros((ndet, ndet), dtype=self.dtype)
        Hguess = np.zeros((ndet, ndet), dtype=self.dtype)
        for i in range(ndet):
            for j in range(i + 1):
                Hguess[i, j] = self.slater_rules.slater_rules(
                    self.sci_params.guess_dets[i], self.sci_params.guess_dets[j]
                )
                Hguess[j, i] = np.conj(Hguess[i, j])
                S2guess[i, j] = det.spin2(
                    self.sci_params.guess_dets[i], self.sci_params.guess_dets[j]
                )
                S2guess[j, i] = np.conj(S2guess[i, j])

        svals, svecs = np.linalg.eigh(S2guess)
        logger.log(f"S^2 values of the guess determinants: {svals}", self.log_level)
        # find the multiplicity of the eigenvalue closest to S(S+1)
        S = (self.state.multiplicity - 1) / 2
        target_s2 = S * (S + 1)
        close_idx = [
            i for i, v in enumerate(svals) if np.isclose(v, target_s2, atol=1e-8)
        ]
        # find the indices of the eigenvalues that are not close to target_s2
        not_close_idx = [i for i in range(len(svals)) if i not in close_idx]

        logger.log(
            f"Target S(S+1) = {target_s2}, found {len(close_idx)} eigenvalues close to it.",
            self.log_level,
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
        logger.log(f"Initial guess energies: {energies}", self.log_level)
        # log the following at a more verbose log level
        logger.log(f"Initial guess states:", self.log_level + 1)
        for r in range(c.shape[1]):
            logger.log(f"  Root {r}:", self.log_level + 1)
            for i in range(c.shape[0]):
                if abs(c[i, r]) > 1e-4:
                    logger.log(
                        f"    {self.sci_params.guess_dets[i].str(self.norb)}: {c[i, r]:20.12f}",
                        self.log_level + 1,
                    )
        return self.sci_params.guess_dets, c, energies, S2project_out

    def _generate_initial_guess_dets(self, window_occ, window_vir):
        logger.log("Generating initial determinant guess", self.log_level)

        na_active = self.state.na - self.ncore
        nb_active = self.state.nb - self.ncore
        nel_active = na_active + nb_active

        if window_occ < 0:
            raise ValueError(
                f"guess_occ_window must be non-negative, got {window_occ}."
            )
        if window_vir < 0:
            raise ValueError(
                f"guess_vir_window must be non-negative, got {window_vir}."
            )

        if window_occ + window_vir == 0:
            logger.log_warning(
                "No guess determinants provided and guess occupation windows set to 0. "
                "Using the Hartree-Fock determinant as the only guess."
                "This is not recommended if spin penalty is used."
            )
            # use the Hartree-Fock determinant as the guess
            d0 = Determinant.zero()
            for i in range(na_active):
                d0.set_na(i, True)
            for i in range(nb_active):
                d0.set_nb(i, True)
            return [d0]

        nocc = nel_active // 2 - window_occ
        if nocc < 0:
            raise ValueError(
                f"guess_occ_window={window_occ} is larger than the number of active "
                f"occupied orbital pairs ({nel_active // 2}). Reduce guess_occ_window "
                "to generate valid guess determinants."
            )
        noccel = 2 * nocc
        nactv = window_occ + window_vir

        if nocc + nactv > self.norb:
            raise ValueError(
                f"Not enough orbitals to generate guess determinants with the specified occupation windows.\n"
                f"Number of occupied orbitals needed: {nocc + nactv}, number of active orbitals available: {self.norb}.\n"
                f"Reduce guess_occ_window and/or guess_vir_window to generate valid guess determinants."
            )

        if noccel == 0:
            ci_strings = CIStrings(
                na_active,
                nb_active,
                0,
                [[0] * nactv],
                [],
                [],
            )
        else:
            ci_strings = CIStrings(
                na_active,
                nb_active,
                0,
                [[0] * nocc, [0] * nactv],
                [noccel],
                [noccel],
            )
        return ci_strings.make_determinants()

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
            na = d.count_alpha()
            nb = d.count_beta()
            if na + self.ncore != self.state.na:
                raise ValueError(
                    f"Guess determinant {d.str(self.norb)} has {na} alpha electrons, expected {self.state.na - self.ncore}."
                )
            if nb + self.ncore != self.state.nb:
                raise ValueError(
                    f"Guess determinant {d.str(self.norb)} has {nb} beta electrons, expected {self.state.nb - self.ncore}."
                )

    def _compute_hdiag(self):
        """Diagonal of the Hamiltonian over the current selected determinant list."""
        return self.sci_helper.Hdiag()

    def _make_sigma_builder(self):
        """Return a sigma-build closure for the Davidson-Liu solver.

        The non-relativistic sigma build is delegated to the real C++ `SelectedCIHelper`.
        """

        def sigma_builder(Bblock, Sblock):
            # Compute the sigma block from the basis block
            ncols = Bblock.shape[1]
            for i in range(ncols):
                self.b_det = Bblock[:, i].copy()
                self.sci_helper.Hamiltonian(self.b_det, self.sigma_det)
                Sblock[:, i] = self.sigma_det.copy()

        return sigma_builder

    def _do_iterative_ci(self):
        """
        Solve CI with an iterative Davidson-Liu solver.

        The diagonal and sigma build are supplied by `_compute_hdiag` /
        `_make_sigma_builder`, which the relativistic subclass overrides to use the
        complex sparse machinery instead of the real C++ `SelectedCIHelper`.
        """
        Hdiag = self._compute_hdiag()

        # If there is only one determinant, we can skip calling the eigensolver
        if self.ndet == 1:
            self.evals = np.array([Hdiag[0]]).real
            self.evecs = np.ones((1, 1), dtype=self.dtype)
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
            dtype=self.dtype,
        )

        # 4. Compute diagonal of the Hamiltonian
        self.eigensolver.add_h_diag(Hdiag)

        # # 5. Build the guess vectors
        self.eigensolver.add_guesses(self.guess_c)

        # Project out any states as needed
        if len(self.project_out) > 0:
            self.eigensolver.add_project_out(self.project_out)

        self.eigensolver.add_sigma_builder(self._make_sigma_builder())

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

    def _selected_dets(self):
        """The current variational determinant list (from the C++ helper)."""
        return self.sci_helper.dets()

    def _do_exact_diagonalization(self):
        logger.log("Using CI algorithm: Exact Diagonalization", self.log_level)

        dets = self._selected_dets()

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

        for root in range(self.nroot):
            root_rdms = {}
            root_rdms["rdm1"] = self.make_sf_1rdm(root)
            rdm2_aa, rdm2_ab, rdm2_bb = self.make_sd_2rdm(root)
            root_rdms["rdm2_aa"] = rdm2_aa
            root_rdms["rdm2_ab"] = rdm2_ab
            root_rdms["rdm2_bb"] = rdm2_bb

            # Convert to full-dimension RDMs
            root_rdms["rdm2_aa_full"] = cpp_helpers.packed_tensor4_to_tensor4(rdm2_aa)
            root_rdms["rdm2_bb_full"] = cpp_helpers.packed_tensor4_to_tensor4(rdm2_bb)

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
                self.e_var[root], rdms_energy
            ), f"CI energy {self.e_var[root]} Eh does not match RDMs energy {rdms_energy} Eh"

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

            assert self.e_var[root] == approx(rdms_energy)

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

            assert self.e_var[root] == approx(rdms_energy)

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
        if right_root is None:
            right_root = left_root
        a = self.sci_helper.a_1rdm(left_root, right_root)
        b = self.sci_helper.b_1rdm(left_root, right_root)
        return a, b

    def make_sd_2rdm(self, left_root: int, right_root: int | None = None):
        r"""
        Make the spin-dependent two-particle RDM for two CI roots.

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
        if right_root is None:
            right_root = left_root
        aa = self.sci_helper.aa_2rdm(left_root, right_root)
        ab = self.sci_helper.ab_2rdm(left_root, right_root)
        bb = self.sci_helper.bb_2rdm(left_root, right_root)
        return aa, ab, bb

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
        if right_root is None:
            right_root = left_root
        return self.sci_helper.sf_1rdm(left_root, right_root)

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
            Spin-free two-particle RDM in chemist's notation.
        """
        if right_root is None:
            right_root = left_root
        return self.sci_helper.sf_2rdm(left_root, right_root)

    # The state-averaged RDM machinery in CIBase calls make_{1,2}rdm on the sub-solvers.
    # Non-relativistic sCI returns spin-free RDMs.
    def make_1rdm(self, left_root: int, right_root: int | None = None):
        return self.make_sf_1rdm(left_root, right_root)

    def make_2rdm(self, left_root: int, right_root: int | None = None):
        return self.make_sf_2rdm(left_root, right_root)

    def compute_natural_occupation_numbers(self):
        """
        Compute the natural occupation numbers from the spin-free 1-RDM.

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
class _RelSelectedCISingleStateSolver(_SelectedCISingleStateSolver):
    """
    Two-component (relativistic) selected CI solver for a single `State`.

    This is the spinor-basis counterpart of `_SelectedCISingleStateSolver`. The CI
    coefficients, integrals, Slater rules, and RDMs are complex, spin is not a good
    quantum number (no S^2 projection, spin penalty, or spin-complement pairing), and all
    active electrons occupy the "alpha" string with the beta string empty (the same
    spinor-occupation convention as the full two-component CI).

    Selection and the sigma-vector build are delegated to the complex C++
    `RelSelectedCIHelper`, the spinor-basis counterpart of `SelectedCIHelper`. It runs the
    same Heat-Bath CI (HBCI) selection and Hamiltonian application on complex Hermitian
    integrals, so this worker inherits the base `run` loop unchanged and only overrides the
    helper / Slater-rule factories, the RDMs, and the (spin-free) guess construction.

    Reduced density matrices are computed on the Python side from the complex CI vectors via
    the tested SparseState helpers (`compute_a_1rdm_complex` / `compute_aa_2rdm_complex`),
    since the C++ helper does not build RDMs.
    """

    def __post_init__(self):
        super().__post_init__()
        self.dtype = complex
        # Spin is not a good quantum number in the two-component (spinor) basis, so the
        # S^2-based guess projection / penalty machinery does not apply.
        self.sci_params.do_spin_penalty = False

    def _make_slater_rules(self):
        # Complex Slater rules over spinor determinants. The scalar energy is real; the
        # one- and two-electron integrals are complex Hermitian. Integrals are stored in
        # physicist notation <pq|rs> (not antisymmetrized), matching SpinorbitalIntegrals.
        return det.RelSlaterRules(self.norb, self.ints.E.real, self.ints.H, self.ints.V)

    def _make_sci_helper(self):
        # Two-component (complex) selected CI helper. The scalar energy is real; the one- and
        # two-electron integrals are complex Hermitian in physicist notation <pq|rs>.
        return ci_helpers.RelSelectedCIHelper(
            self.norb,
            self.guess_determinants,
            self.guess_c,
            self.ints.E.real,
            self.ints.H,
            self.ints.V,
            self.log_level,
            self.sci_params.screening_criterion,
            self.sci_params.frozen_creation,
            self.sci_params.frozen_annihilation,
        )

    def _compute_spin2(self):
        # Spin is not a good quantum number in the two-component (spinor) basis, and the
        # complex helper does not expose compute_spin2.
        return np.zeros(self.nroot)

    def _initial_guess(self):
        """
        Build the initial guess for the two-component (spinor) selected CI.

        Spin is not a good quantum number, so there is no S^2 projection, no spin penalty,
        and no spin-complement pairing. All active electrons occupy the "alpha" string with
        the beta string empty. Guess coefficients come from diagonalizing the complex
        Hermitian Hamiltonian in the guess space.
        """
        window_occ = self.sci_params.guess_occ_window
        window_vir = self.sci_params.guess_vir_window
        if (
            len(self.sci_params.guess_dets) + len(self.sci_params.pinned_guess_dets)
            == 0
        ):
            self.sci_params.guess_dets = self._generate_initial_guess_dets(
                window_occ, window_vir
            )
        else:
            self._check_guess_dets(self.sci_params.guess_dets)
            self._check_guess_dets(self.sci_params.pinned_guess_dets)

        # refine the guess determinants by determinantal energy if there are more than needed
        if len(self.sci_params.guess_dets) > 0:
            guess_hdiag = self.slater_rules.energies(self.sci_params.guess_dets)
            nguess_dets = len(self.sci_params.guess_dets)
            num_guess_states = min(
                self.davidson_liu_params.guess_per_root * self.nroot, nguess_dets
            )
            nguess_dets = min(
                self.davidson_liu_params.ndets_per_guess * num_guess_states,
                nguess_dets,
            )

        if self.sci_params.energy_shift is not None:
            indices = np.argsort(np.abs(guess_hdiag - self.sci_params.energy_shift))[
                :nguess_dets
            ]
        else:
            indices = np.argsort(guess_hdiag)[:nguess_dets]

        self.sci_params.guess_dets = [self.sci_params.guess_dets[i] for i in indices]
        self.sci_params.guess_dets += self.sci_params.pinned_guess_dets
        # deduplicate while preserving order (pinned dets may repeat generated ones)
        self.sci_params.guess_dets = list(
            OrderedDict.fromkeys(self.sci_params.guess_dets)
        )
        logger.log(
            f"Number of guess determinants: {len(self.sci_params.guess_dets)}",
            self.log_level,
        )

        ndet = len(self.sci_params.guess_dets)
        Hguess = np.zeros((ndet, ndet), dtype=self.dtype)
        for i in range(ndet):
            for j in range(i + 1):
                Hguess[i, j] = self.slater_rules.slater_rules(
                    self.sci_params.guess_dets[i], self.sci_params.guess_dets[j]
                )
                Hguess[j, i] = np.conj(Hguess[i, j])

        evals, evecs = np.linalg.eigh(Hguess)
        if self.sci_params.energy_shift is not None:
            argsort = np.argsort(np.abs(evals - self.sci_params.energy_shift))
            evals = evals[argsort]
            evecs = evecs[:, argsort]

        c = evecs[:, : self.nroot].copy()
        energies = evals[: self.nroot].copy()
        logger.log(f"Initial guess energies: {energies}", self.log_level)
        # no states to project out (no spin penalty in the two-component case)
        return self.sci_params.guess_dets, c, energies, []

    def _generate_initial_guess_dets(self, window_occ, window_vir):
        """Generate spinor-occupation guess determinants for two-component selected CI."""
        logger.log(
            "Generating initial determinant guess (two-component)", self.log_level
        )

        nel_active = self.state.nel - self.ncore

        if window_occ < 0:
            raise ValueError(
                f"guess_occ_window must be non-negative, got {window_occ}."
            )
        if window_vir < 0:
            raise ValueError(
                f"guess_vir_window must be non-negative, got {window_vir}."
            )

        if window_occ + window_vir == 0:
            logger.log_warning(
                "No guess determinants provided and guess occupation windows set to 0. "
                "Using the aufbau determinant as the only guess."
            )
            d0 = Determinant.zero()
            for i in range(nel_active):
                d0.set_na(i, True)
            return [d0]

        # spinors are singly occupied, so the occupation window is measured in spinors
        nocc = nel_active - window_occ
        if nocc < 0:
            raise ValueError(
                f"guess_occ_window={window_occ} is larger than the number of active "
                f"occupied spinors ({nel_active}). Reduce guess_occ_window "
                "to generate valid guess determinants."
            )
        nactv = window_occ + window_vir

        if nocc + nactv > self.norb:
            raise ValueError(
                "Not enough orbitals to generate guess determinants with the specified "
                "occupation windows.\n"
                f"Number of spinors needed: {nocc + nactv}, number of active spinors "
                f"available: {self.norb}.\n"
                "Reduce guess_occ_window and/or guess_vir_window to generate valid guess "
                "determinants."
            )

        # all electrons in the alpha string (nb=0); GAS constraints come from the State
        if nocc == 0:
            ci_strings = CIStrings(nel_active, 0, 0, [[0] * nactv], [], [])
        else:
            ci_strings = CIStrings(
                nel_active, 0, 0, [[0] * nocc, [0] * nactv], [nocc], [nocc]
            )
        return ci_strings.make_determinants()

    def _check_guess_dets(self, guess_dets):
        for d in guess_dets:
            na = d.count_alpha()
            nb = d.count_beta()
            if nb != 0:
                raise ValueError(
                    f"Two-component guess determinant {d.str(self.norb)} must place all "
                    f"electrons in the alpha (spinor) string, but has {nb} beta electrons."
                )
            if na + self.ncore != self.state.nel:
                raise ValueError(
                    f"Guess determinant {d.str(self.norb)} has {na} electrons, expected "
                    f"{self.state.nel - self.ncore}."
                )

    def _test_rdms(self):
        # Compute the RDMs from the CI vectors and verify the energy from the RDMs matches
        # the CI energy.
        logger.log("\nComputing RDMs from CI vectors.\n", self.log_level)
        for root in range(self.nroot):
            rdm1 = self.make_1rdm(root)
            rdm2 = self.make_2rdm(root)

            rdms_energy = self.ints.E
            rdms_energy += np.einsum("ij,ij", rdm1, self.ints.H)
            rdms_energy += 0.5 * np.einsum("ijkl,ijkl", rdm2, self.ints.V)
            # the energy is real; the imaginary part should be numerical noise
            rdms_energy = rdms_energy.real
            logger.log(f"CI energy from RDMs: {rdms_energy:.12f} Eh", self.log_level)

            assert self.e_var[root] == approx(rdms_energy)

            logger.log(
                f"RDMs for root {root} validated successfully.\n", self.log_level
            )

    def make_so_1rdm(self, left_root: int, right_root: int | None = None):
        """
        Make the spin-orbital (spinor) one-particle RDM for two CI roots.

        Returns the complex 1-RDM gamma[p][q] = <L| a^+_p a_q |R> over active spinors.
        With left_root != right_root this is the transition 1-RDM. Computed by the C++
        ``RelSelectedCIHelper`` (which conjugates the bra); see ``_make_so_1rdm_ref`` for the
        SparseState reference implementation.
        """
        if right_root is None:
            right_root = left_root
        return self.sci_helper.a_1rdm(left_root, right_root)

    def make_so_2rdm(self, left_root: int, right_root: int | None = None):
        """
        Make the spin-orbital (spinor) two-particle RDM for two CI roots.

        Returns the complex 2-RDM gamma[p][q][r][s] = <L| a^+_p a^+_q a_s a_r |R> over
        active spinors (full antisymmetric tensor), matching RelCISigmaBuilder.compute_2rdm.
        With left_root != right_root this is the transition 2-RDM. Computed by the C++
        ``RelSelectedCIHelper``; see ``_make_so_2rdm_ref`` for the SparseState reference.
        """
        if right_root is None:
            right_root = left_root
        return self.sci_helper.aa_2rdm(left_root, right_root)

    # Reference SparseState-based implementations kept for validation. These build a dense
    # SparseState per root and contract via the tested complex sparse RDM helpers. The
    # production make_so_{1,2}rdm above call the equivalent C++ RelSelectedCIHelper methods.
    def _make_so_1rdm_ref(self, left_root: int, right_root: int | None = None):
        """Reference spin-orbital 1-RDM via the complex SparseState helper."""
        if right_root is None:
            right_root = left_root
        left = SparseState({d: c for d, c in zip(self.dets, self.evecs[:, left_root])})
        right = SparseState(
            {d: c for d, c in zip(self.dets, self.evecs[:, right_root])}
        )
        return rdms.compute_a_1rdm_complex(left, right, self.norb)

    def _make_so_2rdm_ref(self, left_root: int, right_root: int | None = None):
        """Reference spin-orbital 2-RDM via the complex SparseState helper."""
        if right_root is None:
            right_root = left_root
        left = SparseState({d: c for d, c in zip(self.dets, self.evecs[:, left_root])})
        right = SparseState(
            {d: c for d, c in zip(self.dets, self.evecs[:, right_root])}
        )
        return rdms.compute_aa_2rdm_complex(left, right, self.norb)

    # The state-averaged RDM machinery in CIBase calls make_{1,2}rdm on the sub-solvers.
    # Two-component sCI returns spin-orbital (spinor) RDMs, matching the RelCI convention.
    def make_1rdm(self, left_root: int, right_root: int | None = None):
        return self.make_so_1rdm(left_root, right_root)

    def make_2rdm(self, left_root: int, right_root: int | None = None):
        return self.make_so_2rdm(left_root, right_root)

    # The spin-dependent / spin-free RDMs of the base worker rely on the real C++ helper and
    # have no meaning in the spinor basis.
    def make_sd_1rdm(self, *args, **kwargs):
        raise NotImplementedError(
            "Spin-dependent RDMs are not defined for two-component selected CI; "
            "use make_so_1rdm."
        )

    def make_sd_2rdm(self, *args, **kwargs):
        raise NotImplementedError(
            "Spin-dependent RDMs are not defined for two-component selected CI; "
            "use make_so_2rdm."
        )

    def make_sf_1rdm(self, *args, **kwargs):
        raise NotImplementedError(
            "Spin-free RDMs are not defined for two-component selected CI; "
            "use make_so_1rdm."
        )

    def make_sf_2rdm(self, *args, **kwargs):
        raise NotImplementedError(
            "Spin-free RDMs are not defined for two-component selected CI; "
            "use make_so_2rdm."
        )


@dataclass
class SelectedCISolver(CIBase):
    """
    A general configuration interaction (CI) solver class.
    This solver is can be called iteratively, e.g., in a MCSCF loop or a DSRG reference relaxation loop.

    Parameters
    ----------
    sci_params : SelectedCIParams or list[SelectedCIParams], optional
        Parameters specific to the selected CI algorithm.
        If a list is provided, it should have one entry per state.
        If only a single SelectedCIParams is provided, it will be used for all states.
    davidson_liu_params : DavidsonLiuParams or list[DavidsonLiuParams], optional
        Parameters for the Davidson-Liu iterative eigensolver.
        If a list is provided, it should have one entry per state.
        If only a single DavidsonLiuParams is provided, it will be used for all states.
    do_test_rdms : bool, optional, default=False
        If True, compute and test the reduced density matrices (RDMs) after the CI calculation.
    log_level : int, optional
        The logging level for the CI solver. Defaults to the global logger's verbosity level.

    Attributes
    ----------
    sub_solvers : list[_SelectedCISingleStateSolver]
        A list of CI solvers for each state in the state-averaged CI.
    evar/evals_[per_solver/flat] : list[NDArray] / NDArray
        The variational eigenvalues (energies) computed by each sub-solver / concatenated into a single array.
    ept2_var_[per_solver/flat] : list[NDArray] / NDArray
        The PT2 correction due to the new variational determinants, computed by each sub-solver / concatenated into a single array.
    ept2_pt_[per_solver/flat] : list[NDArray] / NDArray
        The PT2 correction due to the perturbative determinants, computed by each sub-solver / concatenated into a single array.
    etot_[per_solver/flat] : list[NDArray] / NDArray
        The total energy (variational + PT2) computed by each sub-solver / concatenated into a single array.
    E : NDArray
        Alias for `evar_flat`, the variational energies of the CI roots.
    E_pt2 : NDArray
        The total PT2 correction (variational + perturbative) for each CI root.
    E_tot : NDArray
        Alias for `etot_flat`, the total energies of the CI roots.
    E_avg : float
        The average variational energy computed from the state-averaged CI roots.
    """

    sci_params: SelectedCIParams | list[SelectedCIParams] = field(
        default_factory=SelectedCIParams
    )
    davidson_liu_params: DavidsonLiuParams | list[DavidsonLiuParams] = field(
        default_factory=DavidsonLiuParams
    )
    do_test_rdms: bool = False
    log_level: int = field(default=logger.get_verbosity_level() + 1)

    def _startup(self):
        super()._startup()
        sci_params, davidson_liu_params = _selected_ci_runtime_params(
            self.sci_params, self.davidson_liu_params, self.sa_info.ncis
        )
        self.norb = self.mo_space.nactv
        # no distinction between core and frozen core in the CI solver
        self.core_indices = (
            self.mo_space.frozen_core_indices + self.mo_space.core_indices
        )
        self.active_indices = self.mo_space.active_indices

        ints = RestrictedMOIntegrals(
            self.system,
            self.mos.C[0],
            self.active_indices,
            self.core_indices,
        )

        self.sub_solvers = []
        active_orbsym = [
            [self.mos.irrep_indices[0][i] for i in active_space]
            for active_space in self.mo_space.active_orbitals
        ]
        for i, state in enumerate(self.sa_info.states):
            # Create a CI solver for each state and MOSpace

            kwargs = self._collect_child_kwargs(_SelectedCISingleStateSolver)
            # these are needed by _SelectedCISingleStateSolver but not present as attributes of SelectedCISolver
            kwargs.update(
                {
                    "sci_params": sci_params[i],
                    "davidson_liu_params": davidson_liu_params[i],
                    "ints": ints,
                    "state": state,
                    "nroot": self.sa_info.nroots[i],
                    "active_orbsym": active_orbsym,
                }
            )
            self.sub_solvers.append(_SelectedCISingleStateSolver(**kwargs))

    def run(self):
        # override the run method in the base class
        if self.first_run:
            self._startup()
            self.first_run = False

        self.evar_per_solver = []
        self.ept2_var_per_solver = []
        self.ept2_pt_per_solver = []
        self.etot_per_solver = []
        for ci_solver in self.sub_solvers:
            ci_solver.run()
            self.evar_per_solver.append(ci_solver.evals)
            self.ept2_var_per_solver.append(ci_solver.ept2_var)
            self.ept2_pt_per_solver.append(ci_solver.ept2_pt)
            self.etot_per_solver.append(ci_solver.e_tot)
        self.evals_per_solver = self.evar_per_solver

        self.evar_flat = np.concatenate(self.evar_per_solver)
        self.evals_flat = self.evar_flat
        self.ept2_var_flat = np.concatenate(self.ept2_var_per_solver)
        self.ept2_pt_flat = np.concatenate(self.ept2_pt_per_solver)
        self.etot_flat = np.concatenate(self.etot_per_solver)
        self.E_avg = self.compute_average_energy()

        self.E = self.evar_flat
        self.E_pt2 = self.ept2_var_flat + self.ept2_pt_flat
        self.E_tot = self.etot_flat

        self.executed = True
        return self

    def make_sd_1rdm(self, left_root: int, right_root: int | None = None):
        """
        Make the spin-dependent one-particle RDM for two absolute CI roots.
        """
        left_state, right_state, left_root_in_state, right_root_in_state = (
            self._validate_rdm_inputs(left_root, right_root)
        )
        if left_state == right_state:
            return self.sub_solvers[left_state].make_sd_1rdm(
                left_root_in_state, right_root_in_state
            )

        left_solver = self.sub_solvers[left_state]
        right_solver = self.sub_solvers[right_state]
        a_1trdm = left_solver.sci_helper.a_1trdm(
            right_solver.sci_helper, left_root_in_state, right_root_in_state
        )
        b_1trdm = left_solver.sci_helper.b_1trdm(
            right_solver.sci_helper, left_root_in_state, right_root_in_state
        )
        return a_1trdm, b_1trdm

    def make_sf_1rdm(self, left_root: int, right_root: int | None = None):
        """
        Make the spin-free one-particle RDM for two absolute CI roots.
        """
        left_state, right_state, left_root_in_state, right_root_in_state = (
            self._validate_rdm_inputs(left_root, right_root)
        )
        if left_state == right_state:
            return self.sub_solvers[left_state].make_sf_1rdm(
                left_root_in_state, right_root_in_state
            )

        left_solver = self.sub_solvers[left_state]
        right_solver = self.sub_solvers[right_state]
        return left_solver.sci_helper.sf_1trdm(
            right_solver.sci_helper, left_root_in_state, right_root_in_state
        )

    def make_sf_2rdm(self, left_root: int, right_root: int | None = None):
        """
        Make the spin-free two-particle RDM for two absolute CI roots.
        """
        left_state, right_state, left_root_in_state, right_root_in_state = (
            self._validate_rdm_inputs(left_root, right_root)
        )
        if left_state == right_state:
            return self.sub_solvers[left_state].make_sf_2rdm(
                left_root_in_state, right_root_in_state
            )
        raise ValueError(
            f"Cross-state 2-RDMs are not supported. Got left_root in state {left_state} and right_root in state {right_state}."
        )

    def make_1rdm(self, left_root: int, right_root: int | None = None):
        """
        Make the spin-free one-particle RDM for two absolute CI roots.
        """
        return self.make_sf_1rdm(left_root, right_root)

    def make_2rdm(self, left_root: int, right_root: int | None = None):
        """
        Make the spin-free two-particle RDM for two absolute CI roots.
        """
        return self.make_sf_2rdm(left_root, right_root)

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
        Compute the natural occupation numbers for the CI states and store them
        in ``self.nat_occs`` (this method returns None).

        The first ``nroots_sum`` columns of ``self.nat_occs`` hold the natural
        occupation numbers for each root. If more than one root is present
        (``nroots_sum > 1``), a final column holds the natural occupation
        numbers from the state-averaged 1-RDM.
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
        The results are stored in `self.transition_dipoles` and `self.oscillator_strengths`.
        """
        if not self.executed:
            raise RuntimeError("CI solver has not been executed yet.")

        if C is None:
            C = self.mos.C[0]

        Cact = C[:, self.active_indices]
        Ccore = C[:, self.core_indices]
        # factor of 2 for spin-summed 1-RDM
        rdm_core = 2 * np.einsum("pi,qi->pq", Ccore, Ccore.conj(), optimize=True)
        # this includes nuclear dipole contribution
        core_dip = get_1e_property(
            self.system, rdm_core, property_name="dipole", unit="au"
        )
        self.transition_dipoles = OrderedDict()
        self.oscillator_strengths = OrderedDict()

        for ici in range(self.sa_info.nroots_sum):
            istate, iroot_in_state = self._get_state_root(ici)
            rdm = self.sub_solvers[istate].make_sf_1rdm(iroot_in_state)
            rdm = np.einsum("ij,pi,qj->pq", rdm, Cact.conj(), Cact, optimize=True)
            dip = get_1e_property(
                self.system, rdm, property_name="electric_dipole", unit="au"
            )
            self.transition_dipoles[(ici, ici)] = dip + core_dip
            self.oscillator_strengths[(ici, ici)] = 0.0
            for jci in range(ici + 1, self.sa_info.nroots_sum):
                jstate, jroot_in_state = self._get_state_root(jci)
                try:
                    vte = (
                        self.evar_per_solver[jstate][jroot_in_state]
                        - self.evar_per_solver[istate][iroot_in_state]
                    )
                    if vte < 0:
                        _ici, _jci = jci, ici
                        vte = -vte
                    else:
                        _ici, _jci = ici, jci
                    tdm = self.make_1rdm(_ici, _jci)
                    tdm = np.einsum(
                        "ij,pi,qj->pq", tdm, Cact.conj(), Cact, optimize=True
                    )
                    tdip = get_1e_property(
                        self.system, tdm, property_name="electric_dipole", unit="au"
                    )
                    self.transition_dipoles[(_ici, _jci)] = tdip
                    self.oscillator_strengths[(_ici, _jci)] = (
                        (2 / 3) * vte * np.linalg.norm(tdip) ** 2
                    )
                except (ValueError, NotImplementedError):
                    continue

        return self.transition_dipoles, self.oscillator_strengths

    def reset_eigensolver(self):
        # sCI eigensolver gets reset every iteration anyway
        pass

    def get_convergence_status(self):
        pass


@dataclass
class SelectedCI(SelectedCISolver):
    """
    Selected CI solver specialized for a single CI calculation. (i.e., not used in a loop).
    See `SelectedCISolver` for all parameters and attributes.
    """

    die_if_not_converged: bool = True
    final_orbitals: str = "original"
    do_transition_dipole: bool = False
    log_level: int = field(default=logger.get_verbosity_level())

    def run(self):
        super().run()
        self._post_process()
        if self.final_orbitals == "semicanonical":
            semi = Semicanonicalizer(
                system=self.system,
                mo_space=self.mo_space,
            )
            semi.semi_canonicalize(g1=self.make_average_1rdm(), C_contig=self.mos.C[0])
            self.mos.C[0] = semi.C_semican.copy()

            # recompute the CI vectors in the semicanonical basis
            ints = RestrictedMOIntegrals(
                self.system,
                self.mos.C[0],
                self.active_indices,
                self.core_indices,
            )
            self.set_ints(ints.E, ints.H, ints.V)
            super().run()

        return self

    def _post_process(self):
        pretty_print_ci_summary(
            self.sa_info,
            self.evar_per_solver,
            header="\nSelected CI energy (variational)",
        )
        pretty_print_ci_summary(
            self.sa_info,
            self.etot_per_solver,
            header="\nSelected CI energy (variational + PT2)",
        )
        top_dets = self.get_top_determinants()
        pretty_print_ci_dets(self.sa_info, self.mo_space, top_dets)

        if self.do_transition_dipole:
            self.compute_transition_properties()
            pretty_print_ci_transition_props(
                self.sa_info,
                self.transition_dipoles,
                self.oscillator_strengths,
                self.evar_per_solver,
            )


@dataclass
class RelSelectedCISolver(RelCIBase):
    """
    A two-component (relativistic) selected configuration interaction (2C selected CI) solver.

    This mirrors `SelectedCISolver` but operates in the molecular spinor basis produced by a
    two-component method (e.g. GHF, optionally with spin-orbit X2C, or a non-relativistic SCF
    promoted with `SpinorUpcaster`). The CI coefficients, integrals, and RDMs are complex, and
    spin is not a good quantum number, so there is no spin adaptation or spin penalty.

    Like `RelCISolver`, it can be used as the inner CI solver of a relativistic MCSCF/CASSCF
    calculation (`MCOptimizer`) or a relativistic DSRG reference.

    Parameters
    ----------
    sci_params : SelectedCIParams or list[SelectedCIParams], optional
        Parameters specific to the selected CI algorithm. One entry per state, or a single
        set shared by all states.
    davidson_liu_params : DavidsonLiuParams or list[DavidsonLiuParams], optional
        Parameters for the Davidson-Liu iterative eigensolver.
    do_test_rdms : bool, optional, default=False
        If True, compute and validate the RDMs against the CI energy after the calculation.
    log_level : int, optional
        The logging level for the CI solver.
    """

    orbital_rotation_invariant: ClassVar[bool] = True

    sci_params: SelectedCIParams | list[SelectedCIParams] = field(
        default_factory=SelectedCIParams
    )
    davidson_liu_params: DavidsonLiuParams | list[DavidsonLiuParams] = field(
        default_factory=DavidsonLiuParams
    )
    do_test_rdms: bool = False
    log_level: int = field(default=logger.get_verbosity_level() + 1)

    # State-averaging orchestration and root bookkeeping are inherited from RelCIBase/CIBase.
    # These dtype-agnostic methods are borrowed from SelectedCISolver unchanged.
    set_ints = SelectedCISolver.set_ints
    reset_eigensolver = SelectedCISolver.reset_eigensolver
    get_convergence_status = SelectedCISolver.get_convergence_status
    get_top_determinants = SelectedCISolver.get_top_determinants
    # aggregates the sub-solvers' (2C-aware) natural occupation numbers
    compute_natural_occupation_numbers = (
        SelectedCISolver.compute_natural_occupation_numbers
    )

    def _startup(self):
        super()._startup()
        sci_params, davidson_liu_params = _selected_ci_runtime_params(
            self.sci_params, self.davidson_liu_params, self.sa_info.ncis
        )

        self.norb = self.mo_space.nactv
        # no distinction between core and frozen core in the CI solver
        self.core_indices = (
            self.mo_space.frozen_core_indices + self.mo_space.core_indices
        )
        self.active_indices = self.mo_space.active_indices

        ints = SpinorbitalIntegrals(
            self.system,
            self.mos.C[0],
            self.active_indices,
            self.core_indices,
        )

        self.sub_solvers = []
        active_orbsym = [
            [self.mos.irrep_indices[0][i] for i in active_space]
            for active_space in self.mo_space.active_orbitals
        ]
        for i, state in enumerate(self.sa_info.states):
            kwargs = self._collect_child_kwargs(_RelSelectedCISingleStateSolver)
            kwargs.update(
                {
                    "sci_params": sci_params[i],
                    "davidson_liu_params": davidson_liu_params[i],
                    "ints": ints,
                    "state": state,
                    "nroot": self.sa_info.nroots[i],
                    "active_orbsym": active_orbsym,
                }
            )
            self.sub_solvers.append(_RelSelectedCISingleStateSolver(**kwargs))

    def run(self):
        if self.first_run:
            self._startup()
            self.first_run = False

        self.evar_per_solver = []
        self.ept2_var_per_solver = []
        self.ept2_pt_per_solver = []
        self.etot_per_solver = []
        for ci_solver in self.sub_solvers:
            ci_solver.run()
            self.evar_per_solver.append(ci_solver.evals)
            self.ept2_var_per_solver.append(ci_solver.ept2_var)
            self.ept2_pt_per_solver.append(ci_solver.ept2_pt)
            self.etot_per_solver.append(ci_solver.e_tot)
        self.evals_per_solver = self.evar_per_solver

        self.evar_flat = np.concatenate(self.evar_per_solver)
        self.evals_flat = self.evar_flat
        self.ept2_var_flat = np.concatenate(self.ept2_var_per_solver)
        self.ept2_pt_flat = np.concatenate(self.ept2_pt_per_solver)
        self.etot_flat = np.concatenate(self.etot_per_solver)
        self.E_avg = self.compute_average_energy()

        self.E = self.evar_flat
        self.E_pt2 = self.ept2_var_flat + self.ept2_pt_flat
        self.E_tot = self.etot_flat

        self.executed = True
        return self

    def make_1rdm(self, left_root: int, right_root: int | None = None):
        """Complex spin-orbital 1-RDM for two absolute CI roots (same state only)."""
        left_state, right_state, left_root_in_state, right_root_in_state = (
            self._validate_rdm_inputs(left_root, right_root)
        )
        if left_state == right_state:
            return self.sub_solvers[left_state].make_so_1rdm(
                left_root_in_state, right_root_in_state
            )
        raise NotImplementedError(
            f"Cross-state 1-RDMs are not supported for RelSelectedCI. Got left_root in "
            f"state {left_state} and right_root in state {right_state}."
        )

    def make_2rdm(self, left_root: int, right_root: int | None = None):
        """Complex spin-orbital 2-RDM for two absolute CI roots (same state only)."""
        left_state, right_state, left_root_in_state, right_root_in_state = (
            self._validate_rdm_inputs(left_root, right_root)
        )
        if left_state == right_state:
            return self.sub_solvers[left_state].make_so_2rdm(
                left_root_in_state, right_root_in_state
            )
        raise NotImplementedError(
            f"Cross-state 2-RDMs are not supported for RelSelectedCI. Got left_root in "
            f"state {left_state} and right_root in state {right_state}."
        )


@dataclass
class RelSelectedCI(RelSelectedCISolver):
    """
    Two-component selected CI specialized for a single calculation (i.e., not used in a loop).
    See `RelSelectedCISolver` for all parameters and attributes.
    """

    die_if_not_converged: bool = True
    final_orbitals: str = "original"
    do_transition_dipole: bool = False
    log_level: int = field(default=logger.get_verbosity_level())

    def __post_init__(self):
        super().__post_init__()
        if self.final_orbitals not in ["original", "semicanonical"]:
            raise ValueError(
                f"Invalid value for final_orbitals: {self.final_orbitals}. "
                "Must be 'original' or 'semicanonical'."
            )

    def run(self):
        super().run()
        self._post_process()
        if self.final_orbitals == "semicanonical":
            semi = Semicanonicalizer(
                mo_space=self.mo_space,
                system=self.system,
                irrep_indices=np.array(self.mos.irrep_indices[0])[
                    self.mo_space.orig_to_contig
                ],
            )
            C_contig = self.mos.C[0][:, self.mo_space.orig_to_contig].copy()
            semi.semi_canonicalize(g1=self.make_average_1rdm(), C_contig=C_contig)
            self.mos.C[0] = semi.C_semican[:, self.mo_space.contig_to_orig].copy()

            # recompute the CI vectors in the semicanonical basis
            ints = SpinorbitalIntegrals(
                self.system,
                self.mos.C[0],
                self.active_indices,
                self.core_indices,
            )
            self.set_ints(ints.E, ints.H, ints.V)
            self.first_run = True
            super().run()

        return self

    def _post_process(self):
        pretty_print_ci_summary(
            self.sa_info,
            self.evar_per_solver,
            header="\nRelativistic selected CI energy (variational)",
        )
        pretty_print_ci_summary(
            self.sa_info,
            self.etot_per_solver,
            header="\nRelativistic selected CI energy (variational + PT2)",
        )
        top_dets = self.get_top_determinants()
        pretty_print_ci_dets(self.sa_info, self.mo_space, top_dets)

        if self.do_transition_dipole:
            self.compute_transition_properties()
            pretty_print_ci_transition_props(
                self.sa_info,
                self.transition_dipoles,
                self.oscillator_strengths,
                self.evar_per_solver,
            )
