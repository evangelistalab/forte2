from dataclasses import dataclass, field
from collections import OrderedDict
from typing import ClassVar

import numpy as np

from forte2.lib import det, ci_helpers, rdms
from forte2.lib.sparse_ops import SparseState
from forte2.lib.det import Determinant
from forte2.lib.ci_helpers import CIStrings
from forte2.helpers.comparisons import approx
from forte2.base_classes import RelCIBase
from forte2.base_classes.params import SelectedCIParams, DavidsonLiuParams
from forte2.helpers import logger
from forte2.jkbuilder import SpinorbitalIntegrals
from forte2.orbitals import FinalOrbitals, validate_final_orbitals
from forte2.ci.ci_utils import (
    pretty_print_ci_summary,
    pretty_print_ci_dets,
    pretty_print_ci_transition_props,
    pretty_print_ci_nat_occ_numbers,
)
from .sci import _SelectedCISingleStateSolver, SelectedCISolver


@dataclass
class _RelSelectedCISingleStateSolver(_SelectedCISingleStateSolver):
    """
    Two-component (relativistic) selected CI solver for a single `State`.
    """

    two_component: ClassVar[bool] = True
    dtype: ClassVar[type] = complex

    def _make_sci_helper(self):
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

    def _update_sci_helper_ints(self):
        # RelSelectedCIHelper.set_Hamiltonian's E parameter is a plain double.
        self.sci_helper.set_Hamiltonian(self.ints.E.real, self.ints.H, self.ints.V)

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
        # local object used only to build initial guess
        # exact diag uses sci_helper's slater_rules
        slater_rules = det.RelSlaterRules(
            self.norb, self.ints.E.real, self.ints.H, self.ints.V
        )
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
            guess_hdiag = slater_rules.energies(self.sci_params.guess_dets)
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
                Hguess[i, j] = slater_rules.slater_rules(
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
class RelSelectedCISolver(RelCIBase, SelectedCISolver):
    """
    A two-component (relativistic) selected configuration interaction (2C selected CI) solver.

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

    # Selected CI is a variational truncation, not the full CI space, so unlike
    # CISolver/RelCISolver the energy is not exactly invariant to which orbital basis
    # the truncation is done in. Inherits False from CIBase, same as SelectedCISolver.

    sci_params: SelectedCIParams | list[SelectedCIParams] = field(
        default_factory=SelectedCIParams
    )
    davidson_liu_params: DavidsonLiuParams | list[DavidsonLiuParams] = field(
        default_factory=DavidsonLiuParams
    )
    do_test_rdms: bool = False
    log_level: int = field(default=logger.get_verbosity_level() + 1)

    # Active-space integral class used by CIBase._make_active_space_ints
    _integrals_cls: ClassVar[type] = SpinorbitalIntegrals
    # Per-state worker class used by CIBase._startup
    _ss_solver_cls: ClassVar[type] = _RelSelectedCISingleStateSolver

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
    final_orbitals: FinalOrbitals = "original"
    do_transition_dipole: bool = False
    log_level: int = field(default=logger.get_verbosity_level())

    def __post_init__(self):
        super().__post_init__()
        validate_final_orbitals(self.final_orbitals)

    def run(self):
        self._solve()
        self._rotate_final_orbitals(self.final_orbitals)
        self._post_process()
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
        self.compute_natural_occupation_numbers()
        pretty_print_ci_nat_occ_numbers(
            self.sa_info,
            self.mo_space,
            self.nat_occs,
            self.nat_occs_avg,
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
