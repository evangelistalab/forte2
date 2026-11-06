from dataclasses import dataclass, field
from typing import ClassVar, Literal

import numpy as np
from numpy.typing import NDArray

from forte2.lib import cpp_helpers
from forte2.lib.ci_helpers import (
    CIStrings,
    CISigmaBuilder,
    CISpinAdapter,
)
from forte2.state import State, MOSpace
from forte2.helpers.comparisons import approx
from forte2.helpers.davidsonliu import DavidsonLiuSolver
from forte2.base_classes import ActiveSpaceDriver, CIBase, Method
from forte2.base_classes.params import DavidsonLiuParams, CIParams
from forte2.helpers import logger
from forte2.jkbuilder import RestrictedMOIntegrals
from forte2.orbitals import FinalOrbitals
from .ci_utils import (
    orthonormalize_guesses,
    pretty_print_gas_info,
    validate_single_state_rdm,
    make_cumulant_from_rdms,
)


@dataclass
class _CISingleStateSolver:
    """
    A general configuration interaction (CI) solver class for a single `State`.
    Although possible, is not recommended to instantiate this class directly.
    Consider using the `CISolver` class instead.

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
    ci_params : CIParams, optional
        Parameters for the CI solver, including choice of algorithm and memory limits.
    davidson_liu_params : DavidsonLiuParams, optional
        Parameters for the Davidson-Liu eigensolver.
    do_test_rdms : bool, optional, default=False
        If True, compute and test the reduced density matrices (RDMs) after the CI calculation.
    log_level : int, optional
        The logging level for the CI solver. Defaults to ``logger.VERBOSITY_DEBUG``.
    die_if_not_converged : bool, optional, default=False
        If True, raise an error if the CI solver does not converge.
    ci_vector_guess : list, optional
        The initial CI vectors of this state.
        One (absolute_index, {Determinant: coefficient}) tuple per root. Each guess is
        projected onto the CSF basis and normalized, so a spin-impure guess, such as a
        single open-shell determinant, keeps the normalized component of the target
        multiplicity. The two-component solver has no CSF basis and only normalizes.
    guess_dets : list[Determinant], optional
        The candidate determinants for the guess space of this state. If None, all
        basis functions are candidates.
    pinned_guess_dets : list[Determinant], optional
        Determinants whose basis functions always join the guess space of this state.

    Attributes
    ----------
    eigensolver : DavidsonLiuSolver
        The eigensolver used to find the roots of the CI problem.
    guess_energies : NDArray
        The eigenvalues of the guess Hamiltonian for the guess vectors. Not set when
        the guess is ``ci_vector_guess``.
    E (evals) : NDArray
        The eigenvalues (energies) of the CI problem.
    evecs : NDArray
        The eigenvectors (CI coefficients) of the CI problem.

    """

    mo_space: MOSpace
    state: State
    ints: RestrictedMOIntegrals
    nroot: int
    active_orbsym: list[int]
    ci_params: CIParams = field(default_factory=CIParams)
    davidson_liu_params: DavidsonLiuParams = field(default_factory=DavidsonLiuParams)
    do_test_rdms: bool = False
    die_if_not_converged: bool = False
    log_level: int = logger.VERBOSITY_DEBUG
    ci_vector_guess: list | None = None
    guess_dets: list | None = None
    pinned_guess_dets: list | None = None

    ### Non-init attributes
    rebuild_guess: bool = field(default=True, init=False)
    guess: NDArray | None = field(default=None, init=False)
    executed: bool = field(default=False, init=False)

    ### These will be overridden by _RelCISingleStateSolver
    two_component: ClassVar[bool] = False
    dtype: ClassVar[type] = float
    _sigma_builder_cls: ClassVar[type] = CISigmaBuilder
    _allowed_algorithms: ClassVar[tuple] = (
        "hz",
        "harrison-zarrabian",
        "kh",
        "knowles-handy",
        "exact",
    )

    def __post_init__(self):
        self.norb = self.mo_space.nactv
        self.ncore = self.mo_space.ncore + self.mo_space.nfrozen_core
        self.ngas = self.mo_space.ngas
        self.gas_min = self.state.gas_min
        self.gas_max = self.state.gas_max
        self.eigensolver = None

        assert self.ci_params.ci_algorithm.lower() in self._allowed_algorithms, (
            f"{type(self).__name__} supports CI algorithms "
            f"{self._allowed_algorithms}. Got '{self.ci_params.ci_algorithm}'."
        )

    def _make_sigma_builder_obj(self):
        """Construct the C++ sigma builder for the current integrals."""
        algorithm = self.ci_params.ci_algorithm.lower()
        if algorithm not in ("kh", "hz", "knowles-handy", "harrison-zarrabian"):
            # e.g. "exact": the iterative sigma-build algorithm is unused on that path
            algorithm = "kh"
        return self._sigma_builder_cls(
            self.ci_strings,
            self.ints.E,
            self.ints.H,
            self.ints.V,
            self.log_level,
            algorithm,
        )

    def _make_ci_strings(self):
        """Build the CI string/determinant space for this state."""
        _nactel_a = self.state.na - self.ncore
        _nactel_b = self.state.nb - self.ncore
        assert (
            _nactel_a >= 0
        ), f"Number of active \u03b1 electrons {_nactel_a} must be non-negative."
        assert (
            _nactel_b >= 0
        ), f"Number of active \u03b2 electrons {_nactel_b} must be non-negative."
        return CIStrings(
            _nactel_a,
            _nactel_b,
            self.state.symmetry,
            self.active_orbsym,
            self.gas_min,
            self.gas_max,
        )

    def _log_ci_strings_info(self):
        logger.log(
            f"\nNumber of \u03b1 electrons: {self.ci_strings.na}", self.log_level
        )
        logger.log(f"Number of \u03b2 electrons: {self.ci_strings.nb}", self.log_level)
        logger.log(f"Number of \u03b1 strings: {self.ci_strings.nas}", self.log_level)
        logger.log(f"Number of \u03b2 strings: {self.ci_strings.nbs}", self.log_level)

    def _setup_basis(self):
        """Build the variational basis and allocate the determinant-basis buffers.

        The non-relativistic solver works in a spin-adapted CSF basis, so
        ``basis_size`` (CSFs) differs from ``ndet``.
        """
        self.spin_adapter = CISpinAdapter(
            self.state.multiplicity - 1, self.state.twice_ms, self.norb
        )
        self.spin_adapter.set_log_level(self.log_level)
        self.dets = self.ci_strings.make_determinants()

        self.spin_adapter.prepare_couplings(self.dets)
        logger.log(
            f"Number of configurations: {self.spin_adapter.nconf}", self.log_level
        )
        logger.log(f"Number of CSFs: {self.spin_adapter.ncsf}", self.log_level)

        self.ndet = self.ci_strings.ndet
        self.basis_size = self.spin_adapter.ncsf

        # CI vectors holding the sigma-builder results in the determinant basis
        self.b_det = np.zeros((self.ndet,), dtype=self.dtype)
        self.sigma_det = np.zeros((self.ndet,), dtype=self.dtype)

    def _ci_solver_startup(self):
        self.ci_strings = self._make_ci_strings()

        pretty_print_gas_info(self.ci_strings)
        self._log_ci_strings_info()
        logger.log(f"Number of determinants: {self.ci_strings.ndet}", self.log_level)

        if self.ci_strings.ndet == 0:
            raise ValueError(
                "No determinants could be generated for the given state and orbitals."
            )

        self._setup_basis()

    def _update_sigma_builder_ints(self):
        """Push the current active-space integrals into the existing sigma builder."""
        self.ci_sigma_builder.set_Hamiltonian(self.ints.E, self.ints.H, self.ints.V)

    def run(self):
        if not self.executed:
            self._ci_solver_startup()
            # Create the sigma builder from the CI strings and integrals.
            self.ci_sigma_builder = self._make_sigma_builder_obj()
        else:
            # Update the integrals (reusing the same sigma builder)
            self._update_sigma_builder_ints()
        self.ci_sigma_builder.set_memory(self.ci_params.ci_builder_memory)
        if self.ci_params.ci_algorithm.lower() == "exact":
            self._do_exact_diagonalization()
        else:
            self._do_iterative_ci()

        self.E = self.evals
        for i, e in enumerate(self.evals):
            logger.log(f"Final CI Energy Root {i}: {e:20.12f} [Eh]", self.log_level)

        if self.do_test_rdms:
            self._test_rdms()

        self.executed = True

        return self

    def _form_hdiag(self):
        """Diagonal of the Hamiltonian in the variational (CSF) basis."""
        return self.ci_sigma_builder.form_Hdiag_csf(
            self.dets, self.spin_adapter, spin_adapt_full_preconditioner=False
        )

    def _make_sigma_builder(self):
        """Return the sigma-build closure for the Davidson-Liu solver.

        Basis vectors arrive in the CSF basis, are transformed to determinants for
        the C++ Hamiltonian application, and transformed back.
        """

        def sigma_builder(Bblock, Sblock):
            ncols = Bblock.shape[1]
            for i in range(ncols):
                self.spin_adapter.csf_C_to_det_C(Bblock[:, i], self.b_det)
                self.ci_sigma_builder.Hamiltonian(self.b_det, self.sigma_det)
                self.spin_adapter.det_C_to_csf_C(self.sigma_det, Sblock[:, i])

        return sigma_builder

    def _log_sigma_build_times(self):
        h_tot, h_aabb, h_aaaa, h_bbbb = self.ci_sigma_builder.avg_build_time()
        logger.log("\nAverage CI Sigma Builder time summary:", self.log_level)
        logger.log(f"h_aabb time:    {h_aabb:.3f} s/build", self.log_level)
        logger.log(f"h_aaaa time:    {h_aaaa:.3f} s/build", self.log_level)
        logger.log(f"h_bbbb time:    {h_bbbb:.3f} s/build", self.log_level)
        logger.log(f"total time:     {h_tot:.3f} s/build\n", self.log_level)

    def _do_iterative_ci(self):
        """
        Solve CI with an iterative Davidson-Liu solver, using either
        Harrison-Zarrabian or Knowles-Handy sigma builder algorithm.
        """
        logger.log(
            f"Using CI algorithm: {self.ci_sigma_builder.get_algorithm()}",
            self.log_level,
        )

        Hdiag = self._form_hdiag()

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
        if self.eigensolver is None:
            self.eigensolver = DavidsonLiuSolver(
                size=self.basis_size,  # size of the basis (number of CSF if we spin adapt)
                nroot=self.nroot,
                davidson_liu_params=self.davidson_liu_params,
                energy_shift=self.ci_params.energy_shift,
                log_level=self.log_level,
                dtype=self.dtype,
            )

        # 4. Compute diagonal of the Hamiltonian
        self.eigensolver.add_h_diag(Hdiag)

        # 5. (Re-)build the guess vectors if requested.
        # This is always done on the first run at least, but can be forced again by e.g. reset_eigensolver.
        if self.rebuild_guess:
            if self.guess is not None:
                self.eigensolver.add_guesses(self.guess)
            else:
                self._build_guess_vectors(Hdiag)
            self.rebuild_guess = False

        self.eigensolver.add_sigma_builder(self._make_sigma_builder())

        # 6. Run Davidson
        self.evals, self.evecs = self.eigensolver.solve()

        if self.eigensolver.converged:
            logger.log("\nDavidson-Liu solver converged.\n", self.log_level)
        else:
            if self.die_if_not_converged:
                raise RuntimeError("Davidson-Liu solver did not converge.")
            else:
                logger.log_warning(
                    f"Davidson-Liu solver did not converge in {self.eigensolver.maxiter} iterations."
                )

        self._log_sigma_build_times()

    def _build_full_hamiltonian(self):
        """Dense Hamiltonian in the variational (CSF) basis."""
        return self.ci_sigma_builder.form_H_csf(self.dets, self.spin_adapter)

    def _do_exact_diagonalization(self):
        logger.log("Using CI algorithm: Exact Diagonalization", self.log_level)

        H = self._build_full_hamiltonian()

        self.evals_full, self.evecs_full = np.linalg.eigh(H)
        if self.ci_params.energy_shift is not None:
            argsort = np.argsort(np.abs(self.evals_full - self.ci_params.energy_shift))
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
            root_rdms["rdm1"] = self.make_rdm(root, order=1, spin_type="sf")
            rdm2_aa, rdm2_ab, rdm2_bb = self.make_rdm(root, order=2, spin_type="sd")
            root_rdms["rdm2_aa"] = rdm2_aa
            root_rdms["rdm2_ab"] = rdm2_ab
            root_rdms["rdm2_bb"] = rdm2_bb

            rdm2_aa_full, _, rdm2_bb_full = self.make_rdm(root, order=2, spin_type="sd")
            # Convert to full-dimension RDMs
            root_rdms["rdm2_aa_full"] = cpp_helpers.packed_tensor4_to_tensor4(
                rdm2_aa_full
            )
            root_rdms["rdm2_bb_full"] = cpp_helpers.packed_tensor4_to_tensor4(
                rdm2_bb_full
            )

            root_rdms["rdm2_sf"] = self.make_rdm(root, order=2, spin_type="sf")

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

    def _basis_matrix_element(self, I, J):
        """<I|H|J> between two vectors of the variational (CSF) basis."""
        return self.ci_sigma_builder.slater_rules_csf(
            self.dets, self.spin_adapter, I, J
        )

    def _build_guess_vectors(self, Hdiag):
        """Build the guess vectors for the CI calculation."""
        # 1. If ci_vector_guess'es are given, add those to the eigensolver and return
        if self.ci_vector_guess is not None:
            guess_mat = self._ci_vector_guess_vectors()
            logger.log(
                f"Number of guess states (ci_vector_guess): {guess_mat.shape[1]}",
                self.log_level,
            )
            self.eigensolver.add_guesses(guess_mat)
            return

        # 2. Else, build a guess Hamiltonian is a basis, which is given by
        #   - CSFs in guess_dets (converted from determinants), if it is provided
        #   - all CSFs in the CI space
        #   both
        # the candidates are narrowed by energy below,
        # while the pinned basis functions alwaus stay in the basis
        if self.guess_dets is None:
            candidates = np.arange(self.basis_size)
        else:
            candidates = self._basis_indices(self.guess_dets, "guess_dets")
        pinned = self._basis_indices(self.pinned_guess_dets or [], "pinned_guess_dets")
        candidates = candidates[~np.isin(candidates, pinned)]

        # determine the number of guess vectors
        num_guess_states = min(
            self.davidson_liu_params.guess_per_root * self.nroot, len(candidates)
        )
        nguess_candidates = min(
            self.davidson_liu_params.ndets_per_guess * num_guess_states,
            len(candidates),
        )

        # find the candidates with the lowest diagonal elements
        if self.ci_params.energy_shift is not None:
            order = np.argsort(np.abs(Hdiag[candidates] - self.ci_params.energy_shift))
        else:
            order = np.argsort(Hdiag[candidates])
        indices = np.concatenate([candidates[order[:nguess_candidates]], pinned])
        nguess_dets = len(indices)

        self.num_guess_states = min(
            self.davidson_liu_params.guess_per_root * self.nroot, nguess_dets
        )
        if self.num_guess_states < self.nroot:
            raise ValueError(
                f"The guess space of {self.state} has {nguess_dets} basis functions, "
                f"fewer than the {self.nroot} roots requested."
            )
        logger.log(f"Number of guess states: {self.num_guess_states}", self.log_level)
        logger.log(f"Number of guess basis: {nguess_dets}", self.log_level)

        _slater_rules = self._basis_matrix_element
        # create the Hamiltonian matrix in the basis of the guess CSFs
        Hguess = np.zeros((nguess_dets, nguess_dets), dtype=self.dtype)
        for i, I in enumerate(indices):
            for j, J in enumerate(indices):
                if i >= j:
                    Hij = _slater_rules(I, J)
                    Hguess[i, j] = Hij
                    Hguess[j, i] = np.conj(Hij)

        # Diagonalize the Hamiltonian to get the initial guess vectors
        evals_guess, evecs_guess = np.linalg.eigh(Hguess)
        if self.ci_params.energy_shift is not None:
            # take the eigenvectors nearest the shift rather than the lowest
            order = np.argsort(np.abs(evals_guess - self.ci_params.energy_shift))
            evals_guess = evals_guess[order]
            evecs_guess = evecs_guess[:, order]
        self.guess_energies = evals_guess[: self.num_guess_states]

        # Select the first eigenvectors in that order
        guess_mat = np.zeros((self.basis_size, self.num_guess_states), dtype=self.dtype)
        for i in range(self.num_guess_states):
            guess = evecs_guess[:, i]
            for j, d in enumerate(indices):
                guess_mat[d, i] = guess[j]

        self.eigensolver.add_guesses(guess_mat)

    def _det_C_to_basis_C(self, det_vec):
        """Project a CI vector from the determinant basis onto the CSF basis."""
        csf_vec = np.zeros(self.basis_size, dtype=self.dtype)
        # this automatically converts a spin-impure det_vec into an eigenfunction of target S2
        self.spin_adapter.det_C_to_csf_C(np.ascontiguousarray(det_vec), csf_vec)
        return csf_vec

    def _det_indices(self, dets, owner):
        """The positions of dets in the CI vector, raising if one is not in the CI space."""
        indices = []
        for d in dets:
            try:
                indices.append(self.ci_strings.determinant_index(d))
            except (IndexError, RuntimeError):
                raise ValueError(
                    f"{owner} has the determinant {d.str(self.norb)}, which is not in "
                    f"the CI space of {self.state}."
                ) from None
        return indices

    def _basis_indices(self, dets, owner):
        """
        The indices of the CSFs corresponding to all configurations of the determinants.
        e.g., for a triplet solver, |ab> corresponds to configuration |11> which then
        correspnods to a CSF of (|ab> + |ba>)/sqrt(2) with some index.
        For a singlet solver, the same determinant would correspond to the same configuration
        but that configuration would correspond to a CSF of (|ab> - |ba>)/sqrt(2) with some
        other index.
        """
        self._det_indices(dets, owner)
        ranges = {self.spin_adapter.configuration_csf_range(d) for d in dets}
        indices = sorted(i for start, stop in ranges for i in range(start, stop))
        return np.array(indices, dtype=int)

    def _ci_vector_guess_vectors(self):
        """ci_vector_guess as orthonormal columns in the CSF basis."""
        vectors = []
        for root, det_coeffs in self.ci_vector_guess:
            det_vec = np.zeros(self.ndet, dtype=self.dtype)
            dets, coeffs = zip(*det_coeffs.items())
            indices = self._det_indices(dets, f"ci_vector_guess[{root}]")
            det_vec[indices] = list(coeffs)
            vec = self._det_C_to_basis_C(det_vec)
            vec_norm = np.linalg.norm(vec)
            if vec_norm < 1e-8 * np.linalg.norm(det_vec):
                raise ValueError(
                    f"The guess for root {root} has no component of multiplicity "
                    f"{self.state.multiplicity}."
                )
            vectors.append((root, vec / vec_norm))
        return orthonormalize_guesses(vectors)

    def csf_C_to_det_C(self, csf_vec):
        """
        Convert a CI vector in the CSF basis to the determinant basis.

        Parameters
        ----------
        csf_vec : NDArray
            CI vector in the CSF basis.

        Returns
        -------
        NDArray
            CI vector in the determinant basis.
        """
        det_vec = np.zeros((self.ndet))
        self.spin_adapter.csf_C_to_det_C(csf_vec, det_vec)
        return det_vec

    def _root_vector_det(self, root: int):
        """CI vector for ``root`` in the determinant basis."""
        return self.csf_C_to_det_C(self.evecs[:, root])

    _rdm_orders: ClassVar[tuple[int, ...]] = (1, 2, 3)
    _rdm_spin_types: ClassVar[tuple[str, ...]] = ("sd", "sf")
    _cumulant_orders: ClassVar[tuple[int, ...]] = (2, 3)
    _cumulant_spin_types: ClassVar[tuple[str, ...]] = ("sf",)

    def make_rdm(
        self,
        left_root: int,
        right_root: int | None = None,
        *,
        order: Literal[1, 2, 3],
        spin_type: Literal["sd", "sf"],
    ):
        """
        Make the RDM of the given order and representation for two CI roots.

        Parameters
        ----------
        left_root : int
            the CI root for the bra state.
        right_root : int | None, optional (default=left_root)
            the CI root for the ket state.
        order : int
            The RDM order (1, 2, or 3).
        spin_type : str
            "sd" (spin-dependent) or "sf" (spin-free). The aliases "spin_dependent",
            "spin-dependent", "spin_free", and "spin-free" are also accepted.

        Returns
        -------
        NDArray or tuple[NDArray, ...]
            spin_type=sd -> (a, b) at order 1, (aa, ab, bb) at order 2, and
            (aaa, aab, abb, bbb) at order 3; spin_type=sf -> a single full tensor.
        """
        spin_type = validate_single_state_rdm(
            self,
            left_root,
            right_root,
            order,
            self._rdm_orders,
            spin_type,
            self._rdm_spin_types,
        )
        left_ci_vec_det = self.csf_C_to_det_C(self.evecs[:, left_root])
        right_ci_vec_det = (
            left_ci_vec_det
            if right_root is None
            else self.csf_C_to_det_C(self.evecs[:, right_root])
        )
        sb = self.ci_sigma_builder
        if spin_type == "sd":
            if order == 1:
                return (
                    sb.a_1rdm(left_ci_vec_det, right_ci_vec_det),
                    sb.b_1rdm(left_ci_vec_det, right_ci_vec_det),
                )
            if order == 2:
                return (
                    sb.aa_2rdm(left_ci_vec_det, right_ci_vec_det),
                    sb.ab_2rdm(left_ci_vec_det, right_ci_vec_det),
                    sb.bb_2rdm(left_ci_vec_det, right_ci_vec_det),
                )
            return (
                sb.aaa_3rdm(left_ci_vec_det, right_ci_vec_det),
                sb.aab_3rdm(left_ci_vec_det, right_ci_vec_det),
                sb.abb_3rdm(left_ci_vec_det, right_ci_vec_det),
                sb.bbb_3rdm(left_ci_vec_det, right_ci_vec_det),
            )
        # spin_type == "sf"
        if order == 1:
            return sb.sf_1rdm(left_ci_vec_det, right_ci_vec_det)
        if order == 2:
            return sb.sf_2rdm(left_ci_vec_det, right_ci_vec_det)
        return sb.sf_3rdm(left_ci_vec_det, right_ci_vec_det)

    def make_cumulant(
        self,
        root: int,
        *,
        order: Literal[2, 3],
        spin_type: Literal["sf", "so"],
    ):
        """
        Make the cumulant of the given order for one CI root.

        Parameters
        ----------
        root : int
            the CI root.
        order : int
            The cumulant order (2 or 3).
        spin_type : str
            "sf" (spin-free) or "so" (spin-orbital), depending on the backend. The
            aliases "spin_free", "spin-free", "spin_orbital", "spin-orbital", and
            "spinorbital" are also accepted.

        Returns
        -------
        NDArray
            The cumulant.
        """
        spin_type = validate_single_state_rdm(
            self,
            root,
            None,
            order,
            self._cumulant_orders,
            spin_type,
            self._cumulant_spin_types,
        )
        return make_cumulant_from_rdms(self, root, order=order, spin_type=spin_type)

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
        spin_type = "so" if self.two_component else "sf"
        no = np.zeros((self.norb, self.nroot))
        for i in range(self.nroot):
            no[:, i] = np.linalg.eigvalsh(
                self.make_rdm(i, order=1, spin_type=spin_type)
            )[::-1]

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
            ci_det = self._root_vector_det(i)
            argsort = np.argsort(np.abs(ci_det))[::-1]  # descending in absolute coeff
            for j in range(n):
                if j < len(argsort):
                    top_dets.append((self.dets[argsort[j]], ci_det[argsort[j]]))
            top_dets_per_root.append(top_dets)

        return top_dets_per_root

    def reset_eigensolver(self, guess=None):
        """
        Drop the eigensolver so the next run rebuilds it.

        Parameters
        ----------
        guess : NDArray, optional
            Guess vectors for the next run, shape ``(basis_size, nguess)``. If
            None, the guess is built from the diagonal of the Hamiltonian.
        """
        self.eigensolver = None
        self.guess = guess
        self.rebuild_guess = True


@dataclass
class CISolver(CIBase):
    """
    A general configuration interaction (CI) solver class.
    This solver is can be called iteratively, e.g., in a MCSCF loop or a DSRG reference relaxation loop.

    Parameters
    ----------
    ci_params : CIParams, optional
        Parameters for the CI solver. If not provided, default parameters are used.
    davidson_liu_params : DavidsonLiuParams, optional
        Parameters for the Davidson-Liu eigensolver. If not provided, default parameters are used.
    do_test_rdms : bool, optional, default=False
        If True, compute and test the reduced density matrices (RDMs) after the CI calculation.
    log_level : int, optional
        The logging level for the CI solver. Defaults to ``logger.VERBOSITY_DEBUG``.
    ci_vector_guess : list[list[tuple[Determinant, float | complex]]], optional
        Initial CI vectors, one per root in absolute root order.
        Each guess is projected onto the configuration state functions (CSFs) of the target multiplicity and normalized.
        The Davidson-Liu solver starts from these vectors, so ``guess_per_root`` and ``ndets_per_guess`` don't apply.
        This option is incompatible with ``guess_dets`` or ``pinned_guess_dets``.
        ``ci_algorithm="exact"`` ignores it.
    guess_dets : None | list[Determinant] | list[list[Determinant] | None], optional
        The candidate determinants for the guess space: a list of ``Determinant``s
        if there is one state, or one entry per state, each a list of ``Determinant``s or None.
        If None, every CSF is a candidate.
        The candidates are narrowed down by diagonal energy, or by distance from ``ci_params.energy_shift``, to at most ``ndets_per_guess * guess_per_root * nroot``.
        The guess vectors are the lowest ``guess_per_root * nroot`` eigenvectors of the Hamiltonian in that space.
    pinned_guess_dets : None | list[Determinant] | list[list[Determinant] | None], optional
        Determinants whose CSFs always join the guess space, without the narrowing, in
        the same per-state format as ``guess_dets``.

    Attributes
    ----------
    sub_solvers : list[_CISingleStateSolver]
        A list of CI solvers for each state in the state-averaged CI (each solver for a different spin/GAS restriction).
    evals_per_solver : list[NDArray]
        The eigenvalues (energies) computed by each sub-solver.
    evals_flat, E : NDArray
        The flattened array of eigenvalues from all sub-solvers.
    E_avg : float
        The average energy computed from the state-averaged CI roots.
    """

    orbital_rotation_invariant: ClassVar[bool] = True

    ci_params: CIParams = field(default_factory=CIParams)
    davidson_liu_params: DavidsonLiuParams = field(default_factory=DavidsonLiuParams)
    do_test_rdms: bool = False

    # Active-space integral class
    _integrals_cls: ClassVar[type] = RestrictedMOIntegrals
    # Single state solver class
    _ss_solver_cls: ClassVar[type] = _CISingleStateSolver

    _rdm_orders: ClassVar[tuple[int, ...]] = (1, 2, 3)
    _rdm_spin_types: ClassVar[tuple[str, ...]] = ("sd", "sf")
    _rdm_cross_state_orders: ClassVar[tuple[int, ...]] = (1,)
    _cumulant_orders: ClassVar[tuple[int, ...]] = (2, 3)
    _cumulant_spin_types: ClassVar[tuple[str, ...]] = ("sf",)

    def make_rdm(
        self,
        left_root: int,
        right_root: int | None = None,
        *,
        order: Literal[1, 2, 3],
        spin_type: Literal["sd", "sf"],
    ):
        """
        Make the RDM of the given order and representation for two absolute CI roots.
        Cross-state (transition) requests are only supported at order=1.

        Parameters
        ----------
        left_root : int
            the absolute CI root for the bra state.
        right_root : int | None, optional (default=left_root)
            the absolute CI root for the ket state.
        order : int
            The RDM order (1, 2, or 3).
        spin_type : str
            "sd" (spin-dependent) or "sf" (spin-free). The aliases "spin_dependent",
            "spin-dependent", "spin_free", and "spin-free" are also accepted.

        Returns
        -------
        NDArray or tuple[NDArray, ...]
            spin_type=sd -> (a, b) at order 1, (aa, ab, bb) at order 2, and
            (aaa, aab, abb, bbb) at order 3; spin_type=sf -> a single full tensor.
        """
        left_state, right_state, left_root_in_state, right_root_in_state, spin_type = (
            self._validate_rdm_inputs(
                left_root,
                right_root,
                order,
                self._rdm_orders,
                spin_type,
                self._rdm_spin_types,
                self._rdm_cross_state_orders,
            )
        )
        if left_state == right_state:
            return self.sub_solvers[left_state].make_rdm(
                left_root_in_state,
                right_root_in_state,
                order=order,
                spin_type=spin_type,
            )
        # Cross-state: the validator above only lets this fall through when order == 1.
        left_solver = self.sub_solvers[left_state]
        right_solver = self.sub_solvers[right_state]
        left_sb = left_solver.ci_sigma_builder
        right_sb = right_solver.ci_sigma_builder
        C_left = left_solver.csf_C_to_det_C(left_solver.evecs[:, left_root_in_state])
        C_right = right_solver.csf_C_to_det_C(
            right_solver.evecs[:, right_root_in_state]
        )
        if spin_type == "sd":
            return (
                left_sb.a_1trdm(right_sb, C_left, C_right),
                left_sb.b_1trdm(right_sb, C_left, C_right),
            )
        return left_sb.sf_1trdm(right_sb, C_left, C_right)


@dataclass
class CI(ActiveSpaceDriver, Method):
    """
    Driver for a single CI calculation, i.e. one not run in a loop.
    This class is agnostic to the solver.

    Parameters
    ----------
    ci_solver : CIBase | RelCIBase
        The solver to drive, e.g. ``CISolver``, ``RelCISolver``,
        ``SelectedCISolver`` or ``RelSelectedCISolver``.
    final_orbitals : str, optional, default="original"
        Orbitals to leave behind: "original", "semicanonical", or "natural".
    do_transition_dipole : bool, optional, default=False
        Whether to compute transition dipoles and oscillator strengths.
    die_if_not_converged : bool, optional, default=True
        Whether a solver that fails to converge raises instead of warning. Set on
        the driver rather than the solver: a one-shot calculation has no later
        iteration to recover from.

    Attributes
    ----------
    E : float
        The state-averaged energy.
    E_ci : NDArray
        A copy of the per-root energies.
    """

    ci_solver: CIBase

    final_orbitals: FinalOrbitals = "original"
    do_transition_dipole: bool = False
    die_if_not_converged: bool = True
