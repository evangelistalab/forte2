from dataclasses import dataclass, field
from typing import ClassVar

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
from forte2.base_classes import CIBase
from forte2.base_classes.params import DavidsonLiuParams, CIParams
from forte2.helpers import logger
from forte2.jkbuilder import RestrictedMOIntegrals
from forte2.orbitals import FinalOrbitals, validate_final_orbitals
from .ci_utils import (
    pretty_print_gas_info,
    pretty_print_ci_summary,
    pretty_print_ci_nat_occ_numbers,
    pretty_print_ci_dets,
    pretty_print_ci_transition_props,
)


@dataclass
class _CISingleStateSolver:
    """
    A general configuration interaction (CI) solver class for a single `State`.
    Although possible, is not recommended to instantiate this class directly.
    Consider using the `CI` class instead.

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

    mo_space: MOSpace
    state: State
    ints: RestrictedMOIntegrals
    nroot: int
    active_orbsym: list[int]
    ci_params: CIParams = field(default_factory=CIParams)
    davidson_liu_params: DavidsonLiuParams = field(default_factory=DavidsonLiuParams)
    do_test_rdms: bool = False
    log_level: int = field(default=logger.get_verbosity_level())
    die_if_not_converged: bool = False

    ### Non-init attributes
    rebuild_guess: bool = field(default=True, init=False)
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
        return self._sigma_builder_cls(
            self.ci_strings,
            self.ints.E,
            self.ints.H,
            self.ints.V,
            self.log_level,
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

    def run(self):
        if not self.executed:
            self._ci_solver_startup()

        # Create the sigma builder from the CI strings and integrals. This object
        # handles some temporary memory deallocated at destruction and is used to
        # compute the Hamiltonian matrix elements in the determinant basis.
        self.ci_sigma_builder = self._make_sigma_builder_obj()
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

    def _select_algorithm(self):
        """Configure the sigma builder's algorithm and return its name."""
        self.ci_sigma_builder.set_algorithm(self.ci_params.ci_algorithm.lower())
        return self.ci_sigma_builder.get_algorithm()

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
            f"Using CI algorithm: {self._select_algorithm()}",
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
                logger.log(
                    f"\nDavidson-Liu solver did not converge in {self.eigensolver.maxiter} iterations.\n",
                    self.log_level,
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

    def _basis_matrix_element(self, I, J):
        """<I|H|J> between two vectors of the variational (CSF) basis."""
        return self.ci_sigma_builder.slater_rules_csf(
            self.dets, self.spin_adapter, I, J
        )

    def _build_guess_vectors(self, Hdiag):
        """Build the guess vectors for the CI calculation."""
        # determine the number of guess vectors
        self.num_guess_states = min(
            self.davidson_liu_params.guess_per_root * self.nroot, self.basis_size
        )
        logger.log(f"Number of guess states: {self.num_guess_states}", self.log_level)
        nguess_dets = min(
            self.davidson_liu_params.ndets_per_guess * self.num_guess_states,
            self.basis_size,
        )
        logger.log(f"Number of guess basis: {nguess_dets}", self.log_level)

        # find the indices of the elements of Hdiag with the lowest values
        if self.ci_params.energy_shift is not None:
            indices = np.argsort(np.abs(Hdiag - self.ci_params.energy_shift))[
                :nguess_dets
            ]
        else:
            indices = np.argsort(Hdiag)[:nguess_dets]

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
        _, evecs_guess = np.linalg.eigh(Hguess)

        # Select the lowest eigenvalues and their corresponding eigenvectors
        guess_mat = np.zeros((self.basis_size, self.num_guess_states), dtype=self.dtype)
        for i in range(self.num_guess_states):
            guess = evecs_guess[:, i]
            for j, d in enumerate(indices):
                guess_mat[d, i] = guess[j]

        self.eigensolver.add_guesses(guess_mat)

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

    def make_1rdm(self, left_root: int, right_root: int | None = None):
        """
        Make the one-particle RDM for two CI roots.
        Spin-free for non-relativistic CI (use make_sd_* for spin-dependent RDMs).
        Spin(or)-orbital for two-component CI.

        Parameters
        ----------
        left_root : int
            the CI root for the bra state.
        right_root : int | None, optional (default=left_root)
            the CI root for the ket state.

        Returns
        -------
        NDArray
            One-particle RDM.
        """
        return self.make_sf_1rdm(left_root, right_root)

    def make_2rdm(self, left_root: int, right_root: int | None = None):
        """
        Make the two-particle RDM for two CI roots.
        Spin-free for non-relativistic CI (use make_sd_* for spin-dependent RDMs).
        Spin(or)-orbital for two-component CI.

        Parameters
        ----------
        left_root : int
            the CI root for the bra state.
        right_root : int | None, optional (default=left_root)
            the CI root for the ket state.

        Returns
        -------
        NDArray
            Two-particle RDM.
        """
        return self.make_sf_2rdm(left_root, right_root)

    def make_2cumulant(self, left_root: int, right_root: int | None = None):
        """
        Make the two-particle cumulant for two CI roots.
        Spin-free for non-relativistic CI (use make_sd_* for spin-dependent RDMs).
        Spin(or)-orbital for two-component CI.

        Parameters
        ----------
        left_root : int
            the CI root for the bra state.
        right_root : int | None, optional (default=left_root)
            the CI root for the ket state.

        Returns
        -------
        NDArray
            Two-particle cumulant.
        """
        return self.make_sf_2cumulant(left_root, right_root)

    def make_3rdm(self, left_root: int, right_root: int | None = None):
        """
        Make the three-particle RDM for two CI roots.
        Spin-free for non-relativistic CI (use make_sd_* for spin-dependent RDMs).
        Spin(or)-orbital for two-component CI.

        Parameters
        ----------
        left_root : int
            the CI root for the bra state.
        right_root : int | None, optional (default=left_root)
            the CI root for the ket state.

        Returns
        -------
        NDArray
            Three-particle RDM.
        """
        return self.make_sf_3rdm(left_root, right_root)

    def make_3cumulant(self, left_root: int, right_root: int | None = None):
        """
        Make the three-particle cumulant for two CI roots.
        Spin-free for non-relativistic CI (use make_sd_* for spin-dependent RDMs).
        Spin(or)-orbital for two-component CI.

        Parameters
        ----------
        left_root : int
            the CI root for the bra state.
        right_root : int | None, optional (default=left_root)
            the CI root for the ket state.

        Returns
        -------
        NDArray
            Three-particle cumulant.
        """
        return self.make_sf_3cumulant(left_root, right_root)

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
        left_ci_vec_det = self.csf_C_to_det_C(self.evecs[:, left_root])
        if right_root is None:
            right_ci_vec_det = left_ci_vec_det
        else:
            right_ci_vec_det = self.csf_C_to_det_C(self.evecs[:, right_root])
        a = self.ci_sigma_builder.a_1rdm(left_ci_vec_det, right_ci_vec_det)
        b = self.ci_sigma_builder.b_1rdm(left_ci_vec_det, right_ci_vec_det)
        return a, b

    def make_sd_2rdm(self, left_root: int, right_root: int | None = None):
        """
        Make the spin-dependent two-particle RDMs (aa, ab, bb) for two CI roots.

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
        left_ci_vec_det = self.csf_C_to_det_C(self.evecs[:, left_root])
        if right_root is None:
            right_ci_vec_det = left_ci_vec_det
        else:
            right_ci_vec_det = self.csf_C_to_det_C(self.evecs[:, right_root])
        aa = self.ci_sigma_builder.aa_2rdm(left_ci_vec_det, right_ci_vec_det)
        ab = self.ci_sigma_builder.ab_2rdm(left_ci_vec_det, right_ci_vec_det)
        bb = self.ci_sigma_builder.bb_2rdm(left_ci_vec_det, right_ci_vec_det)
        return aa, ab, bb

    def make_sd_3rdm(self, left_root: int, right_root: int | None = None):
        """
        Make the spin-dependent three-particle RDMs (aaa, aab, abb, bbb) for two CI roots.

        Parameters
        ----------
        left_root : int
            the CI root for the bra state.
        right_root : int | None, optional (default=left_root)
            the CI root for the ket state.

        Returns
        -------
        tuple[NDArray, NDArray, NDArray, NDArray]:
            Spin-dependent three-particle RDMs (aaa, aab, abb, bbb).
        """
        left_ci_vec_det = self.csf_C_to_det_C(self.evecs[:, left_root])
        if right_root is None:
            right_ci_vec_det = left_ci_vec_det
        else:
            right_ci_vec_det = self.csf_C_to_det_C(self.evecs[:, right_root])

        aaa = self.ci_sigma_builder.aaa_3rdm(left_ci_vec_det, right_ci_vec_det)
        aab = self.ci_sigma_builder.aab_3rdm(left_ci_vec_det, right_ci_vec_det)
        abb = self.ci_sigma_builder.abb_3rdm(left_ci_vec_det, right_ci_vec_det)
        bbb = self.ci_sigma_builder.bbb_3rdm(left_ci_vec_det, right_ci_vec_det)
        return aaa, aab, abb, bbb

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
        left_ci_vec_det = self.csf_C_to_det_C(self.evecs[:, left_root])
        if right_root is None:
            right_ci_vec_det = left_ci_vec_det
        else:
            right_ci_vec_det = self.csf_C_to_det_C(self.evecs[:, right_root])
        return self.ci_sigma_builder.sf_1rdm(left_ci_vec_det, right_ci_vec_det)

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
            Spin-free two-particle RDM.
        """
        left_ci_vec_det = self.csf_C_to_det_C(self.evecs[:, left_root])
        if right_root is None:
            right_ci_vec_det = left_ci_vec_det
        else:
            right_ci_vec_det = self.csf_C_to_det_C(self.evecs[:, right_root])
        return self.ci_sigma_builder.sf_2rdm(left_ci_vec_det, right_ci_vec_det)

    def make_sf_3rdm(self, left_root: int, right_root: int | None = None):
        """
        Make the spin-free three-particle RDM for two CI roots.

        Parameters
        ----------
        left_root : int
            the CI root for the bra state.
        right_root : int | None, optional (default=left_root)
            the CI root for the ket state.

        Returns
        -------
        NDArray
            Spin-free three-particle RDM.
        """
        left_ci_vec_det = self.csf_C_to_det_C(self.evecs[:, left_root])
        if right_root is None:
            right_ci_vec_det = left_ci_vec_det
        else:
            right_ci_vec_det = self.csf_C_to_det_C(self.evecs[:, right_root])
        return self.ci_sigma_builder.sf_3rdm(left_ci_vec_det, right_ci_vec_det)

    def make_sf_2cumulant(self, left_root: int, right_root: int | None = None):
        """
        Make the spin-free cumulant of the two-particle RDM for two CI roots.

        Parameters
        ----------
        left_root : int
            the CI root for the bra state.
        right_root : int | None, optional (default=left_root)
            the CI root for the ket state.

        Returns
        -------
        NDArray
            Spin-free cumulant of the two-particle RDM.
        """
        left_ci_vec_det = self.csf_C_to_det_C(self.evecs[:, left_root])
        if right_root is None:
            right_ci_vec_det = left_ci_vec_det
        else:
            right_ci_vec_det = self.csf_C_to_det_C(self.evecs[:, right_root])
        return self.ci_sigma_builder.sf_2cumulant(left_ci_vec_det, right_ci_vec_det)

    def make_sf_3cumulant(self, left_root: int, right_root: int | None = None):
        """
        Make the spin-free cumulant of the three-particle RDM for two CI roots.

        Parameters
        ----------
        left_root : int
            the CI root for the bra state.
        right_root : int | None, optional (default=left_root)
            the CI root for the ket state.

        Returns
        -------
        NDArray
            Spin-free cumulant of the three-particle RDM.
        """
        left_ci_vec_det = self.csf_C_to_det_C(self.evecs[:, left_root])
        if right_root is None:
            right_ci_vec_det = left_ci_vec_det
        else:
            right_ci_vec_det = self.csf_C_to_det_C(self.evecs[:, right_root])
        return self.ci_sigma_builder.sf_3cumulant(left_ci_vec_det, right_ci_vec_det)

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
        no = np.zeros((self.norb, self.nroot))
        for i in range(self.nroot):
            no[:, i] = np.linalg.eigvalsh(self.make_1rdm(i))[::-1]

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

    def reset_eigensolver(self):
        self.eigensolver = None
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
        The logging level for the CI solver. Defaults to the global logger's verbosity level.

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
    # If used as a solver, log at warning level
    log_level: int = field(default=logger.get_verbosity_level() + 1)

    # Active-space integral class
    _integrals_cls: ClassVar[type] = RestrictedMOIntegrals
    # Single state solver class
    _ss_solver_cls: ClassVar[type] = _CISingleStateSolver

    def make_sd_1rdm(
        self,
        left_root: int,
        right_root: int | None = None,
    ) -> tuple[NDArray, NDArray]:
        """
        Make the spin-dependent 1-RDMs
        """
        left_state, right_state, left_root_in_state, right_root_in_state = (
            self._validate_rdm_inputs(left_root, right_root)
        )
        if left_state == right_state:
            return self.sub_solvers[left_state].make_sd_1rdm(
                left_root_in_state, right_root_in_state
            )
        else:
            left_solver = self.sub_solvers[left_state]
            right_solver = self.sub_solvers[right_state]
            left_sb = left_solver.ci_sigma_builder
            right_sb = right_solver.ci_sigma_builder

            C_left = left_solver.csf_C_to_det_C(
                left_solver.evecs[:, left_root_in_state]
            )
            C_right = right_solver.csf_C_to_det_C(
                right_solver.evecs[:, right_root_in_state]
            )

            a_1trdm = left_sb.a_1trdm(right_sb, C_left, C_right)
            b_1trdm = left_sb.b_1trdm(right_sb, C_left, C_right)
            return a_1trdm, b_1trdm

    def make_sd_2rdm(
        self,
        left_root: int,
        right_root: int | None = None,
    ) -> tuple[NDArray, NDArray, NDArray]:
        left_state, right_state, left_root_in_state, right_root_in_state = (
            self._validate_rdm_inputs(left_root, right_root)
        )

        if left_state != right_state:
            raise ValueError(
                f"Cross-state 2-RDMs are not supported. Got left_root in state {left_state} and right_root in state {right_state}."
            )
        return self.sub_solvers[left_state].make_sd_2rdm(
            left_root_in_state, right_root_in_state
        )

    def make_sd_3rdm(
        self,
        left_root: int,
        right_root: int | None = None,
    ) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        left_state, right_state, left_root_in_state, right_root_in_state = (
            self._validate_rdm_inputs(left_root, right_root)
        )

        if left_state != right_state:
            raise ValueError(
                f"Cross-state 3-RDMs are not supported. Got left_root in state {left_state} and right_root in state {right_state}."
            )
        return self.sub_solvers[left_state].make_sd_3rdm(
            left_root_in_state, right_root_in_state
        )

    def make_sf_1rdm(
        self,
        left_root: int,
        right_root: int | None = None,
    ) -> NDArray:
        left_state, right_state, left_root_in_state, right_root_in_state = (
            self._validate_rdm_inputs(left_root, right_root)
        )
        if left_state == right_state:
            return self.sub_solvers[left_state].make_sf_1rdm(
                left_root_in_state, right_root_in_state
            )
        else:
            left_solver = self.sub_solvers[left_state]
            right_solver = self.sub_solvers[right_state]
            left_sb = left_solver.ci_sigma_builder
            right_sb = right_solver.ci_sigma_builder
            C_left = left_solver.csf_C_to_det_C(
                left_solver.evecs[:, left_root_in_state]
            )
            C_right = right_solver.csf_C_to_det_C(
                right_solver.evecs[:, right_root_in_state]
            )
            return left_sb.sf_1trdm(right_sb, C_left, C_right)

    def make_sf_2rdm(
        self,
        left_root: int,
        right_root: int | None = None,
    ) -> NDArray:
        left_state, right_state, left_root_in_state, right_root_in_state = (
            self._validate_rdm_inputs(left_root, right_root)
        )

        if left_state == right_state:
            return self.sub_solvers[left_state].make_sf_2rdm(
                left_root_in_state, right_root_in_state
            )
        else:
            raise ValueError(
                f"Cross-state 2-RDMs are not supported. Got left_root in state {left_state} and right_root in state {right_state}."
            )

    make_1rdm = make_sf_1rdm
    make_2rdm = make_sf_2rdm


@dataclass
class CI(CISolver):
    """
    CI solver specialized for a single CI calculation. (i.e., not used in a loop).
    See `CISolver` for all parameters and attributes.
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
        pretty_print_ci_summary(self.sa_info, self.evals_per_solver)
        self.compute_natural_occupation_numbers()
        pretty_print_ci_nat_occ_numbers(
            self.sa_info,
            self.mo_space,
            self.nat_occs,
            getattr(self, "nat_occs_avg", None),
        )
        top_dets = self.get_top_determinants()
        pretty_print_ci_dets(self.sa_info, self.mo_space, top_dets)

        if self.do_transition_dipole:
            self.compute_transition_properties()
            pretty_print_ci_transition_props(
                self.sa_info,
                self.transition_dipoles,
                self.oscillator_strengths,
                self.evals_per_solver,
            )
