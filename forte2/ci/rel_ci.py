from dataclasses import dataclass, field
from typing import ClassVar, Literal
import numpy as np

from forte2.lib import sparse_ops
from forte2.lib.sparse_ops import SparseState
from forte2.lib.ci_helpers import (
    CIStrings,
    RelCISigmaBuilder,
)
from forte2.helpers.comparisons import approx
from forte2.base_classes import RelCIBase
from forte2.base_classes.params import DavidsonLiuParams, CIParams
from forte2.helpers import logger
from forte2.jkbuilder import SpinorbitalIntegrals
from forte2.orbitals import FinalOrbitals, validate_final_orbitals
from .ci_utils import (
    pretty_print_ci_summary,
    pretty_print_ci_nat_occ_numbers,
    pretty_print_ci_dets,
    pretty_print_ci_transition_props,
    validate_single_state_rdm,
)
from .ci import _CISingleStateSolver


@dataclass
class _RelCISingleStateSolver(_CISingleStateSolver):
    """
    Two-component (relativistic) CI solver for a single state.
    """

    two_component: ClassVar[bool] = True
    dtype: ClassVar[type] = complex
    _sigma_builder_cls: ClassVar[type] = RelCISigmaBuilder
    _allowed_algorithms: ClassVar[tuple] = (
        "hz",
        "harrison-zarrabian",
        "sparse",
        "exact",
    )

    def _make_sigma_builder_obj(self):
        return self._sigma_builder_cls(
            self.ci_strings,
            self.ints.E.real,
            self.ints.H,
            self.ints.V,
            self.log_level,
        )

    def _update_sigma_builder_ints(self):
        # RelCISigmaBuilder.set_Hamiltonian's E parameter is a plain double.
        self.ci_sigma_builder.set_Hamiltonian(
            self.ints.E.real, self.ints.H, self.ints.V
        )

    def _make_ci_strings(self):
        _nactel = self.state.nel - self.ncore
        assert (
            _nactel >= 0
        ), f"Number of active electrons {_nactel} must be non-negative."
        # all active electrons live in the alpha (spinor) string; nb = 0
        return CIStrings(
            _nactel,
            0,
            self.state.symmetry,
            self.active_orbsym,
            self.state.gas_min,
            self.state.gas_max,
        )

    def _log_ci_strings_info(self):
        logger.log(f"\nNumber of electrons: {self.ci_strings.na}", self.log_level)
        logger.log(f"Number of strings: {self.ci_strings.nas}", self.log_level)

    def _setup_basis(self):
        # no spin adaptation for 2c: the determinant basis is the variational basis
        self.ndet = self.ci_strings.ndet
        self.basis_size = self.ndet
        self.dets = self.ci_strings.make_determinants()
        self.b_det = np.zeros((self.ndet,), dtype=self.dtype)
        self.sigma_det = np.zeros((self.ndet,), dtype=self.dtype)

    def _form_hdiag(self):
        return self.ci_sigma_builder.form_Hdiag(self.dets)

    def _make_sigma_builder(self):
        if self.ci_params.ci_algorithm.lower() == "sparse":
            ham = sparse_ops.sparse_operator_hamiltonian(
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
                    Hpsi = sparse_ops.apply_op(ham, psi, screen_thresh=1e-100)
                    for idet in range(self.ndet):
                        sigma_block[idet, istate] = Hpsi[self.dets[idet]]

            return sigma_builder

        def sigma_builder(Bblock, Sblock):
            ncols = Bblock.shape[1]
            for i in range(ncols):
                # copies ensure contiguous arrays are passed to C++
                self.b_det = Bblock[:, i].copy()
                self.ci_sigma_builder.Hamiltonian(self.b_det, self.sigma_det)
                Sblock[:, i] = self.sigma_det.copy()

        return sigma_builder

    def _log_sigma_build_times(self):
        # the 2c builder does not expose per-block timings
        pass

    def _basis_matrix_element(self, I, J):
        return self.ci_sigma_builder.slater_rules(self.dets, I, J)

    def _build_full_hamiltonian(self):
        H = np.zeros((self.ndet,) * 2, dtype=self.dtype)
        for i in range(self.ndet):
            for j in range(i + 1):
                H[i, j] = self.ci_sigma_builder.slater_rules(self.dets, i, j)
                H[j, i] = np.conj(H[i, j])
        return H

    def _root_vector_det(self, root: int):
        # already in the determinant basis; copy for contiguity
        return self.evecs[:, root].copy()

    def _test_rdms(self):
        logger.log("\nComputing RDMs from CI vectors.\n", self.log_level)
        for root in range(self.nroot):
            rdm1 = self.make_rdm(root, order=1, spin_type="so")
            rdm2 = self.make_rdm(root, order=2, spin_type="so")

            rdms_energy = self.ints.E
            rdms_energy += np.einsum("ij,ij", rdm1, self.ints.H)
            rdms_energy += 0.5 * np.einsum("ijkl,ijkl", rdm2, self.ints.V)
            logger.log(f"CI energy from RDMs: {rdms_energy:.12f} Eh", self.log_level)

            assert self.E[root] == approx(rdms_energy)

            logger.log(
                f"RDMs for root {root} validated successfully.\n", self.log_level
            )

    def csf_C_to_det_C(self, csf_vec):
        raise NotImplementedError(
            "There is no CSF basis in the two-component (spinor) case; the CI "
            "vectors are already in the determinant basis."
        )

    _rdm_orders: ClassVar[tuple[int, ...]] = (1, 2, 3)
    _rdm_spin_types: ClassVar[tuple[str, ...]] = ("so",)
    _cumulant_orders: ClassVar[tuple[int, ...]] = (2, 3)
    _cumulant_spin_types: ClassVar[tuple[str, ...]] = ("so",)

    def make_rdm(
        self,
        left_root: int,
        right_root: int | None = None,
        *,
        order: Literal[1, 2, 3],
        spin_type: Literal["so"],
    ):
        """
        Make the spin-orbital RDM of the given order for two CI roots. For two-component
        CI only.

        Parameters
        ----------
        left_root : int
            the CI root for the bra state.
        right_root : int | None, optional (default=left_root)
            the CI root for the ket state.
        order : int
            The RDM order (1, 2, or 3).
        spin_type : str
            Must be "so" (spin-orbital). The aliases "spin_orbital", "spin-orbital",
            and "spinorbital" are also accepted.

        Returns
        -------
        NDArray
            The spin-orbital RDM.
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
        if right_root is None:
            right_root = left_root
        # copy to ensure contiguous arrays are passed to the sigma builder
        Cl = self.evecs[:, left_root].copy()
        Cr = self.evecs[:, right_root].copy()
        sb = self.ci_sigma_builder
        if order == 1:
            return sb.so_1rdm(Cl, Cr)
        if order == 2:
            return sb.so_2rdm(Cl, Cr)
        return sb.so_3rdm(Cl, Cr)


@dataclass
class RelCISolver(RelCIBase):
    """
    A general two-component configuration interaction (2C-CI) solver class.

    Parameters
    ----------
    davidson_liu_params : DavidsonLiuParams, optional
        Parameters for the Davidson-Liu eigensolver. If not provided, default parameters are used.
    ci_params : CIParams, optional
        Parameters for the CI solver. If not provided, default parameters are used.
    do_test_rdms : bool, optional, default=False
        If True, compute and test the reduced density matrices (RDMs) after the CI calculation.
    log_level : int, optional
        The logging level for the CI solver. Defaults to the global logger's verbosity level.
    """

    orbital_rotation_invariant: ClassVar[bool] = True

    davidson_liu_params: DavidsonLiuParams = field(default_factory=DavidsonLiuParams)
    ci_params: CIParams = field(default_factory=CIParams)

    do_test_rdms: bool = False
    log_level: int = field(default=logger.get_verbosity_level() + 1)

    # Active-space integral class used by CIBase._make_active_space_ints
    _integrals_cls: ClassVar[type] = SpinorbitalIntegrals
    # Per-state worker class used by CIBase._startup
    _ss_solver_cls: ClassVar[type] = _RelCISingleStateSolver

    _rdm_orders: ClassVar[tuple[int, ...]] = (1, 2, 3)
    _rdm_spin_types: ClassVar[tuple[str, ...]] = ("so",)
    # spinor RDMs between roots of *different* states are not implemented
    _rdm_cross_state_orders: ClassVar[tuple[int, ...]] = ()
    _cumulant_orders: ClassVar[tuple[int, ...]] = (2, 3)
    _cumulant_spin_types: ClassVar[tuple[str, ...]] = ("so",)

    def make_rdm(
        self,
        left_root: int,
        right_root: int | None = None,
        *,
        order: Literal[1, 2, 3],
        spin_type: Literal["so"],
    ):
        """Spin-orbital RDM of the given order for two absolute CI roots (same-state only)."""
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
        return self.sub_solvers[left_state].make_rdm(
            left_root_in_state, right_root_in_state, order=order, spin_type=spin_type
        )


@dataclass
class RelCI(RelCISolver):
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
