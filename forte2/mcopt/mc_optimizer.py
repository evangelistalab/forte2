from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
import scipy.sparse.linalg as spla
from numpy.typing import NDArray


from forte2.base_classes import (
    ActiveSpaceSolver,
    CIBase,
    RelCIBase,
    Method,
)
from forte2.orbitals import (
    FinalOrbitals,
    check_final_orbital_energy_invariance,
    make_final_orbitals,
    validate_final_orbitals,
)
from forte2.jkbuilder import RestrictedMOIntegrals, SpinorbitalIntegrals
from forte2.lib.ci_helpers import CISigmaBuilder
from forte2.helpers import logger, LBFGS
from forte2.system.basis_utils import BasisInfo
from forte2.system import ModelSystem
from forte2.ci.ci_utils import (
    pretty_print_ci_summary,
    pretty_print_ci_nat_occ_numbers,
    pretty_print_ci_dets,
    pretty_print_ci_transition_props,
)
from forte2.symmetry import real_sph_to_j_adapted
from .orbital_optimizer import OrbOptimizer, RelOrbOptimizer


@dataclass
class MCOptimizerBase(Method):
    """
    Two-step optimizer for multi-configurational wavefunctions.

    Parameters
    ----------
    ci_solver : CIBase | RelCIBase
        The CI solver to use. This should be an instance of a class that inherits from CIBase or RelCIBase.
    active_frozen_orbitals : list[int], optional
        List of active orbital indices to be frozen in the MCSCF optimization.
        If provided, all gradients involving these orbitals will be zeroed out.
    maxiter : int, optional, default=50
        Maximum number of macroiterations.
    e_tol : float, optional, default=1e-8
        Energy convergence tolerance.
    g_tol : float, optional, default=1e-7
        Gradient convergence tolerance.
    die_if_not_converged : bool, optional, default=True
        If True, raises an error if the optimization does not converge.
    freeze_inter_gas_rots : bool, optional, default=False
        Whether to freeze inter-GAS orbital rotations when multiple GASes are defined.
    micro_maxiter : int, optional, default=6
        Maximum number of microiterations for L-BFGS.
    max_rotation : float, optional, default=0.2
        Maximum orbital rotation size for L-BFGS.
    do_transition_dipole : bool, optional, default=False
        Whether to compute and report transition dipole moments at the end of the optimization.
    final_orbitals : str, optional, default="semicanonical"
        Specify the type of final orbitals. Allowed values are:
        - "semicanonical": The average Fock matrix is diagonal within each orbital subspace.
        - "natural": Same as semicanonical, but the active orbitals are natural orbitals
                     and diagonalize the spin- and state-averaged 1-RDM within the CAS
                     subspace or within each of the GAS subspaces.
        - "original": The orbitals are left in the original basis after the optimization.
                      This option is only for debugging purposes and should generally be avoided
                      as the active orbitals will not be uniquely defined and may not be suitable
                      for subsequent calculations.

    Notes
    -----
    See J. Chem. Phys. 152, 074102 (2020) for the current implementation
    of a unified CASSCF/GASSCF gradient and diagonal Hessian.
    The non-GAS part of diagonal Hessian implementation follows Theor. Chem. Acc. 97, 88-95 (1997).
    An earlier implementation (CASSCF only) used J. Chem. Phys. 142, 224103 (2015).
    """

    ci_solver: CIBase | RelCIBase

    active_frozen_orbitals: list[int] = None
    freeze_inter_gas_rots: bool = False

    ### Macroiteration parameters
    maxiter: int = 50
    e_tol: float = 1e-8
    g_tol: float = 1e-7
    die_if_not_converged: bool = True

    # Same sanity-check tolerance CIBase uses for its own final-orbital invariance
    # check; not a dataclass field of MCOptimizerBase's own, so it stays in sync with
    # ActiveSpaceSolver's single source of truth rather than duplicating the literal.
    _final_orbital_energy_tol: ClassVar[float] = ActiveSpaceSolver._final_orbital_energy_tol

    ### L-BFGS solver (microiteration) parameters
    micro_maxiter: int = 6
    max_rotation: float = 0.2

    ### Post-iteration
    do_transition_dipole: bool = False
    final_orbitals: FinalOrbitals = "semicanonical"

    ### Non-init attributes
    converged: bool = field(default=False, init=False)
    executed: bool = field(default=False, init=False)

    def __post_init__(self):
        if not isinstance(self.ci_solver, (CIBase, RelCIBase)):
            raise ValueError("ci_solver must be an instance of CIBase or RelCIBase.")

        validate_final_orbitals(self.final_orbitals)
        
        self.requires = {"system", "mos"}
        self.provides = {"system", "mos", "mo_space"}

    def __call__(self, method):
        self._register_parent_method(method)
        # make sure we don't print the CI output at INFO1 level
        current_verbosity = logger.get_verbosity_level()
        # only log subproblem if the verbosity is higher than INFO1
        if current_verbosity > 3:
            self.ci_solver_verbosity = current_verbosity
        else:
            self.ci_solver_verbosity = current_verbosity + 1
        return self

    def _startup(self):
        if not self.parent_method.executed:
            self.parent_method.run()

        self.system = self.parent_method.system
        self.mos = self.parent_method.mos.copy()
        # make sure to register parent_method
        self.ci_solver = self.ci_solver(self.parent_method)
        # iteration 0: one step of CI optimization to bootstrap the orbital optimization
        self.iter = 0
        self.ci_solver.run()
        self.mo_space = self.ci_solver.mo_space
        self.dtype = self.ci_solver.dtype

        # make the core, active, and virtual spaces contiguous
        # i.e., [core, gas1, gas2, ..., virt]
        perm = self.mo_space.orig_to_contig
        # this is the contiguous coefficient matrix
        self._C = self.mos.C[0][:, perm].copy()
        # core slice does not include frozen orbitals!
        self.core = self.mo_space.docc
        # self.actv will be a list if multiple GASes are defined
        self.actv = self.mo_space.actv
        # virtual slice does not include frozen orbitals!
        self.virt = self.mo_space.uocc

        # check if all active_frozen_orbitals indices are in the active space
        if self.active_frozen_orbitals is not None:
            assert (
                sorted(self.active_frozen_orbitals) == self.active_frozen_orbitals
            ), "Active frozen orbitals must be sorted."

            missing = set(self.active_frozen_orbitals) - set(
                self.mo_space.active_indices
            )
            if missing:
                raise ValueError(
                    f"selected active frozen indices, {sorted(missing)}, are not in the active space {self.mo_space.active_indices}."
                )

        self.nrr = self._get_nonredundant_rotations()

    def run(self):
        """
        Run the two-step orbital-CI optimization.

        Returns
        -------
        self : MCOptimizer
            The instance of the optimizer with the results stored in its attributes.
        """
        self._startup()
        self.Hcore = self.system.ints_hcore()  # hcore in AO basis
        fock_builder = self.system.fock_builder

        # Intialize the two central objects for the two-step orbital-CI optimization:
        # orbital optimizer and CI optimizer
        # the loop simply proceeds as follows:
        # for i in range(max_macro_iter):
        #     1. minimize energy wrt orbital rotations at current CI expansion
        #       (this is typically done iteratively with micro-iterations using L-BFGS)
        #     2. minimize energy wrt CI expansion at current orbitals
        #       (this is just the diagonalization of the active-space CI Hamiltonian)
        _OrbOptimizer = RelOrbOptimizer if self.system.two_component else OrbOptimizer
        self.orb_opt = _OrbOptimizer(
            self._C,
            (self.core, self.actv, self.virt),
            fock_builder,
            self.Hcore,
            self.system.nuclear_repulsion,
            self.nrr,
            compute_active_hessian=self.mo_space.ngas > 1
            and not self.freeze_inter_gas_rots,
        )

        # Initialize the LBFGS solver that finds the optimal orbital
        # at fixed CI expansion using the gradient and diagonal Hessian
        self.lbfgs_solver = LBFGS(
            epsilon=self.g_tol,
            max_dir=self.max_rotation,
            step_length_method="max_correction",
            maxiter=self.micro_maxiter,
            dtype=self.dtype,
        )

        width = 115

        logger.log_info1(self.mo_space)
        logger.log_info1(f"# of nonredundant rotations: {self.nrr.sum()}\n")

        logger.log_info1("Entering orbital optimization loop")
        logger.log_info1("\nConvergence criteria ('.' if satisfied, 'x' otherwise):")
        logger.log_info1(f"  {'1. RMS(grad)':<32} < {self.g_tol:.1e}")
        logger.log_info1(
            f"  {'2. max(abs(E_CI_i - E_CI_old_i))':<32} < {self.e_tol:.1e}"
        )
        logger.log_info1(f"  {'3. abs(E_avg - E_avg_old)':<32} < {self.e_tol:.1e}\n")

        logger.log_info1("=" * width)
        logger.log_info1(
            f'{"Iteration":>10} {"E_avg":>20} {"E_orb":>20} {"ΔE_avg":>12} {"max(ΔE_ci)":>12} {"RMS(grad)":>12} {"#micro":>8} {"Conv":>8}'
        )
        logger.log_info1("-" * width)

        # CI eigenvalues
        self.E_ci = np.array(self.ci_solver.E)
        self.E_ci_old = self.E_ci.copy()
        # Ensemble average energy
        self.E_avg = self.ci_solver.compute_average_energy()
        self.E_avg_old = self.E_avg
        self.E = self.E_avg
        # Energy after orbital optimization
        self.E_orb = self.E_avg
        self.E_orb_old = self.E_orb

        self.g1_act = self.make_average_1rdm()
        g2_act = self.make_average_2rdm()
        # ci_maxiter_save = self.ci_solver.get_maxiter()
        # self.ci_solver.set_maxiter(self.ci_maxiter)

        # Prepare the orbital optimizer
        self.orb_opt.set_rdms(self.g1_act, g2_act)
        self.orb_opt._compute_Fcore()
        self.orb_opt.get_eri_gaaa()
        self.E_orb = self.E_avg
        self.E_orb_old = self.E_orb

        self.g_old = np.zeros(self.orb_opt.nrot, dtype=self.dtype)

        # This holds the *overall* orbital rotation, C_current = C_0 @ exp(R)
        # It's used as the initial guess at the start of each orbital optimization
        R = np.zeros(self.orb_opt.nrot, dtype=self.dtype)

        if self.orb_opt.nrot == 0:
            logger.log_info1(
                "No nonredundant orbital rotations; skipping macroiterations."
            )
            self.converged = True
        else:
            conv = False
            while self.iter < self.maxiter:
                # 1. Optimize orbitals at fixed CI expansion
                self.E_orb = self.lbfgs_solver.minimize(self.orb_opt, R)
                self._C = self.orb_opt.C.copy()
                # 2. Convergence checks
                _dg = self.lbfgs_solver.g - self.g_old
                self.dg_rms = np.sqrt(np.mean((_dg.conj() * _dg).real))
                self.g_rms = np.sqrt(
                    np.mean((self.lbfgs_solver.g.conj() * self.lbfgs_solver.g).real)
                )
                self.g_old = self.lbfgs_solver.g.copy()
                conv, conv_str = self._check_convergence()
                lbfgs_str = (
                    f"{self.lbfgs_solver.iter}/"
                    f"{'Y' if self.lbfgs_solver.converged else 'N'}"
                )
                iter_info = (
                    f"{self.iter:>10d} {self.E_avg.real:>20.10f} "
                    f"{self.E_orb.real:>20.10f} "
                )
                iter_info += f"{self.delta_ci_avg.real:>12.4e} {self.max_ci_de:>12.4e} {self.g_rms.real:>12.4e} {lbfgs_str:>8} {conv_str:>8}"
                if conv:
                    logger.log_info1(iter_info)
                    self.converged = True
                    break

                logger.log_info1(iter_info)

                # 3. Optimize CI expansion at fixed orbitals
                self.ci_solver.set_ints(
                    self.orb_opt.Ecore + self.system.nuclear_repulsion,
                    self.orb_opt.Fcore[self.actv, self.actv],
                    self.orb_opt.get_active_space_ints(),
                )
                self.ci_solver.run()
                self.E_ci = np.array(self.ci_solver.E)
                self.E_avg = self.ci_solver.compute_average_energy()
                self.E = self.E_avg
                self.g1_act = self.make_average_1rdm()
                g2_act = self.make_average_2rdm()
                self.orb_opt.set_rdms(self.g1_act, g2_act)
                self.iter += 1
            if self.iter >= self.maxiter and not conv:
                logger.log_info1("=" * width)
                if self.die_if_not_converged:
                    raise RuntimeError(
                        f"Orbital optimization did not converge in {self.maxiter} iterations."
                    )
                else:
                    logger.log_warning(
                        f"Orbital optimization did not converge in {self.maxiter} iterations."
                    )
        # self.ci_solver.set_maxiter(ci_maxiter_save)
        self.ci_solver.set_ints(
            self.orb_opt.Ecore + self.system.nuclear_repulsion,
            self.orb_opt.Fcore[self.actv, self.actv],
            self.orb_opt.get_active_space_ints(),
        )

        self.ci_solver.run()
        self.E_ci = np.array(self.ci_solver.E)
        self.E_avg = self.ci_solver.compute_average_energy()
        self.E = self.E_avg
        # Keep the orbital response base point synchronized with the final CI
        # vectors. The last CI diagonalization may slightly change the RDMs
        # after the final orbital microiteration.
        self.g1_act = self.make_average_1rdm()
        g2_act = self.make_average_2rdm()
        self.orb_opt.set_rdms(self.g1_act, g2_act)
        logger.log_info1(
            f"{'Final CI':>10} {self.E_avg:>20.10f} {self.E_orb:>20.10f} {'-':>12} {'-':>12} {'-':>12} {'-':>8} {'':>8}"
        )

        logger.log_info1("=" * width)
        if self.converged:
            logger.log_info1(
                f"Orbital optimization converged in {self.iter} iterations."
            )
        logger.log_info1(f"Final orbital optimized energy: {self.E_avg:.10f}")

        # undo _make_spaces_contiguous
        perm = self.mo_space.contig_to_orig
        self.mos.C[0] = self._C[:, perm].copy()

        # optionally, rotate the final orbitals to semicanonical or natural orbitals
        self._rotate_final_orbitals()

        # print information
        self._post_process()

        convergence_status = self.ci_solver.get_convergence_status()
        if convergence_status and not all(convergence_status):
            logger.log_warning(
                f"CI solver did not converge for all roots: {convergence_status}"
            )
            logger.log_warning("Consider increasing ci_maxiter.")

        self.executed = True
        return self

    def _post_process(self):
        pretty_print_ci_summary(self.ci_solver.sa_info, self.ci_solver.evals_per_solver)
        self.ci_solver.compute_natural_occupation_numbers()
        pretty_print_ci_nat_occ_numbers(
            self.ci_solver.sa_info,
            self.mo_space,
            self.ci_solver.nat_occs,
            getattr(self.ci_solver, "nat_occs_avg", None),
        )
        top_dets = self.ci_solver.get_top_determinants()
        pretty_print_ci_dets(self.ci_solver.sa_info, self.mo_space, top_dets)
        self._print_ao_composition()
        if self.do_transition_dipole:
            self.ci_solver.compute_transition_properties(self.mos.C[0])
            pretty_print_ci_transition_props(
                self.ci_solver.sa_info,
                self.ci_solver.transition_dipoles,
                self.ci_solver.oscillator_strengths,
                self.ci_solver.evals_per_solver,
            )

    def _rotate_final_orbitals(self) -> None:
        if self.final_orbitals not in ["semicanonical", "natural"]:
            return  # no final orbital transformation requested

        C_contig = self.mos.C[0][:, self.mo_space.orig_to_contig].copy()
        g1_act = self.make_average_1rdm()

        # get the final orbitals in contiguous ordering
        C_final = self._make_final_orbitals_contig(g1_act, C_contig)

        # undo contiguous ordering
        self.mos.C[0] = C_final[:, self.mo_space.contig_to_orig].copy()

        # rerun the CI solver in the final orbital basis to get the final energies
        new_E_ci, new_E_avg = self._rerun_ci_in_current_basis()

        check_final_orbital_energy_invariance(
            hard_fail=self.ci_solver.orbital_rotation_invariant,
            tol=self._final_orbital_energy_tol,
            old_E=self.E_ci,
            new_E=new_E_ci,
            old_E_avg=self.E_avg,
            new_E_avg=new_E_avg,
            hard_fail_hint="Consider increasing ci_maxiter.",
        )
        # update energies
        self.E_ci = new_E_ci
        self.E_avg = new_E_avg
        self.E = self.E_avg

    def _final_orbital_irrep_indices(self) -> NDArray:
        """Return the irrep indices of the final orbitals in contiguous ordering."""

        return np.asarray(self.mos.irrep_indices[0], dtype=int)[
            self.mo_space.orig_to_contig
        ]

    def _make_final_orbitals_contig(
        self, g1_act: NDArray, C_contig: NDArray
    ) -> NDArray:
        """Make the final orbitals and return them in contiguous ordering."""

        return make_final_orbitals(
            self.final_orbitals,
            system=self.system,
            mo_space=self.mo_space,
            irrep_indices=self._final_orbital_irrep_indices(),
            C_contig=C_contig,
            g1_act=g1_act,
        )

    def _rerun_ci_in_current_basis(self) -> tuple[NDArray, float]:
        """Rerun the CI solver in the current orbital basis and return the new CI eigenvalues and average energy."""
        if self.system.two_component:
            ints = SpinorbitalIntegrals(
                system=self.system,
                C=self.mos.C[0],
                spinorbitals=self.mo_space.active_indices,
                core_spinorbitals=self.mo_space.docc_indices,
            )
        else:
            ints = RestrictedMOIntegrals(
                system=self.system,
                C=self.mos.C[0],
                orbitals=self.mo_space.active_indices,
                core_orbitals=self.mo_space.docc_indices,
            )
        self.ci_solver.set_ints(ints.E, ints.H, ints.V)

        # due to the basis change, we can't restart from previous CI vectors
        self.ci_solver.reset_eigensolver()
        self.ci_solver.run()
        return np.array(self.ci_solver.E), self.ci_solver.compute_average_energy()

    def _print_ao_composition(self):
        if isinstance(self.system, ModelSystem):
            return
        basis_info = BasisInfo(self.system, self.system.basis)
        if getattr(self.system, "two_component", False):
            if getattr(self.system, "x2c_type", None) == "so":
                if not hasattr(self, "Usph2j"):
                    ua, ub = real_sph_to_j_adapted(self.system.basis)
                    self.Usph2j = np.vstack((ua, ub))
                C = self.Usph2j.conj().T @ self.mos.C[0]
                logger.log_info1("\nSpinor Composition of core MOs:")
                basis_info.print_spinor_composition(C, self.mo_space.docc_indices)
                logger.log_info1("\nSpinor Composition of active MOs:")
                basis_info.print_spinor_composition(C, self.mo_space.active_indices)
            else:
                logger.log_info1("\nAO Composition of core MOs:")
                basis_info.print_ao_composition(
                    self.mos.C[0], self.mo_space.docc_indices, spinorbital=True
                )
                logger.log_info1("\nAO Composition of active MOs:")
                basis_info.print_ao_composition(
                    self.mos.C[0], self.mo_space.active_indices, spinorbital=True
                )
        else:
            logger.log_info1("\nAO Composition of core MOs:")
            basis_info.print_ao_composition(self.mos.C[0], self.mo_space.docc_indices)
            logger.log_info1("\nAO Composition of active MOs:")
            basis_info.print_ao_composition(
                self.mos.C[0], self.mo_space.active_indices
            )

    def _get_nonredundant_rotations(self):
        """Lower triangular matrix of nonredundant rotations"""
        nmo = self._C.shape[1]
        nrr = np.zeros((nmo, nmo), dtype=bool)

        # these do NOT include frozen orbitals!
        _core = self.mo_space.core
        _virt = self.mo_space.virt

        # GASn-GASm rotations
        if self.mo_space.ngas > 1 and not self.freeze_inter_gas_rots:
            for i in range(self.mo_space.ngas):
                for j in range(i + 1, self.mo_space.ngas):
                    nrr[self.mo_space.gas[j], self.mo_space.gas[i]] = True

        nrr[_virt, _core] = True
        nrr[_virt, self.actv] = True
        nrr[self.actv, _core] = True

        # remove active_fronzen indices from nonredundant rotations
        if self.active_frozen_orbitals is not None:
            contig_actv_froz = self.mo_space.contig_to_orig[self.active_frozen_orbitals]
            for idx in contig_actv_froz:
                nrr[idx, :] = False
                nrr[:, idx] = False

        # zero out rotations between orbitals of different irreps
        if self.system.point_group.upper() != "C1":
            _irrid = self._final_orbital_irrep_indices()
            # equivalent to:
            # for i, j in range(nmo):
            #   if i^j != 0:
            #       nrr[i, j] = False
            nrr[(_irrid[:, None] ^ _irrid != 0)] = False

        return nrr

    def _check_convergence(self):
        is_grad_conv = self.g_rms < self.g_tol

        self.max_ci_de = np.max(np.abs(self.E_ci - self.E_ci_old))
        is_ci_eigval_conv = self.max_ci_de < self.e_tol

        self.delta_ci_avg = self.E_avg - self.E_avg_old
        is_ci_avg_conv = abs(self.delta_ci_avg) < self.e_tol

        criteria = [
            is_grad_conv,
            is_ci_eigval_conv,
            is_ci_avg_conv,
        ]

        conv = all(criteria)
        conv_str = "".join(["." if _ else "x" for _ in criteria])

        self.E_ci_old = self.E_ci.copy()
        self.E_avg_old = self.E_avg
        self.E_orb_old = self.E_orb
        return conv, conv_str

    def make_average_1rdm(self):
        return self.ci_solver.make_average_1rdm()

    def make_average_2rdm(self):
        return self.ci_solver.make_average_2rdm()

    def make_average_2cumulant(self):
        return self.ci_solver.make_average_2cumulant()

    def make_average_3rdm(self):
        return self.ci_solver.make_average_3rdm()

    def make_average_3cumulant(self):
        return self.ci_solver.make_average_3cumulant()

    def make_average_cumulants(self):
        return self.ci_solver.make_average_cumulants()


class MCOptimizer(MCOptimizerBase):
    def _validate_orbital_ci_response_request(self):
        if not self.executed:
            raise RuntimeError(
                "The MCSCF calculation must be run before building response blocks."
            )
        if self.system.two_component or np.iscomplexobj(self.orb_opt.C):
            raise NotImplementedError(
                "The orbital--CI response is currently implemented only for "
                "nonrelativistic real wave functions."
            )
        if self.final_orbitals != "original":
            raise NotImplementedError(
                "The orbital--CI response currently requires "
                "final_orbitals='original' so that the orbital optimizer and "
                "final CI vectors use the same active-orbital basis."
            )
        if not hasattr(self.ci_solver, "sub_solvers"):
            raise NotImplementedError(
                "The orbital--CI response currently requires a CISolver with "
                "explicit per-state CI vectors."
            )
        for sub_solver in self.ci_solver.sub_solvers:
            required = (
                "basis_size",
                "evecs",
                "csf_C_to_det_C",
                "ci_sigma_builder",
                "ci_strings",
                "spin_adapter",
                "ndet",
                "ci_params",
            )
            if not all(hasattr(sub_solver, name) for name in required):
                raise NotImplementedError(
                    "The orbital--CI response currently requires spin-adapted "
                    "CISolver sub-solvers."
                )

    def _get_ci_response_layout(self):
        layout = []
        start = 0
        for absolute_root, (state_index, root_in_state) in enumerate(
            self.ci_solver.sa_info.absolute_root_map
        ):
            sub_solver = self.ci_solver.sub_solvers[state_index]
            stop = start + sub_solver.basis_size
            layout.append(
                (
                    absolute_root,
                    state_index,
                    root_in_state,
                    slice(start, stop),
                )
            )
            start = stop
        return tuple(layout), start

    def get_ci_response_layout(self):
        """Return the root-major layout of the flattened CI response vector.

        Returns
        -------
        tuple[tuple[int, int, int, slice], ...]
            One entry per absolute root. Each entry contains
            (absolute_root, state_index, root_in_state, coefficient_slice).
            Coefficients inside each slice use that sub-solver's CSF ordering.
        """
        self._validate_orbital_ci_response_request()
        layout, _ = self._get_ci_response_layout()
        return layout

    def _validate_response_root(self, root):
        if isinstance(root, bool) or not isinstance(root, (int, np.integer)):
            raise TypeError("The target response root must be an integer.")
        nroots = len(self.ci_solver.sa_info.absolute_root_map)
        if root < 0 or root >= nroots:
            raise ValueError(
                f"Expected a target response root in [0, {nroots}), got {root}."
            )
        return int(root)

    def _validate_ci_response_vector(self, ci_vector):
        layout, nci = self._get_ci_response_layout()
        ci_vector = np.asarray(ci_vector)
        if ci_vector.shape != (nci,):
            raise ValueError(
                f"Expected a CI response vector with shape ({nci},), "
                f"got {ci_vector.shape}."
            )
        if np.iscomplexobj(ci_vector):
            raise TypeError(
                "The nonrelativistic orbital--CI response vector must be real."
            )
        return ci_vector.astype(float, copy=False), layout

    def _project_ci_response_vector(self, ci_vector, layout):
        """Project each root block out of its solved-state CI subspace."""
        projected = np.empty_like(ci_vector)
        for _, state_index, _, coefficient_slice in layout:
            sub_solver = self.ci_solver.sub_solvers[state_index]
            solved_roots = sub_solver.evecs[:, : sub_solver.nroot]
            root_vector = ci_vector[coefficient_slice]
            projected[coefficient_slice] = root_vector - solved_roots @ (
                solved_roots.T @ root_vector
            )
        return projected

    def project_ci_response_vector(self, ci_vector):
        r"""Project a root-major vector into the CI response space.

        In every absolute-root block, this method applies the projector for
        that root's state solver,

        .. math::

            \mathbf Q_s
            =\mathbf I_s-\sum_{\gamma\in\mathcal R_s}
             \mathbf c_\gamma\mathbf c_\gamma^T,

        where ``R_s`` contains all solved roots represented in the same CSF
        space.  The projection removes CI normalization directions and
        rotations within the solved-root subspace.

        Parameters
        ----------
        ci_vector : np.ndarray
            Root-major flattened real vector with shape ``(nci,)``.

        Returns
        -------
        np.ndarray
            Projected root-major vector with shape ``(nci,)``.
        """
        self._validate_orbital_ci_response_request()
        ci_vector, layout = self._validate_ci_response_vector(ci_vector)
        return self._project_ci_response_vector(ci_vector, layout)

    def _compute_ci_response_rdms(self, ci_vector, layout):
        nact = self.mo_space.nactv
        overlap_response = 0.0
        g1_response = np.zeros((nact,) * 2, dtype=float)
        g2_response = np.zeros((nact,) * 4, dtype=float)

        for _, state_index, root_in_state, coefficient_slice in layout:
            response = ci_vector[coefficient_slice]
            if not np.any(response):
                continue

            sub_solver = self.ci_solver.sub_solvers[state_index]
            reference = sub_solver.evecs[:, root_in_state]
            overlap_response += np.dot(response, reference)
            overlap_response += np.dot(reference, response)

            response_det = sub_solver.csf_C_to_det_C(response)
            reference_det = sub_solver.csf_C_to_det_C(reference)
            sigma_builder = sub_solver.ci_sigma_builder
            g1_response += sigma_builder.sf_1rdm(response_det, reference_det)
            g1_response += sigma_builder.sf_1rdm(reference_det, response_det)
            g2_response += sigma_builder.sf_2rdm(response_det, reference_det)
            g2_response += sigma_builder.sf_2rdm(reference_det, response_det)

        return overlap_response, g1_response, g2_response

    def compute_orbital_ci_hessian_vector_product(self, ci_vector):
        r"""Apply the CI contribution to the orbital response equation.

        The input is the concatenation of one real CSF vector for each absolute
        root, in the layout returned by get_ci_response_layout. For root
        alpha, its segment x_alpha is contracted with the reference CI vector
        c_alpha to form the bra-plus-ket transition overlap and RDMs. The
        returned vector is

        .. math::

            [\mathcal A^{\mathrm{oc}}\mathbf x]_I
            =
            2\left(A^{\mathrm{oc}}_{p_Iq_I}[\mathbf x]
                   -A^{\mathrm{oc}}_{q_Ip_I}[\mathbf x]\right).

        No state-average weights multiply x_alpha in this block because it is
        the orbital derivative of the unweighted CI residual Lagrangian. For a
        direct derivative of the state-averaged orbital gradient, the
        equivalent physical CI displacement is x_alpha / w_alpha.

        Parameters
        ----------
        ci_vector : np.ndarray
            Root-major flattened real CI multiplier vector.

        Returns
        -------
        np.ndarray
            Orbital response vector with shape (nrot,) and nrr C-order.
        """
        self._validate_orbital_ci_response_request()
        ci_vector, layout = self._validate_ci_response_vector(ci_vector)
        responses = self._compute_ci_response_rdms(ci_vector, layout)
        intermediates = self.orb_opt._build_ci_orbital_response_intermediates()
        return self.orb_opt._compute_ci_orbital_response_from_rdms(
            *responses, intermediates
        )

    def compute_orbital_ci_hessian(self):
        r"""Build the full rectangular orbital--CI response block.

        Column J is the orbital response obtained by applying
        compute_orbital_ci_hessian_vector_product to CI unit vector J. Columns
        use the root-major layout returned by get_ci_response_layout.

        Returns
        -------
        np.ndarray
            The orbital--CI block with shape (nrot, nci).
        """
        self._validate_orbital_ci_response_request()
        layout, nci = self._get_ci_response_layout()
        intermediates = self.orb_opt._build_ci_orbital_response_intermediates()
        hessian = np.empty((self.orb_opt.nrot, nci), dtype=float)
        unit = np.zeros(nci, dtype=float)
        for column in range(nci):
            unit[column] = 1.0
            responses = self._compute_ci_response_rdms(unit, layout)
            hessian[:, column] = self.orb_opt._compute_ci_orbital_response_from_rdms(
                *responses, intermediates
            )
            unit[column] = 0.0
        return hessian

    def _compute_ci_orbital_hessian_vector_product(
        self, orbital_vector, layout, intermediates
    ):
        r"""Apply the CI--orbital block with precomputed base-point tensors.

        The orbital optimizer first produces the response-integral triple
        ``(Ecore_response, H_response, V_response)`` defining
        :math:`\hat H[\mathbf z]`.  One temporary :class:`CISigmaBuilder` is
        constructed for each distinct state sub-solver.  For every absolute
        root, the kernel converts its reference CSF vector to determinants,
        applies the response Hamiltonian, converts the sigma vector back to
        CSFs, and stores

        .. math::

            2w_\alpha\hat H[\mathbf z]\mathbf c_\alpha

        in that root's coefficient slice.

        This private routine assumes that ``orbital_vector`` is a validated
        real ``(nrot,)`` array, ``layout`` is the current root-major layout,
        and ``intermediates`` was built at the current orbitals.  It applies no
        coefficient-space projector.

        Returns
        -------
        np.ndarray
            Root-major raw CSF response with shape ``(nci,)``.
        """
        scalar_response, one_body_response, two_body_response = (
            self.orb_opt._compute_active_space_hamiltonian_response(
                orbital_vector, intermediates
            )
        )
        response = np.empty(layout[-1][-1].stop, dtype=float)
        builders = {}

        for absolute_root, state_index, root_in_state, coefficient_slice in layout:
            sub_solver = self.ci_solver.sub_solvers[state_index]
            if state_index not in builders:
                builder = CISigmaBuilder(
                    sub_solver.ci_strings,
                    scalar_response,
                    one_body_response,
                    two_body_response,
                    sub_solver.log_level,
                )
                builder.set_memory(sub_solver.ci_params.ci_builder_memory)
                algorithm = sub_solver.ci_params.ci_algorithm.lower()
                builder.set_algorithm("kh" if algorithm == "exact" else algorithm)
                builders[state_index] = builder

            reference = sub_solver.evecs[:, root_in_state]
            sigma_csf = self._apply_ci_hamiltonian_to_csf(
                sub_solver, builders[state_index], reference
            )
            weight = self.ci_solver.weights_flat[absolute_root]
            response[coefficient_slice] = 2.0 * weight * sigma_csf

        return response

    @staticmethod
    def _apply_ci_hamiltonian_to_csf(sub_solver, sigma_builder, vector):
        """Apply a determinant-basis sigma builder to a real CSF vector."""
        vector_det = sub_solver.csf_C_to_det_C(vector)
        sigma_det = np.empty(sub_solver.ndet, dtype=float)
        sigma_builder.Hamiltonian(vector_det, sigma_det)
        sigma_csf = np.empty(sub_solver.basis_size, dtype=float)
        sub_solver.spin_adapter.det_C_to_csf_C(sigma_det, sigma_csf)
        return sigma_csf

    def compute_ci_orbital_hessian_vector_product(self, orbital_vector):
        r"""Apply the orbital contribution to the CI response equation.

        For an orbital direction ``z`` in the nonredundant pair ordering, the
        active-space scalar, one-electron, and two-electron integrals are
        differentiated to define the response Hamiltonian ``H[z]``.  The
        root-major CSF result is

        .. math::

            [\mathcal A^{\mathrm{co}}\mathbf z]_\alpha
            =2w_\alpha\hat H[\mathbf z]\mathbf c_\alpha.

        The state-average weight ``w_alpha`` is included once.  The returned
        vector is the raw coefficient-space action; no normalization or
        state-rotation projector is applied.

        Parameters
        ----------
        orbital_vector : np.ndarray
            Real orbital-rotation direction with shape ``(nrot,)`` and the
            same pair ordering as the orbital optimizer.

        Returns
        -------
        np.ndarray
            Root-major flattened CSF response vector with shape ``(nci,)``.
        """
        self._validate_orbital_ci_response_request()
        orbital_vector = self.orb_opt._validate_orbital_response_vector(orbital_vector)
        layout, _ = self._get_ci_response_layout()
        intermediates = self.orb_opt._build_orbital_ci_response_intermediates()
        return self._compute_ci_orbital_hessian_vector_product(
            orbital_vector, layout, intermediates
        )

    def compute_ci_orbital_hessian(self):
        r"""Build the full rectangular CI--orbital response block.

        Column ``I`` is obtained by applying
        :meth:`compute_ci_orbital_hessian_vector_product` to orbital unit
        vector ``I``.  Rows use the root-major CSF layout returned by
        :meth:`get_ci_response_layout`.  With ``W_C`` repeating each root's SA
        weight over its CSF block, the real mixed derivatives obey

        .. math::

            \mathcal A^{\mathrm{co}}
            =W_{\mathrm C}(\mathcal A^{\mathrm{oc}})^T.

        Returns
        -------
        np.ndarray
            The CI--orbital block with shape ``(nci, nrot)``.
        """
        self._validate_orbital_ci_response_request()
        layout, nci = self._get_ci_response_layout()
        intermediates = self.orb_opt._build_orbital_ci_response_intermediates()
        hessian = np.empty((nci, self.orb_opt.nrot), dtype=float)
        unit = np.zeros(self.orb_opt.nrot, dtype=float)
        for column in range(self.orb_opt.nrot):
            unit[column] = 1.0
            hessian[:, column] = self._compute_ci_orbital_hessian_vector_product(
                unit, layout, intermediates
            )
            unit[column] = 0.0
        return hessian

    def _compute_ci_ci_hessian_vector_product(self, ci_vector, layout):
        r"""Apply the raw CI--CI block to a validated coefficient vector.

        For every absolute root, the corresponding final-state CI sigma
        builder evaluates ``H @ x_alpha``.  The root's scalar energy shift is
        then subtracted and the real-coordinate factor of two is applied:

        .. math::

            [\mathcal A^{\mathrm{cc}}\mathbf x]_\alpha
            =2(\mathbf H_\alpha-E_\alpha\mathbf I_\alpha)
             \mathbf x_\alpha.

        Different absolute-root slices do not couple.  This private kernel
        assumes a validated vector and current layout and applies neither SA
        weights nor normalization/state-rotation projectors.

        Returns
        -------
        np.ndarray
            Root-major raw CSF response with shape ``(nci,)``.
        """
        response = np.empty_like(ci_vector)
        for absolute_root, state_index, _, coefficient_slice in layout:
            sub_solver = self.ci_solver.sub_solvers[state_index]
            root_vector = ci_vector[coefficient_slice]
            sigma = self._apply_ci_hamiltonian_to_csf(
                sub_solver, sub_solver.ci_sigma_builder, root_vector
            )
            response[coefficient_slice] = 2.0 * (
                sigma - self.E_ci[absolute_root] * root_vector
            )
        return response

    def compute_ci_ci_hessian_vector_product(self, ci_vector):
        r"""Apply the nonrelativistic CI--CI response block.

        The input and output concatenate one real CSF vector per absolute
        root in the layout returned by :meth:`get_ci_response_layout`.  For
        root ``alpha``, this method computes

        .. math::

            [\mathcal A^{\mathrm{cc}}\mathbf x]_\alpha
            =2(\mathbf H_\alpha-E_\alpha\mathbf I_\alpha)
             \mathbf x_\alpha.

        This is the raw real-coordinate block.  It contains no SA weights and
        no coefficient-space projector, so each reference CI vector is a null
        direction of its root block.

        Parameters
        ----------
        ci_vector : np.ndarray
            Root-major flattened real CI response vector with shape
            ``(nci,)``.

        Returns
        -------
        np.ndarray
            Root-major flattened CI response with shape ``(nci,)``.
        """
        self._validate_orbital_ci_response_request()
        ci_vector, layout = self._validate_ci_response_vector(ci_vector)
        return self._compute_ci_ci_hessian_vector_product(ci_vector, layout)

    def compute_ci_ci_hessian(self):
        r"""Build the full root-major CI--CI response matrix.

        Column ``J`` is obtained by applying
        :meth:`compute_ci_ci_hessian_vector_product` to coefficient-space unit
        vector ``J``.  Equivalently, the result is block diagonal with root
        blocks ``2 * (H_alpha - E_alpha * I_alpha)``.

        Returns
        -------
        np.ndarray
            Dense raw CI--CI response matrix with shape ``(nci, nci)``.
        """
        self._validate_orbital_ci_response_request()
        layout, nci = self._get_ci_response_layout()
        hessian = np.empty((nci, nci), dtype=float)
        unit = np.zeros(nci, dtype=float)
        for column in range(nci):
            unit[column] = 1.0
            hessian[:, column] = self._compute_ci_ci_hessian_vector_product(
                unit, layout
            )
            unit[column] = 0.0
        return hessian

    def compute_ci_response_vector_product(self, orbital_vector, ci_vector):
        r"""Apply the CI row of the coupled CASSCF response matrix.

        .. math::

            \mathbf y^{\mathrm c}
            =\mathcal A^{\mathrm{co}}\mathbf z
             +\mathcal A^{\mathrm{cc}}\mathbf x.

        The result is raw and unprojected in coefficient space.

        Parameters
        ----------
        orbital_vector : np.ndarray
            Real nonredundant orbital direction with shape ``(nrot,)``.
        ci_vector : np.ndarray
            Root-major flattened real CI direction with shape ``(nci,)``.

        Returns
        -------
        np.ndarray
            Combined root-major CI response with shape ``(nci,)``.
        """
        orbital_response = self.compute_ci_orbital_hessian_vector_product(
            orbital_vector
        )
        ci_response = self.compute_ci_ci_hessian_vector_product(ci_vector)
        return orbital_response + ci_response

    def compute_orbital_response_b_vector(self, root):
        r"""Build the target-state orbital ``b`` vector.

        For absolute root ``alpha``, this method evaluates the positive
        target-energy orbital gradient at the converged state-averaged
        orbitals,

        .. math::

            (\mathbf b^{\mathrm o}_\alpha)_I
            =(g^\alpha_{\mathrm{F2}})_I
            =2\left(A^\alpha_{p_Iq_I}-A^\alpha_{q_Ip_I}\right).

        The root-specific spin-free 1- and 2-RDMs are used without an SA
        weight.  The result is the ``b`` vector itself, not the signed linear
        solver right-hand side: the coupled orbital equation is

        .. math::

            \mathcal A^{\mathrm{oo}}\mathbf z
            +\mathcal A^{\mathrm{oc}}\mathbf x
            =-\mathbf b^{\mathrm o}_\alpha.

        Parameters
        ----------
        root : int
            Absolute target-root index in state-average ordering.

        Returns
        -------
        np.ndarray
            Real target-state orbital gradient with shape ``(nrot,)`` and the
            orbital optimizer's nonredundant-pair ordering.
        """
        self._validate_orbital_ci_response_request()
        root = self._validate_response_root(root)
        g1 = self.make_sf_1rdm(root)
        g2 = self.make_sf_2rdm(root)
        intermediates = self.orb_opt._build_ci_orbital_response_intermediates()
        return self.orb_opt._compute_ci_orbital_response_from_rdms(
            1.0, g1, g2, intermediates
        )

    def _compute_raw_ci_response_b_vector(self, root, layout):
        """Build the unprojected real target-energy CI gradient."""
        _, state_index, root_in_state, coefficient_slice = layout[root]
        sub_solver = self.ci_solver.sub_solvers[state_index]
        reference = sub_solver.evecs[:, root_in_state]
        sigma = self._apply_ci_hamiltonian_to_csf(
            sub_solver, sub_solver.ci_sigma_builder, reference
        )
        raw_b = np.zeros(layout[-1][-1].stop, dtype=float)
        raw_b[coefficient_slice] = 2.0 * sigma
        return raw_b

    def compute_ci_response_b_vector(self, root):
        r"""Build the projected target-state CI ``b`` vector.

        For target absolute root ``alpha``, the unprojected positive
        target-energy gradient has only one nonzero root block,

        .. math::

            (\widetilde{\mathbf b}^{\mathrm c}_\alpha)_\beta
            =2\delta_{\alpha\beta}\mathbf H_\beta\mathbf c_\beta.

        This method forms that vector with the final-state CI sigma builder
        and applies :meth:`project_ci_response_vector`.  Because a converged
        target CI vector obeys ``H c_alpha = E_alpha c_alpha``, its gradient is
        entirely in the removed solved-root subspace and
        ``b_ci`` is zero to the CI convergence tolerance.

        The returned vector is the positive projected ``b`` vector, not the
        signed solver right-hand side.  No SA weight is included.  A zero CI
        ``b`` vector does not imply a zero CI response multiplier because the
        orbital--CI block couples it to the nonzero orbital right-hand side.

        Parameters
        ----------
        root : int
            Absolute target-root index in state-average ordering.

        Returns
        -------
        np.ndarray
            Projected root-major CI ``b`` vector with shape ``(nci,)``.
        """
        self._validate_orbital_ci_response_request()
        root = self._validate_response_root(root)
        layout, _ = self._get_ci_response_layout()
        raw_b = self._compute_raw_ci_response_b_vector(root, layout)
        return self._project_ci_response_vector(raw_b, layout)

    def compute_projected_response_vector_product(self, orbital_vector, ci_vector):
        r"""Apply the gauge-fixed projected coupled response operator.

        If ``Q`` is the block-diagonal CI response projector and ``P = I-Q``,
        this method returns

        .. math::

            \begin{pmatrix}
             \mathcal A^{\mathrm{oo}}\mathbf z
             +\mathcal A^{\mathrm{oc}}Q\mathbf x\\
             Q\left(\mathcal A^{\mathrm{co}}\mathbf z
             +\mathcal A^{\mathrm{cc}}Q\mathbf x\right)+P\mathbf x
            \end{pmatrix}.

        The ``P @ x`` term fixes the otherwise singular removed CI directions
        to zero and leaves the physical projected equations unchanged.

        Parameters
        ----------
        orbital_vector : np.ndarray
            Real nonredundant orbital vector with shape ``(nrot,)``.
        ci_vector : np.ndarray
            Root-major real CI vector with shape ``(nci,)``.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Orbital and gauge-fixed projected CI products.
        """
        self._validate_orbital_ci_response_request()
        orbital_vector = self.orb_opt._validate_orbital_response_vector(orbital_vector)
        ci_vector, layout = self._validate_ci_response_vector(ci_vector)
        projected_ci = self._project_ci_response_vector(ci_vector, layout)

        orbital_product = self.compute_orbital_response_vector_product(
            orbital_vector, projected_ci
        )
        raw_ci_product = self.compute_ci_response_vector_product(
            orbital_vector, projected_ci
        )
        ci_product = self._project_ci_response_vector(raw_ci_product, layout)
        ci_product += ci_vector - projected_ci
        return orbital_product, ci_product

    def solve_state_specific_response(
        self,
        root,
        *,
        r_tol=1.0e-10,
        maxiter=None,
    ):
        r"""Solve the projected coupled response equations for one target root.

        GMRES is applied to the matrix-free gauge-fixed system

        .. math::

            \begin{pmatrix}
             \mathcal A^{\mathrm{oo}} & \mathcal A^{\mathrm{oc}}Q\\
             Q\mathcal A^{\mathrm{co}} &
             Q\mathcal A^{\mathrm{cc}}Q+P
            \end{pmatrix}
            \begin{pmatrix}\mathbf z_\alpha\\\mathbf x_\alpha\end{pmatrix}
            =-
            \begin{pmatrix}
             \mathbf b^{\mathrm o}_\alpha\\
             \mathbf b^{\mathrm c}_\alpha
            \end{pmatrix},

        with ``P = I - Q``.  The CI solution is projected once more before it
        is returned to remove roundoff in the gauge components.

        Parameters
        ----------
        root : int
            Absolute target-root index in state-average ordering.
        r_tol : float, optional
            Relative GMRES residual tolerance.
        maxiter : int or None, optional
            Maximum number of GMRES restart cycles.  The SciPy default is used
            when omitted.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            The orbital response ``z_alpha`` with shape ``(nrot,)`` and the
            projected root-major CI response ``x_alpha`` with shape
            ``(nci,)``.

        Raises
        ------
        RuntimeError
            If GMRES does not converge.
        """
        self._validate_orbital_ci_response_request()
        root = self._validate_response_root(root)
        if not np.isscalar(r_tol) or r_tol <= 0.0:
            raise ValueError(f"r_tol must be positive, got {r_tol}.")
        if maxiter is not None and (
            isinstance(maxiter, bool)
            or not isinstance(maxiter, (int, np.integer))
            or maxiter < 1
        ):
            raise ValueError(f"maxiter must be a positive integer, got {maxiter}.")

        layout, nci = self._get_ci_response_layout()
        nrot = self.orb_opt.nrot
        dimension = nrot + nci
        orbital_b = self.compute_orbital_response_b_vector(root)
        ci_b = self.compute_ci_response_b_vector(root)
        rhs = -np.concatenate((orbital_b, ci_b))

        def matvec(vector):
            orbital_product, ci_product = (
                self.compute_projected_response_vector_product(
                    vector[:nrot], vector[nrot:]
                )
            )
            return np.concatenate((orbital_product, ci_product))

        operator = spla.LinearOperator(
            (dimension, dimension), matvec=matvec, dtype=float
        )
        restart = min(dimension, 50)
        solution, info = spla.gmres(
            operator,
            rhs,
            rtol=float(r_tol),
            atol=0.0,
            restart=restart,
            maxiter=maxiter,
        )
        if info != 0:
            if info > 0:
                reason = f"did not converge after {info} iterations"
            else:
                reason = f"failed with status {info}"
            raise RuntimeError(f"The coupled CASSCF response solve {reason}.")

        orbital_response = solution[:nrot]
        ci_response = self._project_ci_response_vector(solution[nrot:], layout)
        return orbital_response, ci_response

    def solve_orbital_response_vector(self, root, *, r_tol=1.0e-10, maxiter=None):
        """Solve and return only the target-root orbital response vector."""
        orbital_response, _ = self.solve_state_specific_response(
            root, r_tol=r_tol, maxiter=maxiter
        )
        return orbital_response

    def compute_omega(
        self,
        root,
        orbital_response=None,
        ci_response=None,
        *,
        r_tol=1.0e-10,
        maxiter=None,
    ):
        r"""Compute the relaxed MO orthogonality multiplier for one root.

        For target root :math:`\alpha`, let :math:`A^\alpha` be its orbital
        Lagrangian, :math:`A^{\mathrm{oc}}[\mathbf x_\alpha]` the contribution
        from the root-major CI multiplier, and :math:`\dot{\bar A}[\mathbf
        z_\alpha]` the directional response of the state-averaged orbital
        Lagrangian.  In the current real nonrelativistic convention this method
        forms

        .. math::

            \Omega_\alpha
            = A^\alpha
            + A^{\mathrm{oc}}[\mathbf x_\alpha]
            + \dot{\bar A}[\mathbf z_\alpha]
            + Z_\alpha\bar A-\bar A Z_\alpha,

            \omega_\alpha
            = \frac{1}{2}(\Omega_\alpha+\Omega_\alpha^T).

        The commutator is required because the orbital-pair generators form a
        moving frame: differentiating ``z.T @ g`` with respect to the full MO
        coefficients is not just the directional derivative of ``A``.  The
        returned matrix is in the current MO basis.  Its AO counterpart for
        the overlap-derivative gradient term is ``C @ omega @ C.T``.

        If neither response vector is supplied, the coupled projected response
        equations are solved first.  If response vectors are supplied, both
        must be given; the CI vector is projected into the physical response
        space before its transition RDMs are formed.

        Parameters
        ----------
        root : int
            Absolute target-root index in state-average ordering.
        orbital_response : np.ndarray or None, optional
            Solved nonredundant orbital multiplier ``z_alpha``.
        ci_response : np.ndarray or None, optional
            Solved root-major CI multiplier ``x_alpha``.
        r_tol : float, optional
            Relative GMRES tolerance used only when solving the response.
        maxiter : int or None, optional
            Maximum GMRES restart cycles used only when solving the response.

        Returns
        -------
        np.ndarray
            Real symmetric ``omega_alpha`` with shape ``(nmo, nmo)`` in the
            current MO basis.
        """
        self._validate_orbital_ci_response_request()
        root = self._validate_response_root(root)
        supplied_orbital = orbital_response is not None
        supplied_ci = ci_response is not None
        if supplied_orbital != supplied_ci:
            raise ValueError(
                "orbital_response and ci_response must either both be supplied "
                "or both be omitted."
            )

        if not supplied_orbital:
            orbital_response, ci_response = self.solve_state_specific_response(
                root, r_tol=r_tol, maxiter=maxiter
            )
        else:
            orbital_response = self.orb_opt._validate_orbital_response_vector(
                orbital_response
            )
            ci_response, layout = self._validate_ci_response_vector(ci_response)
            ci_response = self._project_ci_response_vector(ci_response, layout)

        layout, _ = self._get_ci_response_layout()
        density_intermediates = self.orb_opt._build_ci_orbital_response_intermediates()

        target_A = self.orb_opt._build_orbital_lagrangian_from_rdms(
            1.0,
            self.make_sf_1rdm(root),
            self.make_sf_2rdm(root),
            density_intermediates,
        )
        ci_A = self.orb_opt._build_orbital_lagrangian_from_rdms(
            *self._compute_ci_response_rdms(ci_response, layout),
            density_intermediates,
        )
        average_A = self.orb_opt._build_orbital_lagrangian_from_rdms(
            1.0,
            self.make_average_1rdm(),
            self.make_average_2rdm(),
            density_intermediates,
        )

        orbital_intermediates = self.orb_opt._build_orbital_response_intermediates()
        orbital_A = self.orb_opt._compute_orbital_lagrangian_response(
            orbital_response, orbital_intermediates
        )
        Z = self.orb_opt._vec_to_mat(orbital_response)
        orbital_A += Z @ average_A - average_A @ Z

        Omega = target_A + ci_A + orbital_A
        return 0.5 * (Omega + Omega.T)

    def compute_orbital_response_vector_product(self, orbital_vector, ci_vector):
        r"""Apply the orbital row of the coupled CASSCF response matrix.

        .. math::

            \mathbf y^{\mathrm o}
            =
            \mathcal A^{\mathrm{oo}}\mathbf z
            +
            \mathcal A^{\mathrm{oc}}\mathbf x.

        Parameters
        ----------
        orbital_vector : np.ndarray
            Real nonredundant orbital direction with shape (nrot,).
        ci_vector : np.ndarray
            Root-major flattened real CI multiplier vector.

        Returns
        -------
        np.ndarray
            Combined orbital response with shape (nrot,).
        """
        ci_response = self.compute_orbital_ci_hessian_vector_product(ci_vector)
        orbital_response = self.orb_opt.compute_orbital_hessian_vector_product(
            orbital_vector
        )
        return orbital_response + ci_response

    def make_sd_1rdm(
        self,
        left_root: int,
        right_root: int | None = None,
    ) -> tuple[NDArray, NDArray]:
        return self.ci_solver.make_sd_1rdm(left_root, right_root)

    def make_sd_2rdm(
        self,
        left_root: int,
        right_root: int | None = None,
    ) -> tuple[NDArray, NDArray, NDArray]:
        return self.ci_solver.make_sd_2rdm(left_root, right_root)

    def make_sd_3rdm(
        self,
        left_root: int,
        right_root: int | None = None,
    ) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        return self.ci_solver.make_sd_3rdm(left_root, right_root)

    def make_sf_1rdm(
        self,
        left_root: int,
        right_root: int | None = None,
    ) -> NDArray:
        return self.ci_solver.make_sf_1rdm(left_root, right_root)

    def make_sf_2rdm(
        self,
        left_root: int,
        right_root: int | None = None,
    ) -> NDArray:
        return self.ci_solver.make_sf_2rdm(left_root, right_root)

    def gradient(self, root=None) -> NDArray:
        r"""
        Compute a target-root CASSCF/GASSCF analytic nuclear gradient.

        This implementation supports real nonrelativistic and complex
        two-component state-specific CASSCF/GASSCF wave functions, including
        SF- and SO-X2C-1e Hamiltonians.  It also supports an individual root of
        a real nonrelativistic SA-CASSCF wave function when all roots belong to
        one CI state solver and ``final_orbitals='original'``.  ``root`` must be
        specified for a state average; it defaults to zero only for a
        single-root calculation.  Frozen-core and frozen-virtual response,
        active-frozen rotations, and frozen inter-GAS rotations are not
        supported. Point and Gaussian nuclear charge distributions are
        supported; Gaussian charges require libcint. Requesting any unsupported
        feature raises ``NotImplementedError``.
        Both the orbital optimization and all CI roots must be converged; an
        unconverged wave function raises ``RuntimeError`` because the
        stationary-gradient expression does not apply.

        The gradient is assembled in the same integral-layer form as the RHF
        and UHF gradients:

        .. math::
            E^x =
            E_\mathrm{NN}^x
            + h^x_{\mu\nu}\Gamma_{\mu\nu}
            - S^x_{\mu\nu} W^S_{\mu\nu}
            + W^P_{\mu\nu}(P|\mu\nu)^x
            + W_{PQ}(P|Q)^x.

        Here :math:`\Gamma_{\mu\nu}` is the full spin-free one-particle
        density, :math:`W^S_{\mu\nu}` is the AO representation of the
        symmetric CASSCF/GASSCF orbital Lagrangian, and
        :math:`W^P_{\mu\nu}` and :math:`W_{PQ}` are the density-fitted
        two-electron derivative weights defined in
        ``docs/technical_notes/df_gradients.tex``.

        Parameters
        ----------
        root : int or None, optional
            Absolute target-root index. Required for a state-averaged wave
            function and otherwise defaults to zero.

        Returns
        -------
        NDArray
            Gradient with shape ``(natoms, 3)`` in Hartree/Bohr.
        """
        from .mc_optimizer_grad import _compute_casscf_gradient

        return _compute_casscf_gradient(self, root=root)
