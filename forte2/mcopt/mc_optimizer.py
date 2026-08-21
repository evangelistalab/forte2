from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
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
        SF- and SO-X2C-1e Hamiltonians. It also supports individual roots of
        real nonrelativistic SA-CASSCF/GASSCF wave functions when
        ``final_orbitals='original'``. Frozen-core and frozen-virtual response
        and active-frozen rotations are not supported. Point and Gaussian
        nuclear charge distributions are supported; Gaussian charges require
        libcint. Requesting any unsupported feature raises ``NotImplementedError``.
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
        density, :math:`W^S_{\mu\nu}` is the AO energy-weighted density, and
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
