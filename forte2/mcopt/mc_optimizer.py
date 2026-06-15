from abc import ABC
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


from forte2.base_classes import (
    CIBase,
    RelCIBase,
    SystemMixin,
    MOsMixin,
    MOSpaceMixin,
)
from forte2.orbitals import Semicanonicalizer
from forte2.jkbuilder import RestrictedMOIntegrals, SpinorbitalIntegrals
from forte2.helpers import logger, LBFGS
from forte2.system.basis_utils import BasisInfo
from forte2.system import ModelSystem
from forte2._forte2 import ints
from forte2.gradients import (
    compute_gradient,
    build_metric_inverted_three_center,
)
from forte2.ci.ci_utils import (
    pretty_print_ci_summary,
    pretty_print_ci_nat_occ_numbers,
    pretty_print_ci_dets,
    pretty_print_ci_transition_props,
)
from .orbital_optimizer import OrbOptimizer, RelOrbOptimizer


@dataclass
class MCOptimizerBase(ABC, SystemMixin, MOsMixin, MOSpaceMixin):
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
    final_orbital : str, optional, default="semicanonical"
        Whether to return the final orbitals in the semicanonical basis or the original basis.

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

    ### L-BFGS solver (microiteration) parameters
    micro_maxiter: int = 6
    max_rotation: float = 0.2

    ### Post-iteration
    do_transition_dipole: bool = False
    final_orbital: str = "semicanonical"

    ### Non-init attributes
    converged: bool = field(default=False, init=False)
    executed: bool = field(default=False, init=False)

    def __post_init__(self):
        if not isinstance(self.ci_solver, (CIBase, RelCIBase)):
            raise ValueError("ci_solver must be an instance of CIBase or RelCIBase.")

        if self.final_orbital not in [
            "semicanonical",
            "original",
        ]:
            raise ValueError(
                "final_orbital must be either 'semicanonical' or 'original'."
            )

    def __call__(self, method):
        self.parent_method = method
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

        SystemMixin.copy_from_upstream(self, self.parent_method)
        MOsMixin.copy_from_upstream(self, self.parent_method)
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
        self._C = self.C[0][:, perm].copy()
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

        # _CISolver = RelCISolver if self.two_component else CISolver
        # self.ci_solver = _CISolver(
        #     states=self.states,
        #     core_orbitals=self.mo_space.docc_orbitals,
        #     active_orbitals=self.mo_space.active_orbitals,
        #     nroots=self.sa_info.nroots,
        #     weights=self.sa_info.weights,
        #     log_level=self.ci_solver_verbosity,
        #     die_if_not_converged=False,
        #     ci_params=self.ci_params,
        #     davidson_liu_params=self.davidson_liu_params,
        # )(self.parent_method)

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
        self.C[0] = self._C[:, perm].copy()

        self._post_process()

        if self.final_orbital == "semicanonical":
            semi = Semicanonicalizer(
                mo_space=self.mo_space,
                system=self.system,
                mix_inactive=False,
                mix_active=False,
            )
            C_contig = self.C[0][:, self.mo_space.orig_to_contig].copy()
            semi.semi_canonicalize(
                g1=self.make_average_1rdm(),
                C_contig=C_contig,
            )
            self.C[0] = semi.C_semican[:, self.mo_space.contig_to_orig].copy()

            # recompute the CI vectors in the semicanonical basis
            if self.system.two_component:
                ints = SpinorbitalIntegrals(
                    system=self.system,
                    C=self.C[0],
                    spinorbitals=self.mo_space.active_indices,
                    core_spinorbitals=self.mo_space.docc_indices,
                )
            else:
                ints = RestrictedMOIntegrals(
                    system=self.system,
                    C=self.C[0],
                    orbitals=self.mo_space.active_indices,
                    core_orbitals=self.mo_space.docc_indices,
                )
            self.ci_solver.set_ints(ints.E, ints.H, ints.V)
            # Basis change, can't restart from previous CI vectors *reliably*
            self.ci_solver.reset_eigensolver()
            self.ci_solver.run()

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
            self.ci_solver.sa_info, self.mo_space, self.ci_solver.nat_occs
        )
        top_dets = self.ci_solver.get_top_determinants()
        pretty_print_ci_dets(self.ci_solver.sa_info, self.mo_space, top_dets)
        if not self.system.two_component:
            # TODO: enable AO composition for 2c
            self._print_ao_composition()
        if self.do_transition_dipole:
            self.ci_solver.compute_transition_properties(self.C[0])
            pretty_print_ci_transition_props(
                self.ci_solver.sa_info,
                self.ci_solver.transition_dipoles,
                self.ci_solver.oscillator_strengths,
                self.ci_solver.evals_per_solver,
            )

    def _print_ao_composition(self):
        basis_info = BasisInfo(self.system, self.system.basis)
        logger.log_info1("\nAO Composition of core MOs:")
        basis_info.print_ao_composition(
            self.C[0], list(range(self.core.start, self.core.stop))
        )
        logger.log_info1("\nAO Composition of active MOs:")
        basis_info.print_ao_composition(
            self.C[0], list(range(self.actv.start, self.actv.stop))
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
            _irrid = np.array(self.irrep_indices[0])
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

    def gradient(self):
        r"""
        Compute a state-specific CASSCF analytic nuclear gradient.

        This first implementation is intentionally narrow.  It supports only
        real, nonrelativistic, density-fitted, state-specific CASSCF wave
        functions.  State-averaged gradients, frozen-core response, frozen
        virtual response, active-frozen rotations, X2C, Gaussian nuclear
        charges, and Cholesky-ERI gradients are rejected explicitly.

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
        symmetric CASSCF orbital Lagrangian, and :math:`W^P_{\mu\nu}` and
        :math:`W_{PQ}` are the density-fitted two-electron derivative weights
        defined in ``docs/technical_notes/df_gradients.tex``.

        Returns
        -------
        ndarray
            Gradient with shape ``(natoms, 3)`` in Hartree/Bohr.
        """
        self._validate_casscf_gradient_supported(pre_run=True)

        if not self.executed:
            self.run()

        self._validate_casscf_gradient_supported(pre_run=False)

        C = self.C[0][:, self.mo_space.orig_to_contig].copy()
        gamma1_act = self.make_sf_1rdm(0)
        gamma2_act = self.make_sf_2rdm(0)
        Ccore = C[:, self.mo_space.core]
        Cact = C[:, self.mo_space.actv]

        D1 = self._build_casscf_one_body_density(Ccore, Cact, gamma1_act)
        W1 = self._build_casscf_overlap_weight(C, gamma1_act, gamma2_act)
        W2, W3 = self._build_casscf_df_deriv_weights(
            Ccore, Cact, gamma1_act, gamma2_act
        )

        return compute_gradient(self.system, D1.real, W1.real, W2, W3)

    def _build_casscf_one_body_density(self, Ccore, Cact, gamma1_act):
        r"""
        Build the AO spin-free one-particle density without full MO padding.

        For the first CASSCF gradient implementation, frozen core orbitals are
        rejected and the inactive core is a closed-shell doubly occupied block.
        The spin-free AO density is therefore assembled directly as

        .. math::
            \Gamma_{\mu\nu}
            =
            2 C_{\mu i} C_{\nu i}
            +
            C_{\mu u}\Gamma_{uv} C_{\nu v},

        where :math:`i` labels inactive core orbitals and :math:`u,v` label
        active orbitals.  This avoids constructing a full
        :math:`\Gamma_{pq}` matrix over core, active, and virtual orbitals.

        Parameters
        ----------
        Ccore : NDArray
            Core MO coefficients with shape ``(nbasis, ncore)``.
        Cact : NDArray
            Active MO coefficients with shape ``(nbasis, nactv)``.
        gamma1_act : NDArray
            Active-space spin-free 1-RDM, :math:`\Gamma_{uv}`.

        Returns
        -------
        NDArray
            AO spin-free one-particle density with shape ``(nbasis, nbasis)``.
        """
        nact = Cact.shape[1]
        if gamma1_act.shape != (nact, nact):
            raise ValueError(
                f"Expected active 1-RDM shape {(nact, nact)}, got {gamma1_act.shape}."
            )

        D1 = 2.0 * np.einsum("mi,ni->mn", Ccore, Ccore.conj(), optimize=True)
        D1 += np.einsum("mu,uv,nv->mn", Cact, gamma1_act, Cact.conj(), optimize=True)
        return D1

    def _build_casscf_active_cumulant(self, gamma1_act, gamma2_act):
        r"""
        Build the active-space spin-free two-particle cumulant.

        The CASSCF pair density is written in the notation of
        ``docs/technical_notes/df_gradients.tex`` as

        .. math::
            \Gamma_{pq,rs}
            =
            \Gamma_{pr}\Gamma_{qs}
            -
            \frac{1}{2}\Gamma_{ps}\Gamma_{qr}
            +
            \Lambda_{pq,rs}.

        The inactive core contribution is generated from the one-particle
        density and is never materialized as a full four-index tensor.  The
        genuine correlated contribution is the active cumulant

        .. math::
            \Lambda_{uv,wx}
            =
            \Gamma_{uv,wx}
            -
            \Gamma_{uw}\Gamma_{vx}
            +
            \frac{1}{2}\Gamma_{ux}\Gamma_{vw}.

        Parameters
        ----------
        gamma1_act : NDArray
            Active-space spin-free 1-RDM.
        gamma2_act : NDArray
            Active-space spin-free 2-RDM in the Forte2 CI convention.

        Returns
        -------
        NDArray
            Active-space spin-free two-particle cumulant.
        """
        gamma1_act = np.asarray(gamma1_act)
        gamma2_act = np.asarray(gamma2_act)
        nact = gamma1_act.shape[0]
        if gamma1_act.shape != (nact, nact):
            raise ValueError("Active 1-RDM must be a square matrix.")
        if gamma2_act.shape != (nact, nact, nact, nact):
            raise ValueError(
                "Expected active 2-RDM shape "
                f"{(nact, nact, nact, nact)}, got {gamma2_act.shape}."
            )

        lambda2_act = gamma2_act.copy()
        lambda2_act -= np.einsum("uw,vx->uvwx", gamma1_act, gamma1_act, optimize=True)
        lambda2_act += 0.5 * np.einsum(
            "ux,vw->uvwx", gamma1_act, gamma1_act, optimize=True
        )
        return lambda2_act

    def _build_casscf_df_deriv_weights(self, Ccore, Cact, gamma1_act, gamma2_act):
        r"""
        Build CASSCF DF derivative weights from core and active blocks only.

        This implements the molecular-orbital DF derivative equations from
        ``docs/technical_notes/df_gradients.tex`` without constructing a full
        orbital-space :math:`\Gamma_{pq,rs}` tensor.  Let
        :math:`x,y,z,w` run only over the compact hole space containing the
        inactive core and active orbitals,

        .. math::
            C_h = [C_i\ C_u],
            \qquad
            \Gamma^h =
            \begin{pmatrix}
            2I_\mathrm{core} & 0 \\
            0 & \Gamma^\mathrm{act}
            \end{pmatrix}.

        The metric-applied three-center tensor is transformed only to this
        compact space:

        .. math::
            Z^P_{xy} = C_{\mu x} Z^P_{\mu\nu} C_{\nu y}.

        The product part of the spin-free pair density contributes through
        :math:`\Gamma^h`, while the cumulant correction is nonzero only in the
        active block.  The resulting weights are

        .. math::
            W^P_{xy}
            =
            \Gamma^h_{xy} R^P
            -
            \frac{1}{2}
            \Gamma^h_{xz} Z^P_{wz} \Gamma^h_{wy}
            +
            \delta_{xu}\delta_{yw}
            \Lambda_{uv,wx}Z^P_{vx},

        and

        .. math::
            W_{PQ}
            =
            -\frac{1}{2}R^P R^Q
            +
            \frac{1}{4}
            Z^P_{xy}\Gamma^h_{xz}Z^Q_{wz}\Gamma^h_{wy}
            -
            \frac{1}{2}
            \Lambda_{uv,wx}Z^P_{uw}Z^Q_{vx},

        with :math:`R^P=\Gamma^h_{xy}Z^P_{xy}`.  Only the compact
        :math:`W^P_{xy}` is formed before the final AO back-transformation
        required by the derivative integral kernel.

        Parameters
        ----------
        Ccore : NDArray
            Core MO coefficients with shape ``(nbasis, ncore)``.
        Cact : NDArray
            Active MO coefficients with shape ``(nbasis, nactv)``.
        gamma1_act : NDArray
            Active-space spin-free 1-RDM.
        gamma2_act : NDArray
            Active-space spin-free 2-RDM.

        Returns
        -------
        tuple[NDArray, NDArray]
            ``(W2, W3)`` with shapes ``(naux, naux)`` and
            ``(naux, nbasis, nbasis)``.
        """
        ncore = Ccore.shape[1]
        nact = Cact.shape[1]
        if gamma1_act.shape != (nact, nact):
            raise ValueError(
                f"Expected active 1-RDM shape {(nact, nact)}, got {gamma1_act.shape}."
            )

        Ch = np.hstack((Ccore, Cact))
        nhole = Ch.shape[1]
        gamma_h = np.zeros((nhole, nhole), dtype=np.result_type(gamma1_act, gamma2_act))
        gamma_h[:ncore, :ncore] = 2.0 * np.eye(ncore)
        gamma_h[ncore:, ncore:] = gamma1_act
        lambda2_act = self._build_casscf_active_cumulant(gamma1_act, gamma2_act)

        Z_ao = build_metric_inverted_three_center(self.system)
        Z_h = np.einsum("mx,Pmn,ny->Pxy", Ch.conj(), Z_ao, Ch, optimize=True)

        R = np.einsum("xy,Pxy->P", gamma_h, Z_h, optimize=True)
        W3_h = np.einsum("xy,P->Pxy", gamma_h, R, optimize=True)
        W3_h -= 0.5 * np.einsum("xz,Pwz,wy->Pxy", gamma_h, Z_h, gamma_h, optimize=True)

        Z_act = Z_h[:, ncore:, ncore:]
        W3_h[:, ncore:, ncore:] += np.einsum(
            "uvwx,Pvx->Puw", lambda2_act, Z_act, optimize=True
        )

        W2 = -0.5 * np.einsum("P,Q->PQ", R, R, optimize=True)
        W2 += 0.25 * np.einsum(
            "Pxy,xz,Qwz,wy->PQ", Z_h, gamma_h, Z_h, gamma_h, optimize=True
        )
        W2 -= 0.5 * np.einsum(
            "uvwx,Puw,Qvx->PQ", lambda2_act, Z_act, Z_act, optimize=True
        )

        W3 = np.einsum("mx,Pxy,ny->Pmn", Ch, W3_h, Ch.conj(), optimize=True)
        return W2.real, W3.real

    def _build_casscf_overlap_weight(self, C, gamma1_act, gamma2_act):
        r"""
        Build the AO energy-weighted density for the CASSCF overlap term.

        The existing orbital optimizer constructs the matrix :math:`A_{pq}`
        used in the CASSCF orbital gradient
        :math:`g_{pq}=2(A_{pq}-A_{qp})`.  For a fully optimized
        state-specific CASSCF wave function, the symmetric part of this matrix
        is the orbital Lagrange multiplier that contracts the overlap
        derivative.  This helper recomputes :math:`A` in the current final MO
        basis and transforms

        .. math::
            W^S_{\mu\nu}
            =
            C_{\mu p}
            \frac{1}{2}(A_{pq}+A_{qp})
            C_{\nu q}.

        Parameters
        ----------
        C : NDArray
            Final MO coefficients in contiguous CASSCF order.
        gamma1_act : NDArray
            State-specific active-space spin-free 1-RDM.
        gamma2_act : NDArray
            State-specific active-space spin-free 2-RDM.

        Returns
        -------
        NDArray
            AO energy-weighted density with shape ``(nbasis, nbasis)``.
        """
        orb_opt = OrbOptimizer(
            C,
            (self.mo_space.core, self.mo_space.actv, self.mo_space.virt),
            self.system.fock_builder,
            self.system.ints_hcore(),
            self.system.nuclear_repulsion,
            self.nrr,
            compute_active_hessian=False,
        )
        orb_opt.set_rdms(gamma1_act, gamma2_act)
        orb_opt._compute_Fcore()
        orb_opt.get_eri_gaaa()
        lagrangian_mo = orb_opt.compute_orbital_lagrangian()
        return np.einsum("mp,pq,nq->mn", C, lagrangian_mo, C, optimize=True)

    def _validate_casscf_gradient_supported(self, pre_run=False):
        """Reject CASSCF gradient cases outside the first implementation scope."""
        if isinstance(self.ci_solver, RelCIBase):
            raise NotImplementedError(
                "CASSCF gradients are implemented only for nonrelativistic CASSCF."
            )

        if self.ci_solver.sa_info.ncis != 1 or self.ci_solver.sa_info.nroots_sum != 1:
            raise NotImplementedError(
                "CASSCF gradients are implemented only for state-specific CASSCF."
            )

        if self.active_frozen_orbitals:
            raise NotImplementedError(
                "CASSCF gradients with active frozen orbitals are not implemented."
            )

        system = self.system if hasattr(self, "system") else self.parent_method.system
        if isinstance(system, ModelSystem):
            raise NotImplementedError(
                "CASSCF gradients are not implemented for ModelSystem."
            )
        if system.cholesky_tei:
            raise NotImplementedError(
                "CASSCF gradients are implemented only for density fitting, not cholesky_tei."
            )
        if system.use_gaussian_charges:
            raise NotImplementedError(
                "CASSCF gradients with Gaussian nuclear charges are not implemented."
            )
        if system.x2c_type is not None or system.two_component:
            raise NotImplementedError("CASSCF gradients with X2C are not implemented.")
        if system.auxiliary_basis is None:
            raise NotImplementedError(
                "CASSCF gradients require an auxiliary basis set for density fitting."
            )

        max_l = max(system.basis.max_l, system.auxiliary_basis.max_l)
        if max_l > ints.libint2_max_am:
            raise NotImplementedError(
                "CASSCF gradients require derivative integrals supported by Libint2 "
                f"(max_l = {max_l}, Libint2 max_l = {ints.libint2_max_am})."
            )

        frozen_core = getattr(self.ci_solver, "frozen_core_orbitals", None)
        frozen_virt = getattr(self.ci_solver, "frozen_virtual_orbitals", None)
        if frozen_core:
            raise NotImplementedError(
                "CASSCF gradients with frozen core orbitals are not implemented."
            )
        if frozen_virt:
            raise NotImplementedError(
                "CASSCF gradients with frozen virtual orbitals are not implemented."
            )

        if pre_run:
            return

        if self.mo_space.nfrozen_core > 0:
            raise NotImplementedError(
                "CASSCF gradients with frozen core orbitals are not implemented."
            )
        if self.mo_space.nfrozen_virtual > 0:
            raise NotImplementedError(
                "CASSCF gradients with frozen virtual orbitals are not implemented."
            )
        if self.mo_space.ngas != 1:
            raise NotImplementedError(
                "CASSCF gradients are implemented only for a single active space."
            )
        if np.iscomplexobj(self.C[0]):
            raise NotImplementedError(
                "CASSCF gradients with complex orbitals are not implemented."
            )