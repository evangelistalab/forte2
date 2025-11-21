import numpy as np
from dataclasses import dataclass, field

from forte2.ci import CISolver, RelCISolver
from forte2.base_classes.active_space_solver import (
    ActiveSpaceSolver,
    RelActiveSpaceSolver,
)
from forte2.orbitals import Semicanonicalizer
from forte2.jkbuilder import RestrictedMOIntegrals, SpinorbitalIntegrals
from forte2.helpers import logger, LBFGS, DIIS
from forte2.system.basis_utils import BasisInfo
from forte2.ci.ci_utils import (
    pretty_print_ci_summary,
    pretty_print_ci_nat_occ_numbers,
    pretty_print_ci_dets,
    pretty_print_ci_transition_props,
)
from .orbital_optimizer import OrbOptimizer, RelOrbOptimizer


@dataclass
class MCOptimizer(ActiveSpaceSolver):
    """
    Two-step optimizer for multi-configurational wavefunctions.

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
    active_frozen_orbitals : list[int], optional
        List of active orbital indices to be frozen in the MCSCF optimization.
        If provided, all gradients involving these orbitals will be zeroed out.
    maxiter : int, optional, default=50
        Maximum number of macroiterations.
    econv : float, optional, default=1e-8
        Energy convergence tolerance.
    gconv : float, optional, default=1e-7
        Gradient convergence tolerance.
    die_if_not_converged : bool, optional, default=True
        If True, raises an error if the optimization does not converge.
    freeze_inter_gas_rots : bool, optional, default=False
        Whether to freeze inter-GAS orbital rotations when multiple GASes are defined.
    micro_maxiter : int, optional, default=6
        Maximum number of microiterations for L-BFGS.
    max_rotation : float, optional, default=0.2
        Maximum orbital rotation size for L-BFGS.
    ci_* : various, optional
        Various parameters for the CI solver. See `CISolver` for details.
        Available parameters:
        - ci_guess_per_root
        - ci_ndets_per_guess
        - ci_collapse_per_root
        - ci_basis_per_root
        - ci_maxiter
        - ci_econv
        - ci_rconv
        - ci_energy_shift
        All parameters have the same default values.
    do_diis : bool, optional
        Whether DIIS acceleration is used.
    diis_start : int, optional, default=15
        Start saving DIIS vectors after this many iterations.
    diis_nvec : int, optional, default=8
        The number of vectors to keep in the DIIS.
    diis_min : int, optional, default=4
        The minimum number of vectors to perform extrapolation.
    do_transition_dipole : bool, optional, default=False
        Whether to compute transition dipole moments.

    Notes
    -----
    See J. Chem. Phys. 152, 074102 (2020) for the current implementation
    of a unified CASSCF/GASSCF gradient and diagonal Hessian.
    The non-GAS part of diagonal Hessian implementation follows Theor. Chem. Acc. 97, 88-95 (1997).
    An earlier implementation (CASSCF only) used J. Chem. Phys. 142, 224103 (2015).
    """

    active_frozen_orbitals: list[int] = None
    freeze_inter_gas_rots: bool = False

    ### Macroiteration parameters
    maxiter: int = 50
    econv: float = 1e-8
    gconv: float = 1e-7
    die_if_not_converged: bool = True

    ### L-BFGS solver (microiteration) parameters
    micro_maxiter: int = 6
    max_rotation: float = 0.2

    ### CI solver parameters
    ci_maxiter: int = 100
    ci_econv: float = 1e-12
    ci_rconv: float = 1e-6
    ci_guess_per_root: int = 2
    ci_ndets_per_guess: int = 10
    ci_collapse_per_root: int = 2
    ci_basis_per_root: int = 4
    ci_energy_shift: float = None

    ### DIIS parameters
    do_diis: bool = None
    diis_start: int = 15
    diis_nvec: int = 8
    diis_min: int = 4

    ### Post-iteration
    do_transition_dipole: bool = False

    ### Non-init attributes
    converged: bool = field(default=False, init=False)
    executed: bool = field(default=False, init=False)
    two_component: bool = field(default=False, init=False)

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
        super()._startup()
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
        _OrbOptimizer = RelOrbOptimizer if self.two_component else OrbOptimizer
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

        _CISolver = RelCISolver if self.two_component else CISolver
        self.ci_solver = _CISolver(
            states=self.states,
            core_orbitals=self.mo_space.docc_orbitals,
            active_orbitals=self.mo_space.active_orbitals,
            nroots=self.sa_info.nroots,
            weights=self.sa_info.weights,
            log_level=self.ci_solver_verbosity,
            die_if_not_converged=False,
            econv=self.ci_econv,
            rconv=self.ci_rconv,
            maxiter=self.ci_maxiter,
            guess_per_root=self.ci_guess_per_root,
            ndets_per_guess=self.ci_ndets_per_guess,
            collapse_per_root=self.ci_collapse_per_root,
            basis_per_root=self.ci_basis_per_root,
            energy_shift=self.ci_energy_shift,
        )(self.parent_method)
        # iteration 0: one step of CI optimization to bootstrap the orbital optimization
        self.iter = 0
        self.ci_solver.run()

        # Initialize the LBFGS solver that finds the optimal orbital
        # at fixed CI expansion using the gradient and diagonal Hessian
        self.lbfgs_solver = LBFGS(
            epsilon=self.gconv,
            max_dir=self.max_rotation,
            step_length_method="max_correction",
            maxiter=self.micro_maxiter,
            dtype=self.dtype,
        )

        diis = DIIS(
            diis_start=self.diis_start,
            diis_nvec=self.diis_nvec,
            diis_min=self.diis_min,
            do_diis=self.do_diis,
        )

        width = 115

        logger.log_info1(self.mo_space)
        logger.log_info1(f"# of nonredundant rotations: {self.nrr.sum()}\n")

        logger.log_info1("Entering orbital optimization loop")
        logger.log_info1("\nConvergence criteria ('.' if satisfied, 'x' otherwise):")
        logger.log_info1(f"  {'1. RMS(grad - grad_old)':<25} < {self.gconv:.1e}")
        logger.log_info1(f"  {'2. ||E_CI - E_orb||':<25} < {self.econv:.1e}")
        logger.log_info1(f"  {'3. ||E_CI - E_CI_old||':<25} < {self.econv:.1e}")
        logger.log_info1(f"  {'4. ||E_avg - E_avg_old||':<25} < {self.econv:.1e}")
        logger.log_info1(f"  {'5. ||E_orb - E_orb_old||':<25} < {self.econv:.1e}\n")

        logger.log_info1("=" * width)
        logger.log_info1(
            f'{"Iteration":>10} {"E_CI":>20} {"ΔE_CI":>12} {"E_orb":>20} {"ΔE_orb":>12} {"RMS(Δgrad)":>12} {"#micro":>8} {"Conv":>8} {"DIIS":>5}'
        )
        logger.log_info1("-" * width)

        # E_ci: list[float],CI eigenvalues,
        # E_avg: float, ensemble average energy,
        # E_orb: float, energy after orbital optimization
        self.E_ci = np.array(self.ci_solver.E)
        self.E_ci_old = self.E_ci.copy()
        self.E_avg = self.E_orb = self.ci_solver.compute_average_energy()
        self.E_orb_old = self.E_orb
        self.E_avg_old = self.E_avg

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
        # It's used as the initial guess at the start of each orbital optimization, and also for DIIS
        R = np.zeros(self.orb_opt.nrot, dtype=self.dtype)

        while self.iter < self.maxiter:
            # 1. Optimize orbitals at fixed CI expansion
            self.E_orb = self.lbfgs_solver.minimize(self.orb_opt, R)
            self._C = self.orb_opt.C.copy()
            # 2. Convergence checks
            _dg = self.lbfgs_solver.g - self.g_old
            self.g_rms = np.sqrt(np.mean((_dg.conj() * _dg).real))
            self.g_old = self.lbfgs_solver.g.copy()
            conv, conv_str = self._check_convergence()
            lbfgs_str = f"{self.lbfgs_solver.iter}/{'Y' if self.lbfgs_solver.converged else 'N'}"
            iter_info = f"{self.iter:>10d} {self.E_avg.real:>20.10f} {self.delta_ci_avg.real:>12.4e} "
            iter_info += f"{self.E_orb.real:>20.10f} {self.delta_orb.real:>12.4e} {self.g_rms.real:>12.4e} {lbfgs_str:>8} {conv_str:>8}"
            if conv:
                logger.log_info1(iter_info)
                self.converged = True
                break

            # 3. DIIS Extrapolation
            R = diis.update(R, self.g_old)
            iter_info += f" {diis.status:>5s}"
            logger.log_info1(iter_info)
            # if diis has performed extrapolation
            if "E" in diis.status:
                # orb_opt.evaluate updates the 1 and 2-electron integrals for CI
                _ = self.orb_opt.evaluate(R)

            # 4. Optimize CI expansion at fixed orbitals
            self.ci_solver.set_ints(
                self.orb_opt.Ecore + self.system.nuclear_repulsion,
                self.orb_opt.Fcore[self.actv, self.actv],
                self.orb_opt.get_active_space_ints(),
            )
            self.ci_solver.run()
            self.E_avg = self.ci_solver.compute_average_energy()
            self.E_ci = np.array(self.ci_solver.E)
            self.E = self.E_avg
            self.g1_act = self.make_average_1rdm()
            g2_act = self.make_average_2rdm()
            self.orb_opt.set_rdms(self.g1_act, g2_act)
            self.iter += 1
        else:
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
        logger.log_info1(
            f"{'Final CI':>10} {self.E_avg:>20.10f} {'-':>12} {self.E_orb:>20.10f} {'-':>12} {'-':>12} {'-':>6} {conv_str:>10s}"
        )

        logger.log_info1("=" * width)
        logger.log_info1(f"Orbital optimization converged in {self.iter} iterations.")
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
            self.C[0] = semi.C_semican[self.mo_space.contig_to_orig].copy()

            # recompute the CI vectors in the semicanonical basis
            if self.two_component:
                ints = SpinorbitalIntegrals(
                    system=self.system,
                    C=self.C[0],
                    spinorbitals=self.mo_space.active_indices,
                    core_spinorbitals=self.mo_space.docc_indices,
                    use_aux_corr=True,
                )
            else:
                ints = RestrictedMOIntegrals(
                    system=self.system,
                    C=self.C[0],
                    orbitals=self.mo_space.active_indices,
                    core_orbitals=self.mo_space.docc_indices,
                    use_aux_corr=True,
                )
            self.ci_solver.set_ints(ints.E, ints.H, ints.V)
            self.ci_solver.run()

        self.executed = True
        return self

    def _post_process(self):
        pretty_print_ci_summary(self.sa_info, self.ci_solver.evals_per_solver)
        self.ci_solver.compute_natural_occupation_numbers()
        pretty_print_ci_nat_occ_numbers(
            self.sa_info, self.mo_space, self.ci_solver.nat_occs
        )
        top_dets = self.ci_solver.get_top_determinants()
        pretty_print_ci_dets(self.sa_info, self.mo_space, top_dets)
        if not self.two_component:
            # TODO: enable AO composition for 2c
            self._print_ao_composition()
        if self.do_transition_dipole:
            self.ci_solver.compute_transition_properties(self.C[0])
            pretty_print_ci_transition_props(
                self.sa_info,
                self.ci_solver.tdm_per_solver,
                self.ci_solver.fosc_per_solver,
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
            _irrid = np.array(self.irrep_indices)
            # equivalent to:
            # for i, j in range(nmo):
            #   if i^j != 0:
            #       nrr[i, j] = False
            nrr[(_irrid[:, None] ^ _irrid != 0)] = False

        return nrr

    def _check_convergence(self):
        is_grad_conv = self.g_rms < self.gconv
        is_ci_orb_conv = abs(self.E_orb - self.E_avg) < self.econv
        is_ci_eigval_conv = np.all(abs(self.E_ci - self.E_ci_old) < self.econv)

        self.delta_ci_avg = self.E_avg - self.E_avg_old
        is_ci_avg_conv = abs(self.delta_ci_avg) < self.econv

        self.delta_orb = self.E_orb - self.E_orb_old
        is_orb_conv = abs(self.E_orb - self.E_orb_old) < self.econv

        criteria = [
            is_grad_conv,
            is_ci_orb_conv,
            is_ci_eigval_conv,
            is_ci_avg_conv,
            is_orb_conv,
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


@dataclass
class RelMCOptimizer(RelActiveSpaceSolver, MCOptimizer): ...
