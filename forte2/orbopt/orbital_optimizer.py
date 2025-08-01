import numpy as np
import scipy as sp
from dataclasses import dataclass, field

# from forte2.ci import CISolver
from forte2.base_classes.active_space_solver import ActiveSpaceSolver
from forte2.orbitals import Semicanonicalizer
from forte2.jkbuilder import FockBuilder, RestrictedMOIntegrals
from forte2.helpers import logger, LBFGS, DIIS
from forte2.system.basis_utils import BasisInfo
from forte2.ci.ci_utils import (
    pretty_print_ci_summary,
    pretty_print_ci_nat_occ_numbers,
    pretty_print_ci_dets,
    pretty_print_ci_transition_props,
)


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
    maxiter : int, optional, default=50
        Maximum number of macroiterations.
    econv : float, optional, default=1e-8
        Energy convergence tolerance.
    gconv : float, optional, default=1e-7
        Gradient convergence tolerance.
    micro_maxiter : int, optional, default=6
        Maximum number of microiterations for L-BFGS.
    ci_maxiter : int, optional, default=50
        Maximum number of iterations for CI optimization.
    max_rotation : float, optional, default=0.2
        Maximum orbital rotation size for L-BFGS.
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
    """

    optimize_frozen_orbs: bool = True

    ### Macroiteration parameters
    maxiter: int = 50
    econv: float = 1e-8
    gconv: float = 1e-7

    ### L-BFGS solver (microiteration) parameters
    micro_maxiter: int = 6
    max_rotation: float = 0.2

    ### CI solver parameters
    ci_maxiter: int = 50

    ### DIIS parameters
    do_diis: bool = None
    diis_start: int = 15
    diis_nvec: int = 8
    diis_min: int = 4

    ### Post-iteration
    do_transition_dipole: bool = False

    ### Non-init attributes
    executed: bool = field(default=False, init=False)

    def __call__(self, method):
        self.parent_method = method
        ### make sure we don't print the CI output at INFO1 level
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
        # core slice will include frozen orbitals,
        # if optimize_frozen_orbs is False, then the relevant
        # gradients will be zeroed out by nrr
        self.core = self.mo_space.docc
        # self.actv will be a list if multiple GASes are defined
        self.actv = self.mo_space.actv
        self.virt = self.mo_space.uocc

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
        self.Hcore = self.system.ints_hcore()  # hcore in AO basis (even 1e-sf-X2C)
        fock_builder = FockBuilder(self.system)

        # Intialize the two central objects for the two-step orbital-CI optimization:
        # orbital optimizer and CI optimizer
        # the loop simply proceeds as follows:
        # for i in range(max_macro_iter):
        #     1. minimize energy wrt orbital rotations at current CI expansion
        #       (this is typically done iteratively with micro-iterations using L-BFGS)
        #     2. minimize energy wrt CI expansion at current orbitals
        #       (this is just the diagonalization of the active-space CI Hamiltonian)
        do_gas = self.mo_space.ngas > 1
        self.orb_opt = OrbOptimizer(
            self._C,
            (self.core, self.actv, self.virt),
            fock_builder,
            self.Hcore,
            self.system.nuclear_repulsion,
            self.nrr,
            gas_ref=do_gas,
        )
        from forte2.ci import CISolver

        self.ci_solver = CISolver(
            states=self.states,
            core_orbitals=self.mo_space.docc_orbitals,
            active_orbitals=self.mo_space.active_orbitals,
            nroots=self.sa_info.nroots,
            weights=self.sa_info.weights,
            log_level=self.ci_solver_verbosity,
        )(self.parent_method)
        # iteration 0: one step of CI optimization to bootsrap the orbital optimization
        self.iter = 0
        self.ci_solver.run()

        # Initialize the LBFGS solver that finds the optimal orbital
        # at fixed CI expansion using the gradient and diagonal Hessian
        self.lbfgs_solver = LBFGS(
            epsilon=self.gconv,
            max_dir=self.max_rotation,
            step_length_method="max_correction",
            maxiter=self.micro_maxiter,
        )

        diis = DIIS(
            diis_start=self.diis_start,
            diis_nvec=self.diis_nvec,
            diis_min=self.diis_min,
            do_diis=self.do_diis,
        )

        width = 115
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

        self.g1_act = self.ci_solver.make_average_sf_1rdm()
        g2_act = 0.5 * self.ci_solver.make_average_sf_2rdm()
        # ci_maxiter_save = self.ci_solver.get_maxiter()
        # self.ci_solver.set_maxiter(self.ci_maxiter)

        # Prepare the orbital optimizer
        self.orb_opt.set_rdms(self.g1_act, g2_act)
        self.orb_opt._compute_Fcore()
        self.orb_opt.get_eri_gaaa()
        self.E_orb = self.E_avg
        self.E_orb_old = self.E_orb

        self.g_old = np.zeros(self.orb_opt.nrot, dtype=float)

        # This holds the *overall* orbital rotation, C_current = C_0 @ exp(R)
        # It's as the intial guess at the start of each orbital optimization and for DIIS
        R = np.zeros(self.orb_opt.nrot, dtype=float)

        while self.iter < self.maxiter:
            # 1. Optimize orbitals at fixed CI expansion
            self.E_orb = self.lbfgs_solver.minimize(self.orb_opt, R)
            self._C = self.orb_opt.C.copy()
            # 2. Convergence checks
            self.g_rms = np.sqrt(np.mean((self.lbfgs_solver.g - self.g_old) ** 2))
            self.g_old = self.lbfgs_solver.g.copy()
            conv, conv_str = self._check_convergence()
            lbfgs_str = f"{self.lbfgs_solver.iter}/{'Y' if self.lbfgs_solver.converged else 'N'}"
            iter_info = f"{self.iter:>10d} {self.E_avg:>20.10f} {self.delta_ci_avg:>12.4e} {self.E_orb:>20.10f} {self.delta_orb:>12.4e} {self.g_rms:>12.4e} {lbfgs_str:>8} {conv_str:>8}"
            if conv:
                logger.log_info1(iter_info)
                break

            # 3. DIIS Extrapolation
            R = diis.update(R, self.g_old)
            iter_info += f" {diis.status:>5s}"
            logger.log_info1(iter_info)
            # if diis has performed extrapolation
            if "E" in diis.status:
                # orb_opt.evaluate updates the 1 and 2-electron integrals for CI
                _ = self.orb_opt.evaluate(R, self.lbfgs_solver.g, do_g=False)

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
            self.g1_act = self.ci_solver.make_average_sf_1rdm()
            g2_act = 0.5 * self.ci_solver.make_average_sf_2rdm()
            self.orb_opt.set_rdms(self.g1_act, g2_act)
            self.iter += 1
        else:
            logger.log_info1("=" * width)
            raise RuntimeError(
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
                g1_sf=self.ci_solver.make_average_sf_1rdm(),
                C=self.C[0],
                system=self.system,
                fock_builder=fock_builder,
                mix_inactive=not self.optimize_frozen_orbs,
                mix_active=False,
            )
            semi.run()
            self.C[0] = semi.C_semican.copy()

            # recompute the CI vectors in the semicanonical basis
            ints = RestrictedMOIntegrals(
                system=self.system,
                C=self.C[0],
                orbitals=self.mo_space.active_indices,
                core_orbitals=self.mo_space.docc_indices,
                use_aux_corr=True,
                fock_builder=fock_builder,  # avoid reinitialization of FockBuilder
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
        nrr = np.zeros((self.system.nbf, self.system.nbf), dtype=bool)

        # TODO: handle GAS/RHF/ROHF cases
        if self.optimize_frozen_orbs:
            _core = self.mo_space.docc
            _virt = self.mo_space.uocc
        else:
            _core = self.mo_space.core
            _virt = self.mo_space.virt

        # GASn-GASm rotations
        if self.mo_space.ngas > 1:
            for i in range(self.mo_space.ngas):
                for j in range(i + 1, self.mo_space.ngas):
                    nrr[self.mo_space.gas[i], self.mo_space.gas[j]] = True

        nrr[_core, _virt] = True
        nrr[self.actv, _virt] = True
        nrr[_core, self.actv] = True
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


class OrbOptimizer:
    def __init__(self, C, extents, fock_builder, hcore, e_nuc, nrr, gas_ref=False):
        self.core, self.actv, self.virt = extents
        self.C = C
        self.C0 = C.copy()
        self.Cgen = C
        self.Cact = C[:, self.actv]
        self.Ccore = C[:, self.core]
        self.ncore = self.Ccore.shape[1]
        self.nact = self.Cact.shape[1]
        self.nvirt = self.C.shape[1] - self.ncore - self.nact
        self.fock_builder: FockBuilder = fock_builder
        self.hcore = hcore
        self.nrr = nrr
        self.nrot = self.nrr.sum()
        self.e_nuc = e_nuc
        self.gas_ref = gas_ref

        self.R = np.zeros(self.nrot, dtype=float)
        self.U = np.eye(self.C.shape[1], dtype=float)

    def get_eri_gaaa(self):
        self.eri_gaaa = self.fock_builder.two_electron_integrals_gen_block(
            self.Cgen, *(self.Cact,) * 3
        )
        return self.eri_gaaa

    def set_rdms(self, g1, g2):
        self.g1 = g1
        self.g2 = g2

    def get_active_space_ints(self):
        """
        Returns the active space integrals.
        """
        return self.eri_gaaa[self.actv, ...]

    def evaluate(self, x, g, do_g=True):
        do_update_integrals = self._update_orbitals(x)
        if do_update_integrals:
            self._compute_Fcore()
            self.get_eri_gaaa()

        E_orb = self._compute_reference_energy()

        if do_g:
            grad = -self._compute_orbgrad()
            g = self._mat_to_vec(grad)

        return E_orb, g

    def hess_diag(self, x):
        hess = self._compute_orbhess()
        h0 = self._mat_to_vec(hess)
        return h0

    def _update_orbitals(self, R):
        dR = R - self.R
        if np.max(np.abs(dR)) < 1e-12:
            # no change in orbitals, skip the update
            return False
        self.R += dR
        self.U = self.U @ self._expm(dR)

        self.C = self.C0 @ self.U
        self.Cgen = self.C
        self.Ccore = self.C[:, self.core]
        self.Cact = self.C[:, self.actv]
        return True

    def _expm(self, vec):
        M = self._vec_to_mat(vec)
        eM = sp.linalg.expm(M)
        return eM

    def _vec_to_mat(self, x):
        R = np.zeros_like(self.C)
        R[self.nrr] = x
        R += -R.T
        return R

    def _mat_to_vec(self, R):
        return R[self.nrr]

    def _compute_reference_energy(self):
        energy = self.Ecore + self.e_nuc
        energy += np.einsum("uv,uv->", self.Fcore[self.actv, self.actv], self.g1)
        # factor of 0.5 already included in g2
        energy += np.einsum("uvxy,uvxy->", self.get_active_space_ints(), self.g2)
        return energy

    def _compute_Fcore(self):
        # Compute the core Fock matrix [Eq. (9) or (18)], also return the core energy
        Jcore, Kcore = self.fock_builder.build_JK([self.Ccore])
        self.Fcore = np.einsum(
            "mp,mn,nq->pq",
            self.Cgen.conj(),
            self.hcore + 2 * Jcore[0] - Kcore[0],
            self.Cgen,
            optimize=True,
        )
        self.Ecore = np.einsum(
            "pi,qi,pq->",
            self.Ccore.conj(),
            self.Ccore,
            2 * self.hcore + 2 * Jcore[0] - Kcore[0],
        )

    def _compute_Fact(self):
        Jact, Kact = self.fock_builder.build_JK_generalized(self.Cact, self.g1)

        # [Eq. (20)]
        self.Fact = np.einsum(
            "mp,mn,nq->pq",
            self.Cgen.conj(),
            2 * Jact - Kact,
            self.Cgen,
            optimize=True,
        )

    def _compute_YZ_intermediates(self):
        # compute the Y intermediate [Algorithm 1, line 10]
        Y = np.einsum("pu,tu->pt", self.Fcore[:, self.actv], self.g1)
        # compute the Z intermediate [Algorithm 1, line 11]
        Z = np.einsum("puvw,tuvw->pt", self.eri_gaaa, self.g2)
        return Y, Z

    def _compute_orbgrad(self):
        self._compute_Fact()

        self.Y, self.Z = self._compute_YZ_intermediates()
        orbgrad = np.zeros_like(self.Fcore)

        Fcore_cv = self.Fcore[self.core, self.virt]
        Fcore_ca = self.Fcore[self.core, self.actv]
        Fact_cv = self.Fact[self.core, self.virt]
        Fact_ca = self.Fact[self.core, self.actv]

        Y_va = self.Y[self.virt, :]
        Y_ca = self.Y[self.core, :]
        Z_va = self.Z[self.virt, :]
        Z_ca = self.Z[self.core, :]

        orbgrad[self.core, self.virt] = 4 * Fcore_cv + 2 * Fact_cv
        orbgrad[self.actv, self.virt] = 2 * Y_va.T + 4 * Z_va.T
        orbgrad[self.core, self.actv] = 4 * Fcore_ca + 2 * Fact_ca - 2 * Y_ca - 4 * Z_ca

        if self.gas_ref:
            # tei (xt|vw) => eri_aaaa
            self.eri_aaaa = np.zeros((self.nact,self.nact,self.nact,self.nact))
            for i in range(self.nact):
                for j in range(self.nact):
                    for k in range(self.nact):
                        for l in range(self.nact):
                            self.eri_aaaa[i][j][k][l] = self.eri_gaaa[self.ncore + i][j][k][l]

            # A_xu = sum_v[Fcore_xv * D_vu] + sum_tvw[(xt|vw) * D_tu,vw]
            # A_xu = self.Fcore[self.actv, self.actv] * self.g1 + self.eri_aaaa * self.g2
            Fcore_aa = self.Fcore[self.actv, self.actv]
            self.A_xu = np.zeros((self.nact, self.nact))
            for x in range(self.nact):
                for u in range(self.nact):
                    for v in range(self.nact):
                        self.A_xu[x][u] +=  (Fcore_aa[x][v] * self.g1[v][u])

            for x in range(self.nact):
                for u in range(self.nact):
                    for v in range(self.nact):
                        for t in range(self.nact):
                            for w in range(self.nact):
                                self.A_xu[x][u] += (self.eri_aaaa[x][t][v][w] * self.g2[t][u][v][w])
                        
            
            orbgrad[self.actv, self.actv] = 2 * (self.A_xu - self.A_xu.T)

        return orbgrad

    def _compute_orbhess(self):
        orbhess = np.zeros_like(self.Fcore)

        Fcore_cc = np.diag(self.Fcore[self.core, self.core])
        Fcore_aa = np.diag(self.Fcore[self.actv, self.actv])
        Fcore_vv = np.diag(self.Fcore[self.virt, self.virt])
        Fact_cc = np.diag(self.Fact[self.core, self.core])
        Fact_aa = np.diag(self.Fact[self.actv, self.actv])
        Fact_vv = np.diag(self.Fact[self.virt, self.virt])
        g1_diag = np.diag(self.g1)
        Y_aa = np.diag(self.Y[self.actv, :])
        Z_aa = np.diag(self.Z[self.actv, :])

        # Algorithm 1, line 20
        vdiag = 4 * Fcore_vv + 2 * Fact_vv
        cdiag = 4 * Fcore_cc + 2 * Fact_cc
        orbhess[self.core, self.virt] = vdiag - cdiag[:, None]

        # Algorithm 1, line 21
        av_diag = 2 * np.outer(g1_diag, Fcore_vv) + np.outer(g1_diag, Fact_vv)
        aa_diag = 2 * Y_aa + 4 * Z_aa
        orbhess[self.actv, self.virt] = av_diag - aa_diag[:, None]

        # Algorithm 1, line 22
        ca_diag = 2 * np.outer(Fcore_cc, g1_diag) + np.outer(Fact_cc, g1_diag)
        aa_diag = 4 * Fcore_aa + 2 * Fact_aa - 2 * Y_aa - 4 * Z_aa
        cc_diag = -4 * Fcore_cc - 2 * Fact_cc
        orbhess[self.core, self.actv] = ca_diag + aa_diag[None, :] + cc_diag[:, None]

        # Compute GASn-GASm active-active blocks. [J. Chem. Phys. 152, 074102 (2020)]
        if self.gas_ref:
            # nactv2 = self.nact * self.nact
            # nactv3 = nactv2 * self.nact

            # fc_data = self.Fcore # F^c_{pq} -> Fc_.block("aa").data()
            # v_data = self.eri_gaaa    # tei -> V_.block("aaaa").data()
            # d2_data = self.g2    # 2RDM -> D2_.block("aaaa").data()

            ## G^{uu}_{vv}

            ## (uu|xy)
            jk_internal_ = np.zeros((self.nact, self.nact, self.nact)) # name self.eri_aaa

            for i in range(self.nact):
                for j in range(self.nact):
                    for k in range(self.nact):
                        jk_internal_[i][j][k] = self.eri_gaaa[self.ncore + i][i][j][k]

            # jk_internal_.block("aaa").iterate([&](const std::vector<size_t>& i, double& value) {
            #     auto idx = i[0] * nactv3 + i[0] * nactv2 + i[1] * nactv_ + i[2];
            #     value = v_data[idx];
            # });

            ## D_{vv,xy}
            d2_internal_ = np.zeros((self.nact, self.nact, self.nact))

            for i in range(self.nact):
                for j in range(self.nact):
                    for k in range(self.nact):
                        d2_internal_[i][j][k] = self.g2[i][i][j][k]

            Guu = np.zeros((self.nact, self.nact))

            for i in range(self.nact):
                for j in range(self.nact):
                    for k in range(self.nact):
                        for l in range(self.nact):
                            Guu[i][j] = jk_internal_[i][k][l] * d2_internal_[j][k][l]

            # d2_internal_.block("aaa").iterate([&](const std::vector<size_t>& i, double& value) {
            #     auto idx = i[0] * nactv3 + i[0] * nactv2 + i[1] * nactv_ + i[2];
            #     value = d2_data[idx];
            # });

            # Guu_["uv"] = jk_internal_["uxy"] * d2_internal_["vxy"];

            ## (ux|uy)
            jk_internal_2 = np.zeros((self.nact,self.nact,self.nact))

            for u in range(self.nact):
                for x in range(self.nact):
                    for y in range(self.nact):
                        jk_internal_2[u][x][y] = self.eri_gaaa[self.ncore + u][x][u][y]


            # jk_internal_.block("aaa").iterate([&](const std::vector<size_t>& i, double& value) {
            #     auto idx = i[0] * nactv3 + i[1] * nactv2 + i[0] * nactv_ + i[2];
            #     value = v_data[idx];
            # });

            ## D_{vx,vy}
            d2_internal_2 = np.zeros((self.nact,self.nact,self.nact))

            for v in range(self.nact):
                for x in range(self.nact):
                    for y in range(self.nact):
                        d2_internal_2[v][x][y] = self.g2[v][x][v][y]

            # d2_internal_.block("aaa").iterate([&](const std::vector<size_t>& i, double& value) {
            #     auto idx = i[0] * nactv3 + i[1] * nactv2 + i[0] * nactv_ + i[2];
            #     value = d2_data[idx];
            # });

            for u in range(self.nact):
                for v in range(self.nact):
                    for x in range(self.nact):
                        for y in range(self.nact):
                            Guu[u][v] += 2.0 * jk_internal_2[u][x][y] * d2_internal_2[v][x][y]

            # Guu_["uv"] += 2.0 * jk_internal_["uxy"] * d2_internal_["vxy"];

            # TODO double check
            for u in range(self.nact):
                for v in range(self.nact):
                    Guu[u][v] += self.Fcore[self.ncore + u][self.ncore + u] * self.g1[v][v]

            # Guu_.block("aa").iterate([&](const std::vector<size_t>& i, double& value) {
            #     auto i0 = i[0] * nactv_ + i[0];
            #     auto i1 = i[1] * nactv_ + i[1];
            #     value += fc_data[i0] * d1_data[i1];
            # });

            ## G^{uv}_{vu}
            Guv = np.zeros((self.nact, self.nact))
            for u in range(self.nact):
                for v in range(self.nact):
                    Guv[u][v] = self.Fcore[self.ncore + u][self.ncore + v] * self.g1[v][u]

            # Guv_["uv"] = Fc_["uv"] * D1_["vu"];

            for u in range(self.nact):
                for v in range(self.nact):
                    for x in range(self.nact):
                        for y in range(self.nact):
                            Guv[u][v] += self.eri_gaaa[self.ncore + u][v][x][y] * self.g2[v][u][x][y]

            # Guv_["uv"] += V_["uvxy"] * D2_["vuxy"];

            for u in range(self.nact):
                for v in range(self.nact):
                    for x in range(self.nact):
                        for y in range(self.nact):
                            Guv[u][v] += 2.0 * self.eri_gaaa[self.ncore + u][x][v][y] * self.g2[v][x][u][y]

            # Guv_["uv"] += 2.0 * V_["uxvy"] * D2_["vxuy"];

            ## build diagonal Hessian
            h_diag = np.zeros((self.nact, self.nact))

            for u in range(self.nact):
                for v in range(self.nact):
                    h_diag[u][v] = 2.0 * Guu[u][v] + 2.0 * Guu[v][u] - 2.0 * Guv[u][v] - 2.0 * Guv[v][u]

            # h_diag_["uv"] = 2.0 * Guu_["uv"];
            # h_diag_["uv"] += 2.0 * Guu_["vu"];
            # h_diag_["uv"] -= 2.0 * Guv_["uv"];
            # h_diag_["uv"] -= 2.0 * Guv_["vu"];

            for u in range(self.nact):
                for v in range(self.nact):
                    h_diag[u][v] -= 2.0 * (self.A_xu[u][u] + self.A_xu[v][v])

            # h_diag_.block("aa").iterate([&](const std::vector<size_t>& i, double& value) {
            #     auto i0 = i[0] * nactv_ + i[0];
            #     auto i1 = i[1] * nactv_ + i[1];
            #     value -= 2.0 * (a_data[i0] + a_data[i1]);
            # });

            for u in range(self.nact):
                for v in range(self.nact):
                    orbhess[self.ncore + u][self.ncore + v] = h_diag[u][v]

        return orbhess
