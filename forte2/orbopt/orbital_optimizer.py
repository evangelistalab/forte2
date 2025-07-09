import numpy as np
import scipy as sp
from dataclasses import dataclass, field

from forte2.jkbuilder import FockBuilder
from forte2.helpers.mixins import MOsMixin, SystemMixin
from forte2.helpers import logger


@dataclass
class OrbitalOptimizer(MOsMixin, SystemMixin):
    # placeholder optimization parameters
    maxiter: int = 50
    gradtol: float = 1e-6
    optimizer: str = "L-BFGS"
    max_step_size: float = 0.1

    executed: bool = field(default=False, init=False)

    def __call__(self, method):
        self.parent_method = method
        return self

    def run(self):
        # TODO: skip the first CI step in the MCSCF
        if not self.parent_method.executed:
            self.parent_method.run()

        current_verbosity = logger.get_verbosity_level()
        # only log subproblem if the verbosity is higher than INFO1
        if current_verbosity > 3:
            self.parent_method.log_level = current_verbosity
        else:
            self.parent_method.log_level = current_verbosity + 1
            self.parent_method.solver.log_level = current_verbosity + 1

        SystemMixin.copy_from_upstream(self, self.parent_method)
        MOsMixin.copy_from_upstream(self, self.parent_method)

        self.Hcore = self.system.ints_hcore()  # hcore in AO basis (even 1e-sf-X2C)
        fock_builder = FockBuilder(self.system)

        self._make_spaces_contiguous()
        self.nrr = self._get_nonredundant_rotations()

        self.cas_grad = OrbitalGradient(
            self.C[0],
            (self.core, self.actv, self.virt),
            fock_builder,
            self.Hcore,
            self.system.nuclear_repulsion_energy(),
            self.nrr,
        )
        self.as_solver = ActiveSpaceSolver(self.parent_method)

        logger.log_info1(
            f'{"Iteration":>10} {"E_orb":>20} {"E_CI":>20} {"norm(g)":>20} '
        )
        self.iter = 0
        self.E = self.E_orb = self.E_as = self.parent_method.E

        g1_act, g2_act = self.as_solver.make_rdm12()
        eri_gaaa = self.cas_grad.get_eri_gaaa()

        while self.iter < self.maxiter:
            # Algorithm 1, line 4
            self.cas_grad.set_rdms(g1_act, g2_act)

            self.E_orb = self.cas_grad.run()

            logger.log_info1(
                f"{self.iter:>10d} {self.E_orb:>20.10f} {self.E_as[0]:>20.10f} {abs(np.linalg.norm(self.cas_grad.orbgrad, np.inf)):>20.10f}"
            )

            if abs(np.linalg.norm(self.cas_grad.orbgrad, np.inf)) < self.gradtol:
                break

            # Algorithm 1, line 6-7 - solve the subproblem and get the energies and RDMs
            self.as_solver.set_active_space_ints(
                self.cas_grad.Ecore + self.system.nuclear_repulsion_energy(),
                self.cas_grad.Fcore[self.actv, self.actv].copy(),
                eri_gaaa[self.actv, ...].copy(),
            )
            self.E_as = self.E = self.as_solver.run()
            g1_act, g2_act = self.as_solver.make_rdm12()
            eri_gaaa = self.cas_grad.get_eri_gaaa()
            self.iter += 1
        else:
            logger.log_info1(
                f"Orbital optimization did not converge in {self.maxiter} iterations."
            )
            self.parent_method.log_level = current_verbosity
            self.parent_method.solver.log_level = current_verbosity
            return

        logger.log_info1(f"Orbital optimization converged in {self.iter} iterations.")
        logger.log_info1(f"Final orbital optimized energy: {self.E[0]:.10f}")
        self.parent_method.log_level = current_verbosity
        self.parent_method.solver.log_level = current_verbosity
        self.executed = True

    def _make_spaces_contiguous(self):
        """
        Swap the orbitals to ensure that the core, active, and virtual orbitals
        are contiguous in the flattened orbital array.
        """
        # [todo] handle GAS/RHF/ROHF cases
        core = self.parent_method.core_orbitals
        actv = self.parent_method.flattened_orbitals
        virt = sorted(list(set(range(self.system.nbf())) - set(core) - set(actv)))
        argsort = np.argsort(core + actv + virt)
        self.C[0][:, argsort] = self.C[0].copy()
        self.core = slice(0, len(core))
        self.actv = slice(len(core), len(core) + len(actv))
        self.virt = slice(len(core) + len(actv), None)

    def _get_nonredundant_rotations(self):
        nrr = np.zeros((self.system.nbf(), self.system.nbf()), dtype=bool)

        # TODO: handle GAS/RHF/ROHF cases
        nrr[self.core, self.virt] = True
        nrr[self.actv, self.virt] = True
        nrr[self.core, self.actv] = True
        # make sure the rotations are symmetric
        nrr = nrr | nrr.T

        return nrr


class OrbitalGradient:
    def __init__(self, C, extents, fock_builder, hcore, e_nuc, nrr):
        self.core, self.actv, self.virt = extents
        self.C = C
        self.Cgen = C
        self.Cact = C[:, self.actv]
        self.Ccore = C[:, self.core]
        self.fock_builder = fock_builder
        self.hcore = hcore
        self.nrr = nrr
        self.e_nuc = e_nuc

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

    def run(self):
        self.compute_Fcore()
        self.orbgrad = self.compute_orbgrad()
        self.orbhess = self.compute_orbhess()
        self.C = self.rotate_orbitals()
        self.Cgen = self.C
        self.Ccore = self.C[:, self.core]
        self.Cact = self.C[:, self.actv]
        return self.compute_reference_energy()

    def compute_reference_energy(self):
        energy = self.Ecore + self.e_nuc
        energy += np.einsum("uv,uv->", self.Fcore[self.actv, self.actv], self.g1)
        # factor of 0.5 already included in g2
        energy += np.einsum("uvxy,uvxy->", self.get_active_space_ints(), self.g2)
        return energy

    def compute_Fcore(self):
        # Compute the core Fock matrix [Eq. (9)], also return the core energy
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

    def compute_Fact(self):
        # compute the active Fock matrix [Algorithm 1, line 9]
        # gamma_act (nmo * nmo) is defined in [Eq. (19)]
        # as (gamma_act)_pq = C_pt C_qu g1_tu
        # we decompose g1 and contract it into the coefficients
        # so more efficient coefficient-based J-builder can be used
        try:
            # C @ g1_act C.T = C @ L @ L.T @ C.T == Cp @ Cp.T
            L = np.linalg.cholesky(self.g1, upper=False)
            Cp = self.Cact @ L
        except np.linalg.LinAlgError:  # only if g1_act has very small eigenvalues
            n, L = np.linalg.eigh(self.g1)
            assert np.all(n > -1.0e-11), "g1_act must be positive semi-definite"
            n = np.maximum(n, 0)
            Cp = self.Cact @ L @ np.diag(np.sqrt(n))

        Jact, Kact = self.fock_builder.build_JK([Cp])

        # [Eq. (20)]
        self.Fact = np.einsum(
            "mp,mn,nq->pq",
            self.Cgen.conj(),
            2 * Jact[0] - Kact[0],
            self.Cgen,
            optimize=True,
        )

    def compute_YZ_intermediates(self):
        # compute the Y intermediate [Algorithm 1, line 10]
        Y = np.einsum("pu,tu->pt", self.Fcore[:, self.actv], self.g1)
        # compute the Z intermediate [Algorithm 1, line 11]
        Z = np.einsum("puvw,tuvw->pt", self.eri_gaaa, self.g2)
        return Y, Z

    def compute_orbgrad(self):
        self.compute_Fact()

        self.Y, self.Z = self.compute_YZ_intermediates()
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

        return orbgrad

    def compute_orbhess(self):
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

        return orbhess

    def rotate_orbitals(self):
        # Algorithm 1, line 24
        update = np.divide(
            self.orbgrad,
            self.orbhess,
            out=np.zeros_like(self.orbgrad),
            where=(~np.isclose(self.orbhess, 0)) & self.nrr,
        )
        orbrot = np.triu(update, 1) - np.tril(update.T, -1)

        # Algorithm 1, line 25
        return self.C @ (sp.linalg.expm(orbrot))


class ActiveSpaceSolver:
    def __init__(self, solver):
        self.solver = solver
        self.nroot = self.solver.nroot

    def set_active_space_ints(self, scalar, oei, tei):
        self.solver.ints.E = scalar
        self.solver.ints.H = oei
        self.solver.ints.V = tei

    def run(self):
        self.solver.run()
        return self.solver.E

    def make_rdm12(self):
        g1 = self.solver.make_rdm1_sf(self.solver.evecs[:, 0])
        # [Eq. (5)] has a factor of 0.5
        g2 = 0.5 * self.solver.make_rdm2_sf(self.solver.evecs[:, 0])
        return g1, g2
