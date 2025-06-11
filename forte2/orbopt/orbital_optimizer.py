import numpy as np
import scipy as sp
from dataclasses import dataclass, field

from forte2.jkbuilder import FockBuilder
from forte2.helpers.mixins import MOsMixin, SystemMixin


@dataclass
class OrbitalOptimizer(MOsMixin, SystemMixin):
    # placeholder optimization parameters
    maxiter: int = 50
    gradtol: float = 1e-6
    optimizer: str = "L-BFGS"
    max_step_size: float = 0.1

    def __call__(self, method):
        if not method.executed:
            method.run()

        SystemMixin.copy_from_upstream(self, method)
        MOsMixin.copy_from_upstream(self, method)
        self.parent_method = method

        # [todo] handle GAS/RHF/ROHF cases
        self.core_orbitals = method.core_orbitals
        self.orbitals = method.flattened_orbitals
        self.virtual_orbitals = sorted(
            list(
                set(range(self.system.nbf()))
                - set(self.core_orbitals)
                - set(self.orbitals)
            )
        )

        return self

    def run(self):
        fock_builder = FockBuilder(self.system)
        Hcore = self.system.ints_hcore()  # hcore in AO basis (even 1e-sf-X2C)
        nbf = self.system.nbf()
        self.orbgrad = np.zeros((nbf, nbf))
        self.orbhess = np.zeros((nbf, nbf))
        self.orbrot = np.zeros((nbf, nbf))
        self.active_orbitals = np.array(self.orbitals).flatten()
        print(f'{"Iteration":>10} {"CI Energy":>20} {"norm(g)":>20} ')
        self.iter = 0
        self.E = np.zeros(self.parent_method.nroot)

        cv = np.ix_(self.core_orbitals, self.virtual_orbitals)
        ca = np.ix_(self.core_orbitals, self.active_orbitals)
        av = np.ix_(self.active_orbitals, self.virtual_orbitals)
        cc = np.ix_(self.core_orbitals, self.core_orbitals)
        aa = np.ix_(self.active_orbitals, self.active_orbitals)
        vv = np.ix_(self.virtual_orbitals, self.virtual_orbitals)

        while self.iter < self.maxiter:
            Cgen = self.C[0]
            Ccore = self.C[0][:, self.core_orbitals]
            Cact = self.C[0][:, self.active_orbitals]

            Ecore, Fcore = self._compute_Fcore(fock_builder, Ccore, Cgen, Hcore)

            # TODO: use Eq. (22)?
            eri_gaaa = fock_builder.two_electron_integrals_gen_block(Cgen, *(Cact,) * 3)

            # Set integrals in the CI solver
            self.parent_method.ints.E = Ecore + self.system.nuclear_repulsion_energy()
            self.parent_method.ints.H = Fcore[aa].copy()
            self.parent_method.ints.V = eri_gaaa[self.active_orbitals, ...].copy()

            # Solve the CI problem
            self.parent_method.run()
            ci_vecs = self.parent_method.evecs

            # Compute the active Fock matrix [Eq. (10)]
            # TODO: generalize to multiple states
            g1_act = self.parent_method.make_rdm1_sf(ci_vecs[:, 0])

            Fact = self._compute_Fact(fock_builder, g1_act, Cact, Cgen)

            # compute the Y intermediate [Algorithm 1, line 10]
            Y = np.einsum("pu,tu->pt", Fcore[:, self.active_orbitals], g1_act)
            Y_ca = Y[self.core_orbitals]
            Y_aa = Y[self.active_orbitals]
            Y_va = Y[self.virtual_orbitals]
            # [Eq. (5)] has a factor of 0.5
            g2_act = 0.5 * self.parent_method.make_rdm2_sf(ci_vecs[:, 0])
            # compute the Z intermediate [Algorithm 1, line 11]
            Z = np.einsum("puvw,tuvw->pt", eri_gaaa, g2_act)

            Z_ca = Z[self.core_orbitals]
            Z_aa = Z[self.active_orbitals]
            Z_va = Z[self.virtual_orbitals]

            Fcore_cv = Fcore[cv]
            Fcore_ca = Fcore[ca]
            Fact_cv = Fact[cv]
            Fact_ca = Fact[ca]

            self.orbgrad[cv] = 4 * Fcore_cv + 2 * Fact_cv
            self.orbgrad[av] = 2 * Y_va.T + 4 * Z_va.T
            self.orbgrad[ca] = 4 * Fcore_ca + 2 * Fact_ca - 2 * Y_ca - 4 * Z_ca

            self.E[0] = self.parent_method.E[0]
            print(
                f"{self.iter:>10d} {self.E[0]:>20.10f} {abs(np.linalg.norm(self.orbgrad, np.inf)):>20.10f}"
            )

            if abs(np.linalg.norm(self.orbgrad, np.inf)) < self.gradtol:
                break

            Fcore_cc = np.diag(Fcore[cc])
            Fcore_aa = np.diag(Fcore[aa])
            Fcore_vv = np.diag(Fcore[vv])
            Fact_cc = np.diag(Fact[cc])
            Fact_aa = np.diag(Fact[aa])
            Fact_vv = np.diag(Fact[vv])
            g1_diag = np.diag(g1_act)

            vdiag = 4 * Fcore_vv + 2 * Fact_vv
            cdiag = 4 * Fcore_cc + 2 * Fact_cc

            self.orbhess[cv] = vdiag - cdiag[:, None]

            av_diag = 2 * np.outer(g1_diag, Fcore_vv) + np.outer(g1_diag, Fact_vv)
            aa_diag = 2 * np.diag(Y_aa) + 4 * np.diag(Z_aa)
            self.orbhess[av] = av_diag - aa_diag[:, None]

            ca_diag = 2 * np.outer(Fcore_cc, g1_diag) + np.outer(Fact_cc, g1_diag)
            aa_diag = 4 * Fcore_aa + 2 * Fact_aa - 2 * np.diag(Y_aa) - 4 * np.diag(Z_aa)
            cc_diag = -4 * Fcore_cc - 2 * Fact_cc
            self.orbhess[ca] = ca_diag + aa_diag[None, :] + cc_diag[:, None]

            orbrot = np.triu(
                np.divide(
                    self.orbgrad,
                    self.orbhess,
                    out=np.zeros_like(self.orbgrad),
                    where=(~np.isclose(self.orbhess, 0)),
                ),
                1,
            ) - np.tril(
                np.divide(
                    self.orbgrad.T,
                    self.orbhess.T,
                    out=np.zeros_like(self.orbgrad),
                    where=(~np.isclose(self.orbhess.T, 0)),
                ),
                -1,
            )

            self.C[0] = self.C[0] @ (sp.linalg.expm(orbrot))
            self.iter += 1

    def _compute_Fcore(self, fock_builder, Ccore, Cgen, Hcore):
        # Compute the core Fock matrix [Eq. (9)], also return the core energy
        Jcore, Kcore = fock_builder.build_JK([Ccore])
        Fcore = np.einsum(
            "mp,mn,nq->pq",
            Cgen.conj(),
            Hcore + 2 * Jcore[0] - Kcore[0],
            Cgen,
            optimize=True,
        )
        Ecore = np.einsum(
            "pi,qi,pq->", Ccore.conj(), Ccore, 2 * Hcore + 2 * Jcore[0] - Kcore[0]
        )
        return Ecore, Fcore

    def _compute_Fact(self, fock_builder, g1_act, Cact, Cgen):
        # compute the active Fock matrix [Algorithm 1, line 9]
        # gamma_act (nmo * nmo) is defined in [Eq. (19)]
        # as (gamma_act)_pq = C_pt C_qu g1_tu
        # we decompose g1 and contract it into the coefficients
        # so more efficient coefficient-based J-builder can be used
        n, L = np.linalg.eigh(g1_act)
        Cp = Cact @ L @ np.diag(np.sqrt(n))

        Jact, Kact = fock_builder.build_JK([Cp])

        # [Eq. (20)]
        Fact = np.einsum(
            "mp,mn,nq->pq",
            Cgen.conj(),
            2 * Jact[0] - Kact[0],
            Cgen,
            optimize=True,
        )
        return Fact
