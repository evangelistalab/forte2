import numpy as np
import scipy as sp
from dataclasses import dataclass, field

from forte2.jkbuilder import FockBuilder
from forte2.helpers.mixins import MOsMixin, SystemMixin


@dataclass
class OrbitalOptimizer(MOsMixin, SystemMixin):
    # placeholder optimization parameters
    maxiter: int = 1
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
        print(self.orbitals)
        self.virtual_orbitals = sorted(
            list(
                set(range(self.system.nbf()))
                - set(self.core_orbitals)
                - set(self.orbitals)
            )
        )
        print(self.virtual_orbitals)

        return self

    def run(self):
        fock_builder = FockBuilder(self.system)
        h = self.system.ints_hcore()  # hcore in AO basis (even 1e-sf-X2C)
        self.active_orbitals = np.array(self.orbitals).flatten()
        # gaaa = fock_builder.two_electron_integrals_gen_block(
        #     c_gen, c_act, c_act, c_act
        # ).swapaxes(1, 2)
        print(f'{"Iteration":>10} {"CI Energy":>20} {"E_casci":>20} {"norm(g)":>20} ')
        self.iter = 0
        while self.iter < self.maxiter:
            Cgen = self.C[0]
            Ccore = self.C[0][:, self.core_orbitals]
            Cact = self.C[0][:, self.active_orbitals]

            # Compute the core Fock matrix [Eq. (9)]
            Jcore, Kcore = fock_builder.build_JK([Ccore])
            Fcore = np.einsum(
                "mp,mn,nq->pq",
                Cgen.conj(),
                h + 2 * Jcore[0] - Kcore[0],  # Fcore
                Cgen,
                optimize=True,
            )
            Ecore = np.einsum(
                "pi,qi,pq->", Ccore.conj(), Ccore, 2 * h + 2 * Jcore[0] - Kcore[0]
            )

            # TODO: use Eq. (22)?
            eri_gaaa = fock_builder.two_electron_integrals_gen_block(
                Cgen, Cact, Cact, Cact
            ).swapaxes(1, 2)

            self.parent_method.ints.E = Ecore
            self.parent_method.ints.H = Fcore[self.active_orbitals, :][
                :, self.active_orbitals
            ]
            self.parent_method.ints.V = eri_gaaa[self.active_orbitals, ...].swapaxes(
                1, 2
            )
            self.parent_method.run()
            ci_vecs = self.parent_method.evecs

            # Compute the active Fock matrix [Eq. (10)]
            # TODO: generalize to multiple states
            g1_act = self.parent_method.make_rdm1_sf(ci_vecs[:, 0])

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
                Cact.conj(),
                2 * Jact[0] - Kact[0],  # Fact
                Cact,
                optimize=True,
            )

            # compute the Y intermediate [Algorithm 1, line 10]
            Y = np.einsum("pu,tu->pt", Fcore[:, self.active_orbitals], g1_act)

            # [Eq. (5)] has a factor of 0.5, and differ from our definition of rdm2_sf
            # by a transposition of the middle two indices
            g2_act = 0.5 * self.parent_method.make_rdm2_sf(ci_vecs[:, 0]).swapaxes(1, 2)
            # compute the Z intermediate [Algorithm 1, line 11]
            Z = np.einsum("puvw,tuvw->pt", eri_gaaa, g2_act)

            ### PROGRESS SO FAR ###
            return

            self.orbgrad[self.core_orbitals, self.virt] = (
                4 * Fcore[self.core_orbitals, self.virt]
                + 2 * Fact[self.core_orbitals, self.virt]
            )
            self.orbgrad[self.act, self.virt] = (
                2 * Y[self.virt, :].T + 4 * Z[self.virt, :].T
            )
            self.orbgrad[self.core_orbitals, self.act] = (
                4 * Fcore[self.core_orbitals, self.act]
                + 2 * Fact[self.core_orbitals, self.act]
                - 2 * Y[self.core_orbitals, :]
                - 4 * Z[self.core_orbitals, :]
            )

            print(
                f"{self.iter:>10d} {ci_eigval:>20.10f} {ci_eigval+Ecore+self.mol.energy_nuc():>20.10f} {abs(np.linalg.norm(self.orbgrad, np.inf)):>20.10f}"
            )

            if abs(np.linalg.norm(self.orbgrad, np.inf)) < self.gradtol:
                break

            self.orbhess[self.core, self.virt] = (
                np.diag(
                    4 * Fcore[self.virt, self.virt] + 2 * Fact[self.virt, self.virt]
                )
                - np.diag(
                    4 * Fcore[self.core, self.core] + 2 * Fact[self.core, self.core]
                )[:, None]
            )
            self.orbhess[self.act, self.virt] = (
                np.einsum("tt,aa->ta", dm1, (2 * Fcore + Fact)[self.virt, self.virt])
                - np.diag(2 * Y[self.act, :] + 4 * Z[self.act, :])[:, None]
            )
            self.orbhess[self.core, self.act] = (
                np.einsum("tt,ii->it", dm1, (2 * Fcore + Fact)[self.core, self.core])
                + np.diag(
                    4 * Fcore[self.act, self.act]
                    + 2 * Fact[self.act, self.act]
                    - 2 * Y[self.act, :]
                    - 4 * Z[self.act, :]
                )[None, :]
                - np.diag(
                    4 * Fcore[self.core, self.core] + 2 * Fact[self.core, self.core]
                )[:, None]
            )

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

            self.mo_coeff = self.mo_coeff @ (sp.linalg.expm(orbrot))
            self.iter += 1
