import numpy as np
import scipy as sp
from dataclasses import dataclass, field

from forte2.jkbuilder import FockBuilder
from forte2.helpers.mixins import MOsMixin, SystemMixin


@dataclass
class OrbitalOptimizer(MOsMixin, SystemMixin):
    # placeholder optimization parameters
    maxiter: int = 100
    gradtol: float = 1e-6
    optimizer: str = "L-BFGS"
    max_step_size: float = 0.1

    def __call__(self, method):
        SystemMixin.copy_from_upstream(self, method)
        MOsMixin.copy_from_upstream(self, method)
        assert hasattr(method, "make_rdm1"), "Method must have a make_rdm1 method"
        assert hasattr(method, "make_rdm2"), "Method must have a make_rdm2 method"

        # [todo] handle GAS/RHF/ROHF cases
        self.core_orbitals = method.core_orbitals
        self.actiive_orbitals = method.active_orbitals
        self.virtual_orbitals = sorted(
            list(
                set(range(self.system.nbf()))
                - set(self.core_orbitals)
                - set(self.actiive_orbitals)
            )
        )

        return self

    def run(self):
        fock_builder = FockBuilder(self.system)
        c_gen = self.C[0]
        c_act = self.C[0][:, self.actiive_orbitals]
        gaaa = fock_builder.two_electron_integrals_gen_block(
            c_gen, c_act, c_act, c_act
        ).swapaxes(1, 2)
        print(f'{"Iteration":>10} {"CI Energy":>20} {"E_casci":>20} {"norm(g)":>20} ')
        while self.iter < self.maxiter:
            Dcore = np.einsum(
                "pi,qi->pq", self.C[0][:, self.core_orbitals], self.C[0][:, self.core_orbitals]
            )
            vj, vk = self.mf.get_jk(self.mol, Dcore)
            Ecore = np.einsum(
                "pi,qi,pq->",
                self.C[0][:, self.core_orbitals],
                self.C[0][:, self.core_orbitals],
                2 * self.hcore + 2 * vj - vk,
            )
            Fcore = np.einsum(
                "ip,jq,ij->pq", self.C[0], self.C[0], self.hcore + 2 * vj - vk
            )
            for v in range(self.ncore, self.ncore + self.ncas):
                for w in range(self.ncore, self.ncore + self.ncas):
                    Pact = np.outer(self.C[0][:, v], self.C[0][:, w])
                    vj = self.mf.get_j(self.mol, Pact)
                    self.eri_gaaa[:, :, v - self.ncore, w - self.ncore] = np.einsum(
                        "ip,ju,ij->pu", self.C[0], self.C[0][:, self.act], vj
                    )
            dm1 = self.method.make_rdm1()
            dm2 = self.method.make_rdm2()

            dm1_ao = np.einsum(
                "it,ju,tu->ij",
                self.C[0][:, self.act],
                self.C[0][:, self.act],
                dm1,
            )
            vj = fock_builder.build_J(dm1_ao)[0]
            # todo: change to coefficient-based K
            vk = fock_builder.build_K_density(dm1_ao)[0]
            Fact = np.einsum("ip,jq,ij->pq", self.C[0], self.C[0], 2 * vj - vk)
            Y = np.einsum("pu,tu->pt", Fcore[:, self.act], dm1)
            Z = np.einsum("puvw,tuvw->pt", self.eri_gaaa, dm2)

            self.orbgrad[self.core_orbitals, self.virt] = (
                4 * Fcore[self.core_orbitals, self.virt] + 2 * Fact[self.core_orbitals, self.virt]
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
