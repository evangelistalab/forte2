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
        self.parent_method = method
        return self

    def run(self):
        # TODO: skip the first CI step in the MCSCF
        if not self.parent_method.executed:
            self.parent_method.run()

        SystemMixin.copy_from_upstream(self, self.parent_method)
        MOsMixin.copy_from_upstream(self, self.parent_method)

        nbf = self.system.nbf()
        Hcore = self.system.ints_hcore()  # hcore in AO basis (even 1e-sf-X2C)
        fock_builder = FockBuilder(self.system)

        self._make_spaces_contiguous()

        self.orbgrad = np.zeros((nbf, nbf))
        self.orbhess = np.zeros((nbf, nbf))
        print(f'{"Iteration":>10} {"CI Energy":>20} {"norm(g)":>20} ')
        self.iter = 0
        self.E = np.zeros(self.parent_method.nroot)

        # # these slicers ensure that no copies will be made
        # # inter-space
        # cv = np.ix_(self.core_orbitals, self.virtual_orbitals)
        # ca = np.ix_(self.core_orbitals, self.active_orbitals)
        # av = np.ix_(self.active_orbitals, self.virtual_orbitals)
        # # intra-space
        # cc = np.ix_(self.core_orbitals, self.core_orbitals)
        # aa = np.ix_(self.active_orbitals, self.active_orbitals)
        # vv = np.ix_(self.virtual_orbitals, self.virtual_orbitals)

        # nonredundant_rotations = np.zeros((nbf, nbf), dtype=bool)
        # nonredundant_rotations[cv] = True
        # nonredundant_rotations[ca] = True
        # nonredundant_rotations[av] = True

        while self.iter < self.maxiter:
            Cgen = self.C[0]
            Ccore = self.C[0][:, self.core]
            Cact = self.C[0][:, self.actv]

            # Algorithm 1, line 4
            Ecore, Fcore = self._compute_Fcore(fock_builder, Ccore, Cgen, Hcore)

            # Algorithm 1, line 5
            # TODO: use Eq. (22)?
            eri_gaaa = fock_builder.two_electron_integrals_gen_block(Cgen, *(Cact,) * 3)

            # Algorithm 1, line 6-7 - solve the subproblem and get the energies and RDMs
            self.E, g1_act, g2_act = self._make_rdm_12(
                Ecore,
                Fcore[self.actv, self.actv].copy(),
                eri_gaaa[self.actv, ...].copy(),
            )  # copies ensure contiguous arrays are passed to the CI solver. TODO: is this necessary?

            # Algorithm 1, line 9
            Fact = self._compute_Fact(fock_builder, g1_act, Cact, Cgen)

            # compute the Y and Z intermediates [Algorithm 1, line 10-11]
            Y, Z = self._compute_YZ_intermediates(Fcore, eri_gaaa, g1_act, g2_act)

            self._compute_orbgrad(Fcore, Fact, Y, Z)

            print(
                f"{self.iter:>10d} {self.E[0]:>20.10f} {abs(np.linalg.norm(self.orbgrad, np.inf)):>20.10f}"
            )

            if abs(np.linalg.norm(self.orbgrad, np.inf)) < self.gradtol:
                break

            self._compute_orbhess(Fcore, Fact, g1_act, Y, Z)

            self._rotate_orbitals(self.orbgrad, self.orbhess)
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
        try:
            # C @ g1_act C.T = C @ L @ L.T @ C.T == Cp @ Cp.T
            L = np.linalg.cholesky(g1_act, upper=False)
            Cp = Cact @ L
        except np.linalg.LinAlgError:  # only if g1_act has very small eigenvalues
            n, L = np.linalg.eigh(g1_act)
            assert np.all(n > -1.0e-11), "g1_act must be positive semi-definite"
            n = np.maximum(n, 0)
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

    def _make_spaces_contiguous(self):
        """
        Swap the orbitals to ensure that the core, active, and virtual orbitals
        are contiguous in the flattened orbital array.
        """
        # [todo] handle GAS/RHF/ROHF cases
        core = self.parent_method.core_orbitals
        actv = self.parent_method.flattened_orbitals
        virt = list(set(range(self.system.nbf())) - set(core) - set(actv))
        argssort = np.argsort(core + actv + virt)
        self.C[0] = self.C[0][:, argssort].copy()
        self.core = slice(0, len(core))
        self.actv = slice(len(core), len(core) + len(actv))
        self.virt = slice(len(core) + len(actv), None)

    def _make_rdm_12(self, scalar, oei, tei):
        # Set integrals in the CI solver
        self.parent_method.ints.E = scalar + self.system.nuclear_repulsion_energy()
        self.parent_method.ints.H = oei
        self.parent_method.ints.V = tei
        # Solve the CI problem
        self.parent_method.run()

        # TODO: generalize to multiple states
        g1_act = self.parent_method.make_rdm1_sf(self.parent_method.evecs[:, 0])
        # [Eq. (5)] has a factor of 0.5
        g2_act = 0.5 * self.parent_method.make_rdm2_sf(self.parent_method.evecs[:, 0])

        return self.parent_method.E, g1_act, g2_act

    def _compute_YZ_intermediates(self, Fcore, eri_gaaa, g1_act, g2_act):
        # compute the Y intermediate [Algorithm 1, line 10]
        Y = np.einsum("pu,tu->pt", Fcore[:, self.actv], g1_act)
        # compute the Z intermediate [Algorithm 1, line 11]
        Z = np.einsum("puvw,tuvw->pt", eri_gaaa, g2_act)
        return Y, Z

    def _compute_orbgrad(self, Fcore, Fact, Y, Z):
        Fcore_cv = Fcore[self.core, self.virt]
        Fcore_ca = Fcore[self.core, self.actv]
        Fact_cv = Fact[self.core, self.virt]
        Fact_ca = Fact[self.core, self.actv]

        Y_va = Y[self.virt, :]
        Y_ca = Y[self.core, :]
        Z_va = Z[self.virt, :]
        Z_ca = Z[self.core, :]

        # Algorithm 1, lines 13-15
        self.orbgrad[self.core, self.virt] = 4 * Fcore_cv + 2 * Fact_cv
        self.orbgrad[self.actv, self.virt] = 2 * Y_va.T + 4 * Z_va.T
        self.orbgrad[self.core, self.actv] = (
            4 * Fcore_ca + 2 * Fact_ca - 2 * Y_ca - 4 * Z_ca
        )

    def _compute_orbhess(self, Fcore, Fact, g1_act, Y, Z):
        Fcore_cc = np.diag(Fcore[self.core, self.core])
        Fcore_aa = np.diag(Fcore[self.actv, self.actv])
        Fcore_vv = np.diag(Fcore[self.virt, self.virt])
        Fact_cc = np.diag(Fact[self.core, self.core])
        Fact_aa = np.diag(Fact[self.actv, self.actv])
        Fact_vv = np.diag(Fact[self.virt, self.virt])
        g1_diag = np.diag(g1_act)
        Y_aa = np.diag(Y[self.actv, :])
        Z_aa = np.diag(Z[self.actv, :])

        # Algorithm 1, line 20
        vdiag = 4 * Fcore_vv + 2 * Fact_vv
        cdiag = 4 * Fcore_cc + 2 * Fact_cc
        self.orbhess[self.core, self.virt] = vdiag - cdiag[:, None]

        # Algorithm 1, line 21
        av_diag = 2 * np.outer(g1_diag, Fcore_vv) + np.outer(g1_diag, Fact_vv)
        aa_diag = 2 * Y_aa + 4 * Z_aa
        self.orbhess[self.actv, self.virt] = av_diag - aa_diag[:, None]

        # Algorithm 1, line 22
        ca_diag = 2 * np.outer(Fcore_cc, g1_diag) + np.outer(Fact_cc, g1_diag)
        aa_diag = 4 * Fcore_aa + 2 * Fact_aa - 2 * Y_aa - 4 * Z_aa
        cc_diag = -4 * Fcore_cc - 2 * Fact_cc
        self.orbhess[self.core, self.actv] = (
            ca_diag + aa_diag[None, :] + cc_diag[:, None]
        )

    def _rotate_orbitals(self, orbgrad, orbhess):
        # Algorithm 1, line 24
        update = np.divide(
            orbgrad,
            orbhess,
            out=np.zeros_like(orbgrad),
            where=(~np.isclose(orbhess, 0)),
        )
        orbrot = np.triu(update, 1) - np.tril(update.T, -1)

        # Algorithm 1, line 25
        self.C[0] = self.C[0] @ (sp.linalg.expm(orbrot))
