from dataclasses import dataclass, field

import numpy as np

from forte2 import dsrg_utils
from forte2.jkbuilder.mointegrals import RestrictedMOIntegrals, SpinorbitalIntegrals
from forte2.orbitals import Semicanonicalizer
from .dsrg_base import DSRGBase

MACHEPS = 1e-9
TAYLOR_THRES = 1e-3

regularized_denominator = dsrg_utils.regularized_denominator
taylor_exp = dsrg_utils.taylor_exp


@dataclass
class DSRG_MRPT2(DSRGBase):
    def get_integrals(self):
        # always semi-canonicalize, to get the generalized Fock matrix
        self.semicanonicalizer = Semicanonicalizer(
            system=self.system,
            mo_space=self.mo_space,
            fock_builder=self.fock_builder,
            mix_active=False,
            # do not mix correlated and frozen orbitals after MCSCF
            mix_inactive=False,
        )
        g1 = self.parent_method.make_average_1rdm()
        self.semicanonicalizer.semi_canonicalize(g1=g1, C_contig=self._C)
        self._C = self.semicanonicalizer.C_semican.copy()
        self.fock = self.semicanonicalizer.fock_semican.copy()
        self.eps = self.semicanonicalizer.eps_semican.copy()
        self.delta_actv = self.eps[self.actv][:, None] - self.eps[self.actv][None, :]
        self.Uactv = self.semicanonicalizer.Uactv

        if self.two_component:
            ints = SpinorbitalIntegrals(
                system=self.system,
                C=self._C,
                spinorbitals=list(
                    range(self.mo_space.corr.start, self.mo_space.corr.stop)
                ),
                core_spinorbitals=list(range(0, self.mo_space.frozen_core.stop)),
                antisymmetrize=True,
            )
            ints.H = self.fock - np.diag(np.diag(self.fock))  # remove diagonal
            cumulants = dict()
            g1 = self.parent_method.make_average_1rdm()
            cumulants["gamma1"] = np.einsum(
                "ip,ij,jq->pq", self.Uactv, g1, self.Uactv.conj(), optimize=True
            )
            cumulants["eta1"] = (
                np.eye(cumulants["gamma1"].shape[0], dtype=complex)
                - cumulants["gamma1"]
            )
            l2 = self.parent_method.make_average_2cumulant()
            cumulants["lambda2"] = np.einsum(
                "ip,jq,ijkl,kr,ls->pqrs",
                self.Uactv,
                self.Uactv,
                l2,
                self.Uactv.conj(),
                self.Uactv.conj(),
                optimize=True,
            )
            l3 = self.parent_method.make_average_3cumulant()
            cumulants["lambda3"] = np.einsum(
                "ip,jq,kr,ijklmn,ls,mt,nu->pqrstu",
                self.Uactv,
                self.Uactv,
                self.Uactv,
                l3,
                self.Uactv.conj(),
                self.Uactv.conj(),
                self.Uactv.conj(),
                optimize=True,
            )
            return ints, cumulants
        else:
            raise NotImplementedError("Only two-component integrals are implemented.")

    def solve_dsrg(self):
        self.T1, self.T2 = self._build_tamps()
        self.F_tilde, self.V_tilde = self._renormalize_integrals()
        E = self._compute_pt2_energy(
            self.F_tilde,
            self.V_tilde,
            self.T1,
            self.T2,
            self.cumulants["gamma1"],
            self.cumulants["eta1"],
            self.cumulants["lambda2"],
            self.cumulants["lambda3"],
        )
        self.E = E
        return E

    def do_reference_relaxation(self):
        raise NotImplementedError(
            "Reference relaxation for DSRG-MRPT2 is not yet implemented."
        )

    def _build_tamps(self):
        t2 = np.conj(self.ints.V[self.hole, self.hole, self.part, self.part])
        for i in range(self.nhole):
            for j in range(self.nhole):
                for a in range(self.npart):
                    for b in range(self.npart):
                        denom = (
                            self.eps[i]
                            + self.eps[j]
                            - self.eps[a + self.ncore]
                            - self.eps[b + self.ncore]
                        )
                        t2[i, j, a, b] *= regularized_denominator(
                            denom, self.flow_param
                        )

        t2[self.ha, self.ha, self.pa, self.pa] = 0.0

        t1 = self.ints.H[self.hole, self.part].conj().copy()
        t1 += np.einsum(
            "xu,iuax,xu->ia",
            self.delta_actv,
            t2[:, self.ha, :, self.pa],
            self.cumulants["gamma1"],
            optimize=True,
        )
        for i in range(self.nhole):
            for a in range(self.npart):
                denom = self.eps[i] - self.eps[a + self.ncore]
                t1[i, a] *= regularized_denominator(denom, self.flow_param)
        t1[self.ha, self.pa] = 0.0
        return t1, t2

    def _renormalize_integrals(self):
        F_tilde = np.conj(self.ints.H[self.hole, self.part])
        delta_ia = self.eps[self.hole][:, None] - self.eps[self.part][None, :]
        exp_delta_1 = np.exp(-self.flow_param * delta_ia**2)
        F_tilde += (
            F_tilde * exp_delta_1
            + np.einsum(
                "xu,iuax,xu->ia",
                self.delta_actv,
                self.T2[:, self.ha, :, self.pa],
                self.cumulants["gamma1"],
                optimize=True,
            )
            * exp_delta_1
        )
        np.conj(F_tilde, out=F_tilde)

        V_tilde = np.copy(self.ints.V[self.hole, self.hole, self.part, self.part])
        delta_ijab = (
            self.eps[self.hole][:, None, None, None]
            + self.eps[self.hole][None, :, None, None]
            - self.eps[self.part][None, None, :, None]
            - self.eps[self.part][None, None, None, :]
        )
        exp_delta_2 = np.exp(-self.flow_param * delta_ijab**2)
        V_tilde += V_tilde * exp_delta_2

        return F_tilde, V_tilde

    def _compute_pt2_energy(self, F, V, T1, T2, gamma1, eta1, lambda2, lambda3):
        ha = self.ha
        pa = self.pa
        hc = self.hc
        pv = self.pv
        E = 0.0

        E += +1.000 * np.einsum("iu,iv,vu->", F[hc, pa], T1[hc, pa], eta1, optimize=True)
        E += +1.000 * np.einsum("ia,ia->", F[hc, pv], T1[hc, pv], optimize=True)
        E += +1.000 * np.einsum(
            "ua,va,uv->", F[ha, pv], T1[ha, pv], gamma1, optimize=True
        )
        E += -0.500 * np.einsum(
            "iu,ixvw,vwux->", F[hc, pa], T2[hc, ha, pa, pa], lambda2, optimize=True
        )
        E += -0.500 * np.einsum(
            "ua,wxva,uvwx->", F[ha, pv], T2[ha, ha, pa, pv], lambda2, optimize=True
        )
        E += -0.500 * np.einsum(
            "iu,ivwx,uvwx->", T1[hc, pa], V[hc, ha, pa, pa], lambda2, optimize=True
        )
        E += -0.500 * np.einsum(
            "ua,vwxa,vwux->", T1[ha, pv], V[ha, ha, pa, pv], lambda2, optimize=True
        )
        E += +0.250 * np.einsum(
            "ijuv,ijwx,vx,uw->",
            T2[hc, hc, pa, pa],
            V[hc, hc, pa, pa],
            eta1,
            eta1,
            optimize=True,
        )
        E += +0.125 * np.einsum(
            "ijuv,ijwx,uvwx->",
            T2[hc, hc, pa, pa],
            V[hc, hc, pa, pa],
            lambda2,
            optimize=True,
        )
        E += +0.500 * np.einsum(
            "iwuv,ixyz,vz,uy,xw->",
            T2[hc, ha, pa, pa],
            V[hc, ha, pa, pa],
            eta1,
            eta1,
            gamma1,
            optimize=True,
        )
        E += +1.000 * np.einsum(
            "iwuv,ixyz,vz,uxwy->",
            T2[hc, ha, pa, pa],
            V[hc, ha, pa, pa],
            eta1,
            lambda2,
            optimize=True,
        )
        E += +0.250 * np.einsum(
            "iwuv,ixyz,xw,uvyz->",
            T2[hc, ha, pa, pa],
            V[hc, ha, pa, pa],
            gamma1,
            lambda2,
            optimize=True,
        )
        E += +0.250 * np.einsum(
            "iwuv,ixyz,uvxwyz->",
            T2[hc, ha, pa, pa],
            V[hc, ha, pa, pa],
            lambda3,
            optimize=True,
        )
        E += +0.500 * np.einsum(
            "ijua,ijva,uv->",
            T2[hc, hc, pa, pv],
            V[hc, hc, pa, pv],
            eta1,
            optimize=True,
        )
        E += +1.000 * np.einsum(
            "ivua,iwxa,ux,wv->",
            T2[hc, ha, pa, pv],
            V[hc, ha, pa, pv],
            eta1,
            gamma1,
            optimize=True,
        )
        E += +1.000 * np.einsum(
            "ivua,iwxa,uwvx->",
            T2[hc, ha, pa, pv],
            V[hc, ha, pa, pv],
            lambda2,
            optimize=True,
        )
        E += +0.500 * np.einsum(
            "vwua,xyza,uz,yw,xv->",
            T2[ha, ha, pa, pv],
            V[ha, ha, pa, pv],
            eta1,
            gamma1,
            gamma1,
            optimize=True,
        )
        E += +0.250 * np.einsum(
            "vwua,xyza,uz,xyvw->",
            T2[ha, ha, pa, pv],
            V[ha, ha, pa, pv],
            eta1,
            lambda2,
            optimize=True,
        )
        E += +1.000 * np.einsum(
            "vwua,xyza,yw,uxvz->",
            T2[ha, ha, pa, pv],
            V[ha, ha, pa, pv],
            gamma1,
            lambda2,
            optimize=True,
        )
        E += -0.250 * np.einsum(
            "vwua,xyza,uxyvwz->",
            T2[ha, ha, pa, pv],
            V[ha, ha, pa, pv],
            lambda3,
            optimize=True,
        )
        E += +0.250 * np.einsum(
            "ijab,ijab->", T2[hc, hc, pv, pv], V[hc, hc, pv, pv], optimize=True
        )
        E += +0.500 * np.einsum(
            "iuab,ivab,vu->",
            T2[hc, ha, pv, pv],
            V[hc, ha, pv, pv],
            gamma1,
            optimize=True,
        )
        E += +0.250 * np.einsum(
            "uvab,wxab,xv,wu->",
            T2[ha, ha, pv, pv],
            V[ha, ha, pv, pv],
            gamma1,
            gamma1,
            optimize=True,
        )
        E += +0.125 * np.einsum(
            "uvab,wxab,wxuv->",
            T2[ha, ha, pv, pv],
            V[ha, ha, pv, pv],
            lambda2,
            optimize=True,
        )
        return E
