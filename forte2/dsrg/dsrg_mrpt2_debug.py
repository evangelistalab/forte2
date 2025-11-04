from dataclasses import dataclass

import numpy as np

from forte2 import dsrg_utils
from forte2.jkbuilder.mointegrals import SpinorbitalIntegrals
from .dsrg_base import DSRGBase
from .utils import antisymmetrize_2body

MACHEPS = 1e-9
TAYLOR_THRES = 1e-3

regularized_denominator = dsrg_utils.regularized_denominator
taylor_exp = dsrg_utils.taylor_exp



@dataclass
class DSRG_MRPT2_Reference(DSRGBase):
    """
    Reference implementation of the DSRG-MRPT2 method.
    """

    def get_integrals(self):
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
        E += self.parent_method.E
        return E

    def do_reference_relaxation(self):
        _hbar2 = self.ints.V[self.actv, self.actv, self.actv, self.actv].copy()
        _C2 = 0.5 * self._compute_Hbar_aaaa(
            self.F_tilde,
            self.V_tilde,
            self.T1,
            self.T2,
            self.cumulants["gamma1"],
            self.cumulants["eta1"],
        )
        # 0.5*[H, T-T+] = 0.5*([H, T] + [H, T]+)
        _hbar2 += _C2 + np.einsum("ijab->abij", np.conj(_C2))

        _hbar1 = self.fock[self.actv, self.actv].copy()
        _C1 = 0.5 * self._compute_Hbar_aa(
            self.F_tilde,
            self.V_tilde,
            self.T1,
            self.T2,
            self.cumulants["gamma1"],
            self.cumulants["eta1"],
            self.cumulants["lambda2"],
        )
        # 0.5*[H, T-T+] = 0.5*([H, T] + [H, T]+)
        _hbar1 += _C1 + np.einsum("ia->ai", np.conj(_C1))

        # see eq B.3. of JCP 146, 124132 (2017), but instead of gamma2, use lambda2
        _e_scalar = (
            -np.einsum("uv,uv->", _hbar1, self.cumulants["gamma1"])
            - 0.25 * np.einsum("uvxy,uvxy->", _hbar2, self.cumulants["lambda2"])
            + 0.75
            * np.einsum(
                "uvxy,ux,vy->",
                _hbar2,
                self.cumulants["gamma1"],
                self.cumulants["gamma1"],
            )
            + 0.25
            * np.einsum(
                "uvxy,uy,vx->",
                _hbar2,
                self.cumulants["gamma1"],
                self.cumulants["gamma1"],
            )
        ) + self.E

        _hbar1 -= np.einsum("uxvy,xy->uv", _hbar2, self.cumulants["gamma1"])

        _hbar1_canon = np.einsum(
            "ip,pq,jq->ij", self.Uactv.conj(), _hbar1, self.Uactv, optimize=True
        )
        _hbar2_canon = np.einsum(
            "ip,jq,pqrs,kr,ls->ijkl",
            self.Uactv.conj(),
            self.Uactv.conj(),
            _hbar2,
            self.Uactv,
            self.Uactv,
            optimize=True,
        )

        self.ci_solver.set_ints(_e_scalar, _hbar1_canon, _hbar2_canon)
        self.ci_solver.run(use_asym_ints=True)
        e_relaxed = self.ci_solver.compute_average_energy()
        print(f"Relaxed CI energy: {e_relaxed:.12f} Ha")
        raise

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
        Etot = 0.0

        # 1
        E = +1.000 * np.einsum("iu,iv,vu->", F[hc, pa], T1[hc, pa], eta1, optimize=True)
        Etot += E
        print(E)

        # 2
        E = +1.000 * np.einsum("ia,ia->", F[hc, pv], T1[hc, pv], optimize=True)
        Etot += E
        print(E)

        # 3
        E = +1.000 * np.einsum(
            "ua,va,uv->", F[ha, pv], T1[ha, pv], gamma1, optimize=True
        )
        Etot += E
        print(E)

        # 4
        E = -0.500 * np.einsum(
            "iu,ixvw,vwux->", F[hc, pa], T2[hc, ha, pa, pa], lambda2, optimize=True
        )
        Etot += E
        print(E)

        # 5
        E = -0.500 * np.einsum(
            "ua,wxva,uvwx->", F[ha, pv], T2[ha, ha, pa, pv], lambda2, optimize=True
        )
        Etot += E
        print(E)

        # 6
        E = -0.500 * np.einsum(
            "iu,ivwx,uvwx->", T1[hc, pa], V[hc, ha, pa, pa], lambda2, optimize=True
        )
        Etot += E
        print(E)

        # 7
        E = -0.500 * np.einsum(
            "ua,vwxa,vwux->", T1[ha, pv], V[ha, ha, pa, pv], lambda2, optimize=True
        )
        Etot += E
        print(E)

        # 8
        E = +0.250 * np.einsum(
            "ijuv,ijwx,vx,uw->",
            T2[hc, hc, pa, pa],
            V[hc, hc, pa, pa],
            eta1,
            eta1,
            optimize=True,
        )
        Etot += E
        print(E)

        # 9
        E = +0.125 * np.einsum(
            "ijuv,ijwx,uvwx->",
            T2[hc, hc, pa, pa],
            V[hc, hc, pa, pa],
            lambda2,
            optimize=True,
        )
        Etot += E
        print(E)

        # 10
        E = +0.500 * np.einsum(
            "iwuv,ixyz,vz,uy,xw->",
            T2[hc, ha, pa, pa],
            V[hc, ha, pa, pa],
            eta1,
            eta1,
            gamma1,
            optimize=True,
        )
        Etot += E
        print(E)

        # 11
        E = +1.000 * np.einsum(
            "iwuv,ixyz,vz,uxwy->",
            T2[hc, ha, pa, pa],
            V[hc, ha, pa, pa],
            eta1,
            lambda2,
            optimize=True,
        )
        Etot += E
        print(E)

        # 12
        E = +0.250 * np.einsum(
            "iwuv,ixyz,xw,uvyz->",
            T2[hc, ha, pa, pa],
            V[hc, ha, pa, pa],
            gamma1,
            lambda2,
            optimize=True,
        )
        Etot += E
        print(E)

        # 13
        E = +0.250 * np.einsum(
            "iwuv,ixyz,uvxwyz->",
            T2[hc, ha, pa, pa],
            V[hc, ha, pa, pa],
            lambda3,
            optimize=True,
        )
        Etot += E
        print(E)

        # 14
        E = +0.500 * np.einsum(
            "ijua,ijva,uv->",
            T2[hc, hc, pa, pv],
            V[hc, hc, pa, pv],
            eta1,
            optimize=True,
        )
        Etot += E
        print(E)

        # 15
        E = +1.000 * np.einsum(
            "ivua,iwxa,ux,wv->",
            T2[hc, ha, pa, pv],
            V[hc, ha, pa, pv],
            eta1,
            gamma1,
            optimize=True,
        )
        Etot += E
        print(E)

        # 16
        E = +1.000 * np.einsum(
            "ivua,iwxa,uwvx->",
            T2[hc, ha, pa, pv],
            V[hc, ha, pa, pv],
            lambda2,
            optimize=True,
        )
        Etot += E
        print(E)

        # 17
        E = +0.500 * np.einsum(
            "vwua,xyza,uz,yw,xv->",
            T2[ha, ha, pa, pv],
            V[ha, ha, pa, pv],
            eta1,
            gamma1,
            gamma1,
            optimize=True,
        )
        Etot += E
        print(E)

        # 18
        E = +0.250 * np.einsum(
            "vwua,xyza,uz,xyvw->",
            T2[ha, ha, pa, pv],
            V[ha, ha, pa, pv],
            eta1,
            lambda2,
            optimize=True,
        )
        Etot += E
        print(E)

        # 19
        E = +1.000 * np.einsum(
            "vwua,xyza,yw,uxvz->",
            T2[ha, ha, pa, pv],
            V[ha, ha, pa, pv],
            gamma1,
            lambda2,
            optimize=True,
        )
        Etot += E
        print(E)

        # 20
        E = -0.250 * np.einsum(
            "vwua,xyza,uxyvwz->",
            T2[ha, ha, pa, pv],
            V[ha, ha, pa, pv],
            lambda3,
            optimize=True,
        )
        Etot += E
        print(E)

        # 21
        E = +0.250 * np.einsum(
            "ijab,ijab->", T2[hc, hc, pv, pv], V[hc, hc, pv, pv], optimize=True
        )
        Etot += E
        print(E)

        # 22
        E = +0.500 * np.einsum(
            "iuab,ivab,vu->",
            T2[hc, ha, pv, pv],
            V[hc, ha, pv, pv],
            gamma1,
            optimize=True,
        )
        Etot += E
        print(E)

        # 23
        E = +0.250 * np.einsum(
            "uvab,wxab,xv,wu->",
            T2[ha, ha, pv, pv],
            V[ha, ha, pv, pv],
            gamma1,
            gamma1,
            optimize=True,
        )
        Etot += E
        print(E)

        # 24
        E = +0.125 * np.einsum(
            "uvab,wxab,wxuv->",
            T2[ha, ha, pv, pv],
            V[ha, ha, pv, pv],
            lambda2,
            optimize=True,
        )
        Etot += E
        print(E)
        return Etot

    def _compute_Hbar_aaaa(self, F, V, T1, T2, gamma1, eta1):
        ha = self.ha
        pa = self.pa
        hc = self.hc
        pv = self.pv

        # all quantities are stored ^{hh..}_{pp..}
        # h = {c,a}; p = {a, v}
        _V = np.zeros((self.nact, self.nact, self.nact, self.nact), dtype=complex)
        # Term 6
        _V += -0.500 * np.einsum(
            "ua,wxva->wxuv", F[ha, pv], T2[ha, ha, pa, pv], optimize="optimal"
        )
        # Term 7
        _V += -0.500 * np.einsum(
            "iu,ixvw->uxvw", F[hc, pa], T2[hc, ha, pa, pa], optimize="optimal"
        )
        # Term 8
        _V += -0.500 * np.einsum(
            "iu,ivwx->wxuv", T1[hc, pa], V[hc, ha, pa, pa], optimize="optimal"
        )
        # Term 9
        _V += -0.500 * np.einsum(
            "ua,vwxa->uxvw", T1[ha, pv], V[ha, ha, pa, pv], optimize="optimal"
        )
        # Term 10
        _V += +0.125 * np.einsum(
            "uvab,wxab->uvwx", T2[ha, ha, pv, pv], V[ha, ha, pv, pv], optimize="optimal"
        )
        _V += +0.250 * np.einsum(
            "uvya,wxza,yz->uvwx",
            T2[ha, ha, pa, pv],
            V[ha, ha, pa, pv],
            eta1,
            optimize="optimal",
        )
        # Term 11
        _V += +0.125 * np.einsum(
            "ijuv,ijwx->wxuv", T2[hc, hc, pa, pa], V[hc, hc, pa, pa], optimize="optimal"
        )
        _V += +0.250 * np.einsum(
            "iyuv,izwx,zy->wxuv",
            T2[hc, ha, pa, pa],
            V[hc, ha, pa, pa],
            gamma1,
            optimize="optimal",
        )
        # Term 12
        _V += +1.000 * np.einsum(
            "ivua,iwxa->vxuw", T2[hc, ha, pa, pv], V[hc, ha, pa, pv], optimize="optimal"
        )
        _V += +1.000 * np.einsum(
            "ivuy,iwxz,yz->vxuw",
            T2[hc, ha, pa, pa],
            V[hc, ha, pa, pa],
            eta1,
            optimize="optimal",
        )
        _V += +1.000 * np.einsum(
            "vyua,wzxa,zy->vxuw",
            T2[ha, ha, pa, pv],
            V[ha, ha, pa, pv],
            gamma1,
            optimize="optimal",
        )

        return antisymmetrize_2body(_V.conj(), "aaaa")
        # return _V.conj()

    def _compute_Hbar_aa(self, F, V, T1, T2, gamma1, eta1, lambda2):
        hc = self.hc
        ha = self.ha
        pa = self.pa
        pv = self.pv

        # all quantities are stored ^{hh..}_{pp..}
        # h = {c,a}; p = {a,v}
        _F = np.zeros((self.nact, self.nact), dtype=complex)
        _F += -1.000 * np.einsum("iu,iv->uv", F[hc, pa], T1[hc, pa], optimize="optimal")
        _F += -1.000 * np.einsum(
            "iw,ivux,xw->vu", F[hc, pa], T2[hc, ha, pa, pa], eta1, optimize="optimal"
        )
        _F += -1.000 * np.einsum(
            "ia,ivua->vu", F[hc, pv], T2[hc, ha, pa, pv], optimize="optimal"
        )
        _F += +1.000 * np.einsum("ua,va->vu", F[ha, pv], T1[ha, pv], optimize="optimal")
        _F += +1.000 * np.einsum(
            "wa,vxua,wx->vu", F[ha, pv], T2[ha, ha, pa, pv], gamma1, optimize="optimal"
        )
        _F += -1.000 * np.einsum(
            "iw,iuvx,wx->vu", T1[hc, pa], V[hc, ha, pa, pa], eta1, optimize="optimal"
        )
        _F += -1.000 * np.einsum(
            "ia,iuva->vu", T1[hc, pv], V[hc, ha, pa, pv], optimize="optimal"
        )
        _F += +1.000 * np.einsum(
            "wa,uxva,xw->vu", T1[ha, pv], V[ha, ha, pa, pv], gamma1, optimize="optimal"
        )
        _F += -0.500 * np.einsum(
            "ijuw,ijvx,wx->vu",
            T2[hc, hc, pa, pa],
            V[hc, hc, pa, pa],
            eta1,
            optimize="optimal",
        )
        _F += +0.500 * np.einsum(
            "ivuw,ixyz,wxyz->vu",
            T2[hc, ha, pa, pa],
            V[hc, ha, pa, pa],
            lambda2,
            optimize="optimal",
        )
        _F += -1.000 * np.einsum(
            "ixuw,iyvz,wz,yx->vu",
            T2[hc, ha, pa, pa],
            V[hc, ha, pa, pa],
            eta1,
            gamma1,
            optimize="optimal",
        )
        _F += -1.000 * np.einsum(
            "ixuw,iyvz,wyxz->vu",
            T2[hc, ha, pa, pa],
            V[hc, ha, pa, pa],
            lambda2,
            optimize="optimal",
        )
        _F += -0.500 * np.einsum(
            "ijua,ijva->vu", T2[hc, hc, pa, pv], V[hc, hc, pa, pv], optimize="optimal"
        )
        _F += -1.000 * np.einsum(
            "iwua,ixva,xw->vu",
            T2[hc, ha, pa, pv],
            V[hc, ha, pa, pv],
            gamma1,
            optimize="optimal",
        )
        _F += -0.500 * np.einsum(
            "vwua,xyza,xywz->vu",
            T2[ha, ha, pa, pv],
            V[ha, ha, pa, pv],
            lambda2,
            optimize="optimal",
        )
        _F += -0.500 * np.einsum(
            "wxua,yzva,zx,yw->vu",
            T2[ha, ha, pa, pv],
            V[ha, ha, pa, pv],
            gamma1,
            gamma1,
            optimize="optimal",
        )
        _F += -0.250 * np.einsum(
            "wxua,yzva,yzwx->vu",
            T2[ha, ha, pa, pv],
            V[ha, ha, pa, pv],
            lambda2,
            optimize="optimal",
        )
        _F += +0.500 * np.einsum(
            "iuwx,ivyz,xz,wy->uv",
            T2[hc, ha, pa, pa],
            V[hc, ha, pa, pa],
            eta1,
            eta1,
            optimize="optimal",
        )
        _F += +0.250 * np.einsum(
            "iuwx,ivyz,wxyz->uv",
            T2[hc, ha, pa, pa],
            V[hc, ha, pa, pa],
            lambda2,
            optimize="optimal",
        )
        _F += -0.500 * np.einsum(
            "iywx,iuvz,wxyz->vu",
            T2[hc, ha, pa, pa],
            V[hc, ha, pa, pa],
            lambda2,
            optimize="optimal",
        )
        _F += +1.000 * np.einsum(
            "iuwa,ivxa,wx->uv",
            T2[hc, ha, pa, pv],
            V[hc, ha, pa, pv],
            eta1,
            optimize="optimal",
        )
        _F += +1.000 * np.einsum(
            "uxwa,vyza,wz,yx->uv",
            T2[ha, ha, pa, pv],
            V[ha, ha, pa, pv],
            eta1,
            gamma1,
            optimize="optimal",
        )
        _F += +1.000 * np.einsum(
            "uxwa,vyza,wyxz->uv",
            T2[ha, ha, pa, pv],
            V[ha, ha, pa, pv],
            lambda2,
            optimize="optimal",
        )
        _F += +0.500 * np.einsum(
            "xywa,uzva,wzxy->vu",
            T2[ha, ha, pa, pv],
            V[ha, ha, pa, pv],
            lambda2,
            optimize="optimal",
        )
        _F += +0.500 * np.einsum(
            "iuab,ivab->uv", T2[hc, ha, pv, pv], V[hc, ha, pv, pv], optimize="optimal"
        )
        _F += +0.500 * np.einsum(
            "uwab,vxab,xw->uv",
            T2[ha, ha, pv, pv],
            V[ha, ha, pv, pv],
            gamma1,
            optimize="optimal",
        )

        return _F.conj()
