from dataclasses import dataclass, field

import numpy as np

from forte2 import dsrg_utils
from forte2.jkbuilder.mointegrals import RestrictedMOIntegrals, SpinorbitalIntegrals
from forte2.orbitals import Semicanonicalizer
from .dsrg_base import DSRGBase

MACHEPS = 1e-9
TAYLOR_THRES = 1e-3

compute_t1_block = dsrg_utils.compute_T1_block
compute_t2_block = dsrg_utils.compute_T2_block
renormalize_V_block = dsrg_utils.renormalize_V_block
renormalize_3index = dsrg_utils.renormalize_3index


@dataclass
class DSRG_MRPT2_DF(DSRGBase):
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
            ints = dict()
            B = self.fock_builder.B
            nbf = self.system.nbf
            B_so = np.zeros((B.shape[0], nbf * 2, nbf * 2), dtype=complex)
            B_so[:, :nbf, :nbf] = B
            B_so[:, nbf:, nbf:] = B
            ints["B"] = np.einsum("Bpq,pi,qj->Bij", B_so, self._C.conj(), self._C)
            ints["F"] = self.fock - np.diag(np.diag(self.fock))  # remove diagonal

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

            ints["V"] = dict()
            ints["V"]["caaa"] = np.einsum(
                "Biu,Bvw->ivuw",
                ints["B"][:, self.core, self.actv],
                ints["B"][:, self.actv, self.actv],
                optimize=True,
            )
            ints["V"]["caaa"] -= ints["V"]["caaa"].swapaxes(2, 3)
            ints["V"]["aaav"] = np.einsum(
                "Buv,Bwa->uwva",
                ints["B"][:, self.actv, self.actv],
                ints["B"][:, self.actv, self.virt],
                optimize=True,
            )
            ints["V"]["aaav"] -= ints["V"]["aaav"].swapaxes(0, 1)
            ints["V"]["ccaa"] = np.einsum(
                "Biu,Bjv->ijuv",
                ints["B"][:, self.core, self.actv],
                ints["B"][:, self.core, self.actv],
                optimize=True,
            )
            ints["V"]["ccaa"] -= ints["V"]["ccaa"].swapaxes(2, 3)
            # ints["V"]["ccav"] = np.einsum(
            #     "Biu,Bja->ijua",
            #     ints["B"][:, self.core, self.actv],
            #     ints["B"][:, self.core, self.virt],
            #     optimize=True,
            # )
            # ints["V"]["ccav"] -= ints["V"]["ccav"].swapaxes(0, 1)
            ints["V"]["caav"] = np.einsum(
                "Biu,Bva->ivua",
                ints["B"][:, self.core, self.actv],
                ints["B"][:, self.actv, self.virt],
                optimize=True,
            )
            ints["V"]["caav"] -= np.einsum(
                "Bia,Bvu->ivua",
                ints["B"][:, self.core, self.virt],
                ints["B"][:, self.actv, self.actv],
                optimize=True,
            )
            ints["V"]["aavv"] = np.einsum(
                "Bua,Bvb->uvab",
                ints["B"][:, self.actv, self.virt],
                ints["B"][:, self.actv, self.virt],
                optimize=True,
            )
            ints["V"]["aavv"] -= ints["V"]["aavv"].swapaxes(2, 3)
            ints["eps"] = dict()
            ints["eps"]["core"] = self.eps[self.core].copy()
            ints["eps"]["actv"] = self.eps[self.actv].copy()
            ints["eps"]["virt"] = self.eps[self.virt].copy()
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
        # 1b: ca, cv, av
        # 2b: caaa, aaav, ccaa, ccav, caav, ccvv, aavv, cavv
        t2 = dict()
        t2["caaa"] = self.ints["V"]["caaa"].conj()
        compute_t2_block(
            t2["caaa"],
            self.ints["eps"]["core"],
            self.ints["eps"]["actv"],
            self.ints["eps"]["actv"],
            self.ints["eps"]["actv"],
            self.flow_param,
        )
        t2["aaav"] = self.ints["V"]["aaav"].conj()
        compute_t2_block(
            t2["aaav"],
            self.ints["eps"]["actv"],
            self.ints["eps"]["actv"],
            self.ints["eps"]["actv"],
            self.ints["eps"]["virt"],
            self.flow_param,
        )
        t2["ccaa"] = self.ints["V"]["ccaa"].conj()
        compute_t2_block(
            t2["ccaa"],
            self.ints["eps"]["core"],
            self.ints["eps"]["core"],
            self.ints["eps"]["actv"],
            self.ints["eps"]["actv"],
            self.flow_param,
        )
        # t2["ccav"] = self.ints["V"]["ccav"].conj()
        # compute_t2_block(
        #     t2["ccav"],
        #     self.ints["eps"]["core"],
        #     self.ints["eps"]["core"],
        #     self.ints["eps"]["actv"],
        #     self.ints["eps"]["virt"],
        #     self.flow_param,
        # )
        t2["caav"] = self.ints["V"]["caav"].conj()
        compute_t2_block(
            t2["caav"],
            self.ints["eps"]["core"],
            self.ints["eps"]["actv"],
            self.ints["eps"]["actv"],
            self.ints["eps"]["virt"],
            self.flow_param,
        )
        t2["aavv"] = self.ints["V"]["aavv"].conj()
        compute_t2_block(
            t2["aavv"],
            self.ints["eps"]["actv"],
            self.ints["eps"]["actv"],
            self.ints["eps"]["virt"],
            self.ints["eps"]["virt"],
            self.flow_param,
        )

        t1_tmp = self.ints["F"][self.hole, self.part].conj()
        t2_hapa = np.zeros(
            (self.nhole, self.nact, self.npart, self.nact), dtype=complex
        )
        t2_hapa[self.hc, :, self.pa, :] = t2["caaa"].copy()
        t2_hapa[self.hc, :, self.pv, :] = -t2["caav"].swapaxes(2, 3).copy()
        t2_hapa[self.ha, :, self.pv, :] = -t2["aaav"].swapaxes(2, 3).copy()
        t1_tmp += np.einsum(
            "xu,iuax,xu->ia",
            self.delta_actv,
            t2_hapa,
            self.cumulants["gamma1"],
            optimize=True,
        )
        t1 = dict()
        t1["ca"] = t1_tmp[self.hc, self.pa].copy()
        t1["cv"] = t1_tmp[self.hc, self.pv].copy()
        t1["av"] = t1_tmp[self.ha, self.pv].copy()
        compute_t1_block(
            t1["ca"],
            self.ints["eps"]["core"],
            self.ints["eps"]["actv"],
            self.flow_param,
        )
        compute_t1_block(
            t1["cv"],
            self.ints["eps"]["core"],
            self.ints["eps"]["virt"],
            self.flow_param,
        )
        compute_t1_block(
            t1["av"],
            self.ints["eps"]["actv"],
            self.ints["eps"]["virt"],
            self.flow_param,
        )
        return t1, t2

    def _renormalize_integrals(self):
        f_temp = np.conj(self.ints["F"][self.hole, self.part])
        delta_ia = self.eps[self.hole][:, None] - self.eps[self.part][None, :]
        exp_delta_1 = np.exp(-self.flow_param * delta_ia**2)
        t2_hapa = np.zeros(
            (self.nhole, self.nact, self.npart, self.nact), dtype=complex
        )
        t2_hapa[self.hc, :, self.pa, :] = self.T2["caaa"].copy()
        t2_hapa[self.hc, :, self.pv, :] = -self.T2["caav"].swapaxes(2, 3).copy()
        t2_hapa[self.ha, :, self.pv, :] = -self.T2["aaav"].swapaxes(2, 3).copy()
        f_temp += (
            f_temp * exp_delta_1
            + np.einsum(
                "xu,iuax,xu->ia",
                self.delta_actv,
                t2_hapa,
                self.cumulants["gamma1"],
                optimize=True,
            )
            * exp_delta_1
        )
        np.conj(f_temp, out=f_temp)
        F_tilde = dict()
        F_tilde["ca"] = f_temp[self.hc, self.pa].copy()
        F_tilde["cv"] = f_temp[self.hc, self.pv].copy()
        F_tilde["av"] = f_temp[self.ha, self.pv].copy()

        V_tilde = dict()
        # caaa, aaav, ccaa, ccav, caav, ccvv, aavv
        V_tilde["caaa"] = np.copy(self.ints["V"]["caaa"])
        renormalize_V_block(
            V_tilde["caaa"],
            self.ints["eps"]["core"],
            self.ints["eps"]["actv"],
            self.ints["eps"]["actv"],
            self.ints["eps"]["actv"],
            self.flow_param,
        )
        V_tilde["aaav"] = np.copy(self.ints["V"]["aaav"])
        renormalize_V_block(
            V_tilde["aaav"],
            self.ints["eps"]["actv"],
            self.ints["eps"]["actv"],
            self.ints["eps"]["actv"],
            self.ints["eps"]["virt"],
            self.flow_param,
        )
        V_tilde["ccaa"] = np.copy(self.ints["V"]["ccaa"])
        renormalize_V_block(
            V_tilde["ccaa"],
            self.ints["eps"]["core"],
            self.ints["eps"]["core"],
            self.ints["eps"]["actv"],
            self.ints["eps"]["actv"],
            self.flow_param,
        )
        # V_tilde["ccav"] = np.copy(self.ints["V"]["ccav"])
        # renormalize_V_block(
        #     V_tilde["ccav"],
        #     self.ints["eps"]["core"],
        #     self.ints["eps"]["core"],
        #     self.ints["eps"]["actv"],
        #     self.ints["eps"]["virt"],
        #     self.flow_param,
        # )
        V_tilde["caav"] = np.copy(self.ints["V"]["caav"])
        renormalize_V_block(
            V_tilde["caav"],
            self.ints["eps"]["core"],
            self.ints["eps"]["actv"],
            self.ints["eps"]["actv"],
            self.ints["eps"]["virt"],
            self.flow_param,
        )
        V_tilde["aavv"] = np.copy(self.ints["V"]["aavv"])
        renormalize_V_block(
            V_tilde["aavv"],
            self.ints["eps"]["actv"],
            self.ints["eps"]["actv"],
            self.ints["eps"]["virt"],
            self.ints["eps"]["virt"],
            self.flow_param,
        )
        return F_tilde, V_tilde

    def _compute_pt2_energy(self, F, V, T1, T2, gamma1, eta1, lambda2, lambda3):
        E = 0.0

        E += +1.000 * np.einsum("iu,iv,vu->", F["ca"], T1["ca"], eta1, optimize=True)
        E += +1.000 * np.einsum("ia,ia->", F["cv"], T1["cv"], optimize=True)
        E += +1.000 * np.einsum("ua,va,uv->", F["av"], T1["av"], gamma1, optimize=True)
        E += -0.500 * np.einsum(
            "iu,ixvw,vwux->", F["ca"], T2["caaa"], lambda2, optimize=True
        )
        E += -0.500 * np.einsum(
            "ua,wxva,uvwx->", F["av"], T2["aaav"], lambda2, optimize=True
        )
        E += -0.500 * np.einsum(
            "iu,ivwx,uvwx->", T1["ca"], V["caaa"], lambda2, optimize=True
        )
        E += -0.500 * np.einsum(
            "ua,vwxa,vwux->", T1["av"], V["aaav"], lambda2, optimize=True
        )
        E += +0.250 * np.einsum(
            "ijuv,ijwx,vx,uw->",
            T2["ccaa"],
            V["ccaa"],
            eta1,
            eta1,
            optimize=True,
        )
        E += +0.125 * np.einsum(
            "ijuv,ijwx,uvwx->",
            T2["ccaa"],
            V["ccaa"],
            lambda2,
            optimize=True,
        )
        E += +0.500 * np.einsum(
            "iwuv,ixyz,vz,uy,xw->",
            T2["caaa"],
            V["caaa"],
            eta1,
            eta1,
            gamma1,
            optimize=True,
        )
        E += +1.000 * np.einsum(
            "iwuv,ixyz,vz,uxwy->",
            T2["caaa"],
            V["caaa"],
            eta1,
            lambda2,
            optimize=True,
        )
        E += +0.250 * np.einsum(
            "iwuv,ixyz,xw,uvyz->",
            T2["caaa"],
            V["caaa"],
            gamma1,
            lambda2,
            optimize=True,
        )
        E += +0.250 * np.einsum(
            "iwuv,ixyz,uvxwyz->",
            T2["caaa"],
            V["caaa"],
            lambda3,
            optimize=True,
        )
        E += +1.000 * np.einsum(
            "ivua,iwxa,ux,wv->",
            T2["caav"],
            V["caav"],
            eta1,
            gamma1,
            optimize=True,
        )
        E += +1.000 * np.einsum(
            "ivua,iwxa,uwvx->",
            T2["caav"],
            V["caav"],
            lambda2,
            optimize=True,
        )
        E += +0.500 * np.einsum(
            "vwua,xyza,uz,yw,xv->",
            T2["aaav"],
            V["aaav"],
            eta1,
            gamma1,
            gamma1,
            optimize=True,
        )
        E += +0.250 * np.einsum(
            "vwua,xyza,uz,xyvw->",
            T2["aaav"],
            V["aaav"],
            eta1,
            lambda2,
            optimize=True,
        )
        E += +1.000 * np.einsum(
            "vwua,xyza,yw,uxvz->",
            T2["aaav"],
            V["aaav"],
            gamma1,
            lambda2,
            optimize=True,
        )
        E += -0.250 * np.einsum(
            "vwua,xyza,uxyvwz->",
            T2["aaav"],
            V["aaav"],
            lambda3,
            optimize=True,
        )
        E += +0.250 * np.einsum(
            "uvab,wxab,xv,wu->",
            T2["aavv"],
            V["aavv"],
            gamma1,
            gamma1,
            optimize=True,
        )
        E += +0.125 * np.einsum(
            "uvab,wxab,wxuv->",
            T2["aavv"],
            V["aavv"],
            lambda2,
            optimize=True,
        )
        E += self._compute_pt2_energy_ccvv()
        E += self._compute_pt2_energy_cavv()
        E += self._compute_pt2_energy_ccav()

        return E

    def _compute_pt2_energy_ccvv(self):
        # E += +0.250 * np.einsum("ijab,ijab->", T2["ccvv"], V["ccvv"], optimize=True)
        E = 0.0
        Vbare_i = np.empty((self.ncore, self.nvirt, self.nvirt), dtype=complex)
        Vtmp = np.empty((self.ncore, self.nvirt, self.nvirt), dtype=complex)
        Vr_i = np.empty((self.ncore, self.nvirt, self.nvirt), dtype=complex)
        B_cv = self.ints["B"][:, self.core, self.virt]
        for i in range(self.ncore):
            # T2 = conj(Vbare) * renorm
            # V = Vbare * (1 + exp)
            # So, we compute conj(Vbare) * Vr, where Vr = Vbare * renorm * (1 + exp)
            np.einsum("Ba,Bjb->jab", B_cv[:, i, :], B_cv, optimize=True, out=Vbare_i)
            np.copyto(Vtmp, Vbare_i.swapaxes(1, 2))
            Vbare_i -= Vtmp
            # copy to Vr_i
            Vr_i[:] = Vbare_i
            renormalize_3index(
                Vr_i,
                self.ints["eps"]["core"][i],
                self.ints["eps"]["core"],
                self.ints["eps"]["virt"],
                self.ints["eps"]["virt"],
                self.flow_param,
            )
            E += 0.250 * np.einsum("jab,jab->", Vbare_i.conj(), Vr_i, optimize=True)

        return E

    def _compute_pt2_energy_cavv(self):
        # E += +0.500 * np.einsum(
        #     "iuab,ivab,vu->",
        #     T2["cavv"],
        #     V["cavv"],
        #     gamma1,
        #     optimize=True,
        # )
        E = 0.0
        Vbare_i = np.empty((self.nact, self.nvirt, self.nvirt), dtype=complex)
        Vtmp = np.empty((self.nact, self.nvirt, self.nvirt), dtype=complex)
        Vr_i = np.empty((self.nact, self.nvirt, self.nvirt), dtype=complex)
        B_av = self.ints["B"][:, self.actv, self.virt]
        B_cv = self.ints["B"][:, self.core, self.virt]
        for i in range(self.ncore):
            # T2 = conj(Vbare) * renorm
            # V = Vbare * (1 + exp)
            # So, we compute Vbare * Vr, where Vr = Vbare * renorm * (1 + exp)
            np.einsum("Ba,Bub->uab", B_cv[:, i, :], B_av, optimize=True, out=Vbare_i)
            np.copyto(Vtmp, Vbare_i.swapaxes(1, 2))
            Vbare_i -= Vtmp
            # copy to Vr_i
            Vr_i[:] = Vbare_i
            renormalize_3index(
                Vr_i,
                self.ints["eps"]["core"][i],
                self.ints["eps"]["actv"],
                self.ints["eps"]["virt"],
                self.ints["eps"]["virt"],
                self.flow_param,
            )
            E += 0.500 * np.einsum(
                "uab,vab,uv->",
                Vbare_i.conj(),
                Vr_i,
                self.cumulants["gamma1"],
                optimize=True,
            )

        return E

    def _compute_pt2_energy_ccav(self):
        # E += +0.500 * np.einsum(
        #     "ijua,ijva,uv->",
        #     T2["ccav"],
        #     V["ccav"],
        #     eta1,
        #     optimize=True,
        # )
        E = 0.0
        Vbare_i = np.empty((self.ncore, self.nact, self.nvirt), dtype=complex)
        Vtmp = np.empty((self.ncore, self.nact, self.nvirt), dtype=complex)
        Vr_i = np.empty((self.ncore, self.nact, self.nvirt), dtype=complex)
        B_cv = self.ints["B"][:, self.core, self.virt]
        B_ca = self.ints["B"][:, self.core, self.actv]
        for i in range(self.ncore):
            # T2 = conj(Vbare) * renorm
            # V = Vbare * (1 + exp)
            # So, we compute conj(Vbare) * Vr, where Vr = Vbare * renorm * (1 + exp)
            np.einsum("Bu,Bja->jua", B_ca[:, i, :], B_cv, optimize=True, out=Vbare_i)
            np.einsum("Ba,Bju->jua", B_cv[:, i, :], B_ca, optimize=True, out=Vtmp)
            Vbare_i -= Vtmp
            # copy to Vr_i
            Vr_i[:] = Vbare_i
            renormalize_3index(
                Vr_i,
                self.ints["eps"]["core"][i],
                self.ints["eps"]["core"],
                self.ints["eps"]["actv"],
                self.ints["eps"]["virt"],
                self.flow_param,
            )
            E += 0.500 * np.einsum(
                "jua,jva,uv->",
                Vbare_i.conj(),
                Vr_i,
                self.cumulants["eta1"],
                optimize=True,
            )

        return E
