from dataclasses import dataclass

import numpy as np

from forte2 import dsrg_utils
from forte2.ci.ci_utils import make_2cumulant, make_3cumulant
from .dsrg_base import DSRGBase
from .utils import antisymmetrize_2body, cas_energy_given_cumulants

MACHEPS = 1e-9
TAYLOR_THRES = 1e-3

compute_t1_block = dsrg_utils.compute_T1_block
compute_t2_block = dsrg_utils.compute_T2_block
renormalize_V_block = dsrg_utils.renormalize_V_block
renormalize_3index = dsrg_utils.renormalize_3index


@dataclass
class DSRG_MRPT2(DSRGBase):
    """
    Driven similarity renormalization group second-order multireference perturbation theory (DSRG-MRPT2).
    """

    def get_integrals(self):
        g1 = self.ci_solver.make_average_1rdm()
        # self._C are the MCSCF canonical orbitals. We always use canonical orbitals to build the generalized Fock matrix.
        self.semicanonicalizer.semi_canonicalize(g1=g1, C_contig=self._C)
        self._C_semican = self.semicanonicalizer.C_semican.copy()
        self.fock = self.semicanonicalizer.fock_semican.copy()
        self.eps = self.semicanonicalizer.eps_semican.copy()
        self.delta_actv = self.eps[self.actv][:, None] - self.eps[self.actv][None, :]
        self.Uactv = self.semicanonicalizer.Uactv

        if self.two_component:
            ints = dict()
            ints["F"] = self.fock - np.diag(np.diag(self.fock))  # remove diagonal

            cumulants = dict()
            # g1 = self.ci_solver.make_average_1rdm()
            cumulants["gamma1"] = np.einsum(
                "ip,ij,jq->pq", self.Uactv, g1, self.Uactv.conj(), optimize=True
            )
            cumulants["eta1"] = (
                np.eye(cumulants["gamma1"].shape[0], dtype=complex)
                - cumulants["gamma1"]
            )
            g2 = self.ci_solver.make_average_2rdm()
            g3 = self.ci_solver.make_average_3rdm()
            l2 = make_2cumulant(g1, g2)
            cumulants["lambda2"] = np.einsum(
                "ip,jq,ijkl,kr,ls->pqrs",
                self.Uactv,
                self.Uactv,
                l2,
                self.Uactv.conj(),
                self.Uactv.conj(),
                optimize=True,
            )
            l3 = make_3cumulant(g1, l2, g3)
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

            ints["E"] = cas_energy_given_cumulants(
                self.E_core_orig, self.H_orig, self.V_orig, g1, g2
            )

            # Save blocks of spinorbital basis B tensor
            B_so = dict()
            C_core = self._C_semican[:, self.core]
            C_actv = self._C_semican[:, self.actv]
            C_virt = self._C_semican[:, self.virt]
            B_so["cc"] = self.fock_builder.B_tensor_gen_block_spinor(C_core, C_core)
            B_so["ca"] = self.fock_builder.B_tensor_gen_block_spinor(C_core, C_actv)
            B_so["cv"] = self.fock_builder.B_tensor_gen_block_spinor(C_core, C_virt)
            B_so["aa"] = self.fock_builder.B_tensor_gen_block_spinor(C_actv, C_actv)
            B_so["av"] = self.fock_builder.B_tensor_gen_block_spinor(C_actv, C_virt)

            ints["V"] = dict()
            ints["V"]["aaaa"] = np.einsum(
                "Bux,Bvy->uvxy",
                B_so["aa"],
                B_so["aa"],
                optimize=True,
            )
            ints["V"]["aaaa"] -= ints["V"]["aaaa"].swapaxes(2, 3)
            ints["V"]["caaa"] = np.einsum(
                "Biu,Bvw->ivuw",
                B_so["ca"],
                B_so["aa"],
                optimize=True,
            )
            ints["V"]["caaa"] -= ints["V"]["caaa"].swapaxes(2, 3)
            ints["V"]["aaav"] = np.einsum(
                "Buv,Bwa->uwva",
                B_so["aa"],
                B_so["av"],
                optimize=True,
            )
            ints["V"]["aaav"] -= ints["V"]["aaav"].swapaxes(0, 1)
            ints["V"]["ccaa"] = np.einsum(
                "Biu,Bjv->ijuv",
                B_so["ca"],
                B_so["ca"],
                optimize=True,
            )
            ints["V"]["ccaa"] -= ints["V"]["ccaa"].swapaxes(2, 3)
            ints["V"]["caav"] = np.einsum(
                "Biu,Bva->ivua",
                B_so["ca"],
                B_so["av"],
                optimize=True,
            )
            ints["V"]["caav"] -= np.einsum(
                "Bia,Bvu->ivua",
                B_so["cv"],
                B_so["aa"],
                optimize=True,
            )
            ints["V"]["aavv"] = np.einsum(
                "Bua,Bvb->uvab",
                B_so["av"],
                B_so["av"],
                optimize=True,
            )
            ints["V"]["aavv"] -= ints["V"]["aavv"].swapaxes(2, 3)
            ints["V"]["caca"] = np.einsum(
                "Bij,Buv->iujv",
                B_so["cc"],
                B_so["aa"],
                optimize=True,
            )

            # These are used in on-the-fly energy/Hbar computations
            ints["B"] = dict()
            ints["B"]["ca"] = B_so["ca"].transpose(1, 2, 0).copy()
            ints["B"]["cv"] = B_so["cv"].transpose(1, 2, 0).copy()
            ints["B"]["av"] = B_so["av"].transpose(1, 2, 0).copy()

            ints["eps"] = dict()
            ints["eps"]["core"] = self.eps[self.core].copy()
            ints["eps"]["actv"] = self.eps[self.actv].copy()
            ints["eps"]["virt"] = self.eps[self.virt].copy()
            # <Psi_0 | bare H | Psi_0>, where Psi_0 is the current (possibly relaxed) reference

            return ints, cumulants
        else:
            raise NotImplementedError("Only two-component integrals are implemented.")

    def solve_dsrg(self, form_hbar=False):
        self.T1, self.T2 = self._build_tamps()
        self.F_tilde, self.V_tilde = self._renormalize_integrals()
        if form_hbar:
            self.hbar_aa_df = np.zeros((self.nact, self.nact), dtype=complex)
        E = self._compute_pt2_energy(
            self.F_tilde,
            self.V_tilde,
            self.T1,
            self.T2,
            self.cumulants["gamma1"],
            self.cumulants["eta1"],
            self.cumulants["lambda2"],
            self.cumulants["lambda3"],
            form_hbar=form_hbar,
        )
        E += self.ints["E"]
        return E

    def do_reference_relaxation(self):
        _hbar2 = self.ints["V"]["aaaa"].copy()
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
        _hbar1 += _C1 + _C1.conj().T

        # see eq 29 of Ann. Rev. Phys. Chem.
        _e_scalar = (
            -np.einsum("uv,uv->", _hbar1, self.cumulants["gamma1"])
            - 0.25 * np.einsum("uvxy,uvxy->", _hbar2, self.cumulants["lambda2"])
            + 0.5*np.einsum(
                "uvxy,ux,vy->",
                _hbar2,
                self.cumulants["gamma1"],
                self.cumulants["gamma1"],
            )
        ) + self.E_dsrg

        _hbar1 -= np.einsum("uxvy,xy->uv", _hbar2, self.cumulants["gamma1"])

        _hbar1_canon = np.einsum(
            "ip,pq,jq->ij", self.Uactv, _hbar1, self.Uactv.conj(), optimize=True
        )
        _hbar2_canon = np.einsum(
            "ip,jq,pqrs,kr,ls->ijkl",
            self.Uactv,
            self.Uactv,
            _hbar2,
            self.Uactv.conj(),
            self.Uactv.conj(),
            optimize=True,
        )

        self.ci_solver.set_ints(_e_scalar, _hbar1_canon, _hbar2_canon)
        self.ci_solver.run(use_asym_ints=True)
        e_relaxed = self.ci_solver.compute_average_energy()
        self.relax_eigvals = self.ci_solver.evals_flat.copy()
        return e_relaxed

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

        t1_tmp = self.ints["F"][self.hole, self.part].conj().copy()
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
        f_temp = np.conj(self.ints["F"][self.hole, self.part]).copy()
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

    def _compute_pt2_energy(
        self, F, V, T1, T2, gamma1, eta1, lambda2, lambda3, form_hbar=False
    ):
        E = 0.0

        E += +1.000 * np.einsum(
            "iu,iv,vu->",
            F["ca"],
            T1["ca"],
            eta1,
            optimize=True,
        )
        E += +1.000 * np.einsum(
            "ia,ia->",
            F["cv"],
            T1["cv"],
            optimize=True,
        )
        E += +1.000 * np.einsum(
            "ua,va,uv->",
            F["av"],
            T1["av"],
            gamma1,
            optimize=True,
        )
        E += -0.500 * np.einsum(
            "iu,ixvw,vwux->",
            F["ca"],
            T2["caaa"],
            lambda2,
            optimize=True,
        )
        E += -0.500 * np.einsum(
            "ua,wxva,uvwx->",
            F["av"],
            T2["aaav"],
            lambda2,
            optimize=True,
        )
        E += -0.500 * np.einsum(
            "iu,ivwx,uvwx->",
            T1["ca"],
            V["caaa"],
            lambda2,
            optimize=True,
        )
        E += -0.500 * np.einsum(
            "ua,vwxa,vwux->",
            T1["av"],
            V["aaav"],
            lambda2,
            optimize=True,
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
        E += self._compute_pt2_energy_cavv(form_hbar=form_hbar)
        E += self._compute_pt2_energy_ccav(form_hbar=form_hbar)

        return E

    def _compute_pt2_energy_ccvv(self):
        # This computes the following contribution to the energy:
        # E += +0.250 * np.einsum("ijab,ijab->", T2["ccvv"], V["ccvv"], optimize=True)
        E = 0.0
        Vbare_i = np.empty((self.ncore, self.nvirt, self.nvirt), dtype=complex)
        Vtmp = np.empty((self.ncore, self.nvirt, self.nvirt), dtype=complex)
        Vr_i = np.empty((self.ncore, self.nvirt, self.nvirt), dtype=complex)
        B_cv = self.ints["B"]["cv"]
        for i in range(self.ncore):
            # T2 = conj(Vbare) * renorm
            # V = Vbare * (1 + exp)
            # So, we compute conj(Vbare) * Vr, where Vr = Vbare * renorm * (1 + exp)
            np.einsum("aB,jbB->jba", B_cv[i, :, :], B_cv, optimize=True, out=Vbare_i)
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
            E += 0.250 * np.einsum("jba,jba->", Vbare_i.conj(), Vr_i, optimize=True)

        return E

    def _compute_pt2_energy_cavv(self, form_hbar=False):
        # This computes the following contribution to the energy:
        # E += +0.500 * np.einsum(
        #     "iuab,ivab,vu->",
        #     T2["cavv"],
        #     V["cavv"],
        #     gamma1,
        #     optimize=True,
        # )
        # If relaxing the reference, also compute the cavv contribution to Hbar_aa
        # _F += +0.500 * np.einsum(
        #     "iuab,ivab->uv",
        #     T2["cavv"],
        #     V["cavv"],
        #     optimize=True,
        # )
        E = 0.0
        Vbare_i = np.empty((self.nact, self.nvirt, self.nvirt), dtype=complex)
        Vtmp = np.empty((self.nact, self.nvirt, self.nvirt), dtype=complex)
        Vr_i = np.empty((self.nact, self.nvirt, self.nvirt), dtype=complex)
        B_av = self.ints["B"]["av"]
        B_cv = self.ints["B"]["cv"]
        for i in range(self.ncore):
            # T2 = conj(Vbare) * renorm
            # V = Vbare * (1 + exp)
            # So, we compute Vbare * Vr, where Vr = Vbare * renorm * (1 + exp)
            np.einsum("aB,ubB->uba", B_cv[i, :, :], B_av, optimize=True, out=Vbare_i)
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
                "uba,vba,uv->",
                Vbare_i.conj(),
                Vr_i,
                self.cumulants["gamma1"],
                optimize=True,
            )
            if form_hbar:
                self.hbar_aa_df += 0.500 * np.einsum(
                    "uba,vba->uv",
                    Vbare_i.conj(),
                    Vr_i,
                    optimize=True,
                )

        return E

    def _compute_pt2_energy_ccav(self, form_hbar=False):
        # This computes the following contribution to the energy:
        # E += +0.500 * np.einsum(
        #     "ijua,ijva,uv->",
        #     T2["ccav"],
        #     V["ccav"],
        #     eta1,
        #     optimize=True,
        # )
        # If relaxing the reference, also compute the ccav contribution to Hbar_aa
        # _F += -0.500 * np.einsum(
        #     "ijua,ijva->vu",
        #     T2["ccav"],
        #     V["ccav"],
        #     optimize=True,
        # )

        E = 0.0
        Vbare_i = np.empty((self.ncore, self.nact, self.nvirt), dtype=complex)
        Vtmp = np.empty((self.ncore, self.nact, self.nvirt), dtype=complex)
        Vr_i = np.empty((self.ncore, self.nact, self.nvirt), dtype=complex)
        B_cv = self.ints["B"]["cv"]
        B_ca = self.ints["B"]["ca"]
        for i in range(self.ncore):
            # T2 = conj(Vbare) * renorm
            # V = Vbare * (1 + exp)
            # So, we compute conj(Vbare) * Vr, where Vr = Vbare * renorm * (1 + exp)
            np.einsum("uB,jaB->jua", B_ca[i, :, :], B_cv, optimize=True, out=Vbare_i)
            np.einsum("aB,juB->jua", B_cv[i, :, :], B_ca, optimize=True, out=Vtmp)
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
            if form_hbar:
                self.hbar_aa_df += -0.500 * np.einsum(
                    "jua,jva->vu",
                    Vbare_i.conj(),
                    Vr_i,
                    optimize=True,
                )

        return E

    def _compute_Hbar_aaaa(self, F, V, T1, T2, gamma1, eta1):
        _V = np.zeros((self.nact,) * 4, dtype=complex)
        _V += -0.500 * np.einsum(
            "ua,wxva->wxuv",
            F["av"],
            T2["aaav"],
            optimize=True,
        )
        _V += -0.500 * np.einsum(
            "iu,ixvw->uxvw",
            F["ca"],
            T2["caaa"],
            optimize=True,
        )
        _V += -0.500 * np.einsum(
            "iu,ivwx->wxuv",
            T1["ca"],
            V["caaa"],
            optimize=True,
        )
        _V += -0.500 * np.einsum(
            "ua,vwxa->uxvw",
            T1["av"],
            V["aaav"],
            optimize=True,
        )
        _V += +0.125 * np.einsum(
            "uvab,wxab->uvwx",
            T2["aavv"],
            V["aavv"],
            optimize=True,
        )
        _V += +0.250 * np.einsum(
            "uvya,wxza,yz->uvwx",
            T2["aaav"],
            V["aaav"],
            eta1,
            optimize=True,
        )
        _V += +0.125 * np.einsum(
            "ijuv,ijwx->wxuv",
            T2["ccaa"],
            V["ccaa"],
            optimize=True,
        )
        _V += +0.250 * np.einsum(
            "iyuv,izwx,zy->wxuv",
            T2["caaa"],
            V["caaa"],
            gamma1,
            optimize=True,
        )
        _V += +1.000 * np.einsum(
            "ivua,iwxa->vxuw",
            T2["caav"],
            V["caav"],
            optimize=True,
        )
        _V += +1.000 * np.einsum(
            "ivuy,iwxz,yz->vxuw",
            T2["caaa"],
            V["caaa"],
            eta1,
            optimize=True,
        )
        _V += +1.000 * np.einsum(
            "vyua,wzxa,zy->vxuw",
            T2["aaav"],
            V["aaav"],
            gamma1,
            optimize=True,
        )

        return antisymmetrize_2body(_V.conj(), "aaaa")

    def _compute_Hbar_aa(self, F, V, T1, T2, gamma1, eta1, lambda2):
        _F = self.hbar_aa_df.copy()
        _F += -1.000 * np.einsum(
            "iu,iv->uv",
            F["ca"],
            T1["ca"],
            optimize=True,
        )
        _F += -1.000 * np.einsum(
            "iw,ivux,xw->vu",
            F["ca"],
            T2["caaa"],
            eta1,
            optimize=True,
        )
        _F += -1.000 * np.einsum(
            "ia,ivua->vu",
            F["cv"],
            T2["caav"],
            optimize=True,
        )
        _F += +1.000 * np.einsum(
            "ua,va->vu",
            F["av"],
            T1["av"],
            optimize=True,
        )
        _F += +1.000 * np.einsum(
            "wa,vxua,wx->vu",
            F["av"],
            T2["aaav"],
            gamma1,
            optimize=True,
        )
        _F += -1.000 * np.einsum(
            "iw,iuvx,wx->vu",
            T1["ca"],
            V["caaa"],
            eta1,
            optimize=True,
        )
        _F += -1.000 * np.einsum(
            "ia,iuva->vu",
            T1["cv"],
            V["caav"],
            optimize=True,
        )
        _F += +1.000 * np.einsum(
            "wa,uxva,xw->vu",
            T1["av"],
            V["aaav"],
            gamma1,
            optimize=True,
        )
        _F += -0.500 * np.einsum(
            "ijuw,ijvx,wx->vu",
            T2["ccaa"],
            V["ccaa"],
            eta1,
            optimize=True,
        )
        _F += +0.500 * np.einsum(
            "ivuw,ixyz,wxyz->vu",
            T2["caaa"],
            V["caaa"],
            lambda2,
            optimize=True,
        )
        _F += -1.000 * np.einsum(
            "ixuw,iyvz,wz,yx->vu",
            T2["caaa"],
            V["caaa"],
            eta1,
            gamma1,
            optimize=True,
        )
        _F += -1.000 * np.einsum(
            "ixuw,iyvz,wyxz->vu",
            T2["caaa"],
            V["caaa"],
            lambda2,
            optimize=True,
        )
        # _F += -0.500 * np.einsum(
        #     "ijua,ijva->vu",
        #     T2["ccav"],
        #     V["ccav"],
        #     optimize=True,
        # )
        _F += -1.000 * np.einsum(
            "iwua,ixva,xw->vu",
            T2["caav"],
            V["caav"],
            gamma1,
            optimize=True,
        )
        _F += -0.500 * np.einsum(
            "vwua,xyza,xywz->vu",
            T2["aaav"],
            V["aaav"],
            lambda2,
            optimize=True,
        )
        _F += -0.500 * np.einsum(
            "wxua,yzva,zx,yw->vu",
            T2["aaav"],
            V["aaav"],
            gamma1,
            gamma1,
            optimize=True,
        )
        _F += -0.250 * np.einsum(
            "wxua,yzva,yzwx->vu",
            T2["aaav"],
            V["aaav"],
            lambda2,
            optimize=True,
        )
        _F += +0.500 * np.einsum(
            "iuwx,ivyz,xz,wy->uv",
            T2["caaa"],
            V["caaa"],
            eta1,
            eta1,
            optimize=True,
        )
        _F += +0.250 * np.einsum(
            "iuwx,ivyz,wxyz->uv",
            T2["caaa"],
            V["caaa"],
            lambda2,
            optimize=True,
        )
        _F += -0.500 * np.einsum(
            "iywx,iuvz,wxyz->vu",
            T2["caaa"],
            V["caaa"],
            lambda2,
            optimize=True,
        )
        _F += +1.000 * np.einsum(
            "iuwa,ivxa,wx->uv",
            T2["caav"],
            V["caav"],
            eta1,
            optimize=True,
        )
        _F += +1.000 * np.einsum(
            "uxwa,vyza,wz,yx->uv",
            T2["aaav"],
            V["aaav"],
            eta1,
            gamma1,
            optimize=True,
        )
        _F += +1.000 * np.einsum(
            "uxwa,vyza,wyxz->uv",
            T2["aaav"],
            V["aaav"],
            lambda2,
            optimize=True,
        )
        _F += +0.500 * np.einsum(
            "xywa,uzva,wzxy->vu",
            T2["aaav"],
            V["aaav"],
            lambda2,
            optimize=True,
        )
        # _F += +0.500 * np.einsum(
        #     "iuab,ivab->uv",
        #     T2["cavv"],
        #     V["cavv"],
        #     optimize=True,
        # )
        _F += +0.500 * np.einsum(
            "uwab,vxab,xw->uv",
            T2["aavv"],
            V["aavv"],
            gamma1,
            optimize=True,
        )

        return _F.conj()
