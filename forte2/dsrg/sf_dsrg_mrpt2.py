from dataclasses import dataclass

import numpy as np

from forte2 import dsrg_utils
from forte2.ci.ci_utils import make_2cumulant_sf, make_3cumulant_sf
from .dsrg_base import DSRGBase
from .utils import antisymmetrize_2body, cas_energy_given_cumulants

compute_t1_block = dsrg_utils.compute_T1_block
compute_t2_block = dsrg_utils.compute_T2_block
renormalize_V_block = dsrg_utils.renormalize_V_block_sf
renormalize_3index = dsrg_utils.renormalize_3index

renormalize_F = dsrg_utils.renormalize_F
renormalize_CCVV = dsrg_utils.renormalize_CCVV
renormalize_CAVV = dsrg_utils.renormalize_CAVV
renormalize_CCAV = dsrg_utils.renormalize_CCAV


@dataclass
class SF_DSRG_MRPT2(DSRGBase):
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

        ints = dict()
        # ints["F"] = self.fock - np.diag(np.diag(self.fock))  # remove diagonal

        cumulants = dict()
        # g1 = self.ci_solver.make_average_1rdm()
        cumulants["gamma1"] = np.einsum(
            "ip,ij,jq->pq", self.Uactv, g1, self.Uactv.conj(), optimize=True
        )
        cumulants["eta1"] = (
            2 * np.eye(cumulants["gamma1"].shape[0]) - cumulants["gamma1"]
        )
        g2 = self.ci_solver.make_average_2rdm()
        g3 = self.ci_solver.make_average_3rdm()
        l2 = make_2cumulant_sf(g1, g2)
        cumulants["lambda2"] = np.einsum(
            "ip,jq,ijkl,kr,ls->pqrs",
            self.Uactv,
            self.Uactv,
            l2,
            self.Uactv.conj(),
            self.Uactv.conj(),
            optimize=True,
        )
        l3 = make_3cumulant_sf(g1, g2, g3)
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
        #  ["vvaa", "aacc", "avca", "avac", "vaaa", "aaca", "aaaa"]
        # Save blocks of spinorbital basis B tensor
        B_mo = dict()
        C_core = self._C_semican[:, self.core]
        C_actv = self._C_semican[:, self.actv]
        C_virt = self._C_semican[:, self.virt]
        B_mo["cc"] = self.fock_builder.B_tensor_gen_block(C_core, C_core)
        B_mo["ac"] = self.fock_builder.B_tensor_gen_block(C_actv, C_core)
        B_mo["vc"] = self.fock_builder.B_tensor_gen_block(C_virt, C_core)
        B_mo["aa"] = self.fock_builder.B_tensor_gen_block(C_actv, C_actv)
        B_mo["va"] = self.fock_builder.B_tensor_gen_block(C_virt, C_actv)

        ints["V"] = dict()
        ints["V"]["vvaa"] = np.einsum(
            "Bau,Bbv->abuv",
            B_mo["va"],
            B_mo["va"],
            optimize=True,
        )
        ints["V"]["aacc"] = np.einsum(
            "Bui,Bvj->uvij",
            B_mo["ac"],
            B_mo["ac"],
            optimize=True,
        )
        ints["V"]["avca"] = np.einsum(
            "Bui,Bav->uaiv",
            B_mo["ac"],
            B_mo["va"],
            optimize=True,
        )
        ints["V"]["avac"] = np.einsum(
            "Buv,Bai->uavi",
            B_mo["aa"],
            B_mo["vc"],
            optimize=True,
        )
        ints["V"]["vaaa"] = np.einsum(
            "Bav,Bux->auvx",
            B_mo["va"],
            B_mo["aa"],
            optimize=True,
        )
        ints["V"]["aaca"] = np.einsum(
            "Bui,Bvx->uvix",
            B_mo["ac"],
            B_mo["aa"],
            optimize=True,
        )
        ints["V"]["aaaa"] = np.einsum(
            "Bux,Bvy->uvxy",
            B_mo["aa"],
            B_mo["aa"],
            optimize=True,
        )

        # These are used in on-the-fly energy/Hbar computations
        ints["B"] = dict()
        ints["B"]["ac"] = B_mo["ac"].transpose(2, 1, 0).copy()
        ints["B"]["vc"] = B_mo["vc"].transpose(2, 1, 0).copy()
        ints["B"]["va"] = B_mo["va"].transpose(1, 2, 0).copy()

        ints["eps"] = dict()
        ints["eps"]["c"] = self.eps[self.core].copy()
        ints["eps"]["a"] = self.eps[self.actv].copy()
        ints["eps"]["v"] = self.eps[self.virt].copy()
        # <Psi_0 | bare H | Psi_0>, where Psi_0 is the current (possibly relaxed) reference

        return ints, cumulants

    def solve_dsrg(self, form_hbar=False):
        self.T1, self.T2 = self._build_tamps()
        self.F_tilde = self._renormalize_F()
        # self.ints["V"] gets renormalizes to V_tilde in place for the following blocks:
        # caaa, aaav, ccaa, caav, aavv
        # The aaaa block is remains untouched, and can be safely used in reference relaxation
        self._renormalize_V_in_place()
        if form_hbar:
            self.hbar_aa_df = np.zeros((self.nact, self.nact))
        E = self._compute_pt2_energy(
            self.F_tilde,
            self.ints["V"],
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
            self.ints["V"],
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
            self.ints["V"],
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
            + 0.5
            * np.einsum(
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
        t2 = {"T2": dict(), "S2": dict()}

        for key in ["aavv", "ccaa", "caav", "acav", "aava", "caaa"]:
            vkey = key[2:] + key[:2]  # e.g., caaa -> aaca
            t2["T2"][key] = self.ints["V"][vkey].transpose(2, 3, 0, 1).copy()
            compute_t2_block(
                t2["T2"][key],
                *(self.ints["eps"][_] for _ in key),
                self.flow_param,
            )

        t2["S2"]["aavv"] = 2 * t2["T2"]["aavv"] - t2["T2"]["aavv"].swapaxes(2, 3)
        t2["S2"]["ccaa"] = 2 * t2["T2"]["ccaa"] - t2["T2"]["ccaa"].swapaxes(2, 3)
        t2["S2"]["caav"] = 2 * t2["T2"]["caav"] - t2["T2"]["acav"].swapaxes(0, 1)
        t2["S2"]["acav"] = 2 * t2["T2"]["acav"] - t2["T2"]["caav"].swapaxes(0, 1)
        t2["S2"]["aava"] = 2 * t2["T2"]["aava"] - t2["T2"]["aava"].swapaxes(0, 1)
        t2["S2"]["caaa"] = 2 * t2["T2"]["caaa"] - t2["T2"]["caaa"].swapaxes(2, 3)

        t1_tmp = self.fock[self.hole, self.part].copy()
        faa = self.fock[self.actv, self.actv]
        t1_tmp[self.hc, self.pa] += 0.5 * np.einsum(
            "ivaw,wu,uv->ia",
            t2["S2"]["caaa"],
            faa,
            self.cumulants["gamma1"],
            optimize=True,
        )
        t1_tmp[self.hc, self.pv] += 0.5 * np.einsum(
            "vmwe,wu,uv->me",
            t2["S2"]["acav"],
            faa,
            self.cumulants["gamma1"],
            optimize=True,
        )
        t1_tmp[self.ha, self.pv] += 0.5 * np.einsum(
            "ivaw,wu,uv->ia",
            t2["S2"]["aava"],
            faa,
            self.cumulants["gamma1"],
            optimize=True,
        )

        t1_tmp[self.hc, self.pa] -= 0.5 * np.einsum(
            "iwau,vw,uv->ia",
            t2["S2"]["caaa"],
            faa,
            self.cumulants["gamma1"],
            optimize=True,
        )
        t1_tmp[self.hc, self.pv] -= 0.5 * np.einsum(
            "wmue,vw,uv->me",
            t2["S2"]["acav"],
            faa,
            self.cumulants["gamma1"],
            optimize=True,
        )
        t1_tmp[self.ha, self.pv] -= 0.5 * np.einsum(
            "iwau,vw,uv->ia",
            t2["S2"]["aava"],
            faa,
            self.cumulants["gamma1"],
            optimize=True,
        )

        t1 = {
            "ca": t1_tmp[self.hc, self.pa].copy(),
            "cv": t1_tmp[self.hc, self.pv].copy(),
            "av": t1_tmp[self.ha, self.pv].copy(),
        }

        for key in ["ca", "cv", "av"]:
            compute_t1_block(
                t1[key],
                *(self.ints["eps"][_] for _ in key),
                self.flow_param,
            )

        t1_tmp[:, :] = 0.0
        t1_tmp[self.hc, self.pa] = t1["ca"].copy()
        t1_tmp[self.hc, self.pv] = t1["cv"].copy()
        t1_tmp[self.ha, self.pv] = t1["av"].copy()
        return t1_tmp, t2

    def _renormalize_F(self):
        faa = self.fock[self.actv, self.actv]
        f_tmp = self.fock[self.part, self.hole].copy()
        f_tmp[self.pa, self.hc] += 0.5 * np.einsum(
            "ivaw,wu,uv->ai",
            self.T2["S2"]["caaa"],
            faa,
            self.cumulants["gamma1"],
            optimize=True,
        )
        f_tmp[self.pa, self.hc] -= 0.5 * np.einsum(
            "iwau,vw,uv->ai",
            self.T2["S2"]["caaa"],
            faa,
            self.cumulants["gamma1"],
            optimize=True,
        )

        f_tmp[self.pv, self.hc] += 0.5 * np.einsum(
            "vmwe,wu,uv->em",
            self.T2["S2"]["acav"],
            faa,
            self.cumulants["gamma1"],
            optimize=True,
        )
        f_tmp[self.pv, self.hc] -= 0.5 * np.einsum(
            "wmue,vw,uv->em",
            self.T2["S2"]["acav"],
            faa,
            self.cumulants["gamma1"],
            optimize=True,
        )

        f_tmp[self.pv, self.ha] += 0.5 * np.einsum(
            "ivaw,wu,uv->ai",
            self.T2["S2"]["aava"],
            faa,
            self.cumulants["gamma1"],
            optimize=True,
        )
        f_tmp[self.pv, self.ha] -= 0.5 * np.einsum(
            "iwau,vw,uv->ai",
            self.T2["S2"]["aava"],
            faa,
            self.cumulants["gamma1"],
            optimize=True,
        )

        F_tilde = {
            "ac": f_tmp[self.pa, self.hc].copy(),
            "vc": f_tmp[self.pv, self.hc].copy(),
            "va": f_tmp[self.pv, self.ha].copy(),
        }
        renormalize_F(
            F_tilde["ac"],
            self.ints["eps"]["c"],
            self.ints["eps"]["a"],
            self.flow_param,
        )

        renormalize_F(
            F_tilde["vc"],
            self.ints["eps"]["c"],
            self.ints["eps"]["v"],
            self.flow_param,
        )

        renormalize_F(
            F_tilde["va"],
            self.ints["eps"]["a"],
            self.ints["eps"]["v"],
            self.flow_param,
        )

        f_tmp = self.fock[self.part, self.hole].copy()
        f_tmp[self.pa, self.hc] += F_tilde["ac"].copy()
        f_tmp[self.pv, self.hc] += F_tilde["vc"].copy()
        f_tmp[self.pv, self.ha] += F_tilde["va"].copy()
        return f_tmp

    def _renormalize_V_in_place(self):
        for key in ["vvaa", "aacc", "avca", "avac", "vaaa", "aaca"]:
            renormalize_V_block(
                self.ints["V"][key],
                *(self.ints["eps"][_] for _ in key),
                self.flow_param,
            )

    def _compute_pt2_energy(
        self, F, V, T1, T2, gamma1, eta1, lambda2, lambda3, form_hbar=False
    ):
        E = 0.0
        E += 2.0 * np.einsum("am,ma->", F[:, self.hc], T1[self.hc, :], optimize=True)
        E += np.einsum(
            "ev,ue,uv->",
            F[self.pv, self.ha],
            T1[self.ha, self.pv],
            gamma1,
            optimize=True,
        )
        E -= np.einsum(
            "um,mv,uv->",
            F[self.pa, self.hc],
            T1[self.hc, self.pa],
            gamma1,
            optimize=True,
        )
        E += np.einsum(
            "ex,uvey,uvxy->",
            F[self.pv, self.ha],
            T2["T2"]["aava"],
            lambda2,
            optimize=True,
        )
        E -= np.einsum(
            "vm,muyx,uvxy->",
            F[self.pa, self.hc],
            T2["T2"]["caaa"],
            lambda2,
            optimize=True,
        )
        E += np.einsum(
            "evxy,ue,uvxy->",
            V["vaaa"],
            T1[self.ha, self.pv],
            lambda2,
            optimize=True,
        )
        E -= np.einsum(
            "uvmy,mx,uvxy->",
            V["aaca"],
            T1[self.hc, self.pa],
            lambda2,
            optimize=True,
        )
        E += 0.25 * np.einsum(
            "efxu,yvef,uv,xy->",
            V["vvaa"],
            T2["S2"]["aavv"],
            gamma1,
            gamma1,
            optimize=True,
        )
        E += 0.25 * np.einsum(
            "vymn,mnux,uv,xy->",
            V["aacc"],
            T2["S2"]["ccaa"],
            eta1,
            eta1,
            optimize=True,
        )
        E += 0.5 * np.einsum(
            "vemx,myue,uv,xy->",
            V["avca"],
            T2["S2"]["caav"],
            eta1,
            gamma1,
            optimize=True,
        )
        E += 0.5 * np.einsum(
            "vexm,ymue,uv,xy->",
            V["avac"],
            T2["S2"]["acav"],
            eta1,
            gamma1,
            optimize=True,
        )
        E += 0.25 * np.einsum(
            "evwx,zyeu,wz,uv,xy->",
            V["vaaa"],
            T2["S2"]["aava"],
            gamma1,
            eta1,
            gamma1,
            optimize=True,
        )
        E += 0.25 * np.einsum(
            "vzmx,myuw,wz,uv,xy->",
            V["aaca"],
            T2["S2"]["caaa"],
            eta1,
            eta1,
            gamma1,
            optimize=True,
        )
        E += 0.5 * np.einsum(
            "uvmn,mnxy,uvxy->",
            V["aacc"],
            T2["T2"]["ccaa"],
            lambda2,
            optimize=True,
        )
        E += 0.5 * np.einsum(
            "uvmw,mzxy,wz,uvxy->",
            V["aaca"],
            T2["T2"]["caaa"],
            gamma1,
            lambda2,
            optimize=True,
        )
        E += 0.5 * np.einsum(
            "efxy,uvef,uvxy->",
            V["vvaa"],
            T2["T2"]["aavv"],
            lambda2,
            optimize=True,
        )
        E += 0.5 * np.einsum(
            "ezxy,uvew,wz,uvxy->",
            V["vaaa"],
            T2["T2"]["aava"],
            eta1,
            lambda2,
            optimize=True,
        )
        E += np.einsum(
            "uexm,vmye,uvxy->",
            V["avac"],
            T2["S2"]["acav"],
            lambda2,
            optimize=True,
        )
        E -= np.einsum(
            "uemx,vmye,uvxy->",
            V["avca"],
            T2["T2"]["acav"],
            lambda2,
            optimize=True,
        )
        E -= np.einsum(
            "vemx,muye,uvxy->",
            V["avca"],
            T2["T2"]["caav"],
            lambda2,
            optimize=True,
        )
        E += 0.5 * np.einsum(
            "euwx,zvey,wz,uvxy->",
            V["vaaa"],
            T2["S2"]["aava"],
            gamma1,
            lambda2,
            optimize=True,
        )
        E -= 0.5 * np.einsum(
            "euxw,zvey,wz,uvxy->",
            V["vaaa"],
            T2["T2"]["aava"],
            gamma1,
            lambda2,
            optimize=True,
        )
        E -= 0.5 * np.einsum(
            "evxw,uzey,wz,uvxy->",
            V["vaaa"],
            T2["T2"]["aava"],
            gamma1,
            lambda2,
            optimize=True,
        )
        E += 0.5 * np.einsum(
            "wumx,mvzy,wz,uvxy->",
            V["aaca"],
            T2["S2"]["caaa"],
            eta1,
            lambda2,
            optimize=True,
        )
        E -= 0.5 * np.einsum(
            "uwmx,mvzy,wz,uvxy->",
            V["aaca"],
            T2["T2"]["caaa"],
            eta1,
            lambda2,
            optimize=True,
        )
        E -= 0.5 * np.einsum(
            "vwmx,muyz,wz,uvxy->",
            V["aaca"],
            T2["T2"]["caaa"],
            eta1,
            lambda2,
            optimize=True,
        )
        E += np.einsum(
            "ewxy,uvez,xyzuwv->",
            V["vaaa"],
            T2["T2"]["aava"],
            lambda3,
            optimize=True,
        )
        E -= np.einsum(
            "uvmz,mwxy,xyzuwv->",
            V["aaca"],
            T2["T2"]["caaa"],
            lambda3,
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
        cvB = self.ints["B"]["vc"]

        Vbare_m = np.empty((self.ncore, self.nvirt, self.nvirt))
        Vtmp = np.empty((self.ncore, self.nvirt, self.nvirt))
        Vr_m = np.empty((self.ncore, self.nvirt, self.nvirt))

        for m in range(self.ncore):
            np.einsum(
                "eB,nfB->nfe",
                cvB[m, ...],
                cvB,
                optimize=True,
                out=Vr_m,
            )
            np.copyto(Vtmp, Vr_m.swapaxes(1, 2))
            Vbare_m[:] = 2.0 * Vr_m - Vtmp

            renormalize_3index(
                Vr_m,
                self.ints["eps"]["c"][m],
                self.ints["eps"]["c"],
                self.ints["eps"]["v"],
                self.ints["eps"]["v"],
                self.flow_param,
            )

            E += np.einsum("nfe,nfe->", Vr_m, Vbare_m, optimize=True)
        return E

    def _compute_pt2_energy_cavv(self, form_hbar=False):
        E = 0.0
        vaB = self.ints["B"]["va"]
        cvB = self.ints["B"]["vc"]
        Vbare_m = np.empty((self.nvirt, self.nvirt, self.nact))
        Vtmp = np.empty((self.nvirt, self.nvirt, self.nact))
        Vr_m = np.empty((self.nvirt, self.nvirt, self.nact))

        for m in range(self.ncore):
            np.einsum(
                "eB,fvB->efv",
                cvB[m, ...],
                vaB,
                optimize=True,
                out=Vr_m,
            )
            np.copyto(Vtmp, Vr_m.swapaxes(0, 1))
            Vbare_m[:] = 2.0 * Vr_m - Vtmp

            renormalize_3index(
                Vr_m,
                self.ints["eps"]["c"][m],
                -self.ints["eps"]["v"],
                self.ints["eps"]["v"],
                -self.ints["eps"]["a"],
                self.flow_param,
            )

            E += np.einsum(
                "efu,efv,uv->", Vr_m, Vbare_m, self.cumulants["gamma1"], optimize=True
            )

            # temp += np.einsum("efu,efv->uv", J_m, JK_m, optimize=True)

        # if form_hbar:
        #     self.C1_VT2_CAVV = temp.copy()
        # del temp

        return E

    def _compute_pt2_energy_ccav(self, form_hbar=False):
        E = 0.0

        cvB = self.ints["B"]["vc"]
        caB = self.ints["B"]["ac"]

        Vbare_m = np.empty((self.ncore, self.nact, self.nvirt))
        Vtmp = np.empty((self.ncore, self.nact, self.nvirt))
        Vr_m = np.empty((self.ncore, self.nact, self.nvirt))

        for m in range(self.ncore):
            # mneu
            np.einsum(
                "eB,nuB->nue",
                cvB[m, ...],
                caB,
                optimize=True,
                out=Vr_m,
            )
            # mnue
            np.einsum(
                "uB,neB->nue",
                caB[m, ...],
                cvB,
                optimize=True,
                out=Vtmp,
            )
            Vbare_m[:] = 2.0 * Vr_m - Vtmp

            renormalize_3index(
                Vr_m,
                self.ints["eps"]["c"][m],
                self.ints["eps"]["c"],
                self.ints["eps"]["a"],
                self.ints["eps"]["v"],
                self.flow_param,
            )

            E += np.einsum(
                "nue,nve,uv->", Vr_m, Vbare_m, self.cumulants["eta1"], optimize=True
            )
            # temp += np.einsum("eu,ev->uv", J_mn, JK_mn, optimize="optimal")
        # E += np.einsum("uv,uv->", temp, self.cumulants["eta1"], optimize="optimal")

        # if form_hbar:
        #     self.C1_VT2_CCAV = temp.copy()

        # del temp
        return E

    def _compute_Hbar_aaaa(self, F, V, T1, T2, gamma1, eta1):
        _V = np.zeros((self.nact,) * 4)
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
