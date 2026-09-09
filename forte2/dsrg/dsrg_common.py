import numpy as np


class _DSRGHelper:
    def __init__(self, dsrg_obj):
        self.hc = dsrg_obj.hc
        self.ha = dsrg_obj.ha
        self.pv = dsrg_obj.pv
        self.pa = dsrg_obj.pa
        self.nact = self.pa.stop - self.pa.start

    def H1_T1_C0(self, h1, t1, g1):
        E = 0.0
        E += 2.0 * np.einsum("am,ma->", h1[:, self.hc], t1[self.hc, :], optimize=True)
        E += np.einsum(
            "ev,ue,uv->",
            h1[self.pv, self.ha],
            t1[self.ha, self.pv],
            g1,
            optimize=True,
        )
        E -= np.einsum(
            "um,mv,uv->",
            h1[self.pa, self.hc],
            t1[self.hc, self.pa],
            g1,
            optimize=True,
        )
        return E

    def H1_T2_C0(self, h1, t2, l2):
        E = 0.0

        E += np.einsum(
            "ex,uvey,uvxy->",
            h1[self.pv, self.ha],
            t2["aava"],
            l2,
            optimize=True,
        )
        E -= np.einsum(
            "vm,muyx,uvxy->",
            h1[self.pa, self.hc],
            t2["caaa"],
            l2,
            optimize=True,
        )
        return E

    def H2_T1_C0(self, h2, t1, l2):
        E = 0.0
        E += np.einsum(
            "evxy,ue,uvxy->",
            h2["vaaa"],
            t1[self.ha, self.pv],
            l2,
            optimize=True,
        )
        E -= np.einsum(
            "uvmy,mx,uvxy->",
            h2["aaca"],
            t1[self.hc, self.pa],
            l2,
            optimize=True,
        )
        return E

    def H2_T2_C0(self, h2, t2, s2, g1, e1, l2, l3):
        E = 0.0

        E += 0.25 * np.einsum(
            "efxu,yvef,uv,xy->",
            h2["vvaa"],
            s2["aavv"],
            g1,
            g1,
            optimize=True,
        )
        E += 0.25 * np.einsum(
            "vymn,mnux,uv,xy->",
            h2["aacc"],
            s2["ccaa"],
            e1,
            e1,
            optimize=True,
        )
        E += 0.5 * np.einsum(
            "vemx,myue,uv,xy->",
            h2["avca"],
            s2["caav"],
            e1,
            g1,
            optimize=True,
        )
        E += 0.5 * np.einsum(
            "vexm,ymue,uv,xy->",
            h2["avac"],
            s2["acav"],
            e1,
            g1,
            optimize=True,
        )
        E += 0.25 * np.einsum(
            "evwx,zyeu,wz,uv,xy->",
            h2["vaaa"],
            s2["aava"],
            g1,
            e1,
            g1,
            optimize=True,
        )
        E += 0.25 * np.einsum(
            "vzmx,myuw,wz,uv,xy->",
            h2["aaca"],
            s2["caaa"],
            e1,
            e1,
            g1,
            optimize=True,
        )
        E += 0.5 * np.einsum(
            "uvmn,mnxy,uvxy->",
            h2["aacc"],
            t2["ccaa"],
            l2,
            optimize=True,
        )
        E += 0.5 * np.einsum(
            "uvmw,mzxy,wz,uvxy->",
            h2["aaca"],
            t2["caaa"],
            g1,
            l2,
            optimize=True,
        )
        E += 0.5 * np.einsum(
            "efxy,uvef,uvxy->",
            h2["vvaa"],
            t2["aavv"],
            l2,
            optimize=True,
        )
        E += 0.5 * np.einsum(
            "ezxy,uvew,wz,uvxy->",
            h2["vaaa"],
            t2["aava"],
            e1,
            l2,
            optimize=True,
        )
        E += np.einsum(
            "uexm,vmye,uvxy->",
            h2["avac"],
            s2["acav"],
            l2,
            optimize=True,
        )
        E -= np.einsum(
            "uemx,vmye,uvxy->",
            h2["avca"],
            t2["acav"],
            l2,
            optimize=True,
        )
        E -= np.einsum(
            "vemx,muye,uvxy->",
            h2["avca"],
            t2["caav"],
            l2,
            optimize=True,
        )
        E += 0.5 * np.einsum(
            "euwx,zvey,wz,uvxy->",
            h2["vaaa"],
            s2["aava"],
            g1,
            l2,
            optimize=True,
        )
        E -= 0.5 * np.einsum(
            "euxw,zvey,wz,uvxy->",
            h2["vaaa"],
            t2["aava"],
            g1,
            l2,
            optimize=True,
        )
        E -= 0.5 * np.einsum(
            "evxw,uzey,wz,uvxy->",
            h2["vaaa"],
            t2["aava"],
            g1,
            l2,
            optimize=True,
        )
        E += 0.5 * np.einsum(
            "wumx,mvzy,wz,uvxy->",
            h2["aaca"],
            s2["caaa"],
            e1,
            l2,
            optimize=True,
        )
        E -= 0.5 * np.einsum(
            "uwmx,mvzy,wz,uvxy->",
            h2["aaca"],
            t2["caaa"],
            e1,
            l2,
            optimize=True,
        )
        E -= 0.5 * np.einsum(
            "vwmx,muyz,wz,uvxy->",
            h2["aaca"],
            t2["caaa"],
            e1,
            l2,
            optimize=True,
        )
        E += np.einsum(
            "ewxy,uvez,xyzuwv->",
            h2["vaaa"],
            t2["aava"],
            l3,
            optimize=True,
        )
        E -= np.einsum(
            "uvmz,mwxy,xyzuwv->",
            h2["aaca"],
            t2["caaa"],
            l3,
            optimize=True,
        )
        return E

    def H2_T2_C0_large(self, h2, s2, g1, e1):
        E = 0.0

        E += np.einsum(
            "efmn,mnef->",
            h2["vvcc"],
            s2["ccvv"],
            optimize="optimal",
        )
        E += np.einsum(
            "feum,vmfe,uv->",
            h2["vvac"],
            s2["acvv"],
            g1,
            optimize="optimal",
        )
        E += np.einsum(
            "evnm,nmeu,uv->",
            h2["vacc"],
            s2["ccva"],
            e1,
            optimize="optimal",
        )
        return E

    def evaluate_H_T_C0(self, t1, t2, h1, h2, cumulants, store_large=False):
        E = 0.0
        E += self.H1_T1_C0(h1, t1, cumulants["gamma1"])
        E += self.H1_T2_C0(h1, t2["T2"], cumulants["lambda2"])
        E += self.H2_T1_C0(h2, t1, cumulants["lambda2"])
        E += self.H2_T2_C0(
            h2,
            t2["T2"],
            t2["S2"],
            cumulants["gamma1"],
            cumulants["eta1"],
            cumulants["lambda2"],
            cumulants["lambda3"],
        )
        if store_large:
            E += self.H2_T2_C0_large(
                h2, t2["S2"], cumulants["gamma1"], cumulants["eta1"]
            )
        return E

    def H_T_C1_active(self, t1, t2, s2, h1, h2, g1, e1, l2, store_large=False):
        C1 = np.zeros((self.nact,) * 2)
        C1 += 1.00 * np.einsum(
            "ev,ue->uv",
            h1[self.pv, self.ha],
            t1[self.ha, self.pv],
            optimize=True,
        )
        C1 -= 1.00 * np.einsum(
            "um,mv->uv",
            h1[self.pa, self.hc],
            t1[self.hc, self.pa],
            optimize=True,
        )
        C1 += 1.00 * np.einsum(
            "em,umve->uv",
            h1[self.pv, self.hc],
            s2["acav"],
            optimize=True,
        )
        C1 += 1.00 * np.einsum(
            "xm,muxv->uv",
            h1[self.pa, self.hc],
            s2["caaa"],
            optimize=True,
        )
        C1 += 0.50 * np.einsum(
            "ex,yuev,xy->uv",
            h1[self.pv, self.ha],
            s2["aava"],
            g1,
            optimize=True,
        )
        C1 -= 0.50 * np.einsum(
            "ym,muxv,xy->uv",
            h1[self.pa, self.hc],
            s2["caaa"],
            g1,
            optimize=True,
        )
        C1 += 1.00 * np.einsum(
            "uemz,mwue->wz",
            h2["avca"],
            s2["caav"],
            optimize=True,
        )
        C1 += 1.00 * np.einsum(
            "uezm,wmue->wz",
            h2["avac"],
            s2["acav"],
            optimize=True,
        )
        C1 += 1.00 * np.einsum(
            "vumz,mwvu->wz",
            h2["aaca"],
            s2["caaa"],
            optimize=True,
        )

        C1 -= 1.00 * np.einsum(
            "wemu,muze->wz",
            h2["avca"],
            s2["caav"],
            optimize=True,
        )
        C1 -= 1.00 * np.einsum(
            "weum,umze->wz",
            h2["avac"],
            s2["acav"],
            optimize=True,
        )
        C1 -= 1.00 * np.einsum(
            "ewvu,vuez->wz",
            h2["vaaa"],
            s2["aava"],
            optimize=True,
        )

        temp = 0.5 * np.einsum(
            "wvef,efzu->wzuv",
            s2["aavv"],
            h2["vvaa"],
            optimize=True,
        )
        temp += 0.5 * np.einsum(
            "wvex,exzu->wzuv",
            s2["aava"],
            h2["vaaa"],
            optimize=True,
        )
        temp += 0.5 * np.einsum(
            "vwex,exuz->wzuv",
            s2["aava"],
            h2["vaaa"],
            optimize=True,
        )

        temp -= 0.5 * np.einsum(
            "wmue,vezm->wzuv",
            s2["acav"],
            h2["avac"],
            optimize=True,
        )
        temp -= 0.5 * np.einsum(
            "mwxu,xvmz->wzuv",
            s2["caaa"],
            h2["aaca"],
            optimize=True,
        )

        temp -= 0.5 * np.einsum(
            "mwue,vemz->wzuv",
            s2["caav"],
            h2["avca"],
            optimize=True,
        )
        temp -= 0.5 * np.einsum(
            "mwux,vxmz->wzuv",
            s2["caaa"],
            h2["aaca"],
            optimize=True,
        )

        temp += 0.25 * np.einsum(
            "jwxu,xy,yvjz->wzuv",
            s2["caaa"],
            g1,
            h2["aaca"],
            optimize=True,
        )
        temp -= 0.25 * np.einsum(
            "ywbu,xy,bvxz->wzuv",
            s2["aava"],
            g1,
            h2["vaaa"],
            optimize=True,
        )
        temp -= 0.25 * np.einsum(
            "wybu,xy,bvzx->wzuv",
            s2["aava"],
            g1,
            h2["vaaa"],
            optimize=True,
        )

        C1 += np.einsum("wzuv,uv->wz", temp, g1, optimize=True)
        temp = np.zeros((self.nact,) * 4)

        temp -= 0.5 * np.einsum(
            "mnzu,wvmn->wzuv",
            s2["ccaa"],
            h2["aacc"],
            optimize=True,
        )
        temp -= 0.5 * np.einsum(
            "mxzu,wvmx->wzuv",
            s2["caaa"],
            h2["aaca"],
            optimize=True,
        )
        temp -= 0.5 * np.einsum(
            "mxuz,vwmx->wzuv",
            s2["caaa"],
            h2["aaca"],
            optimize=True,
        )

        temp += 0.5 * np.einsum(
            "vmze,weum->wzuv",
            s2["acav"],
            h2["avac"],
            optimize=True,
        )
        temp += 0.5 * np.einsum(
            "xvez,ewxu->wzuv",
            s2["aava"],
            h2["vaaa"],
            optimize=True,
        )

        temp += 0.5 * np.einsum(
            "mvze,wemu->wzuv",
            s2["caav"],
            h2["avca"],
            optimize=True,
        )
        temp += 0.5 * np.einsum(
            "vxez,ewux->wzuv",
            s2["aava"],
            h2["vaaa"],
            optimize=True,
        )

        temp -= 0.25 * np.einsum(
            "yvbz,xy,bwxu->wzuv",
            s2["aava"],
            e1,
            h2["vaaa"],
            optimize=True,
        )
        temp += 0.25 * np.einsum(
            "jvxz,xy,ywju->wzuv",
            s2["caaa"],
            e1,
            h2["aaca"],
            optimize=True,
        )
        temp += 0.25 * np.einsum(
            "jvzx,xy,wyju->wzuv",
            s2["caaa"],
            e1,
            h2["aaca"],
            optimize=True,
        )

        C1 += np.einsum("wzuv,uv->wz", temp, e1, optimize=True)

        C1 += 0.50 * np.einsum(
            "vujz,jwyx,xyuv->wz",
            h2["aaca"],
            t2["caaa"],
            l2,
            optimize=True,
        )
        C1 += 0.50 * np.einsum(
            "auzx,wvay,xyuv->wz",
            h2["vaaa"],
            s2["aava"],
            l2,
            optimize=True,
        )
        C1 -= 0.50 * np.einsum(
            "auxz,wvay,xyuv->wz",
            h2["vaaa"],
            t2["aava"],
            l2,
            optimize=True,
        )
        C1 -= 0.50 * np.einsum(
            "auxz,vway,xyvu->wz",
            h2["vaaa"],
            t2["aava"],
            l2,
            optimize=True,
        )

        C1 -= 0.50 * np.einsum(
            "bwyx,vubz,xyuv->wz",
            h2["vaaa"],
            t2["aava"],
            l2,
            optimize=True,
        )
        C1 -= 0.50 * np.einsum(
            "wuix,ivzy,xyuv->wz",
            h2["aaca"],
            s2["caaa"],
            l2,
            optimize=True,
        )
        C1 += 0.50 * np.einsum(
            "uwix,ivzy,xyuv->wz",
            h2["aaca"],
            t2["caaa"],
            l2,
            optimize=True,
        )
        C1 += 0.50 * np.einsum(
            "uwix,ivyz,xyvu->wz",
            h2["aaca"],
            t2["caaa"],
            l2,
            optimize=True,
        )

        C1 += 0.50 * np.einsum(
            "avxy,uwaz,xyuv->wz",
            h2["vaaa"],
            s2["aava"],
            l2,
            optimize=True,
        )
        C1 -= 0.50 * np.einsum(
            "uviy,iwxz,xyuv->wz",
            h2["aaca"],
            s2["caaa"],
            l2,
            optimize=True,
        )
        G2 = dict.fromkeys(["avac", "aaac", "avaa"])
        G2["avac"] = 2.0 * h2["avac"] - np.einsum(
            "uemv->uevm", h2["avca"], optimize=True
        )
        G2["aaac"] = 2.0 * np.einsum(
            "vumw->uvwm", h2["aaca"], optimize=True
        ) - np.einsum("uvmw->uvwm", h2["aaca"], optimize=True)
        G2["avaa"] = 2.0 * np.einsum(
            "euyx->uexy", h2["vaaa"], optimize=True
        ) - np.einsum("euxy->uexy", h2["vaaa"], optimize=True)

        C1 += np.einsum("ma,uavm->uv", t1[self.hc, self.pa], G2["aaac"], optimize=True)
        C1 += np.einsum("ma,uavm->uv", t1[self.hc, self.pv], G2["avac"], optimize=True)
        C1 += 0.50 * np.einsum(
            "xe,yx,uevy->uv",
            t1[self.ha, self.pv],
            g1,
            G2["avaa"],
            optimize=True,
        )
        C1 -= 0.50 * np.einsum(
            "mx,xy,uyvm->uv",
            t1[self.hc, self.pa],
            g1,
            G2["aaac"],
            optimize=True,
        )

        C1 += 0.50 * np.einsum(
            "wezx,uvey,xyuv->wz",
            G2["avaa"],
            t2["aava"],
            l2,
            optimize=True,
        )
        C1 -= 0.50 * np.einsum(
            "wuzm,mvxy,xyuv->wz",
            G2["aaac"],
            t2["caaa"],
            l2,
            optimize=True,
        )

        if store_large:
            C1 += np.einsum(
                "efzm,wmef->wz",
                h2["vvac"],
                s2["acvv"],
                optimize="optimal",
            )
            C1 -= np.einsum(
                "ewnm,nmez->wz",
                h2["vacc"],
                s2["ccva"],
                optimize="optimal",
            )

        return C1

    def H_T_C2_active(self, t1, t2, s2, h1, h2, g1, e1):
        C2 = np.zeros((self.nact,) * 4)
        C2 += np.einsum(
            "efxy,uvef->uvxy",
            h2["vvaa"],
            t2["aavv"],
            optimize=True,
        )
        C2 += np.einsum(
            "ewxy,uvew->uvxy",
            h2["vaaa"],
            t2["aava"],
            optimize=True,
        )
        C2 += np.einsum(
            "ewyx,vuew->uvxy",
            h2["vaaa"],
            t2["aava"],
            optimize=True,
        )

        C2 += np.einsum(
            "uvmn,mnxy->uvxy",
            h2["aacc"],
            t2["ccaa"],
            optimize=True,
        )
        C2 += np.einsum(
            "vumw,mwyx->uvxy",
            h2["aaca"],
            t2["caaa"],
            optimize=True,
        )
        C2 += np.einsum(
            "uvmw,mwxy->uvxy",
            h2["aaca"],
            t2["caaa"],
            optimize=True,
        )

        temp = np.einsum(
            "ax,uvay->uvxy",
            h1[self.pv, self.ha],
            t2["aava"],
            optimize=True,
        )
        temp -= np.einsum(
            "ui,ivxy->uvxy",
            h1[self.pa, self.hc],
            t2["caaa"],
            optimize=True,
        )
        temp += np.einsum(
            "ua,avxy->uvxy",
            t1[self.ha, self.pv],
            h2["vaaa"],
            optimize=True,
        )
        temp -= np.einsum(
            "ix,uviy->uvxy",
            t1[self.hc, self.pa],
            h2["aaca"],
            optimize=True,
        )

        temp -= 0.50 * np.einsum(
            "wz,vuaw,azyx->uvxy",
            g1,
            t2["aava"],
            h2["vaaa"],
            optimize=True,
        )
        temp -= 0.50 * np.einsum(
            "wz,izyx,vuiw->uvxy",
            e1,
            t2["caaa"],
            h2["aaca"],
            optimize=True,
        )

        temp += np.einsum(
            "uexm,vmye->uvxy",
            h2["avac"],
            s2["acav"],
            optimize=True,
        )
        temp += np.einsum(
            "wumx,mvwy->uvxy",
            h2["aaca"],
            s2["caaa"],
            optimize=True,
        )

        temp += 0.50 * np.einsum(
            "wz,zvay,auwx->uvxy",
            g1,
            s2["aava"],
            h2["vaaa"],
            optimize=True,
        )
        temp -= 0.50 * np.einsum(
            "wz,ivwy,zuix->uvxy",
            g1,
            s2["caaa"],
            h2["aaca"],
            optimize=True,
        )

        temp -= np.einsum(
            "uemx,vmye->uvxy",
            h2["avca"],
            t2["acav"],
            optimize=True,
        )
        temp -= np.einsum(
            "uwmx,mvwy->uvxy",
            h2["aaca"],
            t2["caaa"],
            optimize=True,
        )

        temp -= 0.50 * np.einsum(
            "wz,zvay,auxw->uvxy",
            g1,
            t2["aava"],
            h2["vaaa"],
            optimize=True,
        )
        temp += 0.50 * np.einsum(
            "wz,ivwy,uzix->uvxy",
            g1,
            t2["caaa"],
            h2["aaca"],
            optimize=True,
        )

        temp -= np.einsum(
            "vemx,muye->uvxy",
            h2["avca"],
            t2["caav"],
            optimize=True,
        )
        temp -= np.einsum(
            "vwmx,muyw->uvxy",
            h2["aaca"],
            t2["caaa"],
            optimize=True,
        )

        temp -= 0.50 * np.einsum(
            "wz,uzay,avxw->uvxy",
            g1,
            t2["aava"],
            h2["vaaa"],
            optimize=True,
        )
        temp += 0.50 * np.einsum(
            "wz,iuyw,vzix->uvxy",
            g1,
            t2["caaa"],
            h2["aaca"],
            optimize=True,
        )

        C2 += temp
        C2 += np.einsum("uvxy->vuyx", temp, optimize=True)
        return C2
