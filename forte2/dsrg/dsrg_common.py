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
        if l3 is not None:
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


class _DSRGDenseHelper:
    """Spin-adapted commutator kernels acting on dense correlated-space operators.

    `_DSRGHelper` only produces scalars and the active-active block of Hbar, which
    is all DSRG-MRPT2 ever needs. DSRG-MRPT3 additionally has to carry one- and
    two-body commutators over the whole correlated space between its stages, which
    is what these kernels are for.

    Every operator is a dense array over the correlated space: one-body terms are
    ``(ncorr, ncorr)`` and two-body terms are ``(ncorr,) * 4``. Amplitudes are
    stored the same way, zero outside their hole-particle blocks. This trades
    memory for a direct term-by-term correspondence with the reference
    implementation; blocking it is a later optimization.

    Index letters follow that reference: ``m, n`` are core, ``u, v, w, x, y, z``
    active, ``e, f`` virtual, ``i, j, k, l`` hole, ``a, b, c, d`` particle, and
    ``p, q, r, s`` general.
    """

    def __init__(self, dsrg_obj):
        self.c = dsrg_obj.core
        self.a = dsrg_obj.actv
        self.v = dsrg_obj.virt
        self.h = dsrg_obj.hole
        self.p = dsrg_obj.part
        self.g = slice(None)
        self.ncorr = dsrg_obj.ncorr
        # Amplitudes are stored blocked: T1 is (nhole, npart) and T2/S2 are
        # (nhole, nhole, npart, npart), so their indices need slices relative
        # to the hole and particle spaces rather than to the correlated space.
        self.hc = dsrg_obj.hc
        self.ha = dsrg_obj.ha
        self.pa = dsrg_obj.pa
        self.pv = dsrg_obj.pv
        self.A = slice(None)
        self.g1 = None
        self.e1 = None
        self.l2 = None

    def set_cumulants(self, cumulants):
        """Bind the reference densities the kernels contract against.

        Called once per solve, since reference relaxation replaces them.
        """
        self.g1 = cumulants["gamma1"]
        self.e1 = cumulants["eta1"]
        self.l2 = cumulants["lambda2"]

    def make_1body(self):
        return np.zeros((self.ncorr,) * 2)

    def make_2body(self):
        return np.zeros((self.ncorr,) * 4)

    def H1_T1_C1(self, C1, H1, T1, alpha=1.0):
        c, a, v, h, p, g = self.c, self.a, self.v, self.h, self.p, self.g

        C1[h, g] += alpha * np.einsum(
            "ap,ia->ip", H1[p, g], T1[self.A, self.A], optimize=True
        )
        C1[g, p] -= alpha * np.einsum(
            "qi,ia->qa", H1[g, h], T1[self.A, self.A], optimize=True
        )

    def H1_T2_C1(self, C1, H1, T2, alpha=1.0):
        c, a, v, h, p, g = self.c, self.a, self.v, self.h, self.p, self.g

        C1[h, p] += (
            2.0
            * alpha
            * np.einsum(
                "bm,imab->ia",
                H1[p, c],
                T2[self.A, self.hc, self.A, self.A],
                optimize=True,
            )
        )
        C1[h, p] -= alpha * np.einsum(
            "bm,miab->ia", H1[p, c], T2[self.hc, self.A, self.A, self.A], optimize=True
        )

        C1[h, p] += alpha * np.einsum(
            "bu,ivab,uv->ia",
            H1[p, a],
            T2[self.A, self.ha, self.A, self.A],
            self.g1,
            optimize=True,
        )
        C1[h, p] -= (
            0.5
            * alpha
            * np.einsum(
                "bu,viab,uv->ia",
                H1[p, a],
                T2[self.ha, self.A, self.A, self.A],
                self.g1,
                optimize=True,
            )
        )

        C1[h, p] -= alpha * np.einsum(
            "vj,ijau,uv->ia",
            H1[a, h],
            T2[self.A, self.A, self.A, self.pa],
            self.g1,
            optimize=True,
        )
        C1[h, p] += (
            0.5
            * alpha
            * np.einsum(
                "vj,jiau,uv->ia",
                H1[a, h],
                T2[self.A, self.A, self.A, self.pa],
                self.g1,
                optimize=True,
            )
        )

    def H2_T1_C1(self, C1, H2, T1, alpha=1.0):
        c, a, v, h, p, g = self.c, self.a, self.v, self.h, self.p, self.g

        C1 += (
            2.0
            * alpha
            * np.einsum(
                "ma,qapm->qp", T1[self.hc, self.A], H2[g, p, g, c], optimize=True
            )
        )
        C1 -= alpha * np.einsum(
            "ma,aqpm->qp", T1[self.hc, self.A], H2[p, g, g, c], optimize=True
        )

        C1 += alpha * np.einsum(
            "xe,yx,qepy->qp",
            T1[self.ha, self.pv],
            self.g1,
            H2[g, v, g, a],
            optimize=True,
        )
        C1 -= (
            0.5
            * alpha
            * np.einsum(
                "xe,yx,eqpy->qp",
                T1[self.ha, self.pv],
                self.g1,
                H2[v, g, g, a],
                optimize=True,
            )
        )

        C1 -= alpha * np.einsum(
            "mu,uv,qvpm->qp",
            T1[self.hc, self.pa],
            self.g1,
            H2[g, a, g, c],
            optimize=True,
        )
        C1 += (
            0.5
            * alpha
            * np.einsum(
                "mu,uv,vqpm->qp",
                T1[self.hc, self.pa],
                self.g1,
                H2[a, g, g, c],
                optimize=True,
            )
        )

    def H2_T2_C1(self, C1, H2, T2, S2, alpha=1.0):
        c, a, v, h, p, g = self.c, self.a, self.v, self.h, self.p, self.g
        g1, e1, l2 = self.g1, self.e1, self.l2

        # particle contractions -> C1["ir"]
        C1[h, g] += alpha * np.einsum(
            "abrm,imab->ir",
            H2[p, p, g, c],
            S2[self.A, self.hc, self.A, self.A],
            optimize=True,
        )
        C1[h, g] += (
            0.5
            * alpha
            * np.einsum(
                "uv,ivab,abru->ir",
                g1,
                S2[self.A, self.ha, self.A, self.A],
                H2[p, p, g, a],
                optimize=True,
            )
        )
        C1[h, g] += (
            0.25
            * alpha
            * np.einsum(
                "ijux,xy,uv,vyrj->ir",
                S2[self.A, self.A, self.pa, self.pa],
                g1,
                g1,
                H2[a, a, g, h],
                optimize=True,
            )
        )
        C1[h, g] -= (
            0.5
            * alpha
            * np.einsum(
                "uv,imub,vbrm->ir",
                g1,
                S2[self.A, self.hc, self.pa, self.A],
                H2[a, p, g, c],
                optimize=True,
            )
        )
        C1[h, g] -= (
            0.5
            * alpha
            * np.einsum(
                "uv,miub,bvrm->ir",
                g1,
                S2[self.hc, self.A, self.pa, self.A],
                H2[p, a, g, c],
                optimize=True,
            )
        )
        C1[h, g] -= (
            0.25
            * alpha
            * np.einsum(
                "iyub,uv,xy,vbrx->ir",
                S2[self.A, self.ha, self.pa, self.A],
                g1,
                g1,
                H2[a, p, g, a],
                optimize=True,
            )
        )
        C1[h, g] -= (
            0.25
            * alpha
            * np.einsum(
                "iybu,uv,xy,bvrx->ir",
                S2[self.A, self.ha, self.A, self.pa],
                g1,
                g1,
                H2[p, a, g, a],
                optimize=True,
            )
        )

        # C_4 C_2 2:2 -> C1["ir"]
        C1[h, g] += (
            0.5
            * alpha
            * np.einsum(
                "ijxy,xyuv,uvrj->ir",
                T2[self.A, self.A, self.pa, self.pa],
                l2,
                H2[a, a, g, h],
                optimize=True,
            )
        )
        C1[h, g] += (
            0.5
            * alpha
            * np.einsum(
                "aurx,ivay,xyuv->ir",
                H2[p, a, g, a],
                S2[self.A, self.ha, self.A, self.pa],
                l2,
                optimize=True,
            )
        )
        C1[h, g] -= (
            0.5
            * alpha
            * np.einsum(
                "uarx,ivay,xyuv->ir",
                H2[a, p, g, a],
                T2[self.A, self.ha, self.A, self.pa],
                l2,
                optimize=True,
            )
        )
        C1[h, g] -= (
            0.5
            * alpha
            * np.einsum(
                "uarx,ivya,xyvu->ir",
                H2[a, p, g, a],
                T2[self.A, self.ha, self.pa, self.A],
                l2,
                optimize=True,
            )
        )

        # hole contractions -> C1["pa"]
        C1[g, p] -= alpha * np.einsum(
            "peij,ijae->pa",
            H2[g, v, h, h],
            S2[self.A, self.A, self.A, self.pv],
            optimize=True,
        )
        C1[g, p] -= (
            0.5
            * alpha
            * np.einsum(
                "uv,ijau,pvij->pa",
                e1,
                S2[self.A, self.A, self.A, self.pa],
                H2[g, a, h, h],
                optimize=True,
            )
        )
        C1[g, p] -= (
            0.25
            * alpha
            * np.einsum(
                "vyab,uv,xy,pbux->pa",
                S2[self.ha, self.ha, self.A, self.A],
                e1,
                e1,
                H2[g, p, a, a],
                optimize=True,
            )
        )
        C1[g, p] += (
            0.5
            * alpha
            * np.einsum(
                "uv,vjae,peuj->pa",
                e1,
                S2[self.ha, self.A, self.A, self.pv],
                H2[g, v, a, h],
                optimize=True,
            )
        )
        C1[g, p] += (
            0.5
            * alpha
            * np.einsum(
                "uv,jvae,peju->pa",
                e1,
                S2[self.A, self.ha, self.A, self.pv],
                H2[g, v, h, a],
                optimize=True,
            )
        )
        C1[g, p] += (
            0.25
            * alpha
            * np.einsum(
                "vjax,uv,xy,pyuj->pa",
                S2[self.ha, self.A, self.A, self.pa],
                e1,
                e1,
                H2[g, a, a, h],
                optimize=True,
            )
        )
        C1[g, p] += (
            0.25
            * alpha
            * np.einsum(
                "jvax,xy,uv,pyju->pa",
                S2[self.A, self.ha, self.A, self.pa],
                e1,
                e1,
                H2[g, a, h, a],
                optimize=True,
            )
        )

        # C_4 C_2 2:2 -> C1["pa"]
        C1[g, p] -= (
            0.5
            * alpha
            * np.einsum(
                "xyuv,uvab,pbxy->pa",
                l2,
                T2[self.ha, self.ha, self.A, self.A],
                H2[g, p, a, a],
                optimize=True,
            )
        )
        C1[g, p] -= (
            0.5
            * alpha
            * np.einsum(
                "puix,ivay,xyuv->pa",
                H2[g, a, h, a],
                S2[self.A, self.ha, self.A, self.pa],
                l2,
                optimize=True,
            )
        )
        C1[g, p] += (
            0.5
            * alpha
            * np.einsum(
                "puxi,ivay,xyuv->pa",
                H2[g, a, a, h],
                T2[self.A, self.ha, self.A, self.pa],
                l2,
                optimize=True,
            )
        )
        C1[g, p] += (
            0.5
            * alpha
            * np.einsum(
                "puxi,viay,xyvu->pa",
                H2[g, a, a, h],
                T2[self.ha, self.A, self.A, self.pa],
                l2,
                optimize=True,
            )
        )

        # C_4 C_2 1:3 -> C1
        C1[h, p] += (
            0.5
            * alpha
            * np.einsum(
                "avxy,ujab,xyuv->jb",
                H2[p, a, a, a],
                S2[self.ha, self.A, self.A, self.A],
                l2,
                optimize=True,
            )
        )
        C1[h, p] -= (
            0.5
            * alpha
            * np.einsum(
                "uviy,ijxb,xyuv->jb",
                H2[a, a, h, a],
                S2[self.A, self.A, self.pa, self.A],
                l2,
                optimize=True,
            )
        )

        C1 += alpha * np.einsum(
            "eqxs,uvey,xyuv->qs",
            H2[v, g, a, g],
            T2[self.ha, self.ha, self.pv, self.pa],
            l2,
            optimize=True,
        )
        C1 -= (
            0.5
            * alpha
            * np.einsum(
                "eqsx,uvey,xyuv->qs",
                H2[v, g, g, a],
                T2[self.ha, self.ha, self.pv, self.pa],
                l2,
                optimize=True,
            )
        )
        C1 -= alpha * np.einsum(
            "uqms,mvxy,xyuv->qs",
            H2[a, g, c, g],
            T2[self.hc, self.ha, self.pa, self.pa],
            l2,
            optimize=True,
        )
        C1 += (
            0.5
            * alpha
            * np.einsum(
                "uqsm,mvxy,xyuv->qs",
                H2[a, g, g, c],
                T2[self.hc, self.ha, self.pa, self.pa],
                l2,
                optimize=True,
            )
        )

    def H1_T2_C2(self, C2, H1, T2, alpha=1.0):
        c, a, v, h, p, g = self.c, self.a, self.v, self.h, self.p, self.g

        temp = alpha * np.einsum(
            "ijab,ap->ijpb", T2[self.A, self.A, self.A, self.A], H1[p, g], optimize=True
        )
        C2[h, h, g, p] += temp
        C2[h, h, p, g] += np.einsum("ijpb->jibp", temp)

        temp = alpha * np.einsum(
            "ijab,qi->qjab", T2[self.A, self.A, self.A, self.A], H1[g, h], optimize=True
        )
        C2[g, h, p, p] -= temp
        C2[h, g, p, p] -= np.einsum("qjab->jqba", temp)

    def H2_T1_C2(self, C2, H2, T1, alpha=1.0):
        c, a, v, h, p, g = self.c, self.a, self.v, self.h, self.p, self.g

        temp = alpha * np.einsum(
            "ia,arpq->irpq", T1[self.A, self.A], H2[p, g, g, g], optimize=True
        )
        C2[h, g, g, g] += temp
        C2[g, h, g, g] += np.einsum("irpq->riqp", temp)

        temp = alpha * np.einsum(
            "ia,rsiq->rsaq", T1[self.A, self.A], H2[g, g, h, g], optimize=True
        )
        C2[g, g, p, g] -= temp
        C2[g, g, g, p] -= np.einsum("rsaq->srqa", temp)

    def H2_T2_C2(self, C2, H2, T2, S2, alpha=1.0):
        c, a, v, h, p, g = self.c, self.a, self.v, self.h, self.p, self.g
        g1, e1 = self.g1, self.e1

        # particle-particle contractions
        C2[h, h, g, g] += alpha * np.einsum(
            "abrs,ijab->ijrs",
            H2[p, p, g, g],
            T2[self.A, self.A, self.A, self.A],
            optimize=True,
        )
        temp = (
            0.5
            * alpha
            * np.einsum(
                "xy,ijxb,ybrs->ijrs",
                g1,
                T2[self.A, self.A, self.pa, self.A],
                H2[a, p, g, g],
                optimize=True,
            )
        )
        C2[h, h, g, g] -= temp
        C2[h, h, g, g] -= np.einsum("ijrs->jisr", temp)

        # hole-hole contractions
        C2[g, g, p, p] += alpha * np.einsum(
            "pqij,ijab->pqab",
            H2[g, g, h, h],
            T2[self.A, self.A, self.A, self.A],
            optimize=True,
        )
        temp = (
            0.5
            * alpha
            * np.einsum(
                "xy,yjab,pqxj->pqab",
                e1,
                T2[self.ha, self.A, self.A, self.A],
                H2[g, g, a, h],
                optimize=True,
            )
        )
        C2[g, g, p, p] -= temp
        C2[g, g, p, p] -= np.einsum("pqab->qpba", temp)

        # hole-particle contractions
        temp = alpha * np.einsum(
            "aqms,mjab->qjsb",
            H2[p, g, c, g],
            S2[self.hc, self.A, self.A, self.A],
            optimize=True,
        )
        temp -= alpha * np.einsum(
            "aqsm,mjab->qjsb",
            H2[p, g, g, c],
            T2[self.hc, self.A, self.A, self.A],
            optimize=True,
        )
        temp += (
            0.5
            * alpha
            * np.einsum(
                "xy,yjab,aqxs->qjsb",
                g1,
                S2[self.ha, self.A, self.A, self.A],
                H2[p, g, a, g],
                optimize=True,
            )
        )
        temp -= (
            0.5
            * alpha
            * np.einsum(
                "xy,yjab,aqsx->qjsb",
                g1,
                T2[self.ha, self.A, self.A, self.A],
                H2[p, g, g, a],
                optimize=True,
            )
        )
        temp -= (
            0.5
            * alpha
            * np.einsum(
                "xy,ijxb,yqis->qjsb",
                g1,
                S2[self.A, self.A, self.pa, self.A],
                H2[a, g, h, g],
                optimize=True,
            )
        )
        temp += (
            0.5
            * alpha
            * np.einsum(
                "xy,ijxb,yqsi->qjsb",
                g1,
                T2[self.A, self.A, self.pa, self.A],
                H2[a, g, g, h],
                optimize=True,
            )
        )
        C2[g, h, g, p] += temp
        C2[h, g, p, g] += np.einsum("qjsb->jqbs", temp)

        temp = -alpha * np.einsum(
            "aqsm,mjba->jqsb",
            H2[p, g, g, c],
            T2[self.hc, self.A, self.A, self.A],
            optimize=True,
        )
        temp -= (
            0.5
            * alpha
            * np.einsum(
                "xy,yjba,aqsx->jqsb",
                g1,
                T2[self.ha, self.A, self.A, self.A],
                H2[p, g, g, a],
                optimize=True,
            )
        )
        temp += (
            0.5
            * alpha
            * np.einsum(
                "xy,ijbx,yqsi->jqsb",
                g1,
                T2[self.A, self.A, self.A, self.pa],
                H2[a, g, g, h],
                optimize=True,
            )
        )
        C2[h, g, g, p] += temp
        C2[g, h, p, g] += np.einsum("jqsb->qjbs", temp)
