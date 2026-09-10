import itertools

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
    """Spin-adapted commutator kernels for DSRG-MRPT3.

    `_DSRGHelper` only produces scalars and the active-active block of Hbar,
    which is all DSRG-MRPT2 ever needs. DSRG-MRPT3 additionally has to carry
    one- and two-body commutators between its stages, which is what these are
    for.

    Each term is written once, in the index letters the reference
    implementation uses: ``m, n`` are core, ``u, v, w, x, y, z`` active,
    ``e, f`` virtual, ``i, j, k, l`` hole, ``a, b`` particle and ``p, q, r, s``
    general. Those letters carry enough information to derive every slice, so
    `_emit` builds them rather than having them spelled out per term.

    Results are accumulated into the two off-diagonal directions separately --
    particle-hole and hole-particle for one-body operators, pphh and hhpp for
    two-body ones -- because that is all a commutator of the Hamiltonian with
    an excitation operator can have. Restricting an output index to one of
    those directions narrows the operands feeding it too, which is where the
    saving comes from: a `pphh`-shaped result can never carry more than two
    virtual indices.
    """

    # index letter -> the space it runs over
    _LETTER = {}
    for _l in "mn":
        _LETTER[_l] = "c"
    for _l in "uvwxyz":
        _LETTER[_l] = "a"
    for _l in "ef":
        _LETTER[_l] = "v"
    for _l in "ijkl":
        _LETTER[_l] = "h"
    for _l in "ab":
        _LETTER[_l] = "p"
    for _l in "pqrs":
        _LETTER[_l] = "g"
    del _l

    # intersection of a space with the hole or virtual space: how the integrals
    # are tiled, correlated = hole + virtual being disjoint and contiguous
    _ISECT_HV = {
        ("c", "h"): "c", ("c", "v"): None,
        ("a", "h"): "a", ("a", "v"): None,
        ("v", "h"): None, ("v", "v"): "v",
        ("h", "h"): "h", ("h", "v"): None,
        ("p", "h"): "a", ("p", "v"): "v",
        ("g", "h"): "h", ("g", "v"): "v",
    }

    # intersection of a space with the hole or particle space
    _ISECT = {
        ("c", "h"): "c",
        ("c", "p"): None,
        ("a", "h"): "a",
        ("a", "p"): "a",
        ("v", "h"): None,
        ("v", "p"): "v",
        ("h", "h"): "h",
        ("h", "p"): "a",
        ("p", "h"): "a",
        ("p", "p"): "p",
        ("g", "h"): "h",
        ("g", "p"): "p",
    }

    def __init__(self, dsrg_obj):
        self.ncorr = dsrg_obj.ncorr
        self.nhole = dsrg_obj.nhole
        self.npart = dsrg_obj.npart
        # a space -> the slice addressing it, in each of the three frames
        self._corr = {
            "c": dsrg_obj.core,
            "a": dsrg_obj.actv,
            "v": dsrg_obj.virt,
            "h": dsrg_obj.hole,
            "p": dsrg_obj.part,
            "g": slice(None),
        }
        self._hole = {
            "c": dsrg_obj.hc,
            "a": dsrg_obj.ha,
            "h": slice(None),
            "p": dsrg_obj.ha,
            "g": slice(None),
        }
        self._part = {
            "a": dsrg_obj.pa,
            "v": dsrg_obj.pv,
            "p": slice(None),
            "h": dsrg_obj.pa,
            "g": slice(None),
        }
        self._frames = {"h": self._hole, "p": self._part}
        # inside a block whose index is already virtual there is nothing to slice
        self._hv = {"h": self._hole, "v": {"v": slice(None)}}
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

    # ------------------------------------------------------------------
    # off-diagonal operators, stored as (particle-hole, hole-particle)
    # ------------------------------------------------------------------

    def make_1body(self):
        return (
            np.zeros((self.npart, self.nhole)),
            np.zeros((self.nhole, self.npart)),
        )

    def make_2body(self):
        return (
            np.zeros((self.npart, self.npart, self.nhole, self.nhole)),
            np.zeros((self.nhole, self.nhole, self.npart, self.npart)),
        )

    # ------------------------------------------------------------------
    # term engine
    # ------------------------------------------------------------------

    def _slices(self, kind, idx, eff):
        """Slice one operand, given the space each of its indices runs over."""
        if kind == "act":
            return None
        frames = {
            "corr": (self._corr,) * len(idx),
            "hp": (self._hole, self._part),
            "hhpp": (self._hole, self._hole, self._part, self._part),
        }[kind]
        return tuple(f[eff.get(L, self._LETTER[L])] for L, f in zip(idx, frames))

    @staticmethod
    def _directions(ndim):
        """The two off-diagonal directions of an operator of this rank."""
        if ndim == 2:
            return ("p", "h"), ("h", "p")
        return ("p", "p", "h", "h"), ("h", "h", "p", "p")

    def _emit(self, out, coef, spec, operands, pair=False, only=None):
        """Accumulate one term into both off-diagonal directions of `out`.

        `only` restricts the term to one direction, for the case where the other
        one is handled separately.

        One operand may be stored in pieces rather than over the whole
        correlated space, in which case the contraction splits over them and any
        combination whose spaces do not intersect is skipped rather than
        computed against zeros. Two kinds occur:

        - an off-diagonal operator, held as its particle-hole and hole-particle
          blocks. About a third of the combinations reached in the first stage
          cannot be occupied by a commutator at all.
        - the two-electron integrals, tiled by which indices are virtual.
          Correlated = hole + virtual is a disjoint, contiguous split, so the
          sixteen combinations cover the integrals exactly, and the ones no
          contraction reaches are never built.

        `pair` also adds the result under the bra/ket exchange of the output
        indices, which is how the reference implementation writes the two index
        orders of a two-body commutator.
        """
        ins, res = spec.split("->")
        ins = ins.split(",")
        ndim = len(res)

        split_at, split_kind, blocks = None, None, [None]
        for i, (tensor, _) in enumerate(operands):
            if isinstance(tensor, tuple):
                split_at, split_kind = i, "od"
                blocks = list(enumerate(self._directions(len(ins[i]))))
                break
            if isinstance(tensor, dict):
                split_at, split_kind = i, "vblk"
                blocks = [(m, m) for m in itertools.product("hv", repeat=len(ins[i]))]
                break
        table = self._ISECT if split_kind == "od" else self._ISECT_HV

        targets = list(zip(out, self._directions(ndim)))
        if only is not None:
            targets = [targets[only]]

        for arr, dirs in targets:
            for block in blocks:
                eff = {}
                for k, L in enumerate(res):
                    r = self._ISECT[(self._LETTER[L], dirs[k])]
                    if r is None:
                        break
                    eff[L] = r
                else:
                    if block is not None:
                        for k, L in enumerate(ins[split_at]):
                            r = table[(eff.get(L, self._LETTER[L]), block[1][k])]
                            if r is None:
                                break
                            eff[L] = r
                        else:
                            self._contract(
                                arr, dirs, coef, spec, ins, res, operands, eff,
                                split_at, split_kind, block, pair,
                            )
                        continue
                    self._contract(
                        arr, dirs, coef, spec, ins, res, operands, eff,
                        split_at, split_kind, block, pair,
                    )

    def _contract(self, arr, dirs, coef, spec, ins, res, operands, eff,
                  split_at, split_kind, block, pair):
        args = []
        for i, (idx, (tensor, kind)) in enumerate(zip(ins, operands)):
            if i == split_at:
                bkey, bdirs = block
                if split_kind == "od":
                    sl = tuple(self._frames[d][eff[L]] for L, d in zip(idx, bdirs))
                else:
                    if bkey not in tensor:
                        raise KeyError(
                            f"contraction {spec} reaches integral block {bkey}, "
                            "which is not stored"
                        )
                    sl = tuple(self._hv[d][eff[L]] for L, d in zip(idx, bdirs))
                args.append(tensor[bkey][sl])
            else:
                sl = self._slices(kind, idx, eff)
                args.append(tensor if sl is None else tensor[sl])
        val = coef * np.einsum(spec, *args, optimize=True)
        osl = tuple(self._frames[dirs[k]][eff[L]] for k, L in enumerate(res))
        arr[osl] += val
        if pair:
            perm = (1, 0) if len(res) == 2 else (1, 0, 3, 2)
            arr[tuple(osl[i] for i in perm)] += val.transpose(perm)

    def H1_T1_C1(self, C1, H1, T1, alpha=1.0):
        H, T = (H1, "corr"), (T1, "hp")
        self._emit(C1, alpha, "ap,ia->ip", (H, T))
        self._emit(C1, -alpha, "qi,ia->qa", (H, T))

    def H1_T2_C1(self, C1, H1, T2, alpha=1.0):
        H, T, G = (H1, "corr"), (T2, "hhpp"), (self.g1, "act")
        self._emit(C1, 2.0 * alpha, "bm,imab->ia", (H, T))
        self._emit(C1, -alpha, "bm,miab->ia", (H, T))
        self._emit(C1, alpha, "bu,ivab,uv->ia", (H, T, G))
        self._emit(C1, -0.5 * alpha, "bu,viab,uv->ia", (H, T, G))
        self._emit(C1, -alpha, "vj,ijau,uv->ia", (H, T, G))
        self._emit(C1, 0.5 * alpha, "vj,jiau,uv->ia", (H, T, G))

    def H2_T1_C1(self, C1, H2, T1, alpha=1.0):
        H, T, G = (H2, "corr"), (T1, "hp"), (self.g1, "act")
        self._emit(C1, 2.0 * alpha, "ma,qapm->qp", (T, H))
        self._emit(C1, -alpha, "ma,aqpm->qp", (T, H))
        self._emit(C1, alpha, "xe,yx,qepy->qp", (T, G, H))
        self._emit(C1, -0.5 * alpha, "xe,yx,eqpy->qp", (T, G, H))
        self._emit(C1, -alpha, "mu,uv,qvpm->qp", (T, G, H))
        self._emit(C1, 0.5 * alpha, "mu,uv,vqpm->qp", (T, G, H))

    def H2_T2_C1(self, C1, H2, T2, S2, alpha=1.0):
        H, T, S = (H2, "corr"), (T2, "hhpp"), (S2, "hhpp")
        G, E, L = (self.g1, "act"), (self.e1, "act"), (self.l2, "act")

        # particle contractions
        self._emit(C1, alpha, "abrm,imab->ir", (H, S))
        self._emit(C1, 0.5 * alpha, "uv,ivab,abru->ir", (G, S, H))
        self._emit(C1, 0.25 * alpha, "ijux,xy,uv,vyrj->ir", (S, G, G, H))
        self._emit(C1, -0.5 * alpha, "uv,imub,vbrm->ir", (G, S, H))
        self._emit(C1, -0.5 * alpha, "uv,miub,bvrm->ir", (G, S, H))
        self._emit(C1, -0.25 * alpha, "iyub,uv,xy,vbrx->ir", (S, G, G, H))
        self._emit(C1, -0.25 * alpha, "iybu,uv,xy,bvrx->ir", (S, G, G, H))
        self._emit(C1, 0.5 * alpha, "ijxy,xyuv,uvrj->ir", (T, L, H))
        self._emit(C1, 0.5 * alpha, "aurx,ivay,xyuv->ir", (H, S, L))
        self._emit(C1, -0.5 * alpha, "uarx,ivay,xyuv->ir", (H, T, L))
        self._emit(C1, -0.5 * alpha, "uarx,ivya,xyvu->ir", (H, T, L))

        # hole contractions
        self._emit(C1, -alpha, "peij,ijae->pa", (H, S))
        self._emit(C1, -0.5 * alpha, "uv,ijau,pvij->pa", (E, S, H))
        self._emit(C1, -0.25 * alpha, "vyab,uv,xy,pbux->pa", (S, E, E, H))
        self._emit(C1, 0.5 * alpha, "uv,vjae,peuj->pa", (E, S, H))
        self._emit(C1, 0.5 * alpha, "uv,jvae,peju->pa", (E, S, H))
        self._emit(C1, 0.25 * alpha, "vjax,uv,xy,pyuj->pa", (S, E, E, H))
        self._emit(C1, 0.25 * alpha, "jvax,xy,uv,pyju->pa", (S, E, E, H))
        self._emit(C1, -0.5 * alpha, "xyuv,uvab,pbxy->pa", (L, T, H))
        self._emit(C1, -0.5 * alpha, "puix,ivay,xyuv->pa", (H, S, L))
        self._emit(C1, 0.5 * alpha, "puxi,ivay,xyuv->pa", (H, T, L))
        self._emit(C1, 0.5 * alpha, "puxi,viay,xyvu->pa", (H, T, L))

        # one active index on the amplitude
        self._emit(C1, 0.5 * alpha, "avxy,ujab,xyuv->jb", (H, S, L))
        self._emit(C1, -0.5 * alpha, "uviy,ijxb,xyuv->jb", (H, S, L))
        self._emit(C1, alpha, "eqxs,uvey,xyuv->qs", (H, T, L))
        self._emit(C1, -0.5 * alpha, "eqsx,uvey,xyuv->qs", (H, T, L))
        self._emit(C1, -alpha, "uqms,mvxy,xyuv->qs", (H, T, L))
        self._emit(C1, 0.5 * alpha, "uqsm,mvxy,xyuv->qs", (H, T, L))

    def _stream_ladder(self, C2h, alpha, B, T2):
        """The particle ladder, contracted from the three-index integrals.

        C2[ijrs] += alpha * sum_ab V[abrs] T2[ijab], with every index in the
        particle space and V[abrs] = sum_Q B[Q,a,r] B[Q,b,s]. Walking the
        auxiliary index in chunks keeps the intermediate small, where forming
        V[abrs] outright would be the largest allocation in the calculation.
        """
        npart = B.shape[1]
        # Walk one particle index rather than the auxiliary one: the slice of the
        # integrals needed for a few values of r is three-index-sized, whereas
        # batching over the auxiliary index would carry the amplitudes along with
        # it and grow with the hole space too.
        chunk = max(1, min(npart, 1_000_000 // max(npart**3, 1)))
        for r0 in range(0, npart, chunk):
            Br = B[:, :, r0 : r0 + chunk]
            Vr = np.einsum("Qar,Qbs->arbs", Br, B, optimize=True)
            C2h[:, :, r0 : r0 + chunk, :] += alpha * np.einsum(
                "arbs,ijab->ijrs", Vr, T2, optimize=True
            )

    def H1_T2_C2(self, C2, H1, T2, alpha=1.0):
        H, T = (H1, "corr"), (T2, "hhpp")
        self._emit(C2, alpha, "ijab,ap->ijpb", (T, H), pair=True)
        self._emit(C2, -alpha, "ijab,qi->qjab", (T, H), pair=True)

    def H2_T1_C2(self, C2, H2, T1, alpha=1.0):
        H, T = (H2, "corr"), (T1, "hp")
        self._emit(C2, alpha, "ia,arpq->irpq", (T, H), pair=True)
        self._emit(C2, -alpha, "ia,rsiq->rsaq", (T, H), pair=True)

    def H2_T2_C2(self, C2, H2, T2, S2, alpha=1.0, B=None):
        H, T, S = (H2, "corr"), (T2, "hhpp"), (S2, "hhpp")
        G, E = (self.g1, "act"), (self.e1, "act")

        # particle-particle. In the hole-hole-particle-particle direction this
        # term reaches every four-virtual integral, which is the single largest
        # array the method would otherwise touch, so it is contracted straight
        # from the three-index integrals when they are available.
        if B is None:
            self._emit(C2, alpha, "abrs,ijab->ijrs", (H, T))
        else:
            self._emit(C2, alpha, "abrs,ijab->ijrs", (H, T), only=0)
            self._stream_ladder(C2[1], alpha, B, T2)
        self._emit(C2, -0.5 * alpha, "xy,ijxb,ybrs->ijrs", (G, T, H), pair=True)

        # hole-hole
        self._emit(C2, alpha, "pqij,ijab->pqab", (H, T))
        self._emit(C2, -0.5 * alpha, "xy,yjab,pqxj->pqab", (E, T, H), pair=True)

        # hole-particle
        self._emit(C2, alpha, "aqms,mjab->qjsb", (H, S), pair=True)
        self._emit(C2, -alpha, "aqsm,mjab->qjsb", (H, T), pair=True)
        self._emit(C2, 0.5 * alpha, "xy,yjab,aqxs->qjsb", (G, S, H), pair=True)
        self._emit(C2, -0.5 * alpha, "xy,yjab,aqsx->qjsb", (G, T, H), pair=True)
        self._emit(C2, -0.5 * alpha, "xy,ijxb,yqis->qjsb", (G, S, H), pair=True)
        self._emit(C2, 0.5 * alpha, "xy,ijxb,yqsi->qjsb", (G, T, H), pair=True)

        self._emit(C2, -alpha, "aqsm,mjba->jqsb", (H, T), pair=True)
        self._emit(C2, -0.5 * alpha, "xy,yjba,aqsx->jqsb", (G, T, H), pair=True)
        self._emit(C2, 0.5 * alpha, "xy,ijbx,yqsi->jqsb", (G, T, H), pair=True)
