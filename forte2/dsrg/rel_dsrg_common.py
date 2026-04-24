import numpy as np
from itertools import product


class _RelDSRGHelper:
    def __init__(self, dsrg_obj):
        self.hc = dsrg_obj.hc
        self.ha = dsrg_obj.ha
        self.pv = dsrg_obj.pv
        self.pa = dsrg_obj.pa
        self.ncore = dsrg_obj.ncore
        self.nact = dsrg_obj.nact
        self.nvirt = dsrg_obj.nvirt

        self.hp_1_labels = set(["".join(_) for _ in product(["c", "a"], ["a", "v"])])
        self.hp_1_labels.remove("aa")
        self.ph_1_labels = set(["".join(_) for _ in product(["a", "v"], ["c", "a"])])
        self.ph_1_labels.remove("aa")
        self.od_1_labels = self.hp_1_labels | self.ph_1_labels

        self.hp_2_labels = set(
            ["".join(_) for _ in product(["cc", "ca", "aa"], ["aa", "av", "vv"])]
        )
        self.hp_2_labels.remove("aaaa")
        self.ph_2_labels = set(
            ["".join(_) for _ in product(["aa", "av", "vv"], ["cc", "ca", "aa"])]
        )
        self.ph_2_labels.remove("aaaa")
        self.od_2_labels = self.hp_2_labels | self.ph_2_labels

        self.all_1_labels = set(
            ["".join(_) for _ in product(["c", "a", "v"], repeat=2)]
        )
        self.non_od_1_labels = self.all_1_labels - self.od_1_labels
        self.all_2_labels = set(
            ["".join(_) for _ in product(["c", "a", "v"], repeat=4)]
        )
        large_labels = set(["vvvv", "avvv", "cvvv", "vvav", "vvcv"])
        self.all_2_labels -= large_labels
        self.non_od_2_labels = self.all_2_labels - self.od_2_labels
        self.dims = {
            "c": self.ncore,
            "a": self.nact,
            "v": self.nvirt,
        }

    def evaluate_H_T_C0(self, t1, t2, h1, h2, cumulants, store_large=False):
        E = 0.0
        g1 = cumulants["gamma1"]
        e1 = cumulants["eta1"]
        l2 = cumulants["lambda2"]
        l3 = cumulants["lambda3"]

        E += +1.000 * np.einsum(
            "iu,iv,vu->",
            h1["ca"],
            t1["ca"],
            e1,
            optimize=True,
        )
        E += +1.000 * np.einsum(
            "ia,ia->",
            h1["cv"],
            t1["cv"],
            optimize=True,
        )
        E += +1.000 * np.einsum(
            "ua,va,uv->",
            h1["av"],
            t1["av"],
            g1,
            optimize=True,
        )
        E += -0.500 * np.einsum(
            "iu,ixvw,vwux->",
            h1["ca"],
            t2["caaa"],
            l2,
            optimize=True,
        )
        E += -0.500 * np.einsum(
            "ua,wxva,uvwx->",
            h1["av"],
            t2["aaav"],
            l2,
            optimize=True,
        )
        E += -0.500 * np.einsum(
            "iu,ivwx,uvwx->",
            t1["ca"],
            h2["caaa"],
            l2,
            optimize=True,
        )
        E += -0.500 * np.einsum(
            "ua,vwxa,vwux->",
            t1["av"],
            h2["aaav"],
            l2,
            optimize=True,
        )
        E += +0.250 * np.einsum(
            "ijuv,ijwx,vx,uw->",
            t2["ccaa"],
            h2["ccaa"],
            e1,
            e1,
            optimize=True,
        )
        E += +0.125 * np.einsum(
            "ijuv,ijwx,uvwx->",
            t2["ccaa"],
            h2["ccaa"],
            l2,
            optimize=True,
        )
        E += +0.500 * np.einsum(
            "iwuv,ixyz,vz,uy,xw->",
            t2["caaa"],
            h2["caaa"],
            e1,
            e1,
            g1,
            optimize=True,
        )
        E += +1.000 * np.einsum(
            "iwuv,ixyz,vz,uxwy->",
            t2["caaa"],
            h2["caaa"],
            e1,
            l2,
            optimize=True,
        )
        E += +0.250 * np.einsum(
            "iwuv,ixyz,xw,uvyz->",
            t2["caaa"],
            h2["caaa"],
            g1,
            l2,
            optimize=True,
        )
        E += +0.250 * np.einsum(
            "iwuv,ixyz,uvxwyz->",
            t2["caaa"],
            h2["caaa"],
            l3,
            optimize=True,
        )
        E += +1.000 * np.einsum(
            "ivua,iwxa,ux,wv->",
            t2["caav"],
            h2["caav"],
            e1,
            g1,
            optimize=True,
        )
        E += +1.000 * np.einsum(
            "ivua,iwxa,uwvx->",
            t2["caav"],
            h2["caav"],
            l2,
            optimize=True,
        )
        E += +0.500 * np.einsum(
            "vwua,xyza,uz,yw,xv->",
            t2["aaav"],
            h2["aaav"],
            e1,
            g1,
            g1,
            optimize=True,
        )
        E += +0.250 * np.einsum(
            "vwua,xyza,uz,xyvw->",
            t2["aaav"],
            h2["aaav"],
            e1,
            l2,
            optimize=True,
        )
        E += +1.000 * np.einsum(
            "vwua,xyza,yw,uxvz->",
            t2["aaav"],
            h2["aaav"],
            g1,
            l2,
            optimize=True,
        )
        E += -0.250 * np.einsum(
            "vwua,xyza,uxyvwz->",
            t2["aaav"],
            h2["aaav"],
            l3,
            optimize=True,
        )
        E += +0.250 * np.einsum(
            "uvab,wxab,xv,wu->",
            t2["aavv"],
            h2["aavv"],
            g1,
            g1,
            optimize=True,
        )
        E += +0.125 * np.einsum(
            "uvab,wxab,wxuv->",
            t2["aavv"],
            h2["aavv"],
            l2,
            optimize=True,
        )
        if store_large:
            E += +0.250 * np.einsum(
                "ijab,ijab->", t2["ccvv"], h2["ccvv"], optimize=True
            )
            E += +0.500 * np.einsum(
                "iuab,ivab,vu->",
                t2["cavv"],
                h2["cavv"],
                g1,
                optimize=True,
            )
            E += +0.500 * np.einsum(
                "ijua,ijva,uv->",
                t2["ccav"],
                h2["ccav"],
                e1,
                optimize=True,
            )
        return E

    def make_tensor(self, labels):
        d = dict()
        for label in labels:
            shape = tuple(self.dims[l] for l in label)
            d[label] = np.zeros(shape, dtype=complex)
        return d

    # fmt: off
    @staticmethod
    def H_T_C0(F, V, T1, T2, cumulants, scale=1.0):
        # 24 lines
        gamma1 = cumulants["gamma1"]
        eta1 = cumulants["eta1"]
        lambda2 = cumulants["lambda2"]
        lambda3 = cumulants["lambda3"]
        C0 = .0j
        C0 += scale * +1.000 * np.einsum('ui,iv,vu->', F['ac'], T1['ca'], eta1, optimize=True)
        C0 += scale * -0.500 * np.einsum('ui,ivwx,wxuv->', F['ac'], T2['caaa'], lambda2, optimize=True)
        C0 += scale * +1.000 * np.einsum('ai,ia->', F['vc'], T1['cv'], optimize=True)
        C0 += scale * +1.000 * np.einsum('au,va,uv->', F['va'], T1['av'], gamma1, optimize=True)
        C0 += scale * -0.500 * np.einsum('au,vwxa,uxvw->', F['va'], T2['aaav'], lambda2, optimize=True)
        C0 += scale * -0.500 * np.einsum('iu,vwix,uxvw->', T1['ca'], V['aaca'], lambda2, optimize=True)
        C0 += scale * -0.500 * np.einsum('ua,vawx,wxuv->', T1['av'], V['avaa'], lambda2, optimize=True)
        C0 += scale * +0.250 * np.einsum('ijuv,wxij,vx,uw->', T2['ccaa'], V['aacc'], eta1, eta1, optimize=True)
        C0 += scale * +0.125 * np.einsum('ijuv,wxij,uvwx->', T2['ccaa'], V['aacc'], lambda2, optimize=True)
        C0 += scale * +0.500 * np.einsum('iuvw,xyiz,wy,vx,zu->', T2['caaa'], V['aaca'], eta1, eta1, gamma1, optimize=True)
        C0 += scale * +1.000 * np.einsum('iuvw,xyiz,wy,vzux->', T2['caaa'], V['aaca'], eta1, lambda2, optimize=True)
        C0 += scale * +0.250 * np.einsum('iuvw,xyiz,zu,vwxy->', T2['caaa'], V['aaca'], gamma1, lambda2, optimize=True)
        C0 += scale * +0.250 * np.einsum('iuvw,xyiz,vwzuxy->', T2['caaa'], V['aaca'], lambda3, optimize=True)
        C0 += scale * +0.500 * np.einsum('ijua,vaij,uv->', T2['ccav'], V['avcc'], eta1, optimize=True)
        C0 += scale * +1.000 * np.einsum('iuva,waix,vw,xu->', T2['caav'], V['avca'], eta1, gamma1, optimize=True)
        C0 += scale * +1.000 * np.einsum('iuva,waix,vxuw->', T2['caav'], V['avca'], lambda2, optimize=True)
        C0 += scale * +0.500 * np.einsum('uvwa,xayz,wx,zv,yu->', T2['aaav'], V['avaa'], eta1, gamma1, gamma1, optimize=True)
        C0 += scale * +0.250 * np.einsum('uvwa,xayz,wx,yzuv->', T2['aaav'], V['avaa'], eta1, lambda2, optimize=True)
        C0 += scale * +1.000 * np.einsum('uvwa,xayz,zv,wyux->', T2['aaav'], V['avaa'], gamma1, lambda2, optimize=True)
        C0 += scale * -0.250 * np.einsum('uvwa,xayz,wyzuvx->', T2['aaav'], V['avaa'], lambda3, optimize=True)
        C0 += scale * +0.250 * np.einsum('ijab,abij->', T2['ccvv'], V['vvcc'], optimize=True)
        C0 += scale * +0.500 * np.einsum('iuab,abiv,vu->', T2['cavv'], V['vvca'], gamma1, optimize=True)
        C0 += scale * +0.250 * np.einsum('uvab,abwx,xv,wu->', T2['aavv'], V['vvaa'], gamma1, gamma1, optimize=True)
        C0 += scale * +0.125 * np.einsum('uvab,abwx,wxuv->', T2['aavv'], V['vvaa'], lambda2, optimize=True)

        return C0

    @staticmethod
    def H_T_C1_aa(C1, F, V, T1, T2, cumulants, scale=1.0):
        # 26 lines
        gamma1 = cumulants["gamma1"]
        eta1 = cumulants["eta1"]
        lambda2 = cumulants["lambda2"]
        C1 += scale * -1.000 * np.einsum('ui,iv->uv', F['ac'], T1['ca'], optimize=True)
        C1 += scale * -1.000 * np.einsum('ui,ivwx,xu->vw', F['ac'], T2['caaa'], eta1, optimize=True)
        C1 += scale * -1.000 * np.einsum('ai,iuva->uv', F['vc'], T2['caav'], optimize=True)
        C1 += scale * +1.000 * np.einsum('au,va->vu', F['va'], T1['av'], optimize=True)
        C1 += scale * +1.000 * np.einsum('au,vwxa,uw->vx', F['va'], T2['aaav'], gamma1, optimize=True)
        C1 += scale * -1.000 * np.einsum('iu,vwix,uw->vx', T1['ca'], V['aaca'], eta1, optimize=True)
        C1 += scale * -1.000 * np.einsum('ia,uaiv->uv', T1['cv'], V['avca'], optimize=True)
        C1 += scale * +1.000 * np.einsum('ua,vawx,xu->vw', T1['av'], V['avaa'], gamma1, optimize=True)
        C1 += scale * -0.500 * np.einsum('ijuv,wxij,vx->wu', T2['ccaa'], V['aacc'], eta1, optimize=True)
        C1 += scale * +0.500 * np.einsum('iuvw,xyiz,wzxy->uv', T2['caaa'], V['aaca'], lambda2, optimize=True)
        C1 += scale * -1.000 * np.einsum('iuvw,xyiz,wy,zu->xv', T2['caaa'], V['aaca'], eta1, gamma1, optimize=True)
        C1 += scale * -1.000 * np.einsum('iuvw,xyiz,wzuy->xv', T2['caaa'], V['aaca'], lambda2, optimize=True)
        C1 += scale * -0.500 * np.einsum('ijua,vaij->vu', T2['ccav'], V['avcc'], optimize=True)
        C1 += scale * -1.000 * np.einsum('iuva,waix,xu->wv', T2['caav'], V['avca'], gamma1, optimize=True)
        C1 += scale * -0.500 * np.einsum('uvwa,xayz,yzvx->uw', T2['aaav'], V['avaa'], lambda2, optimize=True)
        C1 += scale * -0.500 * np.einsum('uvwa,xayz,zv,yu->xw', T2['aaav'], V['avaa'], gamma1, gamma1, optimize=True)
        C1 += scale * -0.250 * np.einsum('uvwa,xayz,yzuv->xw', T2['aaav'], V['avaa'], lambda2, optimize=True)
        C1 += scale * +0.500 * np.einsum('iuvw,xyiz,wy,vx->uz', T2['caaa'], V['aaca'], eta1, eta1, optimize=True)
        C1 += scale * +0.250 * np.einsum('iuvw,xyiz,vwxy->uz', T2['caaa'], V['aaca'], lambda2, optimize=True)
        C1 += scale * -0.500 * np.einsum('iuvw,xyiz,vwuy->xz', T2['caaa'], V['aaca'], lambda2, optimize=True)
        C1 += scale * +1.000 * np.einsum('iuva,waix,vw->ux', T2['caav'], V['avca'], eta1, optimize=True)
        C1 += scale * +1.000 * np.einsum('uvwa,xayz,wx,zv->uy', T2['aaav'], V['avaa'], eta1, gamma1, optimize=True)
        C1 += scale * +1.000 * np.einsum('uvwa,xayz,wzvx->uy', T2['aaav'], V['avaa'], lambda2, optimize=True)
        C1 += scale * +0.500 * np.einsum('uvwa,xayz,wzuv->xy', T2['aaav'], V['avaa'], lambda2, optimize=True)
        C1 += scale * +0.500 * np.einsum('iuab,abiv->uv', T2['cavv'], V['vvca'], optimize=True)
        C1 += scale * +0.500 * np.einsum('uvab,abwx,xv->uw', T2['aavv'], V['vvaa'], gamma1, optimize=True)

    @staticmethod
    def H_T_C2_aaaa(C2, F, V, T1, T2, cumulants, scale=1.0):
        # 11 lines
        gamma1 = cumulants["gamma1"]
        eta1 = cumulants["eta1"]
        C2 += scale * -0.500 * np.einsum('ui,ivwx->uvwx', F['ac'], T2['caaa'], optimize=True)
        C2 += scale * -0.500 * np.einsum('au,vwxa->vwux', F['va'], T2['aaav'], optimize=True)
        C2 += scale * -0.500 * np.einsum('iu,vwix->vwux', T1['ca'], V['aaca'], optimize=True)
        C2 += scale * -0.500 * np.einsum('ua,vawx->uvwx', T1['av'], V['avaa'], optimize=True)
        C2 += scale * +0.125 * np.einsum('ijuv,wxij->wxuv', T2['ccaa'], V['aacc'], optimize=True)
        C2 += scale * +0.250 * np.einsum('iuvw,xyiz,zu->xyvw', T2['caaa'], V['aaca'], gamma1, optimize=True)
        C2 += scale * +1.000 * np.einsum('iuvw,xyiz,wy->uxvz', T2['caaa'], V['aaca'], eta1, optimize=True)
        C2 += scale * +1.000 * np.einsum('iuva,waix->uwvx', T2['caav'], V['avca'], optimize=True)
        C2 += scale * +1.000 * np.einsum('uvwa,xayz,zv->uxwy', T2['aaav'], V['avaa'], gamma1, optimize=True)
        C2 += scale * +0.250 * np.einsum('uvwa,xayz,wx->uvyz', T2['aaav'], V['avaa'], eta1, optimize=True)
        C2 += scale * +0.125 * np.einsum('uvab,abwx->uvwx', T2['aavv'], V['vvaa'], optimize=True)

    @staticmethod
    def H1_T1_C1(C1, F, T1, cumulants, scale=1.0):
        # 6 lines
        C1["av"] += scale * -1.000 * np.einsum('ui,ia->ua', F['ac'], T1['cv'], optimize=True)
        C1["cv"] += scale * -1.000 * np.einsum('iu,ua->ia', F['ca'], T1['av'], optimize=True)
        C1["cv"] += scale * +1.000 * np.einsum('ua,iu->ia', F['av'], T1['ca'], optimize=True)
        C1["va"] += scale * -1.000 * np.einsum('ai,iu->au', F['vc'], T1['ca'], optimize=True)
        C1["ac"] += scale * +1.000 * np.einsum('ai,ua->ui', F['vc'], T1['av'], optimize=True)
        C1["ca"] += scale * +1.000 * np.einsum('au,ia->iu', F['va'], T1['cv'], optimize=True)

    @staticmethod
    def H1_T2_C1(C1, F, T2, cumulants, scale=1.0):
        # 9 lines
        gamma1 = cumulants["gamma1"]
        eta1 = cumulants["eta1"]
        C1["cv"] += scale * -1.000 * np.einsum('ui,jiva,vu->ja', F['ac'], T2['ccav'], eta1, optimize=True)
        C1["cv"] += scale * +1.000 * np.einsum('ai,jiba->jb', F['vc'], T2['ccvv'], optimize=True)
        C1["cv"] += scale * +1.000 * np.einsum('au,ivba,uv->ib', F['va'], T2['cavv'], gamma1, optimize=True)
        C1["ca"] += scale * +1.000 * np.einsum('ui,jivw,wu->jv', F['ac'], T2['ccaa'], eta1, optimize=True)
        C1["ca"] += scale * +1.000 * np.einsum('ai,jiua->ju', F['vc'], T2['ccav'], optimize=True)
        C1["ca"] += scale * +1.000 * np.einsum('au,ivwa,uv->iw', F['va'], T2['caav'], gamma1, optimize=True)
        C1["av"] += scale * +1.000 * np.einsum('ui,ivwa,wu->va', F['ac'], T2['caav'], eta1, optimize=True)
        C1["av"] += scale * -1.000 * np.einsum('ai,iuba->ub', F['vc'], T2['cavv'], optimize=True)
        C1["av"] += scale * +1.000 * np.einsum('au,vwba,uw->vb', F['va'], T2['aavv'], gamma1, optimize=True)

    @staticmethod
    def H2_T1_C1(C1, V, T1, cumulants, scale=1.0):
        # 9 lines
        gamma1 = cumulants["gamma1"]
        eta1 = cumulants["eta1"]
        C1["va"] += scale * +1.000 * np.einsum('iu,vaiw,uv->aw', T1['ca'], V['avca'], eta1, optimize=True)
        C1["va"] += scale * -1.000 * np.einsum('ia,baiu->bu', T1['cv'], V['vvca'], optimize=True)
        C1["va"] += scale * +1.000 * np.einsum('ua,bavw,wu->bv', T1['av'], V['vvaa'], gamma1, optimize=True)
        C1["ac"] += scale * +1.000 * np.einsum('iu,vwji,uw->vj', T1['ca'], V['aacc'], eta1, optimize=True)
        C1["ac"] += scale * +1.000 * np.einsum('ia,uaji->uj', T1['cv'], V['avcc'], optimize=True)
        C1["ac"] += scale * +1.000 * np.einsum('ua,vaiw,wu->vi', T1['av'], V['avca'], gamma1, optimize=True)
        C1["vc"] += scale * -1.000 * np.einsum('iu,vaji,uv->aj', T1['ca'], V['avcc'], eta1, optimize=True)
        C1["vc"] += scale * +1.000 * np.einsum('ia,baji->bj', T1['cv'], V['vvcc'], optimize=True)
        C1["vc"] += scale * +1.000 * np.einsum('ua,baiv,vu->bi', T1['av'], V['vvca'], gamma1, optimize=True)

    @staticmethod
    def H2_T2_C1(C1, V, T2, cumulants, scale=1.0):
        # 52 lines
        gamma1 = cumulants["gamma1"]
        eta1 = cumulants["eta1"]
        lambda2 = cumulants["lambda2"]
        C1["va"] += scale * +0.500 * np.einsum('ijuv,waij,vw->au', T2['ccaa'], V['avcc'], eta1, optimize=True)
        C1["va"] += scale * +1.000 * np.einsum('iuvw,xaiy,wx,yu->av', T2['caaa'], V['avca'], eta1, gamma1, optimize=True)
        C1["va"] += scale * +1.000 * np.einsum('iuvw,xaiy,wyux->av', T2['caaa'], V['avca'], lambda2, optimize=True)
        C1["va"] += scale * -0.500 * np.einsum('ijua,baij->bu', T2['ccav'], V['vvcc'], optimize=True)
        C1["va"] += scale * -1.000 * np.einsum('iuva,baiw,wu->bv', T2['caav'], V['vvca'], gamma1, optimize=True)
        C1["va"] += scale * -0.500 * np.einsum('uvwa,baxy,yv,xu->bw', T2['aaav'], V['vvaa'], gamma1, gamma1, optimize=True)
        C1["va"] += scale * -0.250 * np.einsum('uvwa,baxy,xyuv->bw', T2['aaav'], V['vvaa'], lambda2, optimize=True)
        C1["va"] += scale * +0.500 * np.einsum('iuvw,xaiy,vwux->ay', T2['caaa'], V['avca'], lambda2, optimize=True)
        C1["va"] += scale * +0.500 * np.einsum('uvwa,baxy,wyuv->bx', T2['aaav'], V['vvaa'], lambda2, optimize=True)
        C1["cv"] += scale * -0.500 * np.einsum('iuvw,xyza,zu,wy,vx->ia', T2['caaa'], V['aaav'], eta1, gamma1, gamma1, optimize=True)
        C1["cv"] += scale * -0.250 * np.einsum('iuvw,xyza,zu,vwxy->ia', T2['caaa'], V['aaav'], eta1, lambda2, optimize=True)
        C1["cv"] += scale * -0.500 * np.einsum('iuvw,xyza,wy,vx,zu->ia', T2['caaa'], V['aaav'], eta1, eta1, gamma1, optimize=True)
        C1["cv"] += scale * -1.000 * np.einsum('iuvw,xyza,wy,vzux->ia', T2['caaa'], V['aaav'], eta1, lambda2, optimize=True)
        C1["cv"] += scale * -0.250 * np.einsum('iuvw,xyza,zu,vwxy->ia', T2['caaa'], V['aaav'], gamma1, lambda2, optimize=True)
        C1["cv"] += scale * -1.000 * np.einsum('iuvw,xyza,wy,vzux->ia', T2['caaa'], V['aaav'], gamma1, lambda2, optimize=True)
        C1["cv"] += scale * +0.500 * np.einsum('ijua,vwjx,uxvw->ia', T2['ccav'], V['aaca'], lambda2, optimize=True)
        C1["cv"] += scale * +0.500 * np.einsum('uvwa,ixyz,zv,yu,wx->ia', T2['aaav'], V['caaa'], eta1, eta1, gamma1, optimize=True)
        C1["cv"] += scale * +1.000 * np.einsum('uvwa,ixyz,zv,wyux->ia', T2['aaav'], V['caaa'], eta1, lambda2, optimize=True)
        C1["cv"] += scale * +0.500 * np.einsum('uvwa,ixyz,wx,zv,yu->ia', T2['aaav'], V['caaa'], eta1, gamma1, gamma1, optimize=True)
        C1["cv"] += scale * +0.250 * np.einsum('uvwa,ixyz,wx,yzuv->ia', T2['aaav'], V['caaa'], eta1, lambda2, optimize=True)
        C1["cv"] += scale * +1.000 * np.einsum('uvwa,ixyz,zv,wyux->ia', T2['aaav'], V['caaa'], gamma1, lambda2, optimize=True)
        C1["cv"] += scale * +0.250 * np.einsum('uvwa,ixyz,wx,yzuv->ia', T2['aaav'], V['caaa'], gamma1, lambda2, optimize=True)
        C1["cv"] += scale * -0.500 * np.einsum('iuab,vbwx,wxuv->ia', T2['cavv'], V['avaa'], lambda2, optimize=True)
        C1["ca"] += scale * -0.500 * np.einsum('ijuv,wxjy,vywx->iu', T2['ccaa'], V['aaca'], lambda2, optimize=True)
        C1["ca"] += scale * -0.500 * np.einsum('iuva,waxy,xyuw->iv', T2['caav'], V['avaa'], lambda2, optimize=True)
        C1["ca"] += scale * -0.500 * np.einsum('ijuv,wxjy,vx,uw->iy', T2['ccaa'], V['aaca'], eta1, eta1, optimize=True)
        C1["ca"] += scale * -0.250 * np.einsum('ijuv,wxjy,uvwx->iy', T2['ccaa'], V['aaca'], lambda2, optimize=True)
        C1["ca"] += scale * -1.000 * np.einsum('ijua,vajw,uv->iw', T2['ccav'], V['avca'], eta1, optimize=True)
        C1["ca"] += scale * +1.000 * np.einsum('iuva,waxy,vw,yu->ix', T2['caav'], V['avaa'], eta1, gamma1, optimize=True)
        C1["ca"] += scale * +1.000 * np.einsum('iuva,waxy,vyuw->ix', T2['caav'], V['avaa'], lambda2, optimize=True)
        C1["ca"] += scale * -0.500 * np.einsum('ijab,abju->iu', T2['ccvv'], V['vvca'], optimize=True)
        C1["ca"] += scale * +0.500 * np.einsum('iuab,abvw,wu->iv', T2['cavv'], V['vvaa'], gamma1, optimize=True)
        C1["av"] += scale * +0.500 * np.einsum('ijua,vwij,uw->va', T2['ccav'], V['aacc'], eta1, optimize=True)
        C1["av"] += scale * -0.500 * np.einsum('iuva,wxiy,vywx->ua', T2['caav'], V['aaca'], lambda2, optimize=True)
        C1["av"] += scale * +1.000 * np.einsum('iuva,wxiy,vx,yu->wa', T2['caav'], V['aaca'], eta1, gamma1, optimize=True)
        C1["av"] += scale * +1.000 * np.einsum('iuva,wxiy,vyux->wa', T2['caav'], V['aaca'], lambda2, optimize=True)
        C1["av"] += scale * -0.500 * np.einsum('ijab,ubij->ua', T2['ccvv'], V['avcc'], optimize=True)
        C1["av"] += scale * -1.000 * np.einsum('iuab,vbiw,wu->va', T2['cavv'], V['avca'], gamma1, optimize=True)
        C1["av"] += scale * -0.500 * np.einsum('uvab,wbxy,xyvw->ua', T2['aavv'], V['avaa'], lambda2, optimize=True)
        C1["av"] += scale * -0.500 * np.einsum('uvab,wbxy,yv,xu->wa', T2['aavv'], V['avaa'], gamma1, gamma1, optimize=True)
        C1["av"] += scale * -0.250 * np.einsum('uvab,wbxy,xyuv->wa', T2['aavv'], V['avaa'], lambda2, optimize=True)
        C1["ac"] += scale * -0.500 * np.einsum('iuvw,xyji,wy,vx->uj', T2['caaa'], V['aacc'], eta1, eta1, optimize=True)
        C1["ac"] += scale * -0.250 * np.einsum('iuvw,xyji,vwxy->uj', T2['caaa'], V['aacc'], lambda2, optimize=True)
        C1["ac"] += scale * +0.500 * np.einsum('iuvw,xyji,vwuy->xj', T2['caaa'], V['aacc'], lambda2, optimize=True)
        C1["ac"] += scale * -1.000 * np.einsum('iuva,waji,vw->uj', T2['caav'], V['avcc'], eta1, optimize=True)
        C1["ac"] += scale * +1.000 * np.einsum('uvwa,xaiy,wx,yv->ui', T2['aaav'], V['avca'], eta1, gamma1, optimize=True)
        C1["ac"] += scale * +1.000 * np.einsum('uvwa,xaiy,wyvx->ui', T2['aaav'], V['avca'], lambda2, optimize=True)
        C1["ac"] += scale * +0.500 * np.einsum('uvwa,xaiy,wyuv->xi', T2['aaav'], V['avca'], lambda2, optimize=True)
        C1["ac"] += scale * -0.500 * np.einsum('iuab,abji->uj', T2['cavv'], V['vvcc'], optimize=True)
        C1["ac"] += scale * +0.500 * np.einsum('uvab,abiw,wv->ui', T2['aavv'], V['vvca'], gamma1, optimize=True)
        C1["vc"] += scale * -0.500 * np.einsum('iuvw,xaji,vwux->aj', T2['caaa'], V['avcc'], lambda2, optimize=True)
        C1["vc"] += scale * +0.500 * np.einsum('uvwa,baix,wxuv->bi', T2['aaav'], V['vvca'], lambda2, optimize=True)

    @staticmethod
    def H1_T2_C2(C2, F, T2, cumulants, scale=1.0):
        # 22 lines
        C2["ccav"] += scale * -0.500 * np.einsum('au,ijba->ijub', F['va'], T2['ccvv'], optimize=True)
        C2["ccav"] += scale * +1.000 * np.einsum('iu,juva->ijva', F['ca'], T2['caav'], optimize=True)
        C2["ccav"] += scale * +0.500 * np.einsum('ua,ijvu->ijva', F['av'], T2['ccaa'], optimize=True)
        C2["avaa"] += scale * +0.500 * np.einsum('ai,iuvw->uavw', F['vc'], T2['caaa'], optimize=True)
        C2["ccvv"] += scale * +0.500 * np.einsum('iu,juab->ijab', F['ca'], T2['cavv'], optimize=True)
        C2["ccvv"] += scale * +0.500 * np.einsum('ua,ijub->ijab', F['av'], T2['ccav'], optimize=True)
        C2["caav"] += scale * -1.000 * np.einsum('ui,jiva->juva', F['ac'], T2['ccav'], optimize=True)
        C2["caav"] += scale * -1.000 * np.einsum('au,ivba->ivub', F['va'], T2['cavv'], optimize=True)
        C2["caav"] += scale * +1.000 * np.einsum('iu,vuwa->ivwa', F['ca'], T2['aaav'], optimize=True)
        C2["caav"] += scale * +1.000 * np.einsum('ua,ivwu->ivwa', F['av'], T2['caaa'], optimize=True)
        C2["caaa"] += scale * -0.500 * np.einsum('ui,jivw->juvw', F['ac'], T2['ccaa'], optimize=True)
        C2["caaa"] += scale * -1.000 * np.einsum('au,ivwa->ivuw', F['va'], T2['caav'], optimize=True)
        C2["aavv"] += scale * -0.500 * np.einsum('ui,ivab->uvab', F['ac'], T2['cavv'], optimize=True)
        C2["aavv"] += scale * +0.500 * np.einsum('ua,vwub->vwab', F['av'], T2['aaav'], optimize=True)
        C2["cavv"] += scale * -0.500 * np.einsum('ui,jiab->juab', F['ac'], T2['ccvv'], optimize=True)
        C2["cavv"] += scale * +0.500 * np.einsum('iu,vuab->ivab', F['ca'], T2['aavv'], optimize=True)
        C2["cavv"] += scale * +1.000 * np.einsum('ua,ivub->ivab', F['av'], T2['caav'], optimize=True)
        C2["aaca"] += scale * -0.500 * np.einsum('ai,uvwa->uviw', F['vc'], T2['aaav'], optimize=True)
        C2["ccaa"] += scale * -0.500 * np.einsum('au,ijva->ijuv', F['va'], T2['ccav'], optimize=True)
        C2["ccaa"] += scale * +0.500 * np.einsum('iu,juvw->ijvw', F['ca'], T2['caaa'], optimize=True)
        C2["aaav"] += scale * -1.000 * np.einsum('ui,ivwa->uvwa', F['ac'], T2['caav'], optimize=True)
        C2["aaav"] += scale * -0.500 * np.einsum('au,vwba->vwub', F['va'], T2['aavv'], optimize=True)

    @staticmethod
    def H2_T1_C2(C2, V, T1, cumulants, scale=1.0):
        # 22 lines
        C2["vvaa"] += scale * -0.500 * np.einsum('iu,abiv->abuv', T1['ca'], V['vvca'], optimize=True)
        C2["avcc"] += scale * -0.500 * np.einsum('ua,baij->ubij', T1['av'], V['vvcc'], optimize=True)
        C2["vvca"] += scale * -0.500 * np.einsum('iu,abji->abju', T1['ca'], V['vvcc'], optimize=True)
        C2["ccav"] += scale * -1.000 * np.einsum('iu,juva->ijva', T1['ca'], V['caav'], optimize=True)
        C2["ccav"] += scale * -0.500 * np.einsum('ua,ijvu->ijva', T1['av'], V['ccaa'], optimize=True)
        C2["avaa"] += scale * -1.000 * np.einsum('iu,vaiw->vauw', T1['ca'], V['avca'], optimize=True)
        C2["avaa"] += scale * -0.500 * np.einsum('ua,bavw->ubvw', T1['av'], V['vvaa'], optimize=True)
        C2["aacc"] += scale * -0.500 * np.einsum('ua,vaij->uvij', T1['av'], V['avcc'], optimize=True)
        C2["ccvv"] += scale * -0.500 * np.einsum('iu,juab->ijab', T1['ca'], V['cavv'], optimize=True)
        C2["ccvv"] += scale * -0.500 * np.einsum('ua,ijub->ijab', T1['av'], V['ccav'], optimize=True)
        C2["caav"] += scale * -1.000 * np.einsum('iu,vuwa->ivwa', T1['ca'], V['aaav'], optimize=True)
        C2["caav"] += scale * -1.000 * np.einsum('ua,ivwu->ivwa', T1['av'], V['caaa'], optimize=True)
        C2["avca"] += scale * -1.000 * np.einsum('iu,vaji->vaju', T1['ca'], V['avcc'], optimize=True)
        C2["avca"] += scale * -1.000 * np.einsum('ua,baiv->ubiv', T1['av'], V['vvca'], optimize=True)
        C2["caaa"] += scale * -0.500 * np.einsum('ia,uavw->iuvw', T1['cv'], V['avaa'], optimize=True)
        C2["aavv"] += scale * -0.500 * np.einsum('ua,vwub->vwab', T1['av'], V['aaav'], optimize=True)
        C2["cavv"] += scale * -0.500 * np.einsum('iu,vuab->ivab', T1['ca'], V['aavv'], optimize=True)
        C2["cavv"] += scale * -1.000 * np.einsum('ua,ivub->ivab', T1['av'], V['caav'], optimize=True)
        C2["aaca"] += scale * -0.500 * np.einsum('iu,vwji->vwju', T1['ca'], V['aacc'], optimize=True)
        C2["aaca"] += scale * -1.000 * np.einsum('ua,vaiw->uviw', T1['av'], V['avca'], optimize=True)
        C2["ccaa"] += scale * -0.500 * np.einsum('iu,juvw->ijvw', T1['ca'], V['caaa'], optimize=True)
        C2["aaav"] += scale * +0.500 * np.einsum('ia,uviw->uvwa', T1['cv'], V['aaca'], optimize=True)

    @staticmethod
    def H2_T2_C2(C2, V, T2, cumulants, scale=1.0):
        # 68 lines
        gamma1 = cumulants["gamma1"]
        eta1 = cumulants["eta1"]
        C2["aacc"] += scale * +0.250 * np.einsum('uvwa,xaij,wx->uvij', T2['aaav'], V['avcc'], eta1, optimize=True)
        C2["aacc"] += scale * +0.125 * np.einsum('uvab,abij->uvij', T2['aavv'], V['vvcc'], optimize=True)
        C2["ccvv"] += scale * +0.125 * np.einsum('ijuv,wxab,vx,uw->ijab', T2['ccaa'], V['aavv'], eta1, eta1, optimize=True)
        C2["ccvv"] += scale * -0.125 * np.einsum('ijuv,wxab,vx,uw->ijab', T2['ccaa'], V['aavv'], gamma1, gamma1, optimize=True)
        C2["ccvv"] += scale * -1.000 * np.einsum('iuva,jwxb,xu,vw->ijab', T2['caav'], V['caav'], eta1, gamma1, optimize=True)
        C2["ccvv"] += scale * +1.000 * np.einsum('iuva,jwxb,vw,xu->ijab', T2['caav'], V['caav'], eta1, gamma1, optimize=True)
        C2["ccvv"] += scale * -0.125 * np.einsum('uvab,ijwx,xv,wu->ijab', T2['aavv'], V['ccaa'], eta1, eta1, optimize=True)
        C2["ccvv"] += scale * +0.125 * np.einsum('uvab,ijwx,xv,wu->ijab', T2['aavv'], V['ccaa'], gamma1, gamma1, optimize=True)
        C2["ccav"] += scale * +1.000 * np.einsum('iuvw,jxya,yu,wx->ijva', T2['caaa'], V['caav'], eta1, gamma1, optimize=True)
        C2["ccav"] += scale * -1.000 * np.einsum('iuvw,jxya,wx,yu->ijva', T2['caaa'], V['caav'], eta1, gamma1, optimize=True)
        C2["ccav"] += scale * -0.250 * np.einsum('uvwa,ijxy,yv,xu->ijwa', T2['aaav'], V['ccaa'], eta1, eta1, optimize=True)
        C2["ccav"] += scale * +0.250 * np.einsum('uvwa,ijxy,yv,xu->ijwa', T2['aaav'], V['ccaa'], gamma1, gamma1, optimize=True)
        C2["ccav"] += scale * +0.250 * np.einsum('ijuv,wxya,vx,uw->ijya', T2['ccaa'], V['aaav'], eta1, eta1, optimize=True)
        C2["ccav"] += scale * -0.250 * np.einsum('ijuv,wxya,vx,uw->ijya', T2['ccaa'], V['aaav'], gamma1, gamma1, optimize=True)
        C2["ccav"] += scale * -1.000 * np.einsum('iuva,jwxy,yu,vw->ijxa', T2['caav'], V['caaa'], eta1, gamma1, optimize=True)
        C2["ccav"] += scale * +1.000 * np.einsum('iuva,jwxy,vw,yu->ijxa', T2['caav'], V['caaa'], eta1, gamma1, optimize=True)
        C2["avaa"] += scale * +0.250 * np.einsum('ijuv,waij->wauv', T2['ccaa'], V['avcc'], optimize=True)
        C2["avaa"] += scale * +0.500 * np.einsum('iuvw,xaiy,yu->xavw', T2['caaa'], V['avca'], gamma1, optimize=True)
        C2["avaa"] += scale * -1.000 * np.einsum('iuvw,xaiy,wx->uavy', T2['caaa'], V['avca'], eta1, optimize=True)
        C2["avaa"] += scale * +1.000 * np.einsum('iuva,baiw->ubvw', T2['caav'], V['vvca'], optimize=True)
        C2["avaa"] += scale * +1.000 * np.einsum('uvwa,baxy,yv->ubwx', T2['aaav'], V['vvaa'], gamma1, optimize=True)
        C2["ccaa"] += scale * -1.000 * np.einsum('iuvw,jxyz,zu,wx->ijvy', T2['caaa'], V['caaa'], eta1, gamma1, optimize=True)
        C2["ccaa"] += scale * +1.000 * np.einsum('iuvw,jxyz,wx,zu->ijvy', T2['caaa'], V['caaa'], eta1, gamma1, optimize=True)
        C2["ccaa"] += scale * +0.250 * np.einsum('ijua,vawx,uv->ijwx', T2['ccav'], V['avaa'], eta1, optimize=True)
        C2["ccaa"] += scale * +0.125 * np.einsum('ijab,abuv->ijuv', T2['ccvv'], V['vvaa'], optimize=True)
        C2["aaca"] += scale * +1.000 * np.einsum('iuvw,xyji,wy->uxjv', T2['caaa'], V['aacc'], eta1, optimize=True)
        C2["aaca"] += scale * +1.000 * np.einsum('iuva,waji->uwjv', T2['caav'], V['avcc'], optimize=True)
        C2["aaca"] += scale * -1.000 * np.einsum('uvwa,xaiy,yv->uxiw', T2['aaav'], V['avca'], gamma1, optimize=True)
        C2["aaca"] += scale * +0.500 * np.einsum('uvwa,xaiy,wx->uviy', T2['aaav'], V['avca'], eta1, optimize=True)
        C2["aaca"] += scale * +0.250 * np.einsum('uvab,abiw->uviw', T2['aavv'], V['vvca'], optimize=True)
        C2["caav"] += scale * +1.000 * np.einsum('iuvw,xyza,zu,wy->ixva', T2['caaa'], V['aaav'], eta1, gamma1, optimize=True)
        C2["caav"] += scale * -1.000 * np.einsum('iuvw,xyza,wy,zu->ixva', T2['caaa'], V['aaav'], eta1, gamma1, optimize=True)
        C2["caav"] += scale * -0.500 * np.einsum('uvwa,ixyz,zv,yu->ixwa', T2['aaav'], V['caaa'], eta1, eta1, optimize=True)
        C2["caav"] += scale * +0.500 * np.einsum('uvwa,ixyz,zv,yu->ixwa', T2['aaav'], V['caaa'], gamma1, gamma1, optimize=True)
        C2["caav"] += scale * +0.500 * np.einsum('iuvw,xyza,wy,vx->iuza', T2['caaa'], V['aaav'], eta1, eta1, optimize=True)
        C2["caav"] += scale * -0.500 * np.einsum('iuvw,xyza,wy,vx->iuza', T2['caaa'], V['aaav'], gamma1, gamma1, optimize=True)
        C2["caav"] += scale * -1.000 * np.einsum('ijua,vwjx,uw->ivxa', T2['ccav'], V['aaca'], eta1, optimize=True)
        C2["caav"] += scale * +1.000 * np.einsum('uvwa,ixyz,zv,wx->iuya', T2['aaav'], V['caaa'], eta1, gamma1, optimize=True)
        C2["caav"] += scale * -1.000 * np.einsum('uvwa,ixyz,wx,zv->iuya', T2['aaav'], V['caaa'], eta1, gamma1, optimize=True)
        C2["caav"] += scale * +1.000 * np.einsum('ijab,ubjv->iuva', T2['ccvv'], V['avca'], optimize=True)
        C2["caav"] += scale * -1.000 * np.einsum('iuab,vbwx,xu->ivwa', T2['cavv'], V['avaa'], gamma1, optimize=True)
        C2["avca"] += scale * -1.000 * np.einsum('iuvw,xaji,wx->uajv', T2['caaa'], V['avcc'], eta1, optimize=True)
        C2["avca"] += scale * +1.000 * np.einsum('iuva,baji->ubjv', T2['caav'], V['vvcc'], optimize=True)
        C2["avca"] += scale * -1.000 * np.einsum('uvwa,baix,xv->ubiw', T2['aaav'], V['vvca'], gamma1, optimize=True)
        C2["aavv"] += scale * -1.000 * np.einsum('uvwa,xyzb,zv,wy->uxab', T2['aaav'], V['aaav'], eta1, gamma1, optimize=True)
        C2["aavv"] += scale * +1.000 * np.einsum('uvwa,xyzb,wy,zv->uxab', T2['aaav'], V['aaav'], eta1, gamma1, optimize=True)
        C2["aavv"] += scale * +0.125 * np.einsum('ijab,uvij->uvab', T2['ccvv'], V['aacc'], optimize=True)
        C2["aavv"] += scale * +0.250 * np.einsum('iuab,vwix,xu->vwab', T2['cavv'], V['aaca'], gamma1, optimize=True)
        C2["vvaa"] += scale * +0.125 * np.einsum('ijuv,abij->abuv', T2['ccaa'], V['vvcc'], optimize=True)
        C2["vvaa"] += scale * +0.250 * np.einsum('iuvw,abix,xu->abvw', T2['caaa'], V['vvca'], gamma1, optimize=True)
        C2["caaa"] += scale * -1.000 * np.einsum('ijuv,wxjy,vx->iwuy', T2['ccaa'], V['aaca'], eta1, optimize=True)
        C2["caaa"] += scale * -1.000 * np.einsum('ijua,vajw->ivuw', T2['ccav'], V['avca'], optimize=True)
        C2["caaa"] += scale * +1.000 * np.einsum('iuva,waxy,yu->iwvx', T2['caav'], V['avaa'], gamma1, optimize=True)
        C2["caaa"] += scale * +0.500 * np.einsum('iuva,waxy,vw->iuxy', T2['caav'], V['avaa'], eta1, optimize=True)
        C2["caaa"] += scale * +0.250 * np.einsum('iuab,abvw->iuvw', T2['cavv'], V['vvaa'], optimize=True)
        C2["aaav"] += scale * +0.250 * np.einsum('ijua,vwij->vwua', T2['ccav'], V['aacc'], optimize=True)
        C2["aaav"] += scale * +0.500 * np.einsum('iuva,wxiy,yu->wxva', T2['caav'], V['aaca'], gamma1, optimize=True)
        C2["aaav"] += scale * +1.000 * np.einsum('iuva,wxiy,vx->uwya', T2['caav'], V['aaca'], eta1, optimize=True)
        C2["aaav"] += scale * -1.000 * np.einsum('iuab,vbiw->uvwa', T2['cavv'], V['avca'], optimize=True)
        C2["aaav"] += scale * -1.000 * np.einsum('uvab,wbxy,yv->uwxa', T2['aavv'], V['avaa'], gamma1, optimize=True)
        C2["cavv"] += scale * +0.250 * np.einsum('iuvw,xyab,wy,vx->iuab', T2['caaa'], V['aavv'], eta1, eta1, optimize=True)
        C2["cavv"] += scale * -0.250 * np.einsum('iuvw,xyab,wy,vx->iuab', T2['caaa'], V['aavv'], gamma1, gamma1, optimize=True)
        C2["cavv"] += scale * -1.000 * np.einsum('iuva,wxyb,yu,vx->iwab', T2['caav'], V['aaav'], eta1, gamma1, optimize=True)
        C2["cavv"] += scale * +1.000 * np.einsum('iuva,wxyb,vx,yu->iwab', T2['caav'], V['aaav'], eta1, gamma1, optimize=True)
        C2["cavv"] += scale * +1.000 * np.einsum('uvwa,ixyb,yv,wx->iuab', T2['aaav'], V['caav'], eta1, gamma1, optimize=True)
        C2["cavv"] += scale * -1.000 * np.einsum('uvwa,ixyb,wx,yv->iuab', T2['aaav'], V['caav'], eta1, gamma1, optimize=True)
        C2["cavv"] += scale * -0.250 * np.einsum('uvab,iwxy,yv,xu->iwab', T2['aavv'], V['caaa'], eta1, eta1, optimize=True)
        C2["cavv"] += scale * +0.250 * np.einsum('uvab,iwxy,yv,xu->iwab', T2['aavv'], V['caaa'], gamma1, gamma1, optimize=True)

    @staticmethod
    def H1_T1_C1_non_od(C1, F, T1, cumulants, scale=1.0):
        # 6 lines
        C1["av"] += scale * -1.000 * np.einsum('uv,va->ua', F['aa'], T1['av'], optimize=True)
        C1["av"] += scale * +1.000 * np.einsum('ab,ua->ub', F['vv'], T1['av'], optimize=True)
        C1["cv"] += scale * -1.000 * np.einsum('ij,ja->ia', F['cc'], T1['cv'], optimize=True)
        C1["cv"] += scale * +1.000 * np.einsum('ab,ia->ib', F['vv'], T1['cv'], optimize=True)
        C1["ca"] += scale * -1.000 * np.einsum('ij,ju->iu', F['cc'], T1['ca'], optimize=True)
        C1["ca"] += scale * +1.000 * np.einsum('uv,iu->iv', F['aa'], T1['ca'], optimize=True)

    @staticmethod
    def H2_T1_C1_non_od(C1, V, T1, cumulants, scale=1.0):
        # 9 lines
        gamma1 = cumulants["gamma1"]
        eta1 = cumulants["eta1"]
        C1["cv"] += scale * -1.000 * np.einsum('iu,jvia,uv->ja', T1['ca'], V['cacv'], eta1, optimize=True)
        C1["cv"] += scale * -1.000 * np.einsum('ia,jaib->jb', T1['cv'], V['cvcv'], optimize=True)
        C1["cv"] += scale * -1.000 * np.einsum('ua,iavb,vu->ib', T1['av'], V['cvav'], gamma1, optimize=True)
        C1["ca"] += scale * -1.000 * np.einsum('iu,jviw,uv->jw', T1['ca'], V['caca'], eta1, optimize=True)
        C1["ca"] += scale * -1.000 * np.einsum('ia,jaiu->ju', T1['cv'], V['cvca'], optimize=True)
        C1["ca"] += scale * +1.000 * np.einsum('ua,iavw,wu->iv', T1['av'], V['cvaa'], gamma1, optimize=True)
        C1["av"] += scale * -1.000 * np.einsum('iu,vwia,uw->va', T1['ca'], V['aacv'], eta1, optimize=True)
        C1["av"] += scale * -1.000 * np.einsum('ia,uaib->ub', T1['cv'], V['avcv'], optimize=True)
        C1["av"] += scale * -1.000 * np.einsum('ua,vawb,wu->vb', T1['av'], V['avav'], gamma1, optimize=True)

    @staticmethod
    def H2_T2_C1_non_od(C1, V, T2, cumulants, scale=1.0):
        # 58 lines
        gamma1 = cumulants["gamma1"]
        eta1 = cumulants["eta1"]
        lambda2 = cumulants["lambda2"]
        C1["cv"] += scale * -0.500 * np.einsum('ijuv,wxja,vx,uw->ia', T2['ccaa'], V['aacv'], eta1, eta1, optimize=True)
        C1["cv"] += scale * -0.250 * np.einsum('ijuv,wxja,uvwx->ia', T2['ccaa'], V['aacv'], lambda2, optimize=True)
        C1["cv"] += scale * -0.500 * np.einsum('iuvw,jxia,vwux->ja', T2['caaa'], V['cacv'], lambda2, optimize=True)
        C1["cv"] += scale * -0.500 * np.einsum('iuva,wxyz,zu,vywx->ia', T2['caav'], V['aaaa'], eta1, lambda2, optimize=True)
        C1["cv"] += scale * +0.500 * np.einsum('iuva,wxyz,vx,yzuw->ia', T2['caav'], V['aaaa'], eta1, lambda2, optimize=True)
        C1["cv"] += scale * -0.500 * np.einsum('iuva,wxyz,zu,vywx->ia', T2['caav'], V['aaaa'], gamma1, lambda2, optimize=True)
        C1["cv"] += scale * -0.500 * np.einsum('iuva,wxyz,vw,yzux->ia', T2['caav'], V['aaaa'], gamma1, lambda2, optimize=True)
        C1["cv"] += scale * +0.500 * np.einsum('ijua,kvij,uv->ka', T2['ccav'], V['cacc'], eta1, optimize=True)
        C1["cv"] += scale * +1.000 * np.einsum('iuva,jwix,vw,xu->ja', T2['caav'], V['caca'], eta1, gamma1, optimize=True)
        C1["cv"] += scale * +1.000 * np.einsum('iuva,jwix,vxuw->ja', T2['caav'], V['caca'], lambda2, optimize=True)
        C1["cv"] += scale * -1.000 * np.einsum('ijua,vajb,uv->ib', T2['ccav'], V['avcv'], eta1, optimize=True)
        C1["cv"] += scale * -1.000 * np.einsum('iuva,waxb,vw,xu->ib', T2['caav'], V['avav'], eta1, gamma1, optimize=True)
        C1["cv"] += scale * -1.000 * np.einsum('iuva,waxb,vxuw->ib', T2['caav'], V['avav'], lambda2, optimize=True)
        C1["cv"] += scale * -0.500 * np.einsum('uvwa,iaxb,wxuv->ib', T2['aaav'], V['cvav'], lambda2, optimize=True)
        C1["cv"] += scale * -0.500 * np.einsum('ijab,kbij->ka', T2['ccvv'], V['cvcc'], optimize=True)
        C1["cv"] += scale * -1.000 * np.einsum('iuab,jbiv,vu->ja', T2['cavv'], V['cvca'], gamma1, optimize=True)
        C1["cv"] += scale * -0.500 * np.einsum('uvab,ibwx,xv,wu->ia', T2['aavv'], V['cvaa'], gamma1, gamma1, optimize=True)
        C1["cv"] += scale * -0.250 * np.einsum('uvab,ibwx,wxuv->ia', T2['aavv'], V['cvaa'], lambda2, optimize=True)
        C1["ca"] += scale * +0.500 * np.einsum('iuvw,xyzr,ru,wzxy->iv', T2['caaa'], V['aaaa'], eta1, lambda2, optimize=True)
        C1["ca"] += scale * -0.500 * np.einsum('iuvw,xyzr,wy,zrux->iv', T2['caaa'], V['aaaa'], eta1, lambda2, optimize=True)
        C1["ca"] += scale * +0.500 * np.einsum('iuvw,xyzr,ru,wzxy->iv', T2['caaa'], V['aaaa'], gamma1, lambda2, optimize=True)
        C1["ca"] += scale * +0.500 * np.einsum('iuvw,xyzr,wx,zruy->iv', T2['caaa'], V['aaaa'], gamma1, lambda2, optimize=True)
        C1["ca"] += scale * -0.500 * np.einsum('ijuv,kwij,vw->ku', T2['ccaa'], V['cacc'], eta1, optimize=True)
        C1["ca"] += scale * -1.000 * np.einsum('iuvw,jxiy,wx,yu->jv', T2['caaa'], V['caca'], eta1, gamma1, optimize=True)
        C1["ca"] += scale * -1.000 * np.einsum('iuvw,jxiy,wyux->jv', T2['caaa'], V['caca'], lambda2, optimize=True)
        C1["ca"] += scale * -0.500 * np.einsum('ijua,kaij->ku', T2['ccav'], V['cvcc'], optimize=True)
        C1["ca"] += scale * -1.000 * np.einsum('iuva,jaiw,wu->jv', T2['caav'], V['cvca'], gamma1, optimize=True)
        C1["ca"] += scale * -0.500 * np.einsum('uvwa,iaxy,yv,xu->iw', T2['aaav'], V['cvaa'], gamma1, gamma1, optimize=True)
        C1["ca"] += scale * -0.250 * np.einsum('uvwa,iaxy,xyuv->iw', T2['aaav'], V['cvaa'], lambda2, optimize=True)
        C1["ca"] += scale * +0.500 * np.einsum('iuvw,xyzr,ru,wy,vx->iz', T2['caaa'], V['aaaa'], eta1, gamma1, gamma1, optimize=True)
        C1["ca"] += scale * +0.250 * np.einsum('iuvw,xyzr,ru,vwxy->iz', T2['caaa'], V['aaaa'], eta1, lambda2, optimize=True)
        C1["ca"] += scale * +0.500 * np.einsum('iuvw,xyzr,wy,vx,ru->iz', T2['caaa'], V['aaaa'], eta1, eta1, gamma1, optimize=True)
        C1["ca"] += scale * +1.000 * np.einsum('iuvw,xyzr,wy,vrux->iz', T2['caaa'], V['aaaa'], eta1, lambda2, optimize=True)
        C1["ca"] += scale * +0.250 * np.einsum('iuvw,xyzr,ru,vwxy->iz', T2['caaa'], V['aaaa'], gamma1, lambda2, optimize=True)
        C1["ca"] += scale * +1.000 * np.einsum('iuvw,xyzr,wy,vrux->iz', T2['caaa'], V['aaaa'], gamma1, lambda2, optimize=True)
        C1["ca"] += scale * -0.500 * np.einsum('iuvw,jxiy,vwux->jy', T2['caaa'], V['caca'], lambda2, optimize=True)
        C1["ca"] += scale * +0.500 * np.einsum('uvwa,iaxy,wyuv->ix', T2['aaav'], V['cvaa'], lambda2, optimize=True)
        C1["av"] += scale * +0.500 * np.einsum('iuvw,xyia,wy,vx->ua', T2['caaa'], V['aacv'], eta1, eta1, optimize=True)
        C1["av"] += scale * +0.250 * np.einsum('iuvw,xyia,vwxy->ua', T2['caaa'], V['aacv'], lambda2, optimize=True)
        C1["av"] += scale * -0.500 * np.einsum('iuvw,xyia,vwuy->xa', T2['caaa'], V['aacv'], lambda2, optimize=True)
        C1["av"] += scale * -0.500 * np.einsum('uvwa,xyzr,rv,wzxy->ua', T2['aaav'], V['aaaa'], eta1, lambda2, optimize=True)
        C1["av"] += scale * +0.500 * np.einsum('uvwa,xyzr,wy,zrvx->ua', T2['aaav'], V['aaaa'], eta1, lambda2, optimize=True)
        C1["av"] += scale * -0.500 * np.einsum('uvwa,xyzr,rv,wzxy->ua', T2['aaav'], V['aaaa'], gamma1, lambda2, optimize=True)
        C1["av"] += scale * -0.500 * np.einsum('uvwa,xyzr,wx,zrvy->ua', T2['aaav'], V['aaaa'], gamma1, lambda2, optimize=True)
        C1["av"] += scale * +0.500 * np.einsum('uvwa,xyzr,rv,zu,wy->xa', T2['aaav'], V['aaaa'], eta1, eta1, gamma1, optimize=True)
        C1["av"] += scale * +1.000 * np.einsum('uvwa,xyzr,rv,wzuy->xa', T2['aaav'], V['aaaa'], eta1, lambda2, optimize=True)
        C1["av"] += scale * +0.500 * np.einsum('uvwa,xyzr,wy,rv,zu->xa', T2['aaav'], V['aaaa'], eta1, gamma1, gamma1, optimize=True)
        C1["av"] += scale * +0.250 * np.einsum('uvwa,xyzr,wy,zruv->xa', T2['aaav'], V['aaaa'], eta1, lambda2, optimize=True)
        C1["av"] += scale * +1.000 * np.einsum('uvwa,xyzr,rv,wzuy->xa', T2['aaav'], V['aaaa'], gamma1, lambda2, optimize=True)
        C1["av"] += scale * +0.250 * np.einsum('uvwa,xyzr,wy,zruv->xa', T2['aaav'], V['aaaa'], gamma1, lambda2, optimize=True)
        C1["av"] += scale * +1.000 * np.einsum('iuva,waib,vw->ub', T2['caav'], V['avcv'], eta1, optimize=True)
        C1["av"] += scale * -1.000 * np.einsum('uvwa,xayb,wx,yv->ub', T2['aaav'], V['avav'], eta1, gamma1, optimize=True)
        C1["av"] += scale * -1.000 * np.einsum('uvwa,xayb,wyvx->ub', T2['aaav'], V['avav'], lambda2, optimize=True)
        C1["av"] += scale * -0.500 * np.einsum('uvwa,xayb,wyuv->xb', T2['aaav'], V['avav'], lambda2, optimize=True)

    def H2_T2_C1_large(self, C1, B, T2, cumulants, scale=1.0):
        gamma1 = cumulants["gamma1"]
        for a in range(self.nvirt):
        # C1["cv"] += scale * -0.500 * np.einsum('ijab,abjc->ic', T2['ccvv'], V['vvcv'], optimize=True)
            C1["cv"] += scale * -0.500 * np.einsum('ijb,jP,bcP->ic', T2['ccvv'][..., a, :], B['vc'][a, ...], B['vv'], optimize="optimal")
            C1["cv"] += scale * +0.500 * np.einsum('ijb,cP,bjP->ic', T2['ccvv'][..., a, :], B['vv'][a, ...], B['vc'], optimize="optimal")
        # C1["cv"] += scale * -0.500 * np.einsum('iuab,abvc,vu->ic', T2['cavv'], V['vvav'], gamma1, optimize=True)
            C1["cv"] += scale * -0.500 * np.einsum('iub,vP,bcP,vu->ic', T2['cavv'][..., a, :], B['va'][a, ...], B['vv'], gamma1, optimize="optimal")
            C1["cv"] += scale * +0.500 * np.einsum('iub,cP,bvP,vu->ic', T2['cavv'][..., a, :], B['vv'][a, ...], B['va'], gamma1, optimize="optimal")
        # C1["av"] += scale * +0.500 * np.einsum('iuab,abic->uc', T2['cavv'], V['vvcv'], optimize=True)
            C1["av"] += scale * +0.500 * np.einsum('iub,iP,bcP->uc', T2['cavv'][..., a, :], B['vc'][a, ...], B['vv'], optimize="optimal")
            C1["av"] += scale * -0.500 * np.einsum('iub,cP,biP->uc', T2['cavv'][..., a, :], B['vv'][a, ...], B['vc'], optimize="optimal")
        # C1["av"] += scale * -0.500 * np.einsum('uvab,abwc,wv->uc', T2['aavv'], V['vvav'], gamma1, optimize=True)
            C1["av"] += scale * -0.500 * np.einsum('uvb,wP,bcP,wv->uc', T2['aavv'][..., a, :], B['va'][a, ...], B['vv'], gamma1, optimize=True)
            C1["av"] += scale * +0.500 * np.einsum('uvb,cP,bwP,wv->uc', T2['aavv'][..., a, :], B['vv'][a, ...], B['va'], gamma1, optimize=True)

    @staticmethod
    def H1_T2_C2_non_od(C2, F, T2, cumulants, scale=1.0):
        # 22 lines
        C2["ccav"] += scale * +1.000 * np.einsum('ij,kjua->ikua', F['cc'], T2['ccav'], optimize=True)
        C2["ccav"] += scale * +0.500 * np.einsum('uv,ijua->ijva', F['aa'], T2['ccav'], optimize=True)
        C2["ccav"] += scale * +0.500 * np.einsum('ab,ijua->ijub', F['vv'], T2['ccav'], optimize=True)
        C2["ccvv"] += scale * +0.500 * np.einsum('ij,kjab->ikab', F['cc'], T2['ccvv'], optimize=True)
        C2["ccvv"] += scale * -0.500 * np.einsum('ab,ijca->ijbc', F['vv'], T2['ccvv'], optimize=True)
        C2["caav"] += scale * -1.000 * np.einsum('ij,juva->iuva', F['cc'], T2['caav'], optimize=True)
        C2["caav"] += scale * +1.000 * np.einsum('uv,iwua->iwva', F['aa'], T2['caav'], optimize=True)
        C2["caav"] += scale * -1.000 * np.einsum('uv,ivwa->iuwa', F['aa'], T2['caav'], optimize=True)
        C2["caav"] += scale * +1.000 * np.einsum('ab,iuva->iuvb', F['vv'], T2['caav'], optimize=True)
        C2["caaa"] += scale * -0.500 * np.einsum('ij,juvw->iuvw', F['cc'], T2['caaa'], optimize=True)
        C2["caaa"] += scale * -1.000 * np.einsum('uv,iwxu->iwvx', F['aa'], T2['caaa'], optimize=True)
        C2["caaa"] += scale * -0.500 * np.einsum('uv,ivwx->iuwx', F['aa'], T2['caaa'], optimize=True)
        C2["aavv"] += scale * +0.500 * np.einsum('uv,wvab->uwab', F['aa'], T2['aavv'], optimize=True)
        C2["aavv"] += scale * -0.500 * np.einsum('ab,uvca->uvbc', F['vv'], T2['aavv'], optimize=True)
        C2["cavv"] += scale * -0.500 * np.einsum('ij,juab->iuab', F['cc'], T2['cavv'], optimize=True)
        C2["cavv"] += scale * -0.500 * np.einsum('uv,ivab->iuab', F['aa'], T2['cavv'], optimize=True)
        C2["cavv"] += scale * -1.000 * np.einsum('ab,iuca->iubc', F['vv'], T2['cavv'], optimize=True)
        C2["ccaa"] += scale * +0.500 * np.einsum('ij,kjuv->ikuv', F['cc'], T2['ccaa'], optimize=True)
        C2["ccaa"] += scale * -0.500 * np.einsum('uv,ijwu->ijvw', F['aa'], T2['ccaa'], optimize=True)
        C2["aaav"] += scale * +0.500 * np.einsum('uv,wxua->wxva', F['aa'], T2['aaav'], optimize=True)
        C2["aaav"] += scale * +1.000 * np.einsum('uv,wvxa->uwxa', F['aa'], T2['aaav'], optimize=True)
        C2["aaav"] += scale * +0.500 * np.einsum('ab,uvwa->uvwb', F['vv'], T2['aaav'], optimize=True)

    @staticmethod
    def H2_T1_C2_non_od(C2, V, T1, cumulants, scale=1.0):
        # 22 lines
        C2["ccav"] += scale * -0.500 * np.einsum('iu,jkia->jkua', T1['ca'], V['cccv'], optimize=True)
        C2["ccav"] += scale * +0.500 * np.einsum('ia,jkiu->jkua', T1['cv'], V['ccca'], optimize=True)
        C2["ccav"] += scale * -1.000 * np.einsum('ia,jaub->ijub', T1['cv'], V['cvav'], optimize=True)
        C2["ccvv"] += scale * -0.500 * np.einsum('ia,jkib->jkab', T1['cv'], V['cccv'], optimize=True)
        C2["caav"] += scale * -1.000 * np.einsum('iu,jvia->jvua', T1['ca'], V['cacv'], optimize=True)
        C2["caav"] += scale * +1.000 * np.einsum('ia,juiv->juva', T1['cv'], V['caca'], optimize=True)
        C2["caav"] += scale * -1.000 * np.einsum('ia,uavb->iuvb', T1['cv'], V['avav'], optimize=True)
        C2["caav"] += scale * +1.000 * np.einsum('ua,iavb->iuvb', T1['av'], V['cvav'], optimize=True)
        C2["caaa"] += scale * -1.000 * np.einsum('iu,jviw->jvuw', T1['ca'], V['caca'], optimize=True)
        C2["caaa"] += scale * -0.500 * np.einsum('iu,vuwx->ivwx', T1['ca'], V['aaaa'], optimize=True)
        C2["caaa"] += scale * +0.500 * np.einsum('ua,iavw->iuvw', T1['av'], V['cvaa'], optimize=True)
        C2["aavv"] += scale * -0.500 * np.einsum('ia,uvib->uvab', T1['cv'], V['aacv'], optimize=True)
        C2["cavv"] += scale * -1.000 * np.einsum('ia,juib->juab', T1['cv'], V['cacv'], optimize=True)
        C2["ccaa"] += scale * -0.500 * np.einsum('iu,jkiv->jkuv', T1['ca'], V['ccca'], optimize=True)
        C2["ccaa"] += scale * -0.500 * np.einsum('ia,jauv->ijuv', T1['cv'], V['cvaa'], optimize=True)
        C2["aaav"] += scale * -0.500 * np.einsum('iu,vwia->vwua', T1['ca'], V['aacv'], optimize=True)
        C2["aaav"] += scale * -0.500 * np.einsum('ua,vwxu->vwxa', T1['av'], V['aaaa'], optimize=True)
        C2["aaav"] += scale * -1.000 * np.einsum('ua,vawb->uvwb', T1['av'], V['avav'], optimize=True)

    def H2_T1_C2_large(self, C2, B, T1, cumulants, scale=1.0):
        # C2["aavv"] += scale * -0.500 * np.einsum('ua,vabc->uvbc', T1['av'], V['avvv'], optimize=True)

        for a in range(self.nvirt):
            C2["aavv"] += scale * -0.500 * np.einsum('u,vbP,cP->uvbc', T1['av'][:, a], B['av'], B['vv'][a, ...], optimize="optimal")
            C2["aavv"] += scale * +0.500 * np.einsum('u,vcP,bP->uvbc', T1['av'][:, a], B['av'], B['vv'][a, ...], optimize="optimal")

        # C2["cavv"] += scale * -0.500 * np.einsum('ia,uabc->iubc', T1['cv'], V['avvv'], optimize=True)

            C2["cavv"] += scale * -0.500 * np.einsum('i,ubP,cP->iubc', T1['cv'][:, a], B['av'], B['vv'][a, ...], optimize="optimal")
            C2["cavv"] += scale * +0.500 * np.einsum('i,ucP,bP->iubc', T1['cv'][:, a], B['av'], B['vv'][a, ...], optimize="optimal")

        # C2["ccvv"] += scale * -0.500 * np.einsum('ia,jabc->ijbc', T1['cv'], V['cvvv'], optimize=True)
            C2["ccvv"] += scale * -0.500 * np.einsum('i,jbP,cP->ijbc', T1['cv'][:, a], B['cv'], B['vv'][a, ...], optimize="optimal")
            C2["ccvv"] += scale * +0.500 * np.einsum('i,jcP,bP->ijbc', T1['cv'][:, a], B['cv'], B['vv'][a, ...], optimize="optimal")

        # C2["cavv"] += scale * +0.500 * np.einsum('ua,iabc->iubc', T1['av'], V['cvvv'], optimize=True)
            C2["cavv"] += scale * +0.500 * np.einsum('u,ibP,cP->iubc', T1['av'][:, a], B['cv'], B['vv'][a, ...], optimize=True)
            C2["cavv"] += scale * -0.500 * np.einsum('u,icP,bP->iubc', T1['av'][:, a], B['cv'], B['vv'][a, ...], optimize=True)

    @staticmethod
    def H2_T2_C2_non_od(C2, V, T2, cumulants, scale=1.0):
        # 74 lines
        gamma1 = cumulants["gamma1"]
        eta1 = cumulants["eta1"]
        C2["ccvv"] += scale * +1.000 * np.einsum('ijua,kvjb,uv->ikab', T2['ccav'], V['cacv'], eta1, optimize=True)
        C2["ccvv"] += scale * +0.125 * np.einsum('ijab,klij->klab', T2['ccvv'], V['cccc'], optimize=True)
        C2["ccvv"] += scale * +0.250 * np.einsum('iuab,jkiv,vu->jkab', T2['cavv'], V['ccca'], gamma1, optimize=True)
        C2["ccvv"] += scale * -1.000 * np.einsum('ijab,kbjc->ikac', T2['ccvv'], V['cvcv'], optimize=True)
        C2["ccvv"] += scale * -1.000 * np.einsum('iuab,jbvc,vu->ijac', T2['cavv'], V['cvav'], gamma1, optimize=True)
        C2["ccav"] += scale * -1.000 * np.einsum('ijuv,kwja,vw->ikua', T2['ccaa'], V['cacv'], eta1, optimize=True)
        C2["ccav"] += scale * +0.250 * np.einsum('ijua,klij->klua', T2['ccav'], V['cccc'], optimize=True)
        C2["ccav"] += scale * +0.500 * np.einsum('iuva,jkiw,wu->jkva', T2['caav'], V['ccca'], gamma1, optimize=True)
        C2["ccav"] += scale * -1.000 * np.einsum('ijua,kajb->ikub', T2['ccav'], V['cvcv'], optimize=True)
        C2["ccav"] += scale * -1.000 * np.einsum('iuva,jawb,wu->ijvb', T2['caav'], V['cvav'], gamma1, optimize=True)
        C2["ccav"] += scale * -1.000 * np.einsum('ijua,kvjw,uv->ikwa', T2['ccav'], V['caca'], eta1, optimize=True)
        C2["ccav"] += scale * +0.500 * np.einsum('ijua,vawb,uv->ijwb', T2['ccav'], V['avav'], eta1, optimize=True)
        C2["ccav"] += scale * +1.000 * np.einsum('ijab,kbju->ikua', T2['ccvv'], V['cvca'], optimize=True)
        C2["ccav"] += scale * -1.000 * np.einsum('iuab,jbvw,wu->ijva', T2['cavv'], V['cvaa'], gamma1, optimize=True)
        C2["ccaa"] += scale * +0.125 * np.einsum('ijuv,klij->kluv', T2['ccaa'], V['cccc'], optimize=True)
        C2["ccaa"] += scale * +0.250 * np.einsum('iuvw,jkix,xu->jkvw', T2['caaa'], V['ccca'], gamma1, optimize=True)
        C2["ccaa"] += scale * -1.000 * np.einsum('ijuv,kwjx,vw->ikux', T2['ccaa'], V['caca'], eta1, optimize=True)
        C2["ccaa"] += scale * -1.000 * np.einsum('ijua,kajv->ikuv', T2['ccav'], V['cvca'], optimize=True)
        C2["ccaa"] += scale * +1.000 * np.einsum('iuva,jawx,xu->ijvw', T2['caav'], V['cvaa'], gamma1, optimize=True)
        C2["ccaa"] += scale * +0.125 * np.einsum('ijuv,wxyz,vx,uw->ijyz', T2['ccaa'], V['aaaa'], eta1, eta1, optimize=True)
        C2["ccaa"] += scale * -0.125 * np.einsum('ijuv,wxyz,vx,uw->ijyz', T2['ccaa'], V['aaaa'], gamma1, gamma1, optimize=True)
        C2["caav"] += scale * -1.000 * np.einsum('ijuv,wxja,vx->iwua', T2['ccaa'], V['aacv'], eta1, optimize=True)
        C2["caav"] += scale * -1.000 * np.einsum('iuvw,jxia,wx->juva', T2['caaa'], V['cacv'], eta1, optimize=True)
        C2["caav"] += scale * +0.500 * np.einsum('ijua,kvij->kvua', T2['ccav'], V['cacc'], optimize=True)
        C2["caav"] += scale * +1.000 * np.einsum('iuva,jwix,xu->jwva', T2['caav'], V['caca'], gamma1, optimize=True)
        C2["caav"] += scale * -1.000 * np.einsum('ijua,vajb->ivub', T2['ccav'], V['avcv'], optimize=True)
        C2["caav"] += scale * -1.000 * np.einsum('iuva,waxb,xu->iwvb', T2['caav'], V['avav'], gamma1, optimize=True)
        C2["caav"] += scale * -1.000 * np.einsum('iuva,jaib->juvb', T2['caav'], V['cvcv'], optimize=True)
        C2["caav"] += scale * +1.000 * np.einsum('uvwa,iaxb,xv->iuwb', T2['aaav'], V['cvav'], gamma1, optimize=True)
        C2["caav"] += scale * -1.000 * np.einsum('iuva,wxyz,zu,vx->iwya', T2['caav'], V['aaaa'], eta1, gamma1, optimize=True)
        C2["caav"] += scale * +1.000 * np.einsum('iuva,wxyz,vx,zu->iwya', T2['caav'], V['aaaa'], eta1, gamma1, optimize=True)
        C2["caav"] += scale * -1.000 * np.einsum('iuva,jwix,vw->juxa', T2['caav'], V['caca'], eta1, optimize=True)
        C2["caav"] += scale * +1.000 * np.einsum('iuva,waxb,vw->iuxb', T2['caav'], V['avav'], eta1, optimize=True)
        C2["caav"] += scale * +1.000 * np.einsum('iuab,jbiv->juva', T2['cavv'], V['cvca'], optimize=True)
        C2["caav"] += scale * +1.000 * np.einsum('uvab,ibwx,xv->iuwa', T2['aavv'], V['cvaa'], gamma1, optimize=True)
        C2["aavv"] += scale * -1.000 * np.einsum('iuva,wxib,vx->uwab', T2['caav'], V['aacv'], eta1, optimize=True)
        C2["aavv"] += scale * -0.125 * np.einsum('uvab,wxyz,zv,yu->wxab', T2['aavv'], V['aaaa'], eta1, eta1, optimize=True)
        C2["aavv"] += scale * +0.125 * np.einsum('uvab,wxyz,zv,yu->wxab', T2['aavv'], V['aaaa'], gamma1, gamma1, optimize=True)
        C2["aavv"] += scale * +1.000 * np.einsum('iuab,vbic->uvac', T2['cavv'], V['avcv'], optimize=True)
        C2["aavv"] += scale * -1.000 * np.einsum('uvab,wbxc,xv->uwac', T2['aavv'], V['avav'], gamma1, optimize=True)
        C2["caaa"] += scale * +0.250 * np.einsum('ijuv,kwij->kwuv', T2['ccaa'], V['cacc'], optimize=True)
        C2["caaa"] += scale * +0.500 * np.einsum('iuvw,jxiy,yu->jxvw', T2['caaa'], V['caca'], gamma1, optimize=True)
        C2["caaa"] += scale * -1.000 * np.einsum('iuvw,xyzr,ru,wy->ixvz', T2['caaa'], V['aaaa'], eta1, gamma1, optimize=True)
        C2["caaa"] += scale * +1.000 * np.einsum('iuvw,xyzr,wy,ru->ixvz', T2['caaa'], V['aaaa'], eta1, gamma1, optimize=True)
        C2["caaa"] += scale * -1.000 * np.einsum('iuvw,jxiy,wx->juvy', T2['caaa'], V['caca'], eta1, optimize=True)
        C2["caaa"] += scale * -1.000 * np.einsum('iuva,jaiw->juvw', T2['caav'], V['cvca'], optimize=True)
        C2["caaa"] += scale * -1.000 * np.einsum('uvwa,iaxy,yv->iuwx', T2['aaav'], V['cvaa'], gamma1, optimize=True)
        C2["caaa"] += scale * +0.250 * np.einsum('iuvw,xyzr,wy,vx->iuzr', T2['caaa'], V['aaaa'], eta1, eta1, optimize=True)
        C2["caaa"] += scale * -0.250 * np.einsum('iuvw,xyzr,wy,vx->iuzr', T2['caaa'], V['aaaa'], gamma1, gamma1, optimize=True)
        C2["aaav"] += scale * +1.000 * np.einsum('iuvw,xyia,wy->uxva', T2['caaa'], V['aacv'], eta1, optimize=True)
        C2["aaav"] += scale * -0.250 * np.einsum('uvwa,xyzr,rv,zu->xywa', T2['aaav'], V['aaaa'], eta1, eta1, optimize=True)
        C2["aaav"] += scale * +0.250 * np.einsum('uvwa,xyzr,rv,zu->xywa', T2['aaav'], V['aaaa'], gamma1, gamma1, optimize=True)
        C2["aaav"] += scale * +1.000 * np.einsum('iuva,waib->uwvb', T2['caav'], V['avcv'], optimize=True)
        C2["aaav"] += scale * -1.000 * np.einsum('uvwa,xayb,yv->uxwb', T2['aaav'], V['avav'], gamma1, optimize=True)
        C2["aaav"] += scale * -1.000 * np.einsum('uvwa,xyzr,rv,wy->uxza', T2['aaav'], V['aaaa'], eta1, gamma1, optimize=True)
        C2["aaav"] += scale * +1.000 * np.einsum('uvwa,xyzr,wy,rv->uxza', T2['aaav'], V['aaaa'], eta1, gamma1, optimize=True)
        C2["aaav"] += scale * +0.500 * np.einsum('uvwa,xayb,wx->uvyb', T2['aaav'], V['avav'], eta1, optimize=True)
        C2["cavv"] += scale * +1.000 * np.einsum('ijua,vwjb,uw->ivab', T2['ccav'], V['aacv'], eta1, optimize=True)
        C2["cavv"] += scale * +1.000 * np.einsum('iuva,jwib,vw->juab', T2['caav'], V['cacv'], eta1, optimize=True)
        C2["cavv"] += scale * +0.250 * np.einsum('ijab,kuij->kuab', T2['ccvv'], V['cacc'], optimize=True)
        C2["cavv"] += scale * +0.500 * np.einsum('iuab,jviw,wu->jvab', T2['cavv'], V['caca'], gamma1, optimize=True)
        C2["cavv"] += scale * -1.000 * np.einsum('ijab,ubjc->iuac', T2['ccvv'], V['avcv'], optimize=True)
        C2["cavv"] += scale * -1.000 * np.einsum('iuab,vbwc,wu->ivac', T2['cavv'], V['avav'], gamma1, optimize=True)
        C2["cavv"] += scale * -1.000 * np.einsum('iuab,jbic->juac', T2['cavv'], V['cvcv'], optimize=True)
        C2["cavv"] += scale * +1.000 * np.einsum('uvab,ibwc,wv->iuac', T2['aavv'], V['cvav'], gamma1, optimize=True)

    def H2_T2_C2_large(self, C2, B, T2, cumulants, scale=1.0):
        eta1 = cumulants["eta1"]
        for a in range(self.nvirt):
        # C2["ccvv"] += scale * +0.125 * np.einsum('ijab,abcd->ijcd', T2['ccvv'], V['vvvv'], optimize=True)
            C2["ccvv"] += scale * +0.125 * np.einsum('cP,bdP,ijb->ijcd', B['vv'][a, ...], B['vv'], T2['ccvv'][..., a, :], optimize="optimal")
            C2["ccvv"] += scale * -0.125 * np.einsum('dP,bcP,ijb->ijcd', B['vv'][a, ...], B['vv'], T2['ccvv'][..., a, :], optimize="optimal")

        # C2["cavv"] += scale * +0.250 * np.einsum('iuab,abcd->iucd', T2['cavv'], V['vvvv'], optimize=True)
            C2["cavv"] += scale * +0.250 * np.einsum('iub,cP,bdP->iucd', T2['cavv'][..., a, :], B['vv'][a, ...], B['vv'], optimize="optimal")
            C2["cavv"] += scale * -0.250 * np.einsum('iub,dP,bcP->iucd', T2['cavv'][..., a, :], B['vv'][a, ...], B['vv'], optimize="optimal")

        # C2["aavv"] += scale * +0.125 * np.einsum('uvab,abcd->uvcd', T2['aavv'], V['vvvv'], optimize=True)
            C2["aavv"] += scale * +0.125 * np.einsum('uvb,cP,bdP->uvcd', T2['aavv'][..., a, :], B['vv'][a, ...], B['vv'], optimize="optimal")
            C2["aavv"] += scale * -0.125 * np.einsum('uvb,dP,bcP->uvcd', T2['aavv'][..., a, :], B['vv'][a, ...], B['vv'], optimize="optimal")

        # C2["cavv"] += scale * +0.500 * np.einsum('iuva,wabc,vw->iubc', T2['caav'], V['avvv'], eta1, optimize=True)
            C2["cavv"] += scale * +0.500 * np.einsum('iuv,wbP,cP,vw->iubc', T2['caav'][..., a], B['av'], B['vv'][a, ...], eta1, optimize="optimal")
            C2["cavv"] += scale * -0.500 * np.einsum('iuv,wcP,bP,vw->iubc', T2['caav'][..., a], B['av'], B['vv'][a, ...], eta1, optimize="optimal")

        # C2["aavv"] += scale * +0.250 * np.einsum('uvwa,xabc,wx->uvbc', T2['aaav'], V['avvv'], eta1, optimize=True)
            C2["aavv"] += scale * +0.250 * np.einsum('uvw,xbP,cP,wx->uvbc', T2['aaav'][..., a], B['av'], B['vv'][a, ...], eta1, optimize="optimal")
            C2["aavv"] += scale * -0.250 * np.einsum('uvw,xcP,bP,wx->uvbc', T2['aaav'][..., a], B['av'], B['vv'][a, ...], eta1, optimize="optimal")

        # C2["ccvv"] += scale * +0.250 * np.einsum('ijua,vabc,uv->ijbc', T2['ccav'], V['avvv'], eta1, optimize=True)
            C2["ccvv"] += scale * +0.250 * np.einsum('iju,vbP,cP,uv->ijbc', T2['ccav'][..., a], B['av'], B['vv'][a, ...], eta1, optimize="optimal")
            C2["ccvv"] += scale * -0.250 * np.einsum('iju,vcP,bP,uv->ijbc', T2['ccav'][..., a], B['av'], B['vv'][a, ...], eta1, optimize="optimal")

        # C2["ccav"] += scale * +0.250 * np.einsum('ijab,abuc->ijuc', T2['ccvv'], V['vvav'], optimize=True)
            C2["ccav"] += scale * +0.250 * np.einsum('ijb,uP,bcP->ijuc', T2['ccvv'][..., a, :], B['va'][a, ...], B['vv'], optimize="optimal")
            C2["ccav"] += scale * -0.250 * np.einsum('ijb,cP,buP->ijuc', T2['ccvv'][..., a, :], B['vv'][a, ...], B['va'], optimize="optimal")

        # C2["caav"] += scale * +0.500 * np.einsum('iuab,abvc->iuvc', T2['cavv'], V['vvav'], optimize=True)
            C2["caav"] += scale * +0.500 * np.einsum('iub,vP,bcP->iuvc', T2['cavv'][..., a, :], B['va'][a, ...], B['vv'], optimize="optimal")
            C2["caav"] += scale * -0.500 * np.einsum('iub,cP,bvP->iuvc', T2['cavv'][..., a, :], B['vv'][a, ...], B['va'], optimize="optimal")

        # C2["aaav"] += scale * +0.250 * np.einsum('uvab,abwc->uvwc', T2['aavv'], V['vvav'], optimize=True)
            C2["aaav"] += scale * +0.250 * np.einsum('uvb,wP,bcP->uvwc', T2['aavv'][..., a, :], B['va'][a, ...], B['vv'], optimize="optimal")
            C2["aaav"] += scale * -0.250 * np.einsum('uvb,cP,bwP->uvwc', T2['aavv'][..., a, :], B['vv'][a, ...], B['va'], optimize="optimal")

    # fmt: on
