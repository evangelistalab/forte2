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
        # large_labels = set(["vvvv", "avvv", "cvvv", "vvav", "vvcv"])
        large_labels = set(["vvvv"])
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

        g1 = cumulants['gamma1']
        e1 = cumulants['eta1']
        l2 = cumulants['lambda2']
        l3 = cumulants['lambda3']

        C0 = .0j
        C0 += scale * +1.000 * np.einsum('iu,iv,vu->', F['ca'], T1['ca'], e1, optimize=True)
        C0 += scale * -0.500 * np.einsum('iu,ivwx,wxuv->', F['ca'], T2['caaa'], l2, optimize=True)
        C0 += scale * +1.000 * np.einsum('ia,ia->', F['cv'], T1['cv'], optimize=True)
        C0 += scale * +1.000 * np.einsum('ua,va,uv->', F['av'], T1['av'], g1, optimize=True)
        C0 += scale * -0.500 * np.einsum('ua,vwxa,uxvw->', F['av'], T2['aaav'], l2, optimize=True)
        C0 += scale * -0.500 * np.einsum('iu,ivwx,uvwx->', T1['ca'], V['caaa'], l2, optimize=True)
        C0 += scale * -0.500 * np.einsum('ua,vwxa,vwux->', T1['av'], V['aaav'], l2, optimize=True)
        C0 += scale * +0.250 * np.einsum('ijuv,ijwx,vx,uw->', T2['ccaa'], V['ccaa'], e1, e1, optimize=True)
        C0 += scale * +0.125 * np.einsum('ijuv,ijwx,uvwx->', T2['ccaa'], V['ccaa'], l2, optimize=True)
        C0 += scale * +0.500 * np.einsum('iuvw,ixyz,wz,vy,xu->', T2['caaa'], V['caaa'], e1, e1, g1, optimize=True)
        C0 += scale * +1.000 * np.einsum('iuvw,ixyz,wz,vxuy->', T2['caaa'], V['caaa'], e1, l2, optimize=True)
        C0 += scale * +0.250 * np.einsum('iuvw,ixyz,xu,vwyz->', T2['caaa'], V['caaa'], g1, l2, optimize=True)
        C0 += scale * +0.250 * np.einsum('iuvw,ixyz,vwxuyz->', T2['caaa'], V['caaa'], l3, optimize=True)
        C0 += scale * +0.500 * np.einsum('ijua,ijva,uv->', T2['ccav'], V['ccav'], e1, optimize=True)
        C0 += scale * +1.000 * np.einsum('iuva,iwxa,vx,wu->', T2['caav'], V['caav'], e1, g1, optimize=True)
        C0 += scale * +1.000 * np.einsum('iuva,iwxa,vwux->', T2['caav'], V['caav'], l2, optimize=True)
        C0 += scale * +0.500 * np.einsum('uvwa,xyza,wz,yv,xu->', T2['aaav'], V['aaav'], e1, g1, g1, optimize=True)
        C0 += scale * +0.250 * np.einsum('uvwa,xyza,wz,xyuv->', T2['aaav'], V['aaav'], e1, l2, optimize=True)
        C0 += scale * +1.000 * np.einsum('uvwa,xyza,yv,wxuz->', T2['aaav'], V['aaav'], g1, l2, optimize=True)
        C0 += scale * -0.250 * np.einsum('uvwa,xyza,wxyuvz->', T2['aaav'], V['aaav'], l3, optimize=True)
        C0 += scale * +0.250 * np.einsum('ijab,ijab->', T2['ccvv'], V['ccvv'], optimize=True)
        C0 += scale * +0.500 * np.einsum('iuab,ivab,vu->', T2['cavv'], V['cavv'], g1, optimize=True)
        C0 += scale * +0.250 * np.einsum('uvab,wxab,xv,wu->', T2['aavv'], V['aavv'], g1, g1, optimize=True)
        C0 += scale * +0.125 * np.einsum('uvab,wxab,wxuv->', T2['aavv'], V['aavv'], l2, optimize=True)

        return C0
    
    @staticmethod
    def H_T_C1_aa(C1, F, V, T1, T2, cumulants, scale=1.0):
        # 26 lines

        g1 = cumulants['gamma1']
        e1 = cumulants['eta1']
        l2 = cumulants['lambda2']
        l3 = cumulants['lambda3']

        C1 += scale * -1.000 * np.einsum('iu,iv->uv', F['ca'], T1['ca'], optimize=True)
        C1 += scale * -1.000 * np.einsum('iu,ivwx,xu->vw', F['ca'], T2['caaa'], e1, optimize=True)
        C1 += scale * -1.000 * np.einsum('ia,iuva->uv', F['cv'], T2['caav'], optimize=True)
        C1 += scale * +1.000 * np.einsum('ua,va->vu', F['av'], T1['av'], optimize=True)
        C1 += scale * +1.000 * np.einsum('ua,vwxa,uw->vx', F['av'], T2['aaav'], g1, optimize=True)
        C1 += scale * -1.000 * np.einsum('iu,ivwx,ux->wv', T1['ca'], V['caaa'], e1, optimize=True)
        C1 += scale * -1.000 * np.einsum('ia,iuva->vu', T1['cv'], V['caav'], optimize=True)
        C1 += scale * +1.000 * np.einsum('ua,vwxa,wu->xv', T1['av'], V['aaav'], g1, optimize=True)
        C1 += scale * -0.500 * np.einsum('ijuv,ijwx,vx->wu', T2['ccaa'], V['ccaa'], e1, optimize=True)
        C1 += scale * +0.500 * np.einsum('iuvw,ixyz,wxyz->uv', T2['caaa'], V['caaa'], l2, optimize=True)
        C1 += scale * -1.000 * np.einsum('iuvw,ixyz,wz,xu->yv', T2['caaa'], V['caaa'], e1, g1, optimize=True)
        C1 += scale * -1.000 * np.einsum('iuvw,ixyz,wxuz->yv', T2['caaa'], V['caaa'], l2, optimize=True)
        C1 += scale * -0.500 * np.einsum('ijua,ijva->vu', T2['ccav'], V['ccav'], optimize=True)
        C1 += scale * -1.000 * np.einsum('iuva,iwxa,wu->xv', T2['caav'], V['caav'], g1, optimize=True)
        C1 += scale * -0.500 * np.einsum('uvwa,xyza,xyvz->uw', T2['aaav'], V['aaav'], l2, optimize=True)
        C1 += scale * -0.500 * np.einsum('uvwa,xyza,yv,xu->zw', T2['aaav'], V['aaav'], g1, g1, optimize=True)
        C1 += scale * -0.250 * np.einsum('uvwa,xyza,xyuv->zw', T2['aaav'], V['aaav'], l2, optimize=True)
        C1 += scale * +0.500 * np.einsum('iuvw,ixyz,wz,vy->ux', T2['caaa'], V['caaa'], e1, e1, optimize=True)
        C1 += scale * +0.250 * np.einsum('iuvw,ixyz,vwyz->ux', T2['caaa'], V['caaa'], l2, optimize=True)
        C1 += scale * -0.500 * np.einsum('iuvw,ixyz,vwuz->yx', T2['caaa'], V['caaa'], l2, optimize=True)
        C1 += scale * +1.000 * np.einsum('iuva,iwxa,vx->uw', T2['caav'], V['caav'], e1, optimize=True)
        C1 += scale * +1.000 * np.einsum('uvwa,xyza,wz,yv->ux', T2['aaav'], V['aaav'], e1, g1, optimize=True)
        C1 += scale * +1.000 * np.einsum('uvwa,xyza,wyvz->ux', T2['aaav'], V['aaav'], l2, optimize=True)
        C1 += scale * +0.500 * np.einsum('uvwa,xyza,wyuv->zx', T2['aaav'], V['aaav'], l2, optimize=True)
        C1 += scale * +0.500 * np.einsum('iuab,ivab->uv', T2['cavv'], V['cavv'], optimize=True)
        C1 += scale * +0.500 * np.einsum('uvab,wxab,xv->uw', T2['aavv'], V['aavv'], g1, optimize=True)
    
    @staticmethod
    def H_T_C2_aaaa(C2, F, V, T1, T2, cumulants, scale=1.0):
        # 11 lines

        g1 = cumulants['gamma1']
        e1 = cumulants['eta1']
        l2 = cumulants['lambda2']
        l3 = cumulants['lambda3']

        C2 += scale * -0.500 * np.einsum('iu,ivwx->uvwx', F['ca'], T2['caaa'], optimize=True)
        C2 += scale * -0.500 * np.einsum('ua,vwxa->vwux', F['av'], T2['aaav'], optimize=True)
        C2 += scale * -0.500 * np.einsum('iu,ivwx->wxuv', T1['ca'], V['caaa'], optimize=True)
        C2 += scale * -0.500 * np.einsum('ua,vwxa->uxvw', T1['av'], V['aaav'], optimize=True)
        C2 += scale * +0.125 * np.einsum('ijuv,ijwx->wxuv', T2['ccaa'], V['ccaa'], optimize=True)
        C2 += scale * +0.250 * np.einsum('iuvw,ixyz,xu->yzvw', T2['caaa'], V['caaa'], g1, optimize=True)
        C2 += scale * +1.000 * np.einsum('iuvw,ixyz,wz->uyvx', T2['caaa'], V['caaa'], e1, optimize=True)
        C2 += scale * +1.000 * np.einsum('iuva,iwxa->uxvw', T2['caav'], V['caav'], optimize=True)
        C2 += scale * +1.000 * np.einsum('uvwa,xyza,yv->uzwx', T2['aaav'], V['aaav'], g1, optimize=True)
        C2 += scale * +0.250 * np.einsum('uvwa,xyza,wz->uvxy', T2['aaav'], V['aaav'], e1, optimize=True)
        C2 += scale * +0.125 * np.einsum('uvab,wxab->uvwx', T2['aavv'], V['aavv'], optimize=True)
    
    @staticmethod
    def H1_T1_C1(C1, F, T1, cumulants, scale=1.0):
        # 6 lines

        g1 = cumulants['gamma1']
        e1 = cumulants['eta1']
        l2 = cumulants['lambda2']
        l3 = cumulants['lambda3']

        C1["av"] += scale * -1.000 * np.einsum('iu,ia->ua', F['ca'], T1['cv'], optimize=True)
        C1["cv"] += scale * -1.000 * np.einsum('ui,ua->ia', F['ac'], T1['av'], optimize=True)
        C1["cv"] += scale * +1.000 * np.einsum('au,iu->ia', F['va'], T1['ca'], optimize=True)
        C1["ca"] += scale * +1.000 * np.einsum('ua,ia->iu', F['av'], T1['cv'], optimize=True)
        C1["ac"] += scale * +1.000 * np.einsum('ia,ua->ui', F['cv'], T1['av'], optimize=True)
        C1["va"] += scale * -1.000 * np.einsum('ia,iu->au', F['cv'], T1['ca'], optimize=True)
    
    @staticmethod
    def H1_T2_C1(C1, F, T2, cumulants, scale=1.0):
        # 9 lines

        g1 = cumulants['gamma1']
        e1 = cumulants['eta1']
        l2 = cumulants['lambda2']
        l3 = cumulants['lambda3']

        C1["av"] += scale * +1.000 * np.einsum('iu,ivwa,wu->va', F['ca'], T2['caav'], e1, optimize=True)
        C1["av"] += scale * -1.000 * np.einsum('ia,iuba->ub', F['cv'], T2['cavv'], optimize=True)
        C1["av"] += scale * +1.000 * np.einsum('ua,vwba,uw->vb', F['av'], T2['aavv'], g1, optimize=True)
        C1["cv"] += scale * -1.000 * np.einsum('iu,jiva,vu->ja', F['ca'], T2['ccav'], e1, optimize=True)
        C1["cv"] += scale * +1.000 * np.einsum('ia,jiba->jb', F['cv'], T2['ccvv'], optimize=True)
        C1["cv"] += scale * +1.000 * np.einsum('ua,ivba,uv->ib', F['av'], T2['cavv'], g1, optimize=True)
        C1["ca"] += scale * +1.000 * np.einsum('iu,jivw,wu->jv', F['ca'], T2['ccaa'], e1, optimize=True)
        C1["ca"] += scale * +1.000 * np.einsum('ia,jiua->ju', F['cv'], T2['ccav'], optimize=True)
        C1["ca"] += scale * +1.000 * np.einsum('ua,ivwa,uv->iw', F['av'], T2['caav'], g1, optimize=True)
    
    @staticmethod
    def H2_T1_C1(C1, V, T1, cumulants, scale=1.0):
        # 9 lines

        g1 = cumulants['gamma1']
        e1 = cumulants['eta1']
        l2 = cumulants['lambda2']
        l3 = cumulants['lambda3']

        C1["vc"] += scale * -1.000 * np.einsum('iu,jiva,uv->aj', T1['ca'], V['ccav'], e1, optimize=True)
        C1["vc"] += scale * +1.000 * np.einsum('ia,jiba->bj', T1['cv'], V['ccvv'], optimize=True)
        C1["vc"] += scale * +1.000 * np.einsum('ua,ivba,vu->bi', T1['av'], V['cavv'], g1, optimize=True)
        C1["ac"] += scale * +1.000 * np.einsum('iu,jivw,uw->vj', T1['ca'], V['ccaa'], e1, optimize=True)
        C1["ac"] += scale * +1.000 * np.einsum('ia,jiua->uj', T1['cv'], V['ccav'], optimize=True)
        C1["ac"] += scale * +1.000 * np.einsum('ua,ivwa,vu->wi', T1['av'], V['caav'], g1, optimize=True)
        C1["va"] += scale * +1.000 * np.einsum('iu,ivwa,uw->av', T1['ca'], V['caav'], e1, optimize=True)
        C1["va"] += scale * -1.000 * np.einsum('ia,iuba->bu', T1['cv'], V['cavv'], optimize=True)
        C1["va"] += scale * +1.000 * np.einsum('ua,vwba,wu->bv', T1['av'], V['aavv'], g1, optimize=True)
    
    @staticmethod
    def H2_T2_C1(C1, V, T2, cumulants, scale=1.0):
        # 52 lines

        g1 = cumulants['gamma1']
        e1 = cumulants['eta1']
        l2 = cumulants['lambda2']
        l3 = cumulants['lambda3']

        C1["av"] += scale * +0.500 * np.einsum('ijua,ijvw,uw->va', T2['ccav'], V['ccaa'], e1, optimize=True)
        C1["av"] += scale * -0.500 * np.einsum('iuva,iwxy,vwxy->ua', T2['caav'], V['caaa'], l2, optimize=True)
        C1["av"] += scale * +1.000 * np.einsum('iuva,iwxy,vy,wu->xa', T2['caav'], V['caaa'], e1, g1, optimize=True)
        C1["av"] += scale * +1.000 * np.einsum('iuva,iwxy,vwuy->xa', T2['caav'], V['caaa'], l2, optimize=True)
        C1["av"] += scale * -0.500 * np.einsum('ijab,ijub->ua', T2['ccvv'], V['ccav'], optimize=True)
        C1["av"] += scale * -1.000 * np.einsum('iuab,ivwb,vu->wa', T2['cavv'], V['caav'], g1, optimize=True)
        C1["av"] += scale * -0.500 * np.einsum('uvab,wxyb,wxvy->ua', T2['aavv'], V['aaav'], l2, optimize=True)
        C1["av"] += scale * -0.500 * np.einsum('uvab,wxyb,xv,wu->ya', T2['aavv'], V['aaav'], g1, g1, optimize=True)
        C1["av"] += scale * -0.250 * np.einsum('uvab,wxyb,wxuv->ya', T2['aavv'], V['aaav'], l2, optimize=True)
        C1["vc"] += scale * -0.500 * np.einsum('iuvw,jixa,vwux->aj', T2['caaa'], V['ccav'], l2, optimize=True)
        C1["vc"] += scale * +0.500 * np.einsum('uvwa,ixba,wxuv->bi', T2['aaav'], V['cavv'], l2, optimize=True)
        C1["cv"] += scale * -0.500 * np.einsum('iuvw,xayz,xu,wz,vy->ia', T2['caaa'], V['avaa'], e1, g1, g1, optimize=True)
        C1["cv"] += scale * -0.250 * np.einsum('iuvw,xayz,xu,vwyz->ia', T2['caaa'], V['avaa'], e1, l2, optimize=True)
        C1["cv"] += scale * -0.500 * np.einsum('iuvw,xayz,wz,vy,xu->ia', T2['caaa'], V['avaa'], e1, e1, g1, optimize=True)
        C1["cv"] += scale * -1.000 * np.einsum('iuvw,xayz,wz,vxuy->ia', T2['caaa'], V['avaa'], e1, l2, optimize=True)
        C1["cv"] += scale * -0.250 * np.einsum('iuvw,xayz,xu,vwyz->ia', T2['caaa'], V['avaa'], g1, l2, optimize=True)
        C1["cv"] += scale * -1.000 * np.einsum('iuvw,xayz,wz,vxuy->ia', T2['caaa'], V['avaa'], g1, l2, optimize=True)
        C1["cv"] += scale * +0.500 * np.einsum('ijua,jvwx,uvwx->ia', T2['ccav'], V['caaa'], l2, optimize=True)
        C1["cv"] += scale * +0.500 * np.einsum('uvwa,xyiz,yv,xu,wz->ia', T2['aaav'], V['aaca'], e1, e1, g1, optimize=True)
        C1["cv"] += scale * +1.000 * np.einsum('uvwa,xyiz,yv,wxuz->ia', T2['aaav'], V['aaca'], e1, l2, optimize=True)
        C1["cv"] += scale * +0.500 * np.einsum('uvwa,xyiz,wz,yv,xu->ia', T2['aaav'], V['aaca'], e1, g1, g1, optimize=True)
        C1["cv"] += scale * +0.250 * np.einsum('uvwa,xyiz,wz,xyuv->ia', T2['aaav'], V['aaca'], e1, l2, optimize=True)
        C1["cv"] += scale * +1.000 * np.einsum('uvwa,xyiz,yv,wxuz->ia', T2['aaav'], V['aaca'], g1, l2, optimize=True)
        C1["cv"] += scale * +0.250 * np.einsum('uvwa,xyiz,wz,xyuv->ia', T2['aaav'], V['aaca'], g1, l2, optimize=True)
        C1["cv"] += scale * -0.500 * np.einsum('iuab,vwxb,vwux->ia', T2['cavv'], V['aaav'], l2, optimize=True)
        C1["ca"] += scale * -0.500 * np.einsum('ijuv,jwxy,vwxy->iu', T2['ccaa'], V['caaa'], l2, optimize=True)
        C1["ca"] += scale * -0.500 * np.einsum('iuva,wxya,wxuy->iv', T2['caav'], V['aaav'], l2, optimize=True)
        C1["ca"] += scale * -0.500 * np.einsum('ijuv,jwxy,vy,ux->iw', T2['ccaa'], V['caaa'], e1, e1, optimize=True)
        C1["ca"] += scale * -0.250 * np.einsum('ijuv,jwxy,uvxy->iw', T2['ccaa'], V['caaa'], l2, optimize=True)
        C1["ca"] += scale * -1.000 * np.einsum('ijua,jvwa,uw->iv', T2['ccav'], V['caav'], e1, optimize=True)
        C1["ca"] += scale * +1.000 * np.einsum('iuva,wxya,vy,xu->iw', T2['caav'], V['aaav'], e1, g1, optimize=True)
        C1["ca"] += scale * +1.000 * np.einsum('iuva,wxya,vxuy->iw', T2['caav'], V['aaav'], l2, optimize=True)
        C1["ca"] += scale * -0.500 * np.einsum('ijab,juab->iu', T2['ccvv'], V['cavv'], optimize=True)
        C1["ca"] += scale * +0.500 * np.einsum('iuab,vwab,wu->iv', T2['cavv'], V['aavv'], g1, optimize=True)
        C1["ac"] += scale * -0.500 * np.einsum('iuvw,jixy,wy,vx->uj', T2['caaa'], V['ccaa'], e1, e1, optimize=True)
        C1["ac"] += scale * -0.250 * np.einsum('iuvw,jixy,vwxy->uj', T2['caaa'], V['ccaa'], l2, optimize=True)
        C1["ac"] += scale * +0.500 * np.einsum('iuvw,jixy,vwuy->xj', T2['caaa'], V['ccaa'], l2, optimize=True)
        C1["ac"] += scale * -1.000 * np.einsum('iuva,jiwa,vw->uj', T2['caav'], V['ccav'], e1, optimize=True)
        C1["ac"] += scale * +1.000 * np.einsum('uvwa,ixya,wy,xv->ui', T2['aaav'], V['caav'], e1, g1, optimize=True)
        C1["ac"] += scale * +1.000 * np.einsum('uvwa,ixya,wxvy->ui', T2['aaav'], V['caav'], l2, optimize=True)
        C1["ac"] += scale * +0.500 * np.einsum('uvwa,ixya,wxuv->yi', T2['aaav'], V['caav'], l2, optimize=True)
        C1["ac"] += scale * -0.500 * np.einsum('iuab,jiab->uj', T2['cavv'], V['ccvv'], optimize=True)
        C1["ac"] += scale * +0.500 * np.einsum('uvab,iwab,wv->ui', T2['aavv'], V['cavv'], g1, optimize=True)
        C1["va"] += scale * +0.500 * np.einsum('ijuv,ijwa,vw->au', T2['ccaa'], V['ccav'], e1, optimize=True)
        C1["va"] += scale * +1.000 * np.einsum('iuvw,ixya,wy,xu->av', T2['caaa'], V['caav'], e1, g1, optimize=True)
        C1["va"] += scale * +1.000 * np.einsum('iuvw,ixya,wxuy->av', T2['caaa'], V['caav'], l2, optimize=True)
        C1["va"] += scale * -0.500 * np.einsum('ijua,ijba->bu', T2['ccav'], V['ccvv'], optimize=True)
        C1["va"] += scale * -1.000 * np.einsum('iuva,iwba,wu->bv', T2['caav'], V['cavv'], g1, optimize=True)
        C1["va"] += scale * -0.500 * np.einsum('uvwa,xyba,yv,xu->bw', T2['aaav'], V['aavv'], g1, g1, optimize=True)
        C1["va"] += scale * -0.250 * np.einsum('uvwa,xyba,xyuv->bw', T2['aaav'], V['aavv'], l2, optimize=True)
        C1["va"] += scale * +0.500 * np.einsum('iuvw,ixya,vwuy->ax', T2['caaa'], V['caav'], l2, optimize=True)
        C1["va"] += scale * +0.500 * np.einsum('uvwa,xyba,wyuv->bx', T2['aaav'], V['aavv'], l2, optimize=True)
    
    @staticmethod
    def H1_T2_C2(C2, F, T2, cumulants, scale=1.0):
        # 22 lines

        g1 = cumulants['gamma1']
        e1 = cumulants['eta1']
        l2 = cumulants['lambda2']
        l3 = cumulants['lambda3']

        C2["ccvv"] += scale * +0.500 * np.einsum('ui,juab->ijab', F['ac'], T2['cavv'], optimize=True)
        C2["ccvv"] += scale * +0.500 * np.einsum('au,ijub->ijab', F['va'], T2['ccav'], optimize=True)
        C2["caav"] += scale * -1.000 * np.einsum('iu,jiva->juva', F['ca'], T2['ccav'], optimize=True)
        C2["caav"] += scale * -1.000 * np.einsum('ua,ivba->ivub', F['av'], T2['cavv'], optimize=True)
        C2["caav"] += scale * +1.000 * np.einsum('ui,vuwa->ivwa', F['ac'], T2['aaav'], optimize=True)
        C2["caav"] += scale * +1.000 * np.einsum('au,ivwu->ivwa', F['va'], T2['caaa'], optimize=True)
        C2["avaa"] += scale * +0.500 * np.einsum('ia,iuvw->uavw', F['cv'], T2['caaa'], optimize=True)
        C2["ccav"] += scale * -0.500 * np.einsum('ua,ijba->ijub', F['av'], T2['ccvv'], optimize=True)
        C2["ccav"] += scale * +1.000 * np.einsum('ui,juva->ijva', F['ac'], T2['caav'], optimize=True)
        C2["ccav"] += scale * +0.500 * np.einsum('au,ijvu->ijva', F['va'], T2['ccaa'], optimize=True)
        C2["caaa"] += scale * -0.500 * np.einsum('iu,jivw->juvw', F['ca'], T2['ccaa'], optimize=True)
        C2["caaa"] += scale * -1.000 * np.einsum('ua,ivwa->ivuw', F['av'], T2['caav'], optimize=True)
        C2["aaav"] += scale * -1.000 * np.einsum('iu,ivwa->uvwa', F['ca'], T2['caav'], optimize=True)
        C2["aaav"] += scale * -0.500 * np.einsum('ua,vwba->vwub', F['av'], T2['aavv'], optimize=True)
        C2["ccaa"] += scale * -0.500 * np.einsum('ua,ijva->ijuv', F['av'], T2['ccav'], optimize=True)
        C2["ccaa"] += scale * +0.500 * np.einsum('ui,juvw->ijvw', F['ac'], T2['caaa'], optimize=True)
        C2["cavv"] += scale * -0.500 * np.einsum('iu,jiab->juab', F['ca'], T2['ccvv'], optimize=True)
        C2["cavv"] += scale * +0.500 * np.einsum('ui,vuab->ivab', F['ac'], T2['aavv'], optimize=True)
        C2["cavv"] += scale * +1.000 * np.einsum('au,ivub->ivab', F['va'], T2['caav'], optimize=True)
        C2["aavv"] += scale * -0.500 * np.einsum('iu,ivab->uvab', F['ca'], T2['cavv'], optimize=True)
        C2["aavv"] += scale * +0.500 * np.einsum('au,vwub->vwab', F['va'], T2['aaav'], optimize=True)
        C2["aaca"] += scale * -0.500 * np.einsum('ia,uvwa->uviw', F['cv'], T2['aaav'], optimize=True)
    
    @staticmethod
    def H2_T1_C2(C2, V, T1, cumulants, scale=1.0):
        # 22 lines

        g1 = cumulants['gamma1']
        e1 = cumulants['eta1']
        l2 = cumulants['lambda2']
        l3 = cumulants['lambda3']

        C2["ccvv"] += scale * -0.500 * np.einsum('iu,abju->ijab', T1['ca'], V['vvca'], optimize=True)
        C2["ccvv"] += scale * -0.500 * np.einsum('ua,ubij->ijab', T1['av'], V['avcc'], optimize=True)
        C2["aacc"] += scale * -0.500 * np.einsum('ua,ijva->uvij', T1['av'], V['ccav'], optimize=True)
        C2["caav"] += scale * -1.000 * np.einsum('iu,vawu->iwva', T1['ca'], V['avaa'], optimize=True)
        C2["caav"] += scale * -1.000 * np.einsum('ua,vuiw->iwva', T1['av'], V['aaca'], optimize=True)
        C2["vvaa"] += scale * -0.500 * np.einsum('iu,ivab->abuv', T1['ca'], V['cavv'], optimize=True)
        C2["avaa"] += scale * -1.000 * np.einsum('iu,ivwa->wauv', T1['ca'], V['caav'], optimize=True)
        C2["avaa"] += scale * -0.500 * np.einsum('ua,vwba->ubvw', T1['av'], V['aavv'], optimize=True)
        C2["avca"] += scale * -1.000 * np.einsum('iu,jiva->vaju', T1['ca'], V['ccav'], optimize=True)
        C2["avca"] += scale * -1.000 * np.einsum('ua,ivba->ubiv', T1['av'], V['cavv'], optimize=True)
        C2["ccav"] += scale * -1.000 * np.einsum('iu,vaju->ijva', T1['ca'], V['avca'], optimize=True)
        C2["ccav"] += scale * -0.500 * np.einsum('ua,vuij->ijva', T1['av'], V['aacc'], optimize=True)
        C2["avcc"] += scale * -0.500 * np.einsum('ua,ijba->ubij', T1['av'], V['ccvv'], optimize=True)
        C2["caaa"] += scale * -0.500 * np.einsum('ia,uvwa->iwuv', T1['cv'], V['aaav'], optimize=True)
        C2["aaav"] += scale * +0.500 * np.einsum('ia,iuvw->vwua', T1['cv'], V['caaa'], optimize=True)
        C2["ccaa"] += scale * -0.500 * np.einsum('iu,vwju->ijvw', T1['ca'], V['aaca'], optimize=True)
        C2["cavv"] += scale * -0.500 * np.einsum('iu,abvu->ivab', T1['ca'], V['vvaa'], optimize=True)
        C2["cavv"] += scale * -1.000 * np.einsum('ua,ubiv->ivab', T1['av'], V['avca'], optimize=True)
        C2["aavv"] += scale * -0.500 * np.einsum('ua,ubvw->vwab', T1['av'], V['avaa'], optimize=True)
        C2["aaca"] += scale * -0.500 * np.einsum('iu,jivw->vwju', T1['ca'], V['ccaa'], optimize=True)
        C2["aaca"] += scale * -1.000 * np.einsum('ua,ivwa->uwiv', T1['av'], V['caav'], optimize=True)
        C2["vvca"] += scale * -0.500 * np.einsum('iu,jiab->abju', T1['ca'], V['ccvv'], optimize=True)
    
    @staticmethod
    def H2_T2_C2(C2, V, T2, cumulants, scale=1.0):
        # 68 lines

        g1 = cumulants['gamma1']
        e1 = cumulants['eta1']
        l2 = cumulants['lambda2']
        l3 = cumulants['lambda3']

        C2["ccvv"] += scale * +0.125 * np.einsum('ijuv,abwx,vx,uw->ijab', T2['ccaa'], V['vvaa'], e1, e1, optimize=True)
        C2["ccvv"] += scale * -0.125 * np.einsum('ijuv,abwx,vx,uw->ijab', T2['ccaa'], V['vvaa'], g1, g1, optimize=True)
        C2["ccvv"] += scale * -1.000 * np.einsum('iuva,wbjx,wu,vx->ijab', T2['caav'], V['avca'], e1, g1, optimize=True)
        C2["ccvv"] += scale * +1.000 * np.einsum('iuva,wbjx,vx,wu->ijab', T2['caav'], V['avca'], e1, g1, optimize=True)
        C2["ccvv"] += scale * -0.125 * np.einsum('uvab,wxij,xv,wu->ijab', T2['aavv'], V['aacc'], e1, e1, optimize=True)
        C2["ccvv"] += scale * +0.125 * np.einsum('uvab,wxij,xv,wu->ijab', T2['aavv'], V['aacc'], g1, g1, optimize=True)
        C2["aacc"] += scale * +0.250 * np.einsum('uvwa,ijxa,wx->uvij', T2['aaav'], V['ccav'], e1, optimize=True)
        C2["aacc"] += scale * +0.125 * np.einsum('uvab,ijab->uvij', T2['aavv'], V['ccvv'], optimize=True)
        C2["caav"] += scale * +1.000 * np.einsum('iuvw,xayz,xu,wz->iyva', T2['caaa'], V['avaa'], e1, g1, optimize=True)
        C2["caav"] += scale * -1.000 * np.einsum('iuvw,xayz,wz,xu->iyva', T2['caaa'], V['avaa'], e1, g1, optimize=True)
        C2["caav"] += scale * -0.500 * np.einsum('uvwa,xyiz,yv,xu->izwa', T2['aaav'], V['aaca'], e1, e1, optimize=True)
        C2["caav"] += scale * +0.500 * np.einsum('uvwa,xyiz,yv,xu->izwa', T2['aaav'], V['aaca'], g1, g1, optimize=True)
        C2["caav"] += scale * +0.500 * np.einsum('iuvw,xayz,wz,vy->iuxa', T2['caaa'], V['avaa'], e1, e1, optimize=True)
        C2["caav"] += scale * -0.500 * np.einsum('iuvw,xayz,wz,vy->iuxa', T2['caaa'], V['avaa'], g1, g1, optimize=True)
        C2["caav"] += scale * -1.000 * np.einsum('ijua,jvwx,ux->iwva', T2['ccav'], V['caaa'], e1, optimize=True)
        C2["caav"] += scale * +1.000 * np.einsum('uvwa,xyiz,yv,wz->iuxa', T2['aaav'], V['aaca'], e1, g1, optimize=True)
        C2["caav"] += scale * -1.000 * np.einsum('uvwa,xyiz,wz,yv->iuxa', T2['aaav'], V['aaca'], e1, g1, optimize=True)
        C2["caav"] += scale * +1.000 * np.einsum('ijab,juvb->ivua', T2['ccvv'], V['caav'], optimize=True)
        C2["caav"] += scale * -1.000 * np.einsum('iuab,vwxb,wu->ixva', T2['cavv'], V['aaav'], g1, optimize=True)
        C2["vvaa"] += scale * +0.125 * np.einsum('ijuv,ijab->abuv', T2['ccaa'], V['ccvv'], optimize=True)
        C2["vvaa"] += scale * +0.250 * np.einsum('iuvw,ixab,xu->abvw', T2['caaa'], V['cavv'], g1, optimize=True)
        C2["avaa"] += scale * +0.250 * np.einsum('ijuv,ijwa->wauv', T2['ccaa'], V['ccav'], optimize=True)
        C2["avaa"] += scale * +0.500 * np.einsum('iuvw,ixya,xu->yavw', T2['caaa'], V['caav'], g1, optimize=True)
        C2["avaa"] += scale * -1.000 * np.einsum('iuvw,ixya,wy->uavx', T2['caaa'], V['caav'], e1, optimize=True)
        C2["avaa"] += scale * +1.000 * np.einsum('iuva,iwba->ubvw', T2['caav'], V['cavv'], optimize=True)
        C2["avaa"] += scale * +1.000 * np.einsum('uvwa,xyba,yv->ubwx', T2['aaav'], V['aavv'], g1, optimize=True)
        C2["avca"] += scale * -1.000 * np.einsum('iuvw,jixa,wx->uajv', T2['caaa'], V['ccav'], e1, optimize=True)
        C2["avca"] += scale * +1.000 * np.einsum('iuva,jiba->ubjv', T2['caav'], V['ccvv'], optimize=True)
        C2["avca"] += scale * -1.000 * np.einsum('uvwa,ixba,xv->ubiw', T2['aaav'], V['cavv'], g1, optimize=True)
        C2["ccav"] += scale * +1.000 * np.einsum('iuvw,xajy,xu,wy->ijva', T2['caaa'], V['avca'], e1, g1, optimize=True)
        C2["ccav"] += scale * -1.000 * np.einsum('iuvw,xajy,wy,xu->ijva', T2['caaa'], V['avca'], e1, g1, optimize=True)
        C2["ccav"] += scale * -0.250 * np.einsum('uvwa,xyij,yv,xu->ijwa', T2['aaav'], V['aacc'], e1, e1, optimize=True)
        C2["ccav"] += scale * +0.250 * np.einsum('uvwa,xyij,yv,xu->ijwa', T2['aaav'], V['aacc'], g1, g1, optimize=True)
        C2["ccav"] += scale * +0.250 * np.einsum('ijuv,waxy,vy,ux->ijwa', T2['ccaa'], V['avaa'], e1, e1, optimize=True)
        C2["ccav"] += scale * -0.250 * np.einsum('ijuv,waxy,vy,ux->ijwa', T2['ccaa'], V['avaa'], g1, g1, optimize=True)
        C2["ccav"] += scale * -1.000 * np.einsum('iuva,wxjy,xu,vy->ijwa', T2['caav'], V['aaca'], e1, g1, optimize=True)
        C2["ccav"] += scale * +1.000 * np.einsum('iuva,wxjy,vy,xu->ijwa', T2['caav'], V['aaca'], e1, g1, optimize=True)
        C2["caaa"] += scale * -1.000 * np.einsum('ijuv,jwxy,vy->ixuw', T2['ccaa'], V['caaa'], e1, optimize=True)
        C2["caaa"] += scale * -1.000 * np.einsum('ijua,jvwa->iwuv', T2['ccav'], V['caav'], optimize=True)
        C2["caaa"] += scale * +1.000 * np.einsum('iuva,wxya,xu->iyvw', T2['caav'], V['aaav'], g1, optimize=True)
        C2["caaa"] += scale * +0.500 * np.einsum('iuva,wxya,vy->iuwx', T2['caav'], V['aaav'], e1, optimize=True)
        C2["caaa"] += scale * +0.250 * np.einsum('iuab,vwab->iuvw', T2['cavv'], V['aavv'], optimize=True)
        C2["aaav"] += scale * +0.250 * np.einsum('ijua,ijvw->vwua', T2['ccav'], V['ccaa'], optimize=True)
        C2["aaav"] += scale * +0.500 * np.einsum('iuva,iwxy,wu->xyva', T2['caav'], V['caaa'], g1, optimize=True)
        C2["aaav"] += scale * +1.000 * np.einsum('iuva,iwxy,vy->uxwa', T2['caav'], V['caaa'], e1, optimize=True)
        C2["aaav"] += scale * -1.000 * np.einsum('iuab,ivwb->uwva', T2['cavv'], V['caav'], optimize=True)
        C2["aaav"] += scale * -1.000 * np.einsum('uvab,wxyb,xv->uywa', T2['aavv'], V['aaav'], g1, optimize=True)
        C2["ccaa"] += scale * -1.000 * np.einsum('iuvw,xyjz,yu,wz->ijvx', T2['caaa'], V['aaca'], e1, g1, optimize=True)
        C2["ccaa"] += scale * +1.000 * np.einsum('iuvw,xyjz,wz,yu->ijvx', T2['caaa'], V['aaca'], e1, g1, optimize=True)
        C2["ccaa"] += scale * +0.250 * np.einsum('ijua,vwxa,ux->ijvw', T2['ccav'], V['aaav'], e1, optimize=True)
        C2["ccaa"] += scale * +0.125 * np.einsum('ijab,uvab->ijuv', T2['ccvv'], V['aavv'], optimize=True)
        C2["cavv"] += scale * +0.250 * np.einsum('iuvw,abxy,wy,vx->iuab', T2['caaa'], V['vvaa'], e1, e1, optimize=True)
        C2["cavv"] += scale * -0.250 * np.einsum('iuvw,abxy,wy,vx->iuab', T2['caaa'], V['vvaa'], g1, g1, optimize=True)
        C2["cavv"] += scale * -1.000 * np.einsum('iuva,wbxy,wu,vy->ixab', T2['caav'], V['avaa'], e1, g1, optimize=True)
        C2["cavv"] += scale * +1.000 * np.einsum('iuva,wbxy,vy,wu->ixab', T2['caav'], V['avaa'], e1, g1, optimize=True)
        C2["cavv"] += scale * +1.000 * np.einsum('uvwa,xbiy,xv,wy->iuab', T2['aaav'], V['avca'], e1, g1, optimize=True)
        C2["cavv"] += scale * -1.000 * np.einsum('uvwa,xbiy,wy,xv->iuab', T2['aaav'], V['avca'], e1, g1, optimize=True)
        C2["cavv"] += scale * -0.250 * np.einsum('uvab,wxiy,xv,wu->iyab', T2['aavv'], V['aaca'], e1, e1, optimize=True)
        C2["cavv"] += scale * +0.250 * np.einsum('uvab,wxiy,xv,wu->iyab', T2['aavv'], V['aaca'], g1, g1, optimize=True)
        C2["aavv"] += scale * -1.000 * np.einsum('uvwa,xbyz,xv,wz->uyab', T2['aaav'], V['avaa'], e1, g1, optimize=True)
        C2["aavv"] += scale * +1.000 * np.einsum('uvwa,xbyz,wz,xv->uyab', T2['aaav'], V['avaa'], e1, g1, optimize=True)
        C2["aavv"] += scale * +0.125 * np.einsum('ijab,ijuv->uvab', T2['ccvv'], V['ccaa'], optimize=True)
        C2["aavv"] += scale * +0.250 * np.einsum('iuab,ivwx,vu->wxab', T2['cavv'], V['caaa'], g1, optimize=True)
        C2["aaca"] += scale * +1.000 * np.einsum('iuvw,jixy,wy->uxjv', T2['caaa'], V['ccaa'], e1, optimize=True)
        C2["aaca"] += scale * +1.000 * np.einsum('iuva,jiwa->uwjv', T2['caav'], V['ccav'], optimize=True)
        C2["aaca"] += scale * -1.000 * np.einsum('uvwa,ixya,xv->uyiw', T2['aaav'], V['caav'], g1, optimize=True)
        C2["aaca"] += scale * +0.500 * np.einsum('uvwa,ixya,wy->uvix', T2['aaav'], V['caav'], e1, optimize=True)
        C2["aaca"] += scale * +0.250 * np.einsum('uvab,iwab->uviw', T2['aavv'], V['cavv'], optimize=True)
    
    @staticmethod
    def H1_T1_C1_non_od(C1, F, T1, cumulants, scale=1.0):
        # 8 lines

        g1 = cumulants['gamma1']
        e1 = cumulants['eta1']
        l2 = cumulants['lambda2']
        l3 = cumulants['lambda3']

        C1["av"] += scale * -1.000 * np.einsum('uv,wa,uw->va', F['aa'], T1['av'], e1, optimize=True)
        C1["av"] += scale * -1.000 * np.einsum('uv,wa,uw->va', F['aa'], T1['av'], g1, optimize=True)
        C1["av"] += scale * +1.000 * np.einsum('ab,ub->ua', F['vv'], T1['av'], optimize=True)
        C1["cv"] += scale * -1.000 * np.einsum('ij,ia->ja', F['cc'], T1['cv'], optimize=True)
        C1["cv"] += scale * +1.000 * np.einsum('ab,ib->ia', F['vv'], T1['cv'], optimize=True)
        C1["ca"] += scale * -1.000 * np.einsum('ij,iu->ju', F['cc'], T1['ca'], optimize=True)
        C1["ca"] += scale * +1.000 * np.einsum('uv,iw,wv->iu', F['aa'], T1['ca'], e1, optimize=True)
        C1["ca"] += scale * +1.000 * np.einsum('uv,iw,wv->iu', F['aa'], T1['ca'], g1, optimize=True)
    
    @staticmethod
    def H2_T1_C1_non_od(C1, V, T1, cumulants, scale=1.0):
        # 9 lines

        g1 = cumulants['gamma1']
        e1 = cumulants['eta1']
        l2 = cumulants['lambda2']
        l3 = cumulants['lambda3']

        C1["av"] += scale * -1.000 * np.einsum('iu,iavw,uw->va', T1['ca'], V['cvaa'], e1, optimize=True)
        C1["av"] += scale * -1.000 * np.einsum('ia,ibua->ub', T1['cv'], V['cvav'], optimize=True)
        C1["av"] += scale * -1.000 * np.einsum('ua,vbwa,vu->wb', T1['av'], V['avav'], g1, optimize=True)
        C1["cv"] += scale * -1.000 * np.einsum('iu,iajv,uv->ja', T1['ca'], V['cvca'], e1, optimize=True)
        C1["cv"] += scale * -1.000 * np.einsum('ia,ibja->jb', T1['cv'], V['cvcv'], optimize=True)
        C1["cv"] += scale * -1.000 * np.einsum('ua,vbia,vu->ib', T1['av'], V['avcv'], g1, optimize=True)
        C1["ca"] += scale * -1.000 * np.einsum('iu,ivjw,uw->jv', T1['ca'], V['caca'], e1, optimize=True)
        C1["ca"] += scale * -1.000 * np.einsum('ia,iuja->ju', T1['cv'], V['cacv'], optimize=True)
        C1["ca"] += scale * +1.000 * np.einsum('ua,vwia,wu->iv', T1['av'], V['aacv'], g1, optimize=True)
    
    @staticmethod
    def H2_T2_C1_non_od(C1, V, T2, cumulants, scale=1.0):
        # 58 lines

        g1 = cumulants['gamma1']
        e1 = cumulants['eta1']
        l2 = cumulants['lambda2']
        l3 = cumulants['lambda3']

        C1["av"] += scale * +0.500 * np.einsum('iuvw,iaxy,wy,vx->ua', T2['caaa'], V['cvaa'], e1, e1, optimize=True)
        C1["av"] += scale * +0.250 * np.einsum('iuvw,iaxy,vwxy->ua', T2['caaa'], V['cvaa'], l2, optimize=True)
        C1["av"] += scale * -0.500 * np.einsum('iuvw,iaxy,vwuy->xa', T2['caaa'], V['cvaa'], l2, optimize=True)
        C1["av"] += scale * -0.500 * np.einsum('uvwa,xyzr,yv,wxzr->ua', T2['aaav'], V['aaaa'], e1, l2, optimize=True)
        C1["av"] += scale * +0.500 * np.einsum('uvwa,xyzr,wr,xyvz->ua', T2['aaav'], V['aaaa'], e1, l2, optimize=True)
        C1["av"] += scale * -0.500 * np.einsum('uvwa,xyzr,yv,wxzr->ua', T2['aaav'], V['aaaa'], g1, l2, optimize=True)
        C1["av"] += scale * -0.500 * np.einsum('uvwa,xyzr,wz,xyvr->ua', T2['aaav'], V['aaaa'], g1, l2, optimize=True)
        C1["av"] += scale * +0.500 * np.einsum('uvwa,xyzr,yv,xu,wr->za', T2['aaav'], V['aaaa'], e1, e1, g1, optimize=True)
        C1["av"] += scale * +1.000 * np.einsum('uvwa,xyzr,yv,wxur->za', T2['aaav'], V['aaaa'], e1, l2, optimize=True)
        C1["av"] += scale * +0.500 * np.einsum('uvwa,xyzr,wr,yv,xu->za', T2['aaav'], V['aaaa'], e1, g1, g1, optimize=True)
        C1["av"] += scale * +0.250 * np.einsum('uvwa,xyzr,wr,xyuv->za', T2['aaav'], V['aaaa'], e1, l2, optimize=True)
        C1["av"] += scale * +1.000 * np.einsum('uvwa,xyzr,yv,wxur->za', T2['aaav'], V['aaaa'], g1, l2, optimize=True)
        C1["av"] += scale * +0.250 * np.einsum('uvwa,xyzr,wr,xyuv->za', T2['aaav'], V['aaaa'], g1, l2, optimize=True)
        C1["av"] += scale * +1.000 * np.einsum('iuva,ibwa,vw->ub', T2['caav'], V['cvav'], e1, optimize=True)
        C1["av"] += scale * -1.000 * np.einsum('uvwa,xbya,wy,xv->ub', T2['aaav'], V['avav'], e1, g1, optimize=True)
        C1["av"] += scale * -1.000 * np.einsum('uvwa,xbya,wxvy->ub', T2['aaav'], V['avav'], l2, optimize=True)
        C1["av"] += scale * -0.500 * np.einsum('uvwa,xbya,wxuv->yb', T2['aaav'], V['avav'], l2, optimize=True)
        C1["av"] += scale * +0.500 * np.einsum('iuab,icab->uc', T2['cavv'], V['cvvv'], optimize=True)
        C1["av"] += scale * -0.500 * np.einsum('uvab,wcab,wv->uc', T2['aavv'], V['avvv'], g1, optimize=True)
        C1["cv"] += scale * -0.500 * np.einsum('ijuv,jawx,vx,uw->ia', T2['ccaa'], V['cvaa'], e1, e1, optimize=True)
        C1["cv"] += scale * -0.250 * np.einsum('ijuv,jawx,uvwx->ia', T2['ccaa'], V['cvaa'], l2, optimize=True)
        C1["cv"] += scale * -0.500 * np.einsum('iuvw,iajx,vwux->ja', T2['caaa'], V['cvca'], l2, optimize=True)
        C1["cv"] += scale * -0.500 * np.einsum('iuva,wxyz,xu,vwyz->ia', T2['caav'], V['aaaa'], e1, l2, optimize=True)
        C1["cv"] += scale * +0.500 * np.einsum('iuva,wxyz,vz,wxuy->ia', T2['caav'], V['aaaa'], e1, l2, optimize=True)
        C1["cv"] += scale * -0.500 * np.einsum('iuva,wxyz,xu,vwyz->ia', T2['caav'], V['aaaa'], g1, l2, optimize=True)
        C1["cv"] += scale * -0.500 * np.einsum('iuva,wxyz,vy,wxuz->ia', T2['caav'], V['aaaa'], g1, l2, optimize=True)
        C1["cv"] += scale * +0.500 * np.einsum('ijua,ijkv,uv->ka', T2['ccav'], V['ccca'], e1, optimize=True)
        C1["cv"] += scale * +1.000 * np.einsum('iuva,iwjx,vx,wu->ja', T2['caav'], V['caca'], e1, g1, optimize=True)
        C1["cv"] += scale * +1.000 * np.einsum('iuva,iwjx,vwux->ja', T2['caav'], V['caca'], l2, optimize=True)
        C1["cv"] += scale * -1.000 * np.einsum('ijua,jbva,uv->ib', T2['ccav'], V['cvav'], e1, optimize=True)
        C1["cv"] += scale * -1.000 * np.einsum('iuva,wbxa,vx,wu->ib', T2['caav'], V['avav'], e1, g1, optimize=True)
        C1["cv"] += scale * -1.000 * np.einsum('iuva,wbxa,vwux->ib', T2['caav'], V['avav'], l2, optimize=True)
        C1["cv"] += scale * -0.500 * np.einsum('uvwa,xbia,wxuv->ib', T2['aaav'], V['avcv'], l2, optimize=True)
        C1["cv"] += scale * -0.500 * np.einsum('ijab,ijkb->ka', T2['ccvv'], V['cccv'], optimize=True)
        C1["cv"] += scale * -1.000 * np.einsum('iuab,ivjb,vu->ja', T2['cavv'], V['cacv'], g1, optimize=True)
        C1["cv"] += scale * -0.500 * np.einsum('uvab,wxib,xv,wu->ia', T2['aavv'], V['aacv'], g1, g1, optimize=True)
        C1["cv"] += scale * -0.250 * np.einsum('uvab,wxib,wxuv->ia', T2['aavv'], V['aacv'], l2, optimize=True)
        C1["cv"] += scale * -0.500 * np.einsum('ijab,jcab->ic', T2['ccvv'], V['cvvv'], optimize=True)
        C1["cv"] += scale * -0.500 * np.einsum('iuab,vcab,vu->ic', T2['cavv'], V['avvv'], g1, optimize=True)
        C1["ca"] += scale * +0.500 * np.einsum('iuvw,xyzr,yu,wxzr->iv', T2['caaa'], V['aaaa'], e1, l2, optimize=True)
        C1["ca"] += scale * -0.500 * np.einsum('iuvw,xyzr,wr,xyuz->iv', T2['caaa'], V['aaaa'], e1, l2, optimize=True)
        C1["ca"] += scale * +0.500 * np.einsum('iuvw,xyzr,yu,wxzr->iv', T2['caaa'], V['aaaa'], g1, l2, optimize=True)
        C1["ca"] += scale * +0.500 * np.einsum('iuvw,xyzr,wz,xyur->iv', T2['caaa'], V['aaaa'], g1, l2, optimize=True)
        C1["ca"] += scale * -0.500 * np.einsum('ijuv,ijkw,vw->ku', T2['ccaa'], V['ccca'], e1, optimize=True)
        C1["ca"] += scale * -1.000 * np.einsum('iuvw,ixjy,wy,xu->jv', T2['caaa'], V['caca'], e1, g1, optimize=True)
        C1["ca"] += scale * -1.000 * np.einsum('iuvw,ixjy,wxuy->jv', T2['caaa'], V['caca'], l2, optimize=True)
        C1["ca"] += scale * -0.500 * np.einsum('ijua,ijka->ku', T2['ccav'], V['cccv'], optimize=True)
        C1["ca"] += scale * -1.000 * np.einsum('iuva,iwja,wu->jv', T2['caav'], V['cacv'], g1, optimize=True)
        C1["ca"] += scale * -0.500 * np.einsum('uvwa,xyia,yv,xu->iw', T2['aaav'], V['aacv'], g1, g1, optimize=True)
        C1["ca"] += scale * -0.250 * np.einsum('uvwa,xyia,xyuv->iw', T2['aaav'], V['aacv'], l2, optimize=True)
        C1["ca"] += scale * +0.500 * np.einsum('iuvw,xyzr,yu,wr,vz->ix', T2['caaa'], V['aaaa'], e1, g1, g1, optimize=True)
        C1["ca"] += scale * +0.250 * np.einsum('iuvw,xyzr,yu,vwzr->ix', T2['caaa'], V['aaaa'], e1, l2, optimize=True)
        C1["ca"] += scale * +0.500 * np.einsum('iuvw,xyzr,wr,vz,yu->ix', T2['caaa'], V['aaaa'], e1, e1, g1, optimize=True)
        C1["ca"] += scale * +1.000 * np.einsum('iuvw,xyzr,wr,vyuz->ix', T2['caaa'], V['aaaa'], e1, l2, optimize=True)
        C1["ca"] += scale * +0.250 * np.einsum('iuvw,xyzr,yu,vwzr->ix', T2['caaa'], V['aaaa'], g1, l2, optimize=True)
        C1["ca"] += scale * +1.000 * np.einsum('iuvw,xyzr,wr,vyuz->ix', T2['caaa'], V['aaaa'], g1, l2, optimize=True)
        C1["ca"] += scale * -0.500 * np.einsum('iuvw,ixjy,vwuy->jx', T2['caaa'], V['caca'], l2, optimize=True)
        C1["ca"] += scale * +0.500 * np.einsum('uvwa,xyia,wyuv->ix', T2['aaav'], V['aacv'], l2, optimize=True)
    
    @staticmethod
    def H1_T2_C2_non_od(C2, F, T2, cumulants, scale=1.0):
        # 22 lines

        g1 = cumulants['gamma1']
        e1 = cumulants['eta1']
        l2 = cumulants['lambda2']
        l3 = cumulants['lambda3']

        C2["ccvv"] += scale * +0.500 * np.einsum('ij,kiab->jkab', F['cc'], T2['ccvv'], optimize=True)
        C2["ccvv"] += scale * -0.500 * np.einsum('ab,ijcb->ijac', F['vv'], T2['ccvv'], optimize=True)
        C2["caav"] += scale * -1.000 * np.einsum('ij,iuva->juva', F['cc'], T2['caav'], optimize=True)
        C2["caav"] += scale * +1.000 * np.einsum('uv,iwva->iwua', F['aa'], T2['caav'], optimize=True)
        C2["caav"] += scale * -1.000 * np.einsum('uv,iuwa->ivwa', F['aa'], T2['caav'], optimize=True)
        C2["caav"] += scale * +1.000 * np.einsum('ab,iuvb->iuva', F['vv'], T2['caav'], optimize=True)
        C2["ccav"] += scale * +1.000 * np.einsum('ij,kiua->jkua', F['cc'], T2['ccav'], optimize=True)
        C2["ccav"] += scale * +0.500 * np.einsum('uv,ijva->ijua', F['aa'], T2['ccav'], optimize=True)
        C2["ccav"] += scale * +0.500 * np.einsum('ab,ijub->ijua', F['vv'], T2['ccav'], optimize=True)
        C2["caaa"] += scale * -0.500 * np.einsum('ij,iuvw->juvw', F['cc'], T2['caaa'], optimize=True)
        C2["caaa"] += scale * -1.000 * np.einsum('uv,iwxv->iwux', F['aa'], T2['caaa'], optimize=True)
        C2["caaa"] += scale * -0.500 * np.einsum('uv,iuwx->ivwx', F['aa'], T2['caaa'], optimize=True)
        C2["aaav"] += scale * +0.500 * np.einsum('uv,wxva->wxua', F['aa'], T2['aaav'], optimize=True)
        C2["aaav"] += scale * +1.000 * np.einsum('uv,wuxa->vwxa', F['aa'], T2['aaav'], optimize=True)
        C2["aaav"] += scale * +0.500 * np.einsum('ab,uvwb->uvwa', F['vv'], T2['aaav'], optimize=True)
        C2["ccaa"] += scale * +0.500 * np.einsum('ij,kiuv->jkuv', F['cc'], T2['ccaa'], optimize=True)
        C2["ccaa"] += scale * -0.500 * np.einsum('uv,ijwv->ijuw', F['aa'], T2['ccaa'], optimize=True)
        C2["cavv"] += scale * -0.500 * np.einsum('ij,iuab->juab', F['cc'], T2['cavv'], optimize=True)
        C2["cavv"] += scale * -0.500 * np.einsum('uv,iuab->ivab', F['aa'], T2['cavv'], optimize=True)
        C2["cavv"] += scale * -1.000 * np.einsum('ab,iucb->iuac', F['vv'], T2['cavv'], optimize=True)
        C2["aavv"] += scale * +0.500 * np.einsum('uv,wuab->vwab', F['aa'], T2['aavv'], optimize=True)
        C2["aavv"] += scale * -0.500 * np.einsum('ab,uvcb->uvac', F['vv'], T2['aavv'], optimize=True)
    
    @staticmethod
    def H2_T1_C2_non_od(C2, V, T1, cumulants, scale=1.0):
        # 22 lines

        g1 = cumulants['gamma1']
        e1 = cumulants['eta1']
        l2 = cumulants['lambda2']
        l3 = cumulants['lambda3']

        C2["ccvv"] += scale * -0.500 * np.einsum('ia,ibjk->jkab', T1['cv'], V['cvcc'], optimize=True)
        C2["ccvv"] += scale * -0.500 * np.einsum('ia,bcja->ijbc', T1['cv'], V['vvcv'], optimize=True)
        C2["caav"] += scale * -1.000 * np.einsum('iu,iajv->jvua', T1['ca'], V['cvca'], optimize=True)
        C2["caav"] += scale * +1.000 * np.einsum('ia,iujv->jvua', T1['cv'], V['caca'], optimize=True)
        C2["caav"] += scale * -1.000 * np.einsum('ia,ubva->ivub', T1['cv'], V['avav'], optimize=True)
        C2["caav"] += scale * +1.000 * np.einsum('ua,vbia->iuvb', T1['av'], V['avcv'], optimize=True)
        C2["ccav"] += scale * -0.500 * np.einsum('iu,iajk->jkua', T1['ca'], V['cvcc'], optimize=True)
        C2["ccav"] += scale * +0.500 * np.einsum('ia,iujk->jkua', T1['cv'], V['cacc'], optimize=True)
        C2["ccav"] += scale * -1.000 * np.einsum('ia,ubja->ijub', T1['cv'], V['avcv'], optimize=True)
        C2["caaa"] += scale * -1.000 * np.einsum('iu,ivjw->jwuv', T1['ca'], V['caca'], optimize=True)
        C2["caaa"] += scale * -0.500 * np.einsum('iu,vwxu->ixvw', T1['ca'], V['aaaa'], optimize=True)
        C2["caaa"] += scale * +0.500 * np.einsum('ua,vwia->iuvw', T1['av'], V['aacv'], optimize=True)
        C2["aaav"] += scale * -0.500 * np.einsum('iu,iavw->vwua', T1['ca'], V['cvaa'], optimize=True)
        C2["aaav"] += scale * -0.500 * np.einsum('ua,vuwx->wxva', T1['av'], V['aaaa'], optimize=True)
        C2["aaav"] += scale * -1.000 * np.einsum('ua,vbwa->uwvb', T1['av'], V['avav'], optimize=True)
        C2["ccaa"] += scale * -0.500 * np.einsum('iu,ivjk->jkuv', T1['ca'], V['cacc'], optimize=True)
        C2["ccaa"] += scale * -0.500 * np.einsum('ia,uvja->ijuv', T1['cv'], V['aacv'], optimize=True)
        C2["cavv"] += scale * -1.000 * np.einsum('ia,ibju->juab', T1['cv'], V['cvca'], optimize=True)
        C2["cavv"] += scale * -0.500 * np.einsum('ia,bcua->iubc', T1['cv'], V['vvav'], optimize=True)
        C2["cavv"] += scale * +0.500 * np.einsum('ua,bcia->iubc', T1['av'], V['vvcv'], optimize=True)
        C2["aavv"] += scale * -0.500 * np.einsum('ia,ibuv->uvab', T1['cv'], V['cvaa'], optimize=True)
        C2["aavv"] += scale * -0.500 * np.einsum('ua,bcva->uvbc', T1['av'], V['vvav'], optimize=True)
    
    @staticmethod
    def H2_T2_C2_non_od(C2, V, T2, cumulants, scale=1.0):
        # 74 lines

        g1 = cumulants['gamma1']
        e1 = cumulants['eta1']
        l2 = cumulants['lambda2']
        l3 = cumulants['lambda3']

        C2["ccvv"] += scale * +1.000 * np.einsum('ijua,jbkv,uv->ikab', T2['ccav'], V['cvca'], e1, optimize=True)
        C2["ccvv"] += scale * +0.250 * np.einsum('ijua,bcva,uv->ijbc', T2['ccav'], V['vvav'], e1, optimize=True)
        C2["ccvv"] += scale * +0.125 * np.einsum('ijab,ijkl->klab', T2['ccvv'], V['cccc'], optimize=True)
        C2["ccvv"] += scale * +0.250 * np.einsum('iuab,ivjk,vu->jkab', T2['cavv'], V['cacc'], g1, optimize=True)
        C2["ccvv"] += scale * -1.000 * np.einsum('ijab,jckb->ikac', T2['ccvv'], V['cvcv'], optimize=True)
        C2["ccvv"] += scale * -1.000 * np.einsum('iuab,vcjb,vu->ijac', T2['cavv'], V['avcv'], g1, optimize=True)
        C2["caav"] += scale * -1.000 * np.einsum('ijuv,jawx,vx->iwua', T2['ccaa'], V['cvaa'], e1, optimize=True)
        C2["caav"] += scale * -1.000 * np.einsum('iuvw,iajx,wx->juva', T2['caaa'], V['cvca'], e1, optimize=True)
        C2["caav"] += scale * +0.500 * np.einsum('ijua,ijkv->kvua', T2['ccav'], V['ccca'], optimize=True)
        C2["caav"] += scale * +1.000 * np.einsum('iuva,iwjx,wu->jxva', T2['caav'], V['caca'], g1, optimize=True)
        C2["caav"] += scale * -1.000 * np.einsum('ijua,jbva->ivub', T2['ccav'], V['cvav'], optimize=True)
        C2["caav"] += scale * -1.000 * np.einsum('iuva,wbxa,wu->ixvb', T2['caav'], V['avav'], g1, optimize=True)
        C2["caav"] += scale * -1.000 * np.einsum('iuva,ibja->juvb', T2['caav'], V['cvcv'], optimize=True)
        C2["caav"] += scale * +1.000 * np.einsum('uvwa,xbia,xv->iuwb', T2['aaav'], V['avcv'], g1, optimize=True)
        C2["caav"] += scale * -1.000 * np.einsum('iuva,wxyz,xu,vz->iywa', T2['caav'], V['aaaa'], e1, g1, optimize=True)
        C2["caav"] += scale * +1.000 * np.einsum('iuva,wxyz,vz,xu->iywa', T2['caav'], V['aaaa'], e1, g1, optimize=True)
        C2["caav"] += scale * -1.000 * np.einsum('iuva,iwjx,vx->juwa', T2['caav'], V['caca'], e1, optimize=True)
        C2["caav"] += scale * +1.000 * np.einsum('iuva,wbxa,vx->iuwb', T2['caav'], V['avav'], e1, optimize=True)
        C2["caav"] += scale * +1.000 * np.einsum('iuab,ivjb->juva', T2['cavv'], V['cacv'], optimize=True)
        C2["caav"] += scale * +1.000 * np.einsum('uvab,wxib,xv->iuwa', T2['aavv'], V['aacv'], g1, optimize=True)
        C2["ccav"] += scale * -1.000 * np.einsum('ijuv,jakw,vw->ikua', T2['ccaa'], V['cvca'], e1, optimize=True)
        C2["ccav"] += scale * +0.250 * np.einsum('ijua,ijkl->klua', T2['ccav'], V['cccc'], optimize=True)
        C2["ccav"] += scale * +0.500 * np.einsum('iuva,iwjk,wu->jkva', T2['caav'], V['cacc'], g1, optimize=True)
        C2["ccav"] += scale * -1.000 * np.einsum('ijua,jbka->ikub', T2['ccav'], V['cvcv'], optimize=True)
        C2["ccav"] += scale * -1.000 * np.einsum('iuva,wbja,wu->ijvb', T2['caav'], V['avcv'], g1, optimize=True)
        C2["ccav"] += scale * -1.000 * np.einsum('ijua,jvkw,uw->ikva', T2['ccav'], V['caca'], e1, optimize=True)
        C2["ccav"] += scale * +0.500 * np.einsum('ijua,vbwa,uw->ijvb', T2['ccav'], V['avav'], e1, optimize=True)
        C2["ccav"] += scale * +1.000 * np.einsum('ijab,jukb->ikua', T2['ccvv'], V['cacv'], optimize=True)
        C2["ccav"] += scale * -1.000 * np.einsum('iuab,vwjb,wu->ijva', T2['cavv'], V['aacv'], g1, optimize=True)
        C2["ccav"] += scale * +0.250 * np.einsum('ijab,ucab->ijuc', T2['ccvv'], V['avvv'], optimize=True)
        C2["caaa"] += scale * +0.250 * np.einsum('ijuv,ijkw->kwuv', T2['ccaa'], V['ccca'], optimize=True)
        C2["caaa"] += scale * +0.500 * np.einsum('iuvw,ixjy,xu->jyvw', T2['caaa'], V['caca'], g1, optimize=True)
        C2["caaa"] += scale * -1.000 * np.einsum('iuvw,xyzr,yu,wr->izvx', T2['caaa'], V['aaaa'], e1, g1, optimize=True)
        C2["caaa"] += scale * +1.000 * np.einsum('iuvw,xyzr,wr,yu->izvx', T2['caaa'], V['aaaa'], e1, g1, optimize=True)
        C2["caaa"] += scale * -1.000 * np.einsum('iuvw,ixjy,wy->juvx', T2['caaa'], V['caca'], e1, optimize=True)
        C2["caaa"] += scale * -1.000 * np.einsum('iuva,iwja->juvw', T2['caav'], V['cacv'], optimize=True)
        C2["caaa"] += scale * -1.000 * np.einsum('uvwa,xyia,yv->iuwx', T2['aaav'], V['aacv'], g1, optimize=True)
        C2["caaa"] += scale * +0.250 * np.einsum('iuvw,xyzr,wr,vz->iuxy', T2['caaa'], V['aaaa'], e1, e1, optimize=True)
        C2["caaa"] += scale * -0.250 * np.einsum('iuvw,xyzr,wr,vz->iuxy', T2['caaa'], V['aaaa'], g1, g1, optimize=True)
        C2["aaav"] += scale * +1.000 * np.einsum('iuvw,iaxy,wy->uxva', T2['caaa'], V['cvaa'], e1, optimize=True)
        C2["aaav"] += scale * -0.250 * np.einsum('uvwa,xyzr,yv,xu->zrwa', T2['aaav'], V['aaaa'], e1, e1, optimize=True)
        C2["aaav"] += scale * +0.250 * np.einsum('uvwa,xyzr,yv,xu->zrwa', T2['aaav'], V['aaaa'], g1, g1, optimize=True)
        C2["aaav"] += scale * +1.000 * np.einsum('iuva,ibwa->uwvb', T2['caav'], V['cvav'], optimize=True)
        C2["aaav"] += scale * -1.000 * np.einsum('uvwa,xbya,xv->uywb', T2['aaav'], V['avav'], g1, optimize=True)
        C2["aaav"] += scale * -1.000 * np.einsum('uvwa,xyzr,yv,wr->uzxa', T2['aaav'], V['aaaa'], e1, g1, optimize=True)
        C2["aaav"] += scale * +1.000 * np.einsum('uvwa,xyzr,wr,yv->uzxa', T2['aaav'], V['aaaa'], e1, g1, optimize=True)
        C2["aaav"] += scale * +0.500 * np.einsum('uvwa,xbya,wy->uvxb', T2['aaav'], V['avav'], e1, optimize=True)
        C2["ccaa"] += scale * +0.125 * np.einsum('ijuv,ijkl->kluv', T2['ccaa'], V['cccc'], optimize=True)
        C2["ccaa"] += scale * +0.250 * np.einsum('iuvw,ixjk,xu->jkvw', T2['caaa'], V['cacc'], g1, optimize=True)
        C2["ccaa"] += scale * -1.000 * np.einsum('ijuv,jwkx,vx->ikuw', T2['ccaa'], V['caca'], e1, optimize=True)
        C2["ccaa"] += scale * -1.000 * np.einsum('ijua,jvka->ikuv', T2['ccav'], V['cacv'], optimize=True)
        C2["ccaa"] += scale * +1.000 * np.einsum('iuva,wxja,xu->ijvw', T2['caav'], V['aacv'], g1, optimize=True)
        C2["ccaa"] += scale * +0.125 * np.einsum('ijuv,wxyz,vz,uy->ijwx', T2['ccaa'], V['aaaa'], e1, e1, optimize=True)
        C2["ccaa"] += scale * -0.125 * np.einsum('ijuv,wxyz,vz,uy->ijwx', T2['ccaa'], V['aaaa'], g1, g1, optimize=True)
        C2["cavv"] += scale * +1.000 * np.einsum('ijua,jbvw,uw->ivab', T2['ccav'], V['cvaa'], e1, optimize=True)
        C2["cavv"] += scale * +1.000 * np.einsum('iuva,ibjw,vw->juab', T2['caav'], V['cvca'], e1, optimize=True)
        C2["cavv"] += scale * +0.500 * np.einsum('iuva,bcwa,vw->iubc', T2['caav'], V['vvav'], e1, optimize=True)
        C2["cavv"] += scale * +0.250 * np.einsum('ijab,ijku->kuab', T2['ccvv'], V['ccca'], optimize=True)
        C2["cavv"] += scale * +0.500 * np.einsum('iuab,ivjw,vu->jwab', T2['cavv'], V['caca'], g1, optimize=True)
        C2["cavv"] += scale * -1.000 * np.einsum('ijab,jcub->iuac', T2['ccvv'], V['cvav'], optimize=True)
        C2["cavv"] += scale * -1.000 * np.einsum('iuab,vcwb,vu->iwac', T2['cavv'], V['avav'], g1, optimize=True)
        C2["cavv"] += scale * -1.000 * np.einsum('iuab,icjb->juac', T2['cavv'], V['cvcv'], optimize=True)
        C2["cavv"] += scale * +1.000 * np.einsum('uvab,wcib,wv->iuac', T2['aavv'], V['avcv'], g1, optimize=True)
        C2["aavv"] += scale * -1.000 * np.einsum('iuva,ibwx,vx->uwab', T2['caav'], V['cvaa'], e1, optimize=True)
        C2["aavv"] += scale * -0.125 * np.einsum('uvab,wxyz,xv,wu->yzab', T2['aavv'], V['aaaa'], e1, e1, optimize=True)
        C2["aavv"] += scale * +0.125 * np.einsum('uvab,wxyz,xv,wu->yzab', T2['aavv'], V['aaaa'], g1, g1, optimize=True)
        C2["aavv"] += scale * +1.000 * np.einsum('iuab,icvb->uvac', T2['cavv'], V['cvav'], optimize=True)
        C2["aavv"] += scale * -1.000 * np.einsum('uvab,wcxb,wv->uxac', T2['aavv'], V['avav'], g1, optimize=True)

    def H2_T2_C2_non_od_large(self, C2, B, T2, cumulants, scale=1.0):
        e1 = cumulants['eta1']
        # C2["ccvv"] += scale * +0.125 * np.einsum('ijab,cdab->ijcd', T2['ccvv'], V['vvvv'], optimize=True)
        for i in range(self.ncore):
            Ci = C2["ccvv"][i,...]
            Ti = T2['ccvv'][i,...]
            for j in range(self.ncore):   
                Cij = Ci[j,...] 
                Tij = Ti[j,...]
                Cij += scale * +0.125 * np.einsum('ab,Pca,Pdb->cd', Tij, B['vv'], B['vv'], optimize='optimal')
                Cij += scale * -0.125 * np.einsum('ab,Pcb,Pda->cd', Tij, B['vv'], B['vv'], optimize='optimal')
        # C2["cavv"] += scale * +0.250 * np.einsum('iuab,cdab->iucd', T2['cavv'], V['vvvv'], optimize=True)
        for i in range(self.ncore):
            Ci = C2["cavv"][i,...]
            Ti = T2['cavv'][i,...]
            for u in range(self.nact):
                Ciu = Ci[u,...]
                Tiu = Ti[u,...]
                Ciu += scale * +0.250 * np.einsum('ab,Pca,Pdb->cd', Tiu, B['vv'], B['vv'], optimize='optimal')
                Ciu += scale * -0.250 * np.einsum('ab,Pcb,Pda->cd', Tiu, B['vv'], B['vv'], optimize='optimal')
        # C2["aavv"] += scale * +0.125 * np.einsum('uvab,cdab->uvcd', T2['aavv'], V['vvvv'], optimize=True)
        for u in range(self.nact):
            Cu = C2["aavv"][u,...]
            Tu = T2['aavv'][u,...]
            for v in range(self.nact):
                Cuv = Cu[v,...]
                Tuv = Tu[v,...]
                Cuv += scale * +0.125 * np.einsum('ab,Pca,Pdb->cd', Tuv, B['vv'], B['vv'], optimize='optimal')
                Cuv += scale * -0.125 * np.einsum('ab,Pcb,Pda->cd', Tuv, B['vv'], B['vv'], optimize='optimal')

        # C2["caav"] += scale * +0.500 * np.einsum('iuab,vcab->iuvc', T2['cavv'], V['avvv'], optimize=True)
        for i in range(self.ncore):
            Ci = C2["caav"][i,...]
            Ti = T2['cavv'][i,...]
            for u in range(self.nact):
                Ciu = Ci[u,...]
                Tiu = Ti[u,...]
                Ciu += scale * +0.500 * np.einsum('ab,Pva,Pcb->vc', Tiu, B["av"], B["vv"], optimize="optimal")
                Ciu += scale * -0.500 * np.einsum('ab,Pvb,Pca->vc', Tiu, B["av"], B["vv"], optimize="optimal")
        

        # C2["aaav"] += scale * +0.250 * np.einsum('uvab,wcab->uvwc', T2['aavv'], V['avvv'], optimize=True)
        for u in range(self.nact):
            Cu = C2["aaav"][u,...]
            Tu = T2['aavv'][u,...]
            for v in range(self.nact):
                Cuv = Cu[v,...]
                Tuv = Tu[v,...]
                Cuv += scale * +0.250 * np.einsum('ab,Pwa,Pcb->wc', Tuv, B['av'], B['vv'], optimize="optimal")
                Cuv += scale * -0.250 * np.einsum('ab,Pwb,Pca->wc', Tuv, B['av'], B['vv'], optimize="optimal")

        # C2["aavv"] += scale * +0.250 * np.einsum('uvwa,bcxa,wx->uvbc', T2['aaav'], V['vvav'], e1, optimize=True)
        for u in range(self.nact):
            Cu = C2["aavv"][u,...]
            Tu = T2['aaav'][u,...]
            for v in range(self.nact):
                Cuv = Cu[v,...]
                Tuv = Tu[v,...]
                Cuv += scale * +0.250 * np.einsum('wa,Pbx,Pca,wx->bc', Tuv, B['va'], B['vv'], e1, optimize="optimal")
                Cuv += scale * -0.250 * np.einsum('wa,Pba,Pcx,wx->bc', Tuv, B['vv'], B['va'], e1, optimize="optimal")

    # fmt: on
