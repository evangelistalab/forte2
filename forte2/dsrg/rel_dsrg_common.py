import numpy as np
from itertools import product

einsum = lambda *args, **kwargs: np.einsum(*args, **kwargs, optimize=True)


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
        # Only 31 of the 81 two-body blocks are ever contracted, so only those are
        # built. Taking every label and subtracting a handful of "large" ones used to
        # leave 76, of which 45 were never read -- including vavv, vcvv, vvva and vvvc,
        # which are O(nvirt**3) and were allocated twice (once in ints["V"], again in
        # the non-od copy). The blocks with three or four virtual indices are all
        # handled by the *_non_od_large kernels, which stream them from the B tensors
        # instead of materialising them.
        # Every od block is required: they are added into H0A1_2b wholesale. The non-od
        # blocks below are those the contraction kernels index by name; to regenerate,
        # collect the V["...."] literals in the uncommented body of this file.
        self.contracted_non_od_2_labels = set(
            [
                "aaaa",
                "aacv",
                "avav",
                "avcv",
                "caca",
                "cacc",
                "cacv",
                "ccca",
                "cccc",
                "cccv",
                "cvaa",
                "cvav",
                "cvca",
                "cvcc",
                "cvcv",
            ]
        )
        self.all_2_labels = self.od_2_labels | self.contracted_non_od_2_labels
        self.non_od_2_labels = self.all_2_labels - self.od_2_labels
        self.dims = {
            "c": self.ncore,
            "a": self.nact,
            "v": self.nvirt,
        }

    def make_tensor(self, labels):
        d = dict()
        for label in labels:
            shape = tuple(self.dims[l] for l in label)
            d[label] = np.zeros(shape, dtype=complex)
        return d

    # fmt: off
    @staticmethod
    def H_T_C0(F, V, T1, T2, cumulants, scale=1.0, store_large=True):
        # 24 lines

        g1 = cumulants['gamma1']
        e1 = cumulants['eta1']
        l2 = cumulants['lambda2']
        l3 = cumulants['lambda3']

        C0 = .0j
        C0 += scale * +1.000 * einsum('iu,iv,vu->', F['ca'], T1['ca'], e1)
        C0 += scale * -0.500 * einsum('iu,ivwx,wxuv->', F['ca'], T2['caaa'], l2)
        C0 += scale * +1.000 * einsum('ia,ia->', F['cv'], T1['cv'])
        C0 += scale * +1.000 * einsum('ua,va,uv->', F['av'], T1['av'], g1)
        C0 += scale * -0.500 * einsum('ua,vwxa,uxvw->', F['av'], T2['aaav'], l2)
        C0 += scale * -0.500 * einsum('iu,ivwx,uvwx->', T1['ca'], V['caaa'], l2)
        C0 += scale * -0.500 * einsum('ua,vwxa,vwux->', T1['av'], V['aaav'], l2)
        C0 += scale * +0.250 * einsum('ijuv,ijwx,vx,uw->', T2['ccaa'], V['ccaa'], e1, e1)
        C0 += scale * +0.125 * einsum('ijuv,ijwx,uvwx->', T2['ccaa'], V['ccaa'], l2)
        C0 += scale * +0.500 * einsum('iuvw,ixyz,wz,vy,xu->', T2['caaa'], V['caaa'], e1, e1, g1)
        C0 += scale * +1.000 * einsum('iuvw,ixyz,wz,vxuy->', T2['caaa'], V['caaa'], e1, l2)
        C0 += scale * +0.250 * einsum('iuvw,ixyz,xu,vwyz->', T2['caaa'], V['caaa'], g1, l2)
        C0 += scale * +0.250 * einsum('iuvw,ixyz,vwxuyz->', T2['caaa'], V['caaa'], l3)
        C0 += scale * +1.000 * einsum('iuva,iwxa,vx,wu->', T2['caav'], V['caav'], e1, g1)
        C0 += scale * +1.000 * einsum('iuva,iwxa,vwux->', T2['caav'], V['caav'], l2)
        C0 += scale * +0.500 * einsum('uvwa,xyza,wz,yv,xu->', T2['aaav'], V['aaav'], e1, g1, g1)
        C0 += scale * +0.250 * einsum('uvwa,xyza,wz,xyuv->', T2['aaav'], V['aaav'], e1, l2)
        C0 += scale * +1.000 * einsum('uvwa,xyza,yv,wxuz->', T2['aaav'], V['aaav'], g1, l2)
        C0 += scale * -0.250 * einsum('uvwa,xyza,wxyuvz->', T2['aaav'], V['aaav'], l3)
        C0 += scale * +0.250 * einsum('uvab,wxab,xv,wu->', T2['aavv'], V['aavv'], g1, g1)
        C0 += scale * +0.125 * einsum('uvab,wxab,wxuv->', T2['aavv'], V['aavv'], l2)

        if store_large:
            C0 += scale * +0.500 * einsum('ijua,ijva,uv->', T2['ccav'], V['ccav'], e1)
            C0 += scale * +0.250 * einsum('ijab,ijab->', T2['ccvv'], V['ccvv'])
            C0 += scale * +0.500 * einsum('iuab,ivab,vu->', T2['cavv'], V['cavv'], g1)

        return C0
    
    @staticmethod
    def H_T_C1_aa(C1, F, V, T1, T2, cumulants, scale=1.0, store_large=True):
        # 26 lines

        g1 = cumulants['gamma1']
        e1 = cumulants['eta1']
        l2 = cumulants['lambda2']
        l3 = cumulants['lambda3']

        C1 += scale * -1.000 * einsum('iu,iv->uv', F['ca'], T1['ca'])
        C1 += scale * -1.000 * einsum('iu,ivwx,xu->vw', F['ca'], T2['caaa'], e1)
        C1 += scale * -1.000 * einsum('ia,iuva->uv', F['cv'], T2['caav'])
        C1 += scale * +1.000 * einsum('ua,va->vu', F['av'], T1['av'])
        C1 += scale * +1.000 * einsum('ua,vwxa,uw->vx', F['av'], T2['aaav'], g1)
        C1 += scale * -1.000 * einsum('iu,ivwx,ux->wv', T1['ca'], V['caaa'], e1)
        C1 += scale * -1.000 * einsum('ia,iuva->vu', T1['cv'], V['caav'])
        C1 += scale * +1.000 * einsum('ua,vwxa,wu->xv', T1['av'], V['aaav'], g1)
        C1 += scale * -0.500 * einsum('ijuv,ijwx,vx->wu', T2['ccaa'], V['ccaa'], e1)
        C1 += scale * +0.500 * einsum('iuvw,ixyz,wxyz->uv', T2['caaa'], V['caaa'], l2)
        C1 += scale * -1.000 * einsum('iuvw,ixyz,wz,xu->yv', T2['caaa'], V['caaa'], e1, g1)
        C1 += scale * -1.000 * einsum('iuvw,ixyz,wxuz->yv', T2['caaa'], V['caaa'], l2)
        C1 += scale * -1.000 * einsum('iuva,iwxa,wu->xv', T2['caav'], V['caav'], g1)
        C1 += scale * -0.500 * einsum('uvwa,xyza,xyvz->uw', T2['aaav'], V['aaav'], l2)
        C1 += scale * -0.500 * einsum('uvwa,xyza,yv,xu->zw', T2['aaav'], V['aaav'], g1, g1)
        C1 += scale * -0.250 * einsum('uvwa,xyza,xyuv->zw', T2['aaav'], V['aaav'], l2)
        C1 += scale * +0.500 * einsum('iuvw,ixyz,wz,vy->ux', T2['caaa'], V['caaa'], e1, e1)
        C1 += scale * +0.250 * einsum('iuvw,ixyz,vwyz->ux', T2['caaa'], V['caaa'], l2)
        C1 += scale * -0.500 * einsum('iuvw,ixyz,vwuz->yx', T2['caaa'], V['caaa'], l2)
        C1 += scale * +1.000 * einsum('iuva,iwxa,vx->uw', T2['caav'], V['caav'], e1)
        C1 += scale * +1.000 * einsum('uvwa,xyza,wz,yv->ux', T2['aaav'], V['aaav'], e1, g1)
        C1 += scale * +1.000 * einsum('uvwa,xyza,wyvz->ux', T2['aaav'], V['aaav'], l2)
        C1 += scale * +0.500 * einsum('uvwa,xyza,wyuv->zx', T2['aaav'], V['aaav'], l2)
        C1 += scale * +0.500 * einsum('uvab,wxab,xv->uw', T2['aavv'], V['aavv'], g1)

        if store_large:
            C1 += scale * -0.500 * einsum('ijua,ijva->vu', T2['ccav'], V['ccav'])
            C1 += scale * +0.500 * einsum('iuab,ivab->uv', T2['cavv'], V['cavv'])

    
    @staticmethod
    def H_T_C2_aaaa(C2, F, V, T1, T2, cumulants, scale=1.0):
        # 11 lines

        g1 = cumulants['gamma1']
        e1 = cumulants['eta1']
        l2 = cumulants['lambda2']
        l3 = cumulants['lambda3']

        C2 += scale * -0.500 * einsum('iu,ivwx->uvwx', F['ca'], T2['caaa'])
        C2 += scale * -0.500 * einsum('ua,vwxa->vwux', F['av'], T2['aaav'])
        C2 += scale * -0.500 * einsum('iu,ivwx->wxuv', T1['ca'], V['caaa'])
        C2 += scale * -0.500 * einsum('ua,vwxa->uxvw', T1['av'], V['aaav'])
        C2 += scale * +0.125 * einsum('ijuv,ijwx->wxuv', T2['ccaa'], V['ccaa'])
        C2 += scale * +0.250 * einsum('iuvw,ixyz,xu->yzvw', T2['caaa'], V['caaa'], g1)
        C2 += scale * +1.000 * einsum('iuvw,ixyz,wz->uyvx', T2['caaa'], V['caaa'], e1)
        C2 += scale * +1.000 * einsum('iuva,iwxa->uxvw', T2['caav'], V['caav'])
        C2 += scale * +1.000 * einsum('uvwa,xyza,yv->uzwx', T2['aaav'], V['aaav'], g1)
        C2 += scale * +0.250 * einsum('uvwa,xyza,wz->uvxy', T2['aaav'], V['aaav'], e1)
        C2 += scale * +0.125 * einsum('uvab,wxab->uvwx', T2['aavv'], V['aavv'])
    # fmt: on
