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
    
    @staticmethod
    def H1_T1_C1(C1, F, T1, cumulants, scale=1.0):
        # 6 lines

        g1 = cumulants['gamma1']
        e1 = cumulants['eta1']
        l2 = cumulants['lambda2']
        l3 = cumulants['lambda3']

        C1["av"] += scale * -1.000 * einsum('iu,ia->ua', F['ca'], T1['cv'])
        C1["cv"] += scale * -1.000 * einsum('ui,ua->ia', F['ac'], T1['av'])
        C1["cv"] += scale * +1.000 * einsum('au,iu->ia', F['va'], T1['ca'])
        C1["ca"] += scale * +1.000 * einsum('ua,ia->iu', F['av'], T1['cv'])
        C1["ac"] += scale * +1.000 * einsum('ia,ua->ui', F['cv'], T1['av'])
        C1["va"] += scale * -1.000 * einsum('ia,iu->au', F['cv'], T1['ca'])
    
    @staticmethod
    def H1_T2_C1(C1, F, T2, cumulants, scale=1.0):
        # 9 lines

        g1 = cumulants['gamma1']
        e1 = cumulants['eta1']
        l2 = cumulants['lambda2']
        l3 = cumulants['lambda3']

        C1["av"] += scale * +1.000 * einsum('iu,ivwa,wu->va', F['ca'], T2['caav'], e1)
        C1["av"] += scale * -1.000 * einsum('ia,iuba->ub', F['cv'], T2['cavv'])
        C1["av"] += scale * +1.000 * einsum('ua,vwba,uw->vb', F['av'], T2['aavv'], g1)
        C1["cv"] += scale * -1.000 * einsum('iu,jiva,vu->ja', F['ca'], T2['ccav'], e1)
        C1["cv"] += scale * +1.000 * einsum('ia,jiba->jb', F['cv'], T2['ccvv'])
        C1["cv"] += scale * +1.000 * einsum('ua,ivba,uv->ib', F['av'], T2['cavv'], g1)
        C1["ca"] += scale * +1.000 * einsum('iu,jivw,wu->jv', F['ca'], T2['ccaa'], e1)
        C1["ca"] += scale * +1.000 * einsum('ia,jiua->ju', F['cv'], T2['ccav'])
        C1["ca"] += scale * +1.000 * einsum('ua,ivwa,uv->iw', F['av'], T2['caav'], g1)
    
    @staticmethod
    def H2_T1_C1(C1, V, T1, cumulants, scale=1.0):
        # 9 lines

        g1 = cumulants['gamma1']
        e1 = cumulants['eta1']
        l2 = cumulants['lambda2']
        l3 = cumulants['lambda3']

        C1["vc"] += scale * -1.000 * einsum('iu,jiva,uv->aj', T1['ca'], V['ccav'], e1)
        C1["vc"] += scale * +1.000 * einsum('ia,jiba->bj', T1['cv'], V['ccvv'])
        C1["vc"] += scale * +1.000 * einsum('ua,ivba,vu->bi', T1['av'], V['cavv'], g1)
        C1["ac"] += scale * +1.000 * einsum('iu,jivw,uw->vj', T1['ca'], V['ccaa'], e1)
        C1["ac"] += scale * +1.000 * einsum('ia,jiua->uj', T1['cv'], V['ccav'])
        C1["ac"] += scale * +1.000 * einsum('ua,ivwa,vu->wi', T1['av'], V['caav'], g1)
        C1["va"] += scale * +1.000 * einsum('iu,ivwa,uw->av', T1['ca'], V['caav'], e1)
        C1["va"] += scale * -1.000 * einsum('ia,iuba->bu', T1['cv'], V['cavv'])
        C1["va"] += scale * +1.000 * einsum('ua,vwba,wu->bv', T1['av'], V['aavv'], g1)
    
    @staticmethod
    def H2_T2_C1(C1, V, T2, cumulants, scale=1.0):
        # 52 lines

        g1 = cumulants['gamma1']
        e1 = cumulants['eta1']
        l2 = cumulants['lambda2']
        l3 = cumulants['lambda3']

        C1["av"] += scale * +0.500 * einsum('ijua,ijvw,uw->va', T2['ccav'], V['ccaa'], e1)
        C1["av"] += scale * -0.500 * einsum('iuva,iwxy,vwxy->ua', T2['caav'], V['caaa'], l2)
        C1["av"] += scale * +1.000 * einsum('iuva,iwxy,vy,wu->xa', T2['caav'], V['caaa'], e1, g1)
        C1["av"] += scale * +1.000 * einsum('iuva,iwxy,vwuy->xa', T2['caav'], V['caaa'], l2)
        C1["av"] += scale * -0.500 * einsum('ijab,ijub->ua', T2['ccvv'], V['ccav'])
        C1["av"] += scale * -1.000 * einsum('iuab,ivwb,vu->wa', T2['cavv'], V['caav'], g1)
        C1["av"] += scale * -0.500 * einsum('uvab,wxyb,wxvy->ua', T2['aavv'], V['aaav'], l2)
        C1["av"] += scale * -0.500 * einsum('uvab,wxyb,xv,wu->ya', T2['aavv'], V['aaav'], g1, g1)
        C1["av"] += scale * -0.250 * einsum('uvab,wxyb,wxuv->ya', T2['aavv'], V['aaav'], l2)
        C1["vc"] += scale * -0.500 * einsum('iuvw,jixa,vwux->aj', T2['caaa'], V['ccav'], l2)
        C1["vc"] += scale * +0.500 * einsum('uvwa,ixba,wxuv->bi', T2['aaav'], V['cavv'], l2)
        C1["cv"] += scale * -0.500 * einsum('iuvw,xayz,xu,wz,vy->ia', T2['caaa'], V['avaa'], e1, g1, g1)
        C1["cv"] += scale * -0.250 * einsum('iuvw,xayz,xu,vwyz->ia', T2['caaa'], V['avaa'], e1, l2)
        C1["cv"] += scale * -0.500 * einsum('iuvw,xayz,wz,vy,xu->ia', T2['caaa'], V['avaa'], e1, e1, g1)
        C1["cv"] += scale * -1.000 * einsum('iuvw,xayz,wz,vxuy->ia', T2['caaa'], V['avaa'], e1, l2)
        C1["cv"] += scale * -0.250 * einsum('iuvw,xayz,xu,vwyz->ia', T2['caaa'], V['avaa'], g1, l2)
        C1["cv"] += scale * -1.000 * einsum('iuvw,xayz,wz,vxuy->ia', T2['caaa'], V['avaa'], g1, l2)
        C1["cv"] += scale * +0.500 * einsum('ijua,jvwx,uvwx->ia', T2['ccav'], V['caaa'], l2)
        C1["cv"] += scale * +0.500 * einsum('uvwa,xyiz,yv,xu,wz->ia', T2['aaav'], V['aaca'], e1, e1, g1)
        C1["cv"] += scale * +1.000 * einsum('uvwa,xyiz,yv,wxuz->ia', T2['aaav'], V['aaca'], e1, l2)
        C1["cv"] += scale * +0.500 * einsum('uvwa,xyiz,wz,yv,xu->ia', T2['aaav'], V['aaca'], e1, g1, g1)
        C1["cv"] += scale * +0.250 * einsum('uvwa,xyiz,wz,xyuv->ia', T2['aaav'], V['aaca'], e1, l2)
        C1["cv"] += scale * +1.000 * einsum('uvwa,xyiz,yv,wxuz->ia', T2['aaav'], V['aaca'], g1, l2)
        C1["cv"] += scale * +0.250 * einsum('uvwa,xyiz,wz,xyuv->ia', T2['aaav'], V['aaca'], g1, l2)
        C1["cv"] += scale * -0.500 * einsum('iuab,vwxb,vwux->ia', T2['cavv'], V['aaav'], l2)
        C1["ca"] += scale * -0.500 * einsum('ijuv,jwxy,vwxy->iu', T2['ccaa'], V['caaa'], l2)
        C1["ca"] += scale * -0.500 * einsum('iuva,wxya,wxuy->iv', T2['caav'], V['aaav'], l2)
        C1["ca"] += scale * -0.500 * einsum('ijuv,jwxy,vy,ux->iw', T2['ccaa'], V['caaa'], e1, e1)
        C1["ca"] += scale * -0.250 * einsum('ijuv,jwxy,uvxy->iw', T2['ccaa'], V['caaa'], l2)
        C1["ca"] += scale * -1.000 * einsum('ijua,jvwa,uw->iv', T2['ccav'], V['caav'], e1)
        C1["ca"] += scale * +1.000 * einsum('iuva,wxya,vy,xu->iw', T2['caav'], V['aaav'], e1, g1)
        C1["ca"] += scale * +1.000 * einsum('iuva,wxya,vxuy->iw', T2['caav'], V['aaav'], l2)
        C1["ca"] += scale * -0.500 * einsum('ijab,juab->iu', T2['ccvv'], V['cavv'])
        C1["ca"] += scale * +0.500 * einsum('iuab,vwab,wu->iv', T2['cavv'], V['aavv'], g1)
        C1["ac"] += scale * -0.500 * einsum('iuvw,jixy,wy,vx->uj', T2['caaa'], V['ccaa'], e1, e1)
        C1["ac"] += scale * -0.250 * einsum('iuvw,jixy,vwxy->uj', T2['caaa'], V['ccaa'], l2)
        C1["ac"] += scale * +0.500 * einsum('iuvw,jixy,vwuy->xj', T2['caaa'], V['ccaa'], l2)
        C1["ac"] += scale * -1.000 * einsum('iuva,jiwa,vw->uj', T2['caav'], V['ccav'], e1)
        C1["ac"] += scale * +1.000 * einsum('uvwa,ixya,wy,xv->ui', T2['aaav'], V['caav'], e1, g1)
        C1["ac"] += scale * +1.000 * einsum('uvwa,ixya,wxvy->ui', T2['aaav'], V['caav'], l2)
        C1["ac"] += scale * +0.500 * einsum('uvwa,ixya,wxuv->yi', T2['aaav'], V['caav'], l2)
        C1["ac"] += scale * -0.500 * einsum('iuab,jiab->uj', T2['cavv'], V['ccvv'])
        C1["ac"] += scale * +0.500 * einsum('uvab,iwab,wv->ui', T2['aavv'], V['cavv'], g1)
        C1["va"] += scale * +0.500 * einsum('ijuv,ijwa,vw->au', T2['ccaa'], V['ccav'], e1)
        C1["va"] += scale * +1.000 * einsum('iuvw,ixya,wy,xu->av', T2['caaa'], V['caav'], e1, g1)
        C1["va"] += scale * +1.000 * einsum('iuvw,ixya,wxuy->av', T2['caaa'], V['caav'], l2)
        C1["va"] += scale * -0.500 * einsum('ijua,ijba->bu', T2['ccav'], V['ccvv'])
        C1["va"] += scale * -1.000 * einsum('iuva,iwba,wu->bv', T2['caav'], V['cavv'], g1)
        C1["va"] += scale * -0.500 * einsum('uvwa,xyba,yv,xu->bw', T2['aaav'], V['aavv'], g1, g1)
        C1["va"] += scale * -0.250 * einsum('uvwa,xyba,xyuv->bw', T2['aaav'], V['aavv'], l2)
        C1["va"] += scale * +0.500 * einsum('iuvw,ixya,vwuy->ax', T2['caaa'], V['caav'], l2)
        C1["va"] += scale * +0.500 * einsum('uvwa,xyba,wyuv->bx', T2['aaav'], V['aavv'], l2)
    
    @staticmethod
    def H1_T2_C2(C2, F, T2, cumulants, scale=1.0):
        # 22 lines

        g1 = cumulants['gamma1']
        e1 = cumulants['eta1']
        l2 = cumulants['lambda2']
        l3 = cumulants['lambda3']

        C2["ccvv"] += scale * +0.500 * einsum('ui,juab->ijab', F['ac'], T2['cavv'])
        C2["ccvv"] += scale * +0.500 * einsum('au,ijub->ijab', F['va'], T2['ccav'])
        C2["caav"] += scale * -1.000 * einsum('iu,jiva->juva', F['ca'], T2['ccav'])
        C2["caav"] += scale * -1.000 * einsum('ua,ivba->ivub', F['av'], T2['cavv'])
        C2["caav"] += scale * +1.000 * einsum('ui,vuwa->ivwa', F['ac'], T2['aaav'])
        C2["caav"] += scale * +1.000 * einsum('au,ivwu->ivwa', F['va'], T2['caaa'])
        C2["avaa"] += scale * +0.500 * einsum('ia,iuvw->uavw', F['cv'], T2['caaa'])
        C2["ccav"] += scale * -0.500 * einsum('ua,ijba->ijub', F['av'], T2['ccvv'])
        C2["ccav"] += scale * +1.000 * einsum('ui,juva->ijva', F['ac'], T2['caav'])
        C2["ccav"] += scale * +0.500 * einsum('au,ijvu->ijva', F['va'], T2['ccaa'])
        C2["caaa"] += scale * -0.500 * einsum('iu,jivw->juvw', F['ca'], T2['ccaa'])
        C2["caaa"] += scale * -1.000 * einsum('ua,ivwa->ivuw', F['av'], T2['caav'])
        C2["aaav"] += scale * -1.000 * einsum('iu,ivwa->uvwa', F['ca'], T2['caav'])
        C2["aaav"] += scale * -0.500 * einsum('ua,vwba->vwub', F['av'], T2['aavv'])
        C2["ccaa"] += scale * -0.500 * einsum('ua,ijva->ijuv', F['av'], T2['ccav'])
        C2["ccaa"] += scale * +0.500 * einsum('ui,juvw->ijvw', F['ac'], T2['caaa'])
        C2["cavv"] += scale * -0.500 * einsum('iu,jiab->juab', F['ca'], T2['ccvv'])
        C2["cavv"] += scale * +0.500 * einsum('ui,vuab->ivab', F['ac'], T2['aavv'])
        C2["cavv"] += scale * +1.000 * einsum('au,ivub->ivab', F['va'], T2['caav'])
        C2["aavv"] += scale * -0.500 * einsum('iu,ivab->uvab', F['ca'], T2['cavv'])
        C2["aavv"] += scale * +0.500 * einsum('au,vwub->vwab', F['va'], T2['aaav'])
        C2["aaca"] += scale * -0.500 * einsum('ia,uvwa->uviw', F['cv'], T2['aaav'])
    
    @staticmethod
    def H2_T1_C2(C2, V, T1, cumulants, scale=1.0):
        # 22 lines

        g1 = cumulants['gamma1']
        e1 = cumulants['eta1']
        l2 = cumulants['lambda2']
        l3 = cumulants['lambda3']

        C2["ccvv"] += scale * -0.500 * einsum('iu,abju->ijab', T1['ca'], V['vvca'])
        C2["ccvv"] += scale * -0.500 * einsum('ua,ubij->ijab', T1['av'], V['avcc'])
        C2["aacc"] += scale * -0.500 * einsum('ua,ijva->uvij', T1['av'], V['ccav'])
        C2["caav"] += scale * -1.000 * einsum('iu,vawu->iwva', T1['ca'], V['avaa'])
        C2["caav"] += scale * -1.000 * einsum('ua,vuiw->iwva', T1['av'], V['aaca'])
        C2["vvaa"] += scale * -0.500 * einsum('iu,ivab->abuv', T1['ca'], V['cavv'])
        C2["avaa"] += scale * -1.000 * einsum('iu,ivwa->wauv', T1['ca'], V['caav'])
        C2["avaa"] += scale * -0.500 * einsum('ua,vwba->ubvw', T1['av'], V['aavv'])
        C2["avca"] += scale * -1.000 * einsum('iu,jiva->vaju', T1['ca'], V['ccav'])
        C2["avca"] += scale * -1.000 * einsum('ua,ivba->ubiv', T1['av'], V['cavv'])
        C2["ccav"] += scale * -1.000 * einsum('iu,vaju->ijva', T1['ca'], V['avca'])
        C2["ccav"] += scale * -0.500 * einsum('ua,vuij->ijva', T1['av'], V['aacc'])
        C2["avcc"] += scale * -0.500 * einsum('ua,ijba->ubij', T1['av'], V['ccvv'])
        C2["caaa"] += scale * -0.500 * einsum('ia,uvwa->iwuv', T1['cv'], V['aaav'])
        C2["aaav"] += scale * +0.500 * einsum('ia,iuvw->vwua', T1['cv'], V['caaa'])
        C2["ccaa"] += scale * -0.500 * einsum('iu,vwju->ijvw', T1['ca'], V['aaca'])
        C2["cavv"] += scale * -0.500 * einsum('iu,abvu->ivab', T1['ca'], V['vvaa'])
        C2["cavv"] += scale * -1.000 * einsum('ua,ubiv->ivab', T1['av'], V['avca'])
        C2["aavv"] += scale * -0.500 * einsum('ua,ubvw->vwab', T1['av'], V['avaa'])
        C2["aaca"] += scale * -0.500 * einsum('iu,jivw->vwju', T1['ca'], V['ccaa'])
        C2["aaca"] += scale * -1.000 * einsum('ua,ivwa->uwiv', T1['av'], V['caav'])
        C2["vvca"] += scale * -0.500 * einsum('iu,jiab->abju', T1['ca'], V['ccvv'])
    
    @staticmethod
    def H2_T2_C2(C2, V, T2, cumulants, scale=1.0):
        # 68 lines

        g1 = cumulants['gamma1']
        e1 = cumulants['eta1']
        l2 = cumulants['lambda2']
        l3 = cumulants['lambda3']

        C2["ccvv"] += scale * +0.125 * einsum('ijuv,abwx,vx,uw->ijab', T2['ccaa'], V['vvaa'], e1, e1)
        C2["ccvv"] += scale * -0.125 * einsum('ijuv,abwx,vx,uw->ijab', T2['ccaa'], V['vvaa'], g1, g1)
        C2["ccvv"] += scale * -1.000 * einsum('iuva,wbjx,wu,vx->ijab', T2['caav'], V['avca'], e1, g1)
        C2["ccvv"] += scale * +1.000 * einsum('iuva,wbjx,vx,wu->ijab', T2['caav'], V['avca'], e1, g1)
        C2["ccvv"] += scale * -0.125 * einsum('uvab,wxij,xv,wu->ijab', T2['aavv'], V['aacc'], e1, e1)
        C2["ccvv"] += scale * +0.125 * einsum('uvab,wxij,xv,wu->ijab', T2['aavv'], V['aacc'], g1, g1)
        C2["aacc"] += scale * +0.250 * einsum('uvwa,ijxa,wx->uvij', T2['aaav'], V['ccav'], e1)
        C2["aacc"] += scale * +0.125 * einsum('uvab,ijab->uvij', T2['aavv'], V['ccvv'])
        C2["caav"] += scale * +1.000 * einsum('iuvw,xayz,xu,wz->iyva', T2['caaa'], V['avaa'], e1, g1)
        C2["caav"] += scale * -1.000 * einsum('iuvw,xayz,wz,xu->iyva', T2['caaa'], V['avaa'], e1, g1)
        C2["caav"] += scale * -0.500 * einsum('uvwa,xyiz,yv,xu->izwa', T2['aaav'], V['aaca'], e1, e1)
        C2["caav"] += scale * +0.500 * einsum('uvwa,xyiz,yv,xu->izwa', T2['aaav'], V['aaca'], g1, g1)
        C2["caav"] += scale * +0.500 * einsum('iuvw,xayz,wz,vy->iuxa', T2['caaa'], V['avaa'], e1, e1)
        C2["caav"] += scale * -0.500 * einsum('iuvw,xayz,wz,vy->iuxa', T2['caaa'], V['avaa'], g1, g1)
        C2["caav"] += scale * -1.000 * einsum('ijua,jvwx,ux->iwva', T2['ccav'], V['caaa'], e1)
        C2["caav"] += scale * +1.000 * einsum('uvwa,xyiz,yv,wz->iuxa', T2['aaav'], V['aaca'], e1, g1)
        C2["caav"] += scale * -1.000 * einsum('uvwa,xyiz,wz,yv->iuxa', T2['aaav'], V['aaca'], e1, g1)
        C2["caav"] += scale * +1.000 * einsum('ijab,juvb->ivua', T2['ccvv'], V['caav'])
        C2["caav"] += scale * -1.000 * einsum('iuab,vwxb,wu->ixva', T2['cavv'], V['aaav'], g1)
        C2["vvaa"] += scale * +0.125 * einsum('ijuv,ijab->abuv', T2['ccaa'], V['ccvv'])
        C2["vvaa"] += scale * +0.250 * einsum('iuvw,ixab,xu->abvw', T2['caaa'], V['cavv'], g1)
        C2["avaa"] += scale * +0.250 * einsum('ijuv,ijwa->wauv', T2['ccaa'], V['ccav'])
        C2["avaa"] += scale * +0.500 * einsum('iuvw,ixya,xu->yavw', T2['caaa'], V['caav'], g1)
        C2["avaa"] += scale * -1.000 * einsum('iuvw,ixya,wy->uavx', T2['caaa'], V['caav'], e1)
        C2["avaa"] += scale * +1.000 * einsum('iuva,iwba->ubvw', T2['caav'], V['cavv'])
        C2["avaa"] += scale * +1.000 * einsum('uvwa,xyba,yv->ubwx', T2['aaav'], V['aavv'], g1)
        C2["avca"] += scale * -1.000 * einsum('iuvw,jixa,wx->uajv', T2['caaa'], V['ccav'], e1)
        C2["avca"] += scale * +1.000 * einsum('iuva,jiba->ubjv', T2['caav'], V['ccvv'])
        C2["avca"] += scale * -1.000 * einsum('uvwa,ixba,xv->ubiw', T2['aaav'], V['cavv'], g1)
        C2["ccav"] += scale * +1.000 * einsum('iuvw,xajy,xu,wy->ijva', T2['caaa'], V['avca'], e1, g1)
        C2["ccav"] += scale * -1.000 * einsum('iuvw,xajy,wy,xu->ijva', T2['caaa'], V['avca'], e1, g1)
        C2["ccav"] += scale * -0.250 * einsum('uvwa,xyij,yv,xu->ijwa', T2['aaav'], V['aacc'], e1, e1)
        C2["ccav"] += scale * +0.250 * einsum('uvwa,xyij,yv,xu->ijwa', T2['aaav'], V['aacc'], g1, g1)
        C2["ccav"] += scale * +0.250 * einsum('ijuv,waxy,vy,ux->ijwa', T2['ccaa'], V['avaa'], e1, e1)
        C2["ccav"] += scale * -0.250 * einsum('ijuv,waxy,vy,ux->ijwa', T2['ccaa'], V['avaa'], g1, g1)
        C2["ccav"] += scale * -1.000 * einsum('iuva,wxjy,xu,vy->ijwa', T2['caav'], V['aaca'], e1, g1)
        C2["ccav"] += scale * +1.000 * einsum('iuva,wxjy,vy,xu->ijwa', T2['caav'], V['aaca'], e1, g1)
        C2["caaa"] += scale * -1.000 * einsum('ijuv,jwxy,vy->ixuw', T2['ccaa'], V['caaa'], e1)
        C2["caaa"] += scale * -1.000 * einsum('ijua,jvwa->iwuv', T2['ccav'], V['caav'])
        C2["caaa"] += scale * +1.000 * einsum('iuva,wxya,xu->iyvw', T2['caav'], V['aaav'], g1)
        C2["caaa"] += scale * +0.500 * einsum('iuva,wxya,vy->iuwx', T2['caav'], V['aaav'], e1)
        C2["caaa"] += scale * +0.250 * einsum('iuab,vwab->iuvw', T2['cavv'], V['aavv'])
        C2["aaav"] += scale * +0.250 * einsum('ijua,ijvw->vwua', T2['ccav'], V['ccaa'])
        C2["aaav"] += scale * +0.500 * einsum('iuva,iwxy,wu->xyva', T2['caav'], V['caaa'], g1)
        C2["aaav"] += scale * +1.000 * einsum('iuva,iwxy,vy->uxwa', T2['caav'], V['caaa'], e1)
        C2["aaav"] += scale * -1.000 * einsum('iuab,ivwb->uwva', T2['cavv'], V['caav'])
        C2["aaav"] += scale * -1.000 * einsum('uvab,wxyb,xv->uywa', T2['aavv'], V['aaav'], g1)
        C2["ccaa"] += scale * -1.000 * einsum('iuvw,xyjz,yu,wz->ijvx', T2['caaa'], V['aaca'], e1, g1)
        C2["ccaa"] += scale * +1.000 * einsum('iuvw,xyjz,wz,yu->ijvx', T2['caaa'], V['aaca'], e1, g1)
        C2["ccaa"] += scale * +0.250 * einsum('ijua,vwxa,ux->ijvw', T2['ccav'], V['aaav'], e1)
        C2["ccaa"] += scale * +0.125 * einsum('ijab,uvab->ijuv', T2['ccvv'], V['aavv'])
        C2["cavv"] += scale * +0.250 * einsum('iuvw,abxy,wy,vx->iuab', T2['caaa'], V['vvaa'], e1, e1)
        C2["cavv"] += scale * -0.250 * einsum('iuvw,abxy,wy,vx->iuab', T2['caaa'], V['vvaa'], g1, g1)
        C2["cavv"] += scale * -1.000 * einsum('iuva,wbxy,wu,vy->ixab', T2['caav'], V['avaa'], e1, g1)
        C2["cavv"] += scale * +1.000 * einsum('iuva,wbxy,vy,wu->ixab', T2['caav'], V['avaa'], e1, g1)
        C2["cavv"] += scale * +1.000 * einsum('uvwa,xbiy,xv,wy->iuab', T2['aaav'], V['avca'], e1, g1)
        C2["cavv"] += scale * -1.000 * einsum('uvwa,xbiy,wy,xv->iuab', T2['aaav'], V['avca'], e1, g1)
        C2["cavv"] += scale * -0.250 * einsum('uvab,wxiy,xv,wu->iyab', T2['aavv'], V['aaca'], e1, e1)
        C2["cavv"] += scale * +0.250 * einsum('uvab,wxiy,xv,wu->iyab', T2['aavv'], V['aaca'], g1, g1)
        C2["aavv"] += scale * -1.000 * einsum('uvwa,xbyz,xv,wz->uyab', T2['aaav'], V['avaa'], e1, g1)
        C2["aavv"] += scale * +1.000 * einsum('uvwa,xbyz,wz,xv->uyab', T2['aaav'], V['avaa'], e1, g1)
        C2["aavv"] += scale * +0.125 * einsum('ijab,ijuv->uvab', T2['ccvv'], V['ccaa'])
        C2["aavv"] += scale * +0.250 * einsum('iuab,ivwx,vu->wxab', T2['cavv'], V['caaa'], g1)
        C2["aaca"] += scale * +1.000 * einsum('iuvw,jixy,wy->uxjv', T2['caaa'], V['ccaa'], e1)
        C2["aaca"] += scale * +1.000 * einsum('iuva,jiwa->uwjv', T2['caav'], V['ccav'])
        C2["aaca"] += scale * -1.000 * einsum('uvwa,ixya,xv->uyiw', T2['aaav'], V['caav'], g1)
        C2["aaca"] += scale * +0.500 * einsum('uvwa,ixya,wy->uvix', T2['aaav'], V['caav'], e1)
        C2["aaca"] += scale * +0.250 * einsum('uvab,iwab->uviw', T2['aavv'], V['cavv'])
    
    @staticmethod
    def H1_T1_C1_non_od(C1, F, T1, cumulants, scale=1.0):
        # 8 lines

        g1 = cumulants['gamma1']
        e1 = cumulants['eta1']
        l2 = cumulants['lambda2']
        l3 = cumulants['lambda3']

        C1["av"] += scale * -1.000 * einsum('uv,wa,uw->va', F['aa'], T1['av'], e1)
        C1["av"] += scale * -1.000 * einsum('uv,wa,uw->va', F['aa'], T1['av'], g1)
        C1["av"] += scale * +1.000 * einsum('ab,ub->ua', F['vv'], T1['av'])
        C1["cv"] += scale * -1.000 * einsum('ij,ia->ja', F['cc'], T1['cv'])
        C1["cv"] += scale * +1.000 * einsum('ab,ib->ia', F['vv'], T1['cv'])
        C1["ca"] += scale * -1.000 * einsum('ij,iu->ju', F['cc'], T1['ca'])
        C1["ca"] += scale * +1.000 * einsum('uv,iw,wv->iu', F['aa'], T1['ca'], e1)
        C1["ca"] += scale * +1.000 * einsum('uv,iw,wv->iu', F['aa'], T1['ca'], g1)
    
    @staticmethod
    def H2_T1_C1_non_od(C1, V, T1, cumulants, scale=1.0):
        # 9 lines

        g1 = cumulants['gamma1']
        e1 = cumulants['eta1']
        l2 = cumulants['lambda2']
        l3 = cumulants['lambda3']

        C1["av"] += scale * -1.000 * einsum('iu,iavw,uw->va', T1['ca'], V['cvaa'], e1)
        C1["av"] += scale * -1.000 * einsum('ia,ibua->ub', T1['cv'], V['cvav'])
        C1["av"] += scale * -1.000 * einsum('ua,vbwa,vu->wb', T1['av'], V['avav'], g1)
        C1["cv"] += scale * -1.000 * einsum('iu,iajv,uv->ja', T1['ca'], V['cvca'], e1)
        C1["cv"] += scale * -1.000 * einsum('ia,ibja->jb', T1['cv'], V['cvcv'])
        C1["cv"] += scale * -1.000 * einsum('ua,vbia,vu->ib', T1['av'], V['avcv'], g1)
        C1["ca"] += scale * -1.000 * einsum('iu,ivjw,uw->jv', T1['ca'], V['caca'], e1)
        C1["ca"] += scale * -1.000 * einsum('ia,iuja->ju', T1['cv'], V['cacv'])
        C1["ca"] += scale * +1.000 * einsum('ua,vwia,wu->iv', T1['av'], V['aacv'], g1)
    
    @staticmethod
    def H2_T2_C1_non_od(C1, V, T2, cumulants, scale=1.0):
        # 58 lines

        g1 = cumulants['gamma1']
        e1 = cumulants['eta1']
        l2 = cumulants['lambda2']
        l3 = cumulants['lambda3']

        C1["av"] += scale * +0.500 * einsum('iuvw,iaxy,wy,vx->ua', T2['caaa'], V['cvaa'], e1, e1)
        C1["av"] += scale * +0.250 * einsum('iuvw,iaxy,vwxy->ua', T2['caaa'], V['cvaa'], l2)
        C1["av"] += scale * -0.500 * einsum('iuvw,iaxy,vwuy->xa', T2['caaa'], V['cvaa'], l2)
        C1["av"] += scale * -0.500 * einsum('uvwa,xyzr,yv,wxzr->ua', T2['aaav'], V['aaaa'], e1, l2)
        C1["av"] += scale * +0.500 * einsum('uvwa,xyzr,wr,xyvz->ua', T2['aaav'], V['aaaa'], e1, l2)
        C1["av"] += scale * -0.500 * einsum('uvwa,xyzr,yv,wxzr->ua', T2['aaav'], V['aaaa'], g1, l2)
        C1["av"] += scale * -0.500 * einsum('uvwa,xyzr,wz,xyvr->ua', T2['aaav'], V['aaaa'], g1, l2)
        C1["av"] += scale * +0.500 * einsum('uvwa,xyzr,yv,xu,wr->za', T2['aaav'], V['aaaa'], e1, e1, g1)
        C1["av"] += scale * +1.000 * einsum('uvwa,xyzr,yv,wxur->za', T2['aaav'], V['aaaa'], e1, l2)
        C1["av"] += scale * +0.500 * einsum('uvwa,xyzr,wr,yv,xu->za', T2['aaav'], V['aaaa'], e1, g1, g1)
        C1["av"] += scale * +0.250 * einsum('uvwa,xyzr,wr,xyuv->za', T2['aaav'], V['aaaa'], e1, l2)
        C1["av"] += scale * +1.000 * einsum('uvwa,xyzr,yv,wxur->za', T2['aaav'], V['aaaa'], g1, l2)
        C1["av"] += scale * +0.250 * einsum('uvwa,xyzr,wr,xyuv->za', T2['aaav'], V['aaaa'], g1, l2)
        C1["av"] += scale * +1.000 * einsum('iuva,ibwa,vw->ub', T2['caav'], V['cvav'], e1)
        C1["av"] += scale * -1.000 * einsum('uvwa,xbya,wy,xv->ub', T2['aaav'], V['avav'], e1, g1)
        C1["av"] += scale * -1.000 * einsum('uvwa,xbya,wxvy->ub', T2['aaav'], V['avav'], l2)
        C1["av"] += scale * -0.500 * einsum('uvwa,xbya,wxuv->yb', T2['aaav'], V['avav'], l2)
        C1["cv"] += scale * -0.500 * einsum('ijuv,jawx,vx,uw->ia', T2['ccaa'], V['cvaa'], e1, e1)
        C1["cv"] += scale * -0.250 * einsum('ijuv,jawx,uvwx->ia', T2['ccaa'], V['cvaa'], l2)
        C1["cv"] += scale * -0.500 * einsum('iuvw,iajx,vwux->ja', T2['caaa'], V['cvca'], l2)
        C1["cv"] += scale * -0.500 * einsum('iuva,wxyz,xu,vwyz->ia', T2['caav'], V['aaaa'], e1, l2)
        C1["cv"] += scale * +0.500 * einsum('iuva,wxyz,vz,wxuy->ia', T2['caav'], V['aaaa'], e1, l2)
        C1["cv"] += scale * -0.500 * einsum('iuva,wxyz,xu,vwyz->ia', T2['caav'], V['aaaa'], g1, l2)
        C1["cv"] += scale * -0.500 * einsum('iuva,wxyz,vy,wxuz->ia', T2['caav'], V['aaaa'], g1, l2)
        C1["cv"] += scale * +0.500 * einsum('ijua,ijkv,uv->ka', T2['ccav'], V['ccca'], e1)
        C1["cv"] += scale * +1.000 * einsum('iuva,iwjx,vx,wu->ja', T2['caav'], V['caca'], e1, g1)
        C1["cv"] += scale * +1.000 * einsum('iuva,iwjx,vwux->ja', T2['caav'], V['caca'], l2)
        C1["cv"] += scale * -1.000 * einsum('ijua,jbva,uv->ib', T2['ccav'], V['cvav'], e1)
        C1["cv"] += scale * -1.000 * einsum('iuva,wbxa,vx,wu->ib', T2['caav'], V['avav'], e1, g1)
        C1["cv"] += scale * -1.000 * einsum('iuva,wbxa,vwux->ib', T2['caav'], V['avav'], l2)
        C1["cv"] += scale * -0.500 * einsum('uvwa,xbia,wxuv->ib', T2['aaav'], V['avcv'], l2)
        C1["cv"] += scale * -0.500 * einsum('ijab,ijkb->ka', T2['ccvv'], V['cccv'])
        C1["cv"] += scale * -1.000 * einsum('iuab,ivjb,vu->ja', T2['cavv'], V['cacv'], g1)
        C1["cv"] += scale * -0.500 * einsum('uvab,wxib,xv,wu->ia', T2['aavv'], V['aacv'], g1, g1)
        C1["cv"] += scale * -0.250 * einsum('uvab,wxib,wxuv->ia', T2['aavv'], V['aacv'], l2)
        C1["ca"] += scale * +0.500 * einsum('iuvw,xyzr,yu,wxzr->iv', T2['caaa'], V['aaaa'], e1, l2)
        C1["ca"] += scale * -0.500 * einsum('iuvw,xyzr,wr,xyuz->iv', T2['caaa'], V['aaaa'], e1, l2)
        C1["ca"] += scale * +0.500 * einsum('iuvw,xyzr,yu,wxzr->iv', T2['caaa'], V['aaaa'], g1, l2)
        C1["ca"] += scale * +0.500 * einsum('iuvw,xyzr,wz,xyur->iv', T2['caaa'], V['aaaa'], g1, l2)
        C1["ca"] += scale * -0.500 * einsum('ijuv,ijkw,vw->ku', T2['ccaa'], V['ccca'], e1)
        C1["ca"] += scale * -1.000 * einsum('iuvw,ixjy,wy,xu->jv', T2['caaa'], V['caca'], e1, g1)
        C1["ca"] += scale * -1.000 * einsum('iuvw,ixjy,wxuy->jv', T2['caaa'], V['caca'], l2)
        C1["ca"] += scale * -0.500 * einsum('ijua,ijka->ku', T2['ccav'], V['cccv'])
        C1["ca"] += scale * -1.000 * einsum('iuva,iwja,wu->jv', T2['caav'], V['cacv'], g1)
        C1["ca"] += scale * -0.500 * einsum('uvwa,xyia,yv,xu->iw', T2['aaav'], V['aacv'], g1, g1)
        C1["ca"] += scale * -0.250 * einsum('uvwa,xyia,xyuv->iw', T2['aaav'], V['aacv'], l2)
        C1["ca"] += scale * +0.500 * einsum('iuvw,xyzr,yu,wr,vz->ix', T2['caaa'], V['aaaa'], e1, g1, g1)
        C1["ca"] += scale * +0.250 * einsum('iuvw,xyzr,yu,vwzr->ix', T2['caaa'], V['aaaa'], e1, l2)
        C1["ca"] += scale * +0.500 * einsum('iuvw,xyzr,wr,vz,yu->ix', T2['caaa'], V['aaaa'], e1, e1, g1)
        C1["ca"] += scale * +1.000 * einsum('iuvw,xyzr,wr,vyuz->ix', T2['caaa'], V['aaaa'], e1, l2)
        C1["ca"] += scale * +0.250 * einsum('iuvw,xyzr,yu,vwzr->ix', T2['caaa'], V['aaaa'], g1, l2)
        C1["ca"] += scale * +1.000 * einsum('iuvw,xyzr,wr,vyuz->ix', T2['caaa'], V['aaaa'], g1, l2)
        C1["ca"] += scale * -0.500 * einsum('iuvw,ixjy,vwuy->jx', T2['caaa'], V['caca'], l2)
        C1["ca"] += scale * +0.500 * einsum('uvwa,xyia,wyuv->ix', T2['aaav'], V['aacv'], l2)
    
    def H2_T2_C1_non_od_large(self, C1, B, T2, cumulants, scale=1.0):
        g1 = cumulants['gamma1']
        temp = np.zeros((max(self.nact, self.ncore), self.nvirt, self.nvirt), dtype=complex)

        # C1["av"] += scale * +0.500 * einsum('iuab,icab->uc', T2['cavv'], V['cvvv'])
        for u in range(self.nact):
            tmp = temp[:self.ncore]
            Cu = C1["av"][u]
            Tu = T2['cavv'][:, u, ...]
            np.subtract(Tu, Tu.swapaxes(1, 2), out=tmp)
            Cu += scale * +0.500 * einsum('iab,Pia,Pcb->c', tmp, B['cv'], B['vv'])

        # C1["av"] += scale * -0.500 * einsum('uvab,wcab,wv->uc', T2['aavv'], V['avvv'], g1)
            tmp = temp[:self.nact]
            Tu = T2['aavv'][u]
            np.subtract(Tu, Tu.swapaxes(1, 2), out=tmp)
            Cu += scale * -0.500 * einsum('vab,Pwa,Pcb,wv->c', tmp, B['av'], B['vv'], g1)

        # C1["cv"] += scale * -0.500 * einsum('ijab,jcab->ic', T2['ccvv'], V['cvvv'])
        for i in range(self.ncore):
            tmp = temp[:self.ncore]
            Ci = C1["cv"][i]
            Ti = T2['ccvv'][i]
            np.subtract(Ti, Ti.swapaxes(1, 2), out=tmp)
            Ci += scale * -0.500 * einsum('jab,Pja,Pcb->c', tmp, B['cv'], B['vv'])

        # C1["cv"] += scale * -0.500 * einsum('iuab,vcab,vu->ic', T2['cavv'], V['avvv'], g1)
            tmp = temp[:self.nact]
            Ti = T2['cavv'][i]
            np.subtract(Ti, Ti.swapaxes(1, 2), out=tmp)
            Ci += scale * -0.500 * einsum('uab,Pva,Pcb,vu->c', tmp, B['av'], B['vv'], g1)

    @staticmethod
    def H1_T2_C2_non_od(C2, F, T2, cumulants, scale=1.0):
        # 22 lines

        g1 = cumulants['gamma1']
        e1 = cumulants['eta1']
        l2 = cumulants['lambda2']
        l3 = cumulants['lambda3']

        C2["ccvv"] += scale * +0.500 * einsum('ij,kiab->jkab', F['cc'], T2['ccvv'])
        C2["ccvv"] += scale * -0.500 * einsum('ab,ijcb->ijac', F['vv'], T2['ccvv'])
        C2["caav"] += scale * -1.000 * einsum('ij,iuva->juva', F['cc'], T2['caav'])
        C2["caav"] += scale * +1.000 * einsum('uv,iwva->iwua', F['aa'], T2['caav'])
        C2["caav"] += scale * -1.000 * einsum('uv,iuwa->ivwa', F['aa'], T2['caav'])
        C2["caav"] += scale * +1.000 * einsum('ab,iuvb->iuva', F['vv'], T2['caav'])
        C2["ccav"] += scale * +1.000 * einsum('ij,kiua->jkua', F['cc'], T2['ccav'])
        C2["ccav"] += scale * +0.500 * einsum('uv,ijva->ijua', F['aa'], T2['ccav'])
        C2["ccav"] += scale * +0.500 * einsum('ab,ijub->ijua', F['vv'], T2['ccav'])
        C2["caaa"] += scale * -0.500 * einsum('ij,iuvw->juvw', F['cc'], T2['caaa'])
        C2["caaa"] += scale * -1.000 * einsum('uv,iwxv->iwux', F['aa'], T2['caaa'])
        C2["caaa"] += scale * -0.500 * einsum('uv,iuwx->ivwx', F['aa'], T2['caaa'])
        C2["aaav"] += scale * +0.500 * einsum('uv,wxva->wxua', F['aa'], T2['aaav'])
        C2["aaav"] += scale * +1.000 * einsum('uv,wuxa->vwxa', F['aa'], T2['aaav'])
        C2["aaav"] += scale * +0.500 * einsum('ab,uvwb->uvwa', F['vv'], T2['aaav'])
        C2["ccaa"] += scale * +0.500 * einsum('ij,kiuv->jkuv', F['cc'], T2['ccaa'])
        C2["ccaa"] += scale * -0.500 * einsum('uv,ijwv->ijuw', F['aa'], T2['ccaa'])
        C2["cavv"] += scale * -0.500 * einsum('ij,iuab->juab', F['cc'], T2['cavv'])
        C2["cavv"] += scale * -0.500 * einsum('uv,iuab->ivab', F['aa'], T2['cavv'])
        C2["cavv"] += scale * -1.000 * einsum('ab,iucb->iuac', F['vv'], T2['cavv'])
        C2["aavv"] += scale * +0.500 * einsum('uv,wuab->vwab', F['aa'], T2['aavv'])
        C2["aavv"] += scale * -0.500 * einsum('ab,uvcb->uvac', F['vv'], T2['aavv'])
    
    @staticmethod
    def H2_T1_C2_non_od(C2, V, T1, cumulants, scale=1.0):
        # 22 lines

        g1 = cumulants['gamma1']
        e1 = cumulants['eta1']
        l2 = cumulants['lambda2']
        l3 = cumulants['lambda3']

        C2["ccvv"] += scale * -0.500 * einsum('ia,ibjk->jkab', T1['cv'], V['cvcc'])
        C2["caav"] += scale * -1.000 * einsum('iu,iajv->jvua', T1['ca'], V['cvca'])
        C2["caav"] += scale * +1.000 * einsum('ia,iujv->jvua', T1['cv'], V['caca'])
        C2["caav"] += scale * -1.000 * einsum('ia,ubva->ivub', T1['cv'], V['avav'])
        C2["caav"] += scale * +1.000 * einsum('ua,vbia->iuvb', T1['av'], V['avcv'])
        C2["ccav"] += scale * -0.500 * einsum('iu,iajk->jkua', T1['ca'], V['cvcc'])
        C2["ccav"] += scale * +0.500 * einsum('ia,iujk->jkua', T1['cv'], V['cacc'])
        C2["ccav"] += scale * -1.000 * einsum('ia,ubja->ijub', T1['cv'], V['avcv'])
        C2["caaa"] += scale * -1.000 * einsum('iu,ivjw->jwuv', T1['ca'], V['caca'])
        C2["caaa"] += scale * -0.500 * einsum('iu,vwxu->ixvw', T1['ca'], V['aaaa'])
        C2["caaa"] += scale * +0.500 * einsum('ua,vwia->iuvw', T1['av'], V['aacv'])
        C2["aaav"] += scale * -0.500 * einsum('iu,iavw->vwua', T1['ca'], V['cvaa'])
        C2["aaav"] += scale * -0.500 * einsum('ua,vuwx->wxva', T1['av'], V['aaaa'])
        C2["aaav"] += scale * -1.000 * einsum('ua,vbwa->uwvb', T1['av'], V['avav'])
        C2["ccaa"] += scale * -0.500 * einsum('iu,ivjk->jkuv', T1['ca'], V['cacc'])
        C2["ccaa"] += scale * -0.500 * einsum('ia,uvja->ijuv', T1['cv'], V['aacv'])
        C2["cavv"] += scale * -1.000 * einsum('ia,ibju->juab', T1['cv'], V['cvca'])
        C2["aavv"] += scale * -0.500 * einsum('ia,ibuv->uvab', T1['cv'], V['cvaa'])
    
    def H2_T1_C2_non_od_large(self, C2, B, T1, cumulants, scale=1.0):
        # C2["ccvv"] += scale * -0.500 * einsum('ia,bcja->ijbc', T1['cv'], V['vvcv'])
        temp = np.zeros((max(self.ncore, self.nact), self.nvirt, self.nvirt), dtype=complex)
        for i in range(self.ncore):
            tmp = temp[:self.ncore]
            Ti = T1['cv'][i]
            Ci = C2["ccvv"][i]
            einsum('a,Pbj,Pca->jbc', Ti, B['vc'], B['vv'], out=tmp)
            tmp *= -0.500 * scale
            Ci += tmp
            Ci -= tmp.swapaxes(1,2)

        # C2["cavv"] += scale * -0.500 * einsum('ia,bcua->iubc', T1['cv'], V['vvav'])
            tmp = temp[:self.nact]
            Ci = C2["cavv"][i]
            einsum('a,Pbu,Pca->ubc', Ti, B['va'], B['vv'], out=tmp)
            tmp *= -0.500 * scale
            Ci += tmp 
            Ci -= tmp.swapaxes(1,2)

        # C2["cavv"] += scale * +0.500 * einsum('ua,bcia->iubc', T1['av'], V['vvcv'])
        for u in range(self.nact):
            tmp = temp[:self.ncore]
            Tu = T1['av'][u]
            Cu = C2["cavv"][:,u,...]
            einsum('a,Pbi,Pca->ibc', Tu, B['vc'], B['vv'], out=tmp)
            tmp *= +0.500 * scale
            Cu += tmp
            Cu -= tmp.swapaxes(1,2)

        # C2["aavv"] += scale * -0.500 * einsum('ua,bcva->uvbc', T1['av'], V['vvav'])
            tmp = temp[:self.nact]
            Cu = C2["aavv"][u]
            einsum('a,Pbv,Pca->vbc', Tu, B['va'], B['vv'], out=tmp)
            tmp *= -0.500 * scale
            Cu += tmp
            Cu -= tmp.swapaxes(1,2)


    @staticmethod
    def H2_T2_C2_non_od(C2, V, T2, cumulants, scale=1.0):
        # 74 lines

        g1 = cumulants['gamma1']
        e1 = cumulants['eta1']
        l2 = cumulants['lambda2']
        l3 = cumulants['lambda3']

        C2["ccvv"] += scale * +1.000 * einsum('ijua,jbkv,uv->ikab', T2['ccav'], V['cvca'], e1)
        C2["ccvv"] += scale * +0.125 * einsum('ijab,ijkl->klab', T2['ccvv'], V['cccc'])
        C2["ccvv"] += scale * +0.250 * einsum('iuab,ivjk,vu->jkab', T2['cavv'], V['cacc'], g1)
        C2["ccvv"] += scale * -1.000 * einsum('ijab,jckb->ikac', T2['ccvv'], V['cvcv'])
        C2["ccvv"] += scale * -1.000 * einsum('iuab,vcjb,vu->ijac', T2['cavv'], V['avcv'], g1)
        C2["caav"] += scale * -1.000 * einsum('ijuv,jawx,vx->iwua', T2['ccaa'], V['cvaa'], e1)
        C2["caav"] += scale * -1.000 * einsum('iuvw,iajx,wx->juva', T2['caaa'], V['cvca'], e1)
        C2["caav"] += scale * +0.500 * einsum('ijua,ijkv->kvua', T2['ccav'], V['ccca'])
        C2["caav"] += scale * +1.000 * einsum('iuva,iwjx,wu->jxva', T2['caav'], V['caca'], g1)
        C2["caav"] += scale * -1.000 * einsum('ijua,jbva->ivub', T2['ccav'], V['cvav'])
        C2["caav"] += scale * -1.000 * einsum('iuva,wbxa,wu->ixvb', T2['caav'], V['avav'], g1)
        C2["caav"] += scale * -1.000 * einsum('iuva,ibja->juvb', T2['caav'], V['cvcv'])
        C2["caav"] += scale * +1.000 * einsum('uvwa,xbia,xv->iuwb', T2['aaav'], V['avcv'], g1)
        C2["caav"] += scale * -1.000 * einsum('iuva,wxyz,xu,vz->iywa', T2['caav'], V['aaaa'], e1, g1)
        C2["caav"] += scale * +1.000 * einsum('iuva,wxyz,vz,xu->iywa', T2['caav'], V['aaaa'], e1, g1)
        C2["caav"] += scale * -1.000 * einsum('iuva,iwjx,vx->juwa', T2['caav'], V['caca'], e1)
        C2["caav"] += scale * +1.000 * einsum('iuva,wbxa,vx->iuwb', T2['caav'], V['avav'], e1)
        C2["caav"] += scale * +1.000 * einsum('iuab,ivjb->juva', T2['cavv'], V['cacv'])
        C2["caav"] += scale * +1.000 * einsum('uvab,wxib,xv->iuwa', T2['aavv'], V['aacv'], g1)
        C2["ccav"] += scale * -1.000 * einsum('ijuv,jakw,vw->ikua', T2['ccaa'], V['cvca'], e1)
        C2["ccav"] += scale * +0.250 * einsum('ijua,ijkl->klua', T2['ccav'], V['cccc'])
        C2["ccav"] += scale * +0.500 * einsum('iuva,iwjk,wu->jkva', T2['caav'], V['cacc'], g1)
        C2["ccav"] += scale * -1.000 * einsum('ijua,jbka->ikub', T2['ccav'], V['cvcv'])
        C2["ccav"] += scale * -1.000 * einsum('iuva,wbja,wu->ijvb', T2['caav'], V['avcv'], g1)
        C2["ccav"] += scale * -1.000 * einsum('ijua,jvkw,uw->ikva', T2['ccav'], V['caca'], e1)
        C2["ccav"] += scale * +0.500 * einsum('ijua,vbwa,uw->ijvb', T2['ccav'], V['avav'], e1)
        C2["ccav"] += scale * +1.000 * einsum('ijab,jukb->ikua', T2['ccvv'], V['cacv'])
        C2["ccav"] += scale * -1.000 * einsum('iuab,vwjb,wu->ijva', T2['cavv'], V['aacv'], g1)
        C2["caaa"] += scale * +0.250 * einsum('ijuv,ijkw->kwuv', T2['ccaa'], V['ccca'])
        C2["caaa"] += scale * +0.500 * einsum('iuvw,ixjy,xu->jyvw', T2['caaa'], V['caca'], g1)
        C2["caaa"] += scale * -1.000 * einsum('iuvw,xyzr,yu,wr->izvx', T2['caaa'], V['aaaa'], e1, g1)
        C2["caaa"] += scale * +1.000 * einsum('iuvw,xyzr,wr,yu->izvx', T2['caaa'], V['aaaa'], e1, g1)
        C2["caaa"] += scale * -1.000 * einsum('iuvw,ixjy,wy->juvx', T2['caaa'], V['caca'], e1)
        C2["caaa"] += scale * -1.000 * einsum('iuva,iwja->juvw', T2['caav'], V['cacv'])
        C2["caaa"] += scale * -1.000 * einsum('uvwa,xyia,yv->iuwx', T2['aaav'], V['aacv'], g1)
        C2["caaa"] += scale * +0.250 * einsum('iuvw,xyzr,wr,vz->iuxy', T2['caaa'], V['aaaa'], e1, e1)
        C2["caaa"] += scale * -0.250 * einsum('iuvw,xyzr,wr,vz->iuxy', T2['caaa'], V['aaaa'], g1, g1)
        C2["aaav"] += scale * +1.000 * einsum('iuvw,iaxy,wy->uxva', T2['caaa'], V['cvaa'], e1)
        C2["aaav"] += scale * -0.250 * einsum('uvwa,xyzr,yv,xu->zrwa', T2['aaav'], V['aaaa'], e1, e1)
        C2["aaav"] += scale * +0.250 * einsum('uvwa,xyzr,yv,xu->zrwa', T2['aaav'], V['aaaa'], g1, g1)
        C2["aaav"] += scale * +1.000 * einsum('iuva,ibwa->uwvb', T2['caav'], V['cvav'])
        C2["aaav"] += scale * -1.000 * einsum('uvwa,xbya,xv->uywb', T2['aaav'], V['avav'], g1)
        C2["aaav"] += scale * -1.000 * einsum('uvwa,xyzr,yv,wr->uzxa', T2['aaav'], V['aaaa'], e1, g1)
        C2["aaav"] += scale * +1.000 * einsum('uvwa,xyzr,wr,yv->uzxa', T2['aaav'], V['aaaa'], e1, g1)
        C2["aaav"] += scale * +0.500 * einsum('uvwa,xbya,wy->uvxb', T2['aaav'], V['avav'], e1)
        C2["ccaa"] += scale * +0.125 * einsum('ijuv,ijkl->kluv', T2['ccaa'], V['cccc'])
        C2["ccaa"] += scale * +0.250 * einsum('iuvw,ixjk,xu->jkvw', T2['caaa'], V['cacc'], g1)
        C2["ccaa"] += scale * -1.000 * einsum('ijuv,jwkx,vx->ikuw', T2['ccaa'], V['caca'], e1)
        C2["ccaa"] += scale * -1.000 * einsum('ijua,jvka->ikuv', T2['ccav'], V['cacv'])
        C2["ccaa"] += scale * +1.000 * einsum('iuva,wxja,xu->ijvw', T2['caav'], V['aacv'], g1)
        C2["ccaa"] += scale * +0.125 * einsum('ijuv,wxyz,vz,uy->ijwx', T2['ccaa'], V['aaaa'], e1, e1)
        C2["ccaa"] += scale * -0.125 * einsum('ijuv,wxyz,vz,uy->ijwx', T2['ccaa'], V['aaaa'], g1, g1)
        C2["cavv"] += scale * +1.000 * einsum('ijua,jbvw,uw->ivab', T2['ccav'], V['cvaa'], e1)
        C2["cavv"] += scale * +1.000 * einsum('iuva,ibjw,vw->juab', T2['caav'], V['cvca'], e1)
        C2["cavv"] += scale * +0.250 * einsum('ijab,ijku->kuab', T2['ccvv'], V['ccca'])
        C2["cavv"] += scale * +0.500 * einsum('iuab,ivjw,vu->jwab', T2['cavv'], V['caca'], g1)
        C2["cavv"] += scale * -1.000 * einsum('ijab,jcub->iuac', T2['ccvv'], V['cvav'])
        C2["cavv"] += scale * -1.000 * einsum('iuab,vcwb,vu->iwac', T2['cavv'], V['avav'], g1)
        C2["cavv"] += scale * -1.000 * einsum('iuab,icjb->juac', T2['cavv'], V['cvcv'])
        C2["cavv"] += scale * +1.000 * einsum('uvab,wcib,wv->iuac', T2['aavv'], V['avcv'], g1)
        C2["aavv"] += scale * -1.000 * einsum('iuva,ibwx,vx->uwab', T2['caav'], V['cvaa'], e1)
        C2["aavv"] += scale * -0.125 * einsum('uvab,wxyz,xv,wu->yzab', T2['aavv'], V['aaaa'], e1, e1)
        C2["aavv"] += scale * +0.125 * einsum('uvab,wxyz,xv,wu->yzab', T2['aavv'], V['aaaa'], g1, g1)
        C2["aavv"] += scale * +1.000 * einsum('iuab,icvb->uvac', T2['cavv'], V['cvav'])
        C2["aavv"] += scale * -1.000 * einsum('uvab,wcxb,wv->uxac', T2['aavv'], V['avav'], g1)

    def H2_T2_C2_non_od_large(self, C2, B, T2, cumulants, scale=1.0):
        e1 = cumulants['eta1']
        temp = np.zeros((self.nvirt, self.nvirt), dtype=complex)
        # C2["ccvv"] += scale * +0.125 * einsum('ijab,cdab->ijcd', T2['ccvv'], V['vvvv'])
        for i in range(self.ncore):
            Ci = C2["ccvv"][i]
            Ti = T2['ccvv'][i]
            for j in range(self.ncore):   
                Cij = Ci[j] 
                Tij = Ti[j]
                np.subtract(Tij, Tij.T, out=temp)
                Cij += scale * +0.125 * einsum('ab,Pca,Pdb->cd', temp, B['vv'], B['vv'])
        # C2["cavv"] += scale * +0.250 * einsum('iuab,cdab->iucd', T2['cavv'], V['vvvv'])
        for i in range(self.ncore):
            Ci = C2["cavv"][i]
            Ti = T2['cavv'][i]
            for u in range(self.nact):
                Ciu = Ci[u]
                Tiu = Ti[u]
                np.subtract(Tiu, Tiu.T, out=temp)
                Ciu += scale * +0.250 * einsum('ab,Pca,Pdb->cd', temp, B['vv'], B['vv'])
        # C2["aavv"] += scale * +0.125 * einsum('uvab,cdab->uvcd', T2['aavv'], V['vvvv'])
        for u in range(self.nact):
            Cu = C2["aavv"][u]
            Tu = T2['aavv'][u]
            for v in range(self.nact):
                Cuv = Cu[v]
                Tuv = Tu[v]
                np.subtract(Tuv, Tuv.T, out=temp)
                Cuv += scale * +0.125 * einsum('ab,Pca,Pdb->cd', temp, B['vv'], B['vv'])

        # C2["caav"] += scale * +0.500 * einsum('iuab,vcab->iuvc', T2['cavv'], V['avvv'])
        for i in range(self.ncore):
            Ci = C2["caav"][i]
            Ti = T2['cavv'][i]
            for u in range(self.nact):
                Ciu = Ci[u]
                Tiu = Ti[u]
                np.subtract(Tiu, Tiu.T, out=temp)
                Ciu += scale * +0.500 * einsum('ab,Pva,Pcb->vc', temp, B["av"], B["vv"])        

        # C2["aaav"] += scale * +0.250 * einsum('uvab,wcab->uvwc', T2['aavv'], V['avvv'])
        for u in range(self.nact):
            Cu = C2["aaav"][u]
            Tu = T2['aavv'][u]
            for v in range(self.nact):
                Cuv = Cu[v]
                Tuv = Tu[v]
                np.subtract(Tuv, Tuv.T, out=temp)
                Cuv += scale * +0.250 * einsum('ab,Pwa,Pcb->wc', temp, B['av'], B['vv'])


        # C2["ccav"] += scale * +0.250 * einsum('ijab,ucab->ijuc', T2['ccvv'], V['avvv'])
        for i in range(self.ncore):
            Ci = C2["ccav"][i]
            Ti = T2['ccvv'][i]
            for j in range(self.ncore):
                Cij = Ci[j]
                Tij = Ti[j]
                temp = np.subtract(Tij, Tij.T, out=temp)
                Cij += scale * +0.250 * einsum('ab,Pua,Pcb->uc', temp, B['av'], B['vv'])

        # C2["cavv"] += scale * +0.500 * einsum('iuva,bcwa,vw->iubc', T2['caav'], V['vvav'], e1)
        for i in range(self.ncore):
            Ci = C2["cavv"][i]
            Ti = T2['caav'][i]
            for u in range(self.nact):
                Ciu = Ci[u]
                Tiu = Ti[u]
                einsum('va,Pbw,Pca,vw->bc', Tiu, B['va'], B['vv'], e1, out=temp)
                temp *= scale * +0.500
                Ciu += temp
                Ciu -= temp.T

        # C2["aavv"] += scale * +0.250 * einsum('uvwa,bcxa,wx->uvbc', T2['aaav'], V['vvav'], e1)
        for u in range(self.nact):
            Cu = C2["aavv"][u]
            Tu = T2['aaav'][u]
            for v in range(self.nact):
                Cuv = Cu[v]
                Tuv = Tu[v]
                einsum('wa,Pbx,Pca,wx->bc', Tuv, B['va'], B['vv'], e1, out=temp)
                temp *= scale * +0.250
                Cuv += temp
                Cuv -= temp.T

        # C2["ccvv"] += scale * +0.250 * einsum('ijua,bcva,uv->ijbc', T2['ccav'], V['vvav'], e1)
        for i in range(self.ncore):
            Ci = C2["ccvv"][i]
            Ti = T2['ccav'][i]
            for j in range(self.ncore):
                Cij = Ci[j]
                Tij = Ti[j]
                einsum('ua,Pbv,Pca,uv->bc', Tij, B['va'], B['vv'], e1, out=temp)
                temp *= scale * +0.250
                Cij += temp
                Cij -= temp.T

    # fmt: on
