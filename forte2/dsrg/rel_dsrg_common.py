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

        self.all_1_labels = set(["".join(_) for _ in product(["c", "a", "v"], repeat=2)])
        self.non_od_1_labels = self.all_1_labels - self.od_1_labels
        self.all_2_labels = set(["".join(_) for _ in product(["c", "a", "v"], repeat=4)])
        large_labels = set(["vvvv", "avvv", "cvvv"])
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
