"""The DSRG-MRPT3 commutator terms, in the index letters of the reference.

Each entry is (coefficient, einsum subscripts, operand kinds, pair).
Kinds: H1/H2 correlated-space operators, T1/T2/S2 amplitudes, G/E/L densities.
"""

TERMS = {
    "H1_T1_C1": [
        (1.0, "ap,ia->ip", ("H1", "T1"), False),
        (-1.0, "qi,ia->qa", ("H1", "T1"), False),
    ],
    "H1_T2_C1": [
        (2.0, "bm,imab->ia", ("H1", "T2"), False),
        (-1.0, "bm,miab->ia", ("H1", "T2"), False),
        (1.0, "bu,ivab,uv->ia", ("H1", "T2", "G"), False),
        (-0.5, "bu,viab,uv->ia", ("H1", "T2", "G"), False),
        (-1.0, "vj,ijau,uv->ia", ("H1", "T2", "G"), False),
        (0.5, "vj,jiau,uv->ia", ("H1", "T2", "G"), False),
    ],
    "H2_T1_C1": [
        (2.0, "ma,qapm->qp", ("T1", "H2"), False),
        (-1.0, "ma,aqpm->qp", ("T1", "H2"), False),
        (1.0, "xe,yx,qepy->qp", ("T1", "G", "H2"), False),
        (-0.5, "xe,yx,eqpy->qp", ("T1", "G", "H2"), False),
        (-1.0, "mu,uv,qvpm->qp", ("T1", "G", "H2"), False),
        (0.5, "mu,uv,vqpm->qp", ("T1", "G", "H2"), False),
    ],
    "H2_T2_C1": [
        (1.0, "abrm,imab->ir", ("H2", "S2"), False),
        (0.5, "uv,ivab,abru->ir", ("G", "S2", "H2"), False),
        (0.25, "ijux,xy,uv,vyrj->ir", ("S2", "G", "G", "H2"), False),
        (-0.5, "uv,imub,vbrm->ir", ("G", "S2", "H2"), False),
        (-0.5, "uv,miub,bvrm->ir", ("G", "S2", "H2"), False),
        (-0.25, "iyub,uv,xy,vbrx->ir", ("S2", "G", "G", "H2"), False),
        (-0.25, "iybu,uv,xy,bvrx->ir", ("S2", "G", "G", "H2"), False),
        (0.5, "ijxy,xyuv,uvrj->ir", ("T2", "L", "H2"), False),
        (0.5, "aurx,ivay,xyuv->ir", ("H2", "S2", "L"), False),
        (-0.5, "uarx,ivay,xyuv->ir", ("H2", "T2", "L"), False),
        (-0.5, "uarx,ivya,xyvu->ir", ("H2", "T2", "L"), False),
        (-1.0, "peij,ijae->pa", ("H2", "S2"), False),
        (-0.5, "uv,ijau,pvij->pa", ("E", "S2", "H2"), False),
        (-0.25, "vyab,uv,xy,pbux->pa", ("S2", "E", "E", "H2"), False),
        (0.5, "uv,vjae,peuj->pa", ("E", "S2", "H2"), False),
        (0.5, "uv,jvae,peju->pa", ("E", "S2", "H2"), False),
        (0.25, "vjax,uv,xy,pyuj->pa", ("S2", "E", "E", "H2"), False),
        (0.25, "jvax,xy,uv,pyju->pa", ("S2", "E", "E", "H2"), False),
        (-0.5, "xyuv,uvab,pbxy->pa", ("L", "T2", "H2"), False),
        (-0.5, "puix,ivay,xyuv->pa", ("H2", "S2", "L"), False),
        (0.5, "puxi,ivay,xyuv->pa", ("H2", "T2", "L"), False),
        (0.5, "puxi,viay,xyvu->pa", ("H2", "T2", "L"), False),
        (0.5, "avxy,ujab,xyuv->jb", ("H2", "S2", "L"), False),
        (-0.5, "uviy,ijxb,xyuv->jb", ("H2", "S2", "L"), False),
        (1.0, "eqxs,uvey,xyuv->qs", ("H2", "T2", "L"), False),
        (-0.5, "eqsx,uvey,xyuv->qs", ("H2", "T2", "L"), False),
        (-1.0, "uqms,mvxy,xyuv->qs", ("H2", "T2", "L"), False),
        (0.5, "uqsm,mvxy,xyuv->qs", ("H2", "T2", "L"), False),
    ],
    "H1_T2_C2": [
        (1.0, "ijab,ap->ijpb", ("T2", "H1"), True),
        (-1.0, "ijab,qi->qjab", ("T2", "H1"), True),
    ],
    "H2_T1_C2": [
        (1.0, "ia,arpq->irpq", ("T1", "H2"), True),
        (-1.0, "ia,rsiq->rsaq", ("T1", "H2"), True),
    ],
    "H2_T2_C2": [
        (1.0, "abrs,ijab->ijrs", ("H2", "T2"), False),
        (-0.5, "xy,ijxb,ybrs->ijrs", ("G", "T2", "H2"), True),
        (1.0, "pqij,ijab->pqab", ("H2", "T2"), False),
        (-0.5, "xy,yjab,pqxj->pqab", ("E", "T2", "H2"), True),
        (1.0, "aqms,mjab->qjsb", ("H2", "S2"), True),
        (-1.0, "aqsm,mjab->qjsb", ("H2", "T2"), True),
        (0.5, "xy,yjab,aqxs->qjsb", ("G", "S2", "H2"), True),
        (-0.5, "xy,yjab,aqsx->qjsb", ("G", "T2", "H2"), True),
        (-0.5, "xy,ijxb,yqis->qjsb", ("G", "S2", "H2"), True),
        (0.5, "xy,ijxb,yqsi->qjsb", ("G", "T2", "H2"), True),
        (-1.0, "aqsm,mjba->jqsb", ("H2", "T2"), True),
        (-0.5, "xy,yjba,aqsx->jqsb", ("G", "T2", "H2"), True),
        (0.5, "xy,ijbx,yqsi->jqsb", ("G", "T2", "H2"), True),
    ],
}

LETTER = {}
for _l in "mn":
    LETTER[_l] = "c"
for _l in "uvwxyz":
    LETTER[_l] = "a"
for _l in "ef":
    LETTER[_l] = "v"
for _l in "ijkl":
    LETTER[_l] = "h"
for _l in "ab":
    LETTER[_l] = "p"
for _l in "pqrs":
    LETTER[_l] = "g"
del _l

EXPAND = {"c": "c", "a": "a", "v": "v", "h": "ca", "p": "av", "g": "cav"}
