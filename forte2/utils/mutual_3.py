import numpy as np
import itertools
from matplotlib import pyplot as plt
from itertools import combinations
import pandas as pd

def C_x(cumu2):
    DIM = cumu2.shape[0]
    norm = 0.25 * np.einsum('pqrs,pqrs->', np.conjugate(cumu2), cumu2)
    return norm


def M_2(cumu2):
    
    DIM = cumu2.shape[0]
    abs2 = np.abs(cumu2)**2
    M_pq = np.zeros((DIM,DIM),dtype=complex)
    
    M_pq +=  np.einsum('pqqq-> pq',abs2)
    M_pq += np.einsum('ppqq-> pq',abs2)*0.5
    M_pq += np.einsum('pqpq-> pq',abs2)
    M_pq += np.einsum('pppq-> pq',abs2)
    return  M_pq   


def M_3(cumu2):
    DIM = cumu2.shape[0]
    # M_pqr = np.zeros((DIM,DIM,DIM),dtype=complex)
    # M_pqr += np.einsum('pqrr,pqrr-> pqr',np.conjugate(cumu2),cumu2)
    # M_pqr += 2*np.einsum('prqr,prqr-> pqr',np.conjugate(cumu2),cumu2)
    # M_pqr += np.einsum('qrpp,qrpp-> pqr',np.conjugate(cumu2),cumu2)
    # M_pqr += 2*np.einsum('qprp,qprp-> pqr',np.conjugate(cumu2),cumu2)
    # M_pqr += np.einsum('prqq,prqq-> pqr',np.conjugate(cumu2),cumu2)
    # M_pqr += 2*np.einsum('pqrq,pqrq-> pqr',np.conjugate(cumu2),cumu2)
    abs2 = np.abs(cumu2)**2
    M_pqr = np.zeros((DIM, DIM, DIM), dtype=complex)
    M_pqr += np.einsum("pqrr->pqr", abs2)
    M_pqr += 2.0 * np.einsum("prqr->pqr", abs2)
    M_pqr += np.einsum("qrpp->pqr", abs2)
    M_pqr += 2.0 * np.einsum("qprp->pqr", abs2)
    M_pqr += np.einsum("prqq->pqr", abs2)
    M_pqr += 2.0 * np.einsum("pqrq->pqr", abs2)
    
    return M_pqr    

def M3old(cumu2):
    DIM = cumu2.shape[0]
    M_pqr = np.zeros((DIM,DIM,DIM),dtype=complex)
    for p,q,r in itertools.product(range(DIM), repeat=3):
        M_pqr[p,q,r] = cumu2[p,q,r,r]*cumu2[p,q,r,r]+2*cumu2[p,r,q,r]*cumu2[p,r,q,r]+cumu2[q,r,p,p]*cumu2[q,r,p,p]+2*cumu2[q,p,r,p]*cumu2[q,p,r,p]+cumu2[p,r,q,q]*cumu2[p,r,q,q]+2*cumu2[p,q,r,q]*cumu2[p,q,r,q]
    return M_pqr

def M_4(cumu2):
    DIM = cumu2.shape[0]
    abs2 = np.abs(cumu2)**2
    M_pqrs = np.zeros((DIM,DIM,DIM,DIM),dtype=complex)
    # Mpqrs2 = np.zeros((DIM,DIM,DIM,DIM),dtype=complex)
    M_pqrs += 2*np.einsum('pqrs->pqrs',abs2)
    M_pqrs += 2*np.einsum('prqs->pqrs',abs2)
    M_pqrs += 2*np.einsum('psqr->pqrs',abs2)

    return M_pqrs


def print_M2_table(M_pq, use_abs=False):
    rows = []

    for (p, q), val in np.ndenumerate(M_pq):

        if p < q:

            sort_val = abs(val) if use_abs else np.real(val)

            rows.append((p, q, np.real(val), sort_val))

    df = pd.DataFrame(rows, columns=["p", "q", "M_pq", "sort_val"])

    df = df.sort_values("sort_val", ascending=False).reset_index(drop=True)

    return df



def print_M3_table(M_pqr, use_abs=False):
    rows = []

    for (p, q, r), val in np.ndenumerate(M_pqr):

        if p < q < r:

            sort_val = abs(val) if use_abs else np.real(val)

            rows.append((p, q, r, np.real(val), sort_val))

    df = pd.DataFrame(rows, columns=["p", "q", "r", "M_pqr", "sort_val"])

    df = df.sort_values("sort_val", ascending=False).reset_index(drop=True)

    return df

    
    

def print_M4_table(M_pqrs, use_abs=False):
    rows = []
    for (p, q, r, s), val in np.ndenumerate(M_pqrs):
        if p < q < r < s:
            sort_val = abs(val) if use_abs else np.real(val)
        
            rows.append((p, q, r, s, np.real(val), sort_val))
        
    
    df = pd.DataFrame(rows, columns=["p", "q", "r", "s", "M_pqrs", "sort_val"])
    df = df.sort_values("sort_val", ascending=False).reset_index(drop=True)
    return df


def test_M(cumu2):
    M_pqr = M_3(cumu2)
    M_pq = M_2(cumu2)
    M_pqrs = M_4(cumu2)
    Cx = C_x(cumu2)
    DIM = cumu2.shape[0]
    
    one = 0.0
    two = 0.0
    three = 0.0
    four = 0.0
    four_6 = 0.0
    for p in range(DIM):
        one += cumu2[p, p, p, p]*np.conjugate(cumu2[p, p, p, p])

    for p, q in combinations(range(DIM), 2):
        two += M_pq[p, q]

    for p, q, r in combinations(range(DIM), 3):
        three += M_pqr[p, q, r]

    for p, q, r, s in itertools.combinations(range(DIM), 4):
        four += M_pqrs[p, q, r, s]

    return np.isclose(Cx, one + two + three + four)
    
def get_MC(cumu2):
    M1 = C_x(cumu2)
    M2 = M_2(cumu2)
    M3 = M_3(cumu2)
    M4 = M_4(cumu2)
    return M1, M2, M3, M4


def make_NO_cumulant(rdm1,rdm2):
    from forte2.ci.ci_utils import make_2cumulant_sf, make_2cumulant_so

    occs, evecs = np.linalg.eigh(rdm1)
    occs = occs[::-1]
    evecs = evecs[:, ::-1]
    rdm1_no = np.einsum(
        "pP,pq,qQ->PQ",
        evecs.conj(),
        rdm1,
        evecs,
        optimize=True,
    )
    rdm2_no = np.einsum(
            "pP,qQ,pqrs,rR,sS->PQRS",
            evecs.conj(),
            evecs.conj(),
            rdm2,
            evecs,
            evecs,
            optimize=True,
        )
    
    cumulant2_no = rdm2_no - np.einsum("ik,jl->ijkl", rdm1_no, rdm1_no, optimize=True) + np.einsum("il,jk->ijkl", rdm1_no, rdm1_no, optimize=True)
    return cumulant2_no
    