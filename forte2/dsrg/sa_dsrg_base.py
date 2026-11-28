from dataclasses import dataclass

import numpy as np

from .dsrg_base import DSRGBase
from .dsrg_common import _DSRGHelper
from .utils import (
    cas_energy_given_RDMs,
    compute_t1_block,
    compute_t2_block,
    degno_active_sf,
    renormalize_V_block,
    rotate_active_ints,
)


@dataclass
class SADSRGBase(DSRGBase):
    """
    Base class for the spin-adapted DSRG perturbation theories.

    Operators are dicts of the orbital-space blocks that `_DSRGHelper`
    describes. The routines here act on whichever two-body blocks an operator
    holds, so a method can stream a block instead of storing it.
    """

    def __post_init__(self):
        super().__post_init__()
        self.requires_attrs |= {"two_component": False}

    def get_integrals(self):
        """The reference energy, Fock blocks, orbital energies and cumulants.

        Subclasses add the two-body integrals they need.
        """
        self.dsrg_helper = _DSRGHelper(self)

        g1 = self.ci_solver.make_average_rdm(1)
        g2 = self.ci_solver.make_average_rdm(2)
        l2 = self.ci_solver.make_average_cumulant(2)
        l3 = self.ci_solver.make_average_cumulant(3)
        # self._C are the MCSCF canonical orbitals. We always use canonical orbitals to build the generalized Fock matrix.
        self.semicanonicalizer.semi_canonicalize(g1=g1, C_contig=self._C)
        # Freeze core orbitals by removing them from the semicanonicalized quantities
        # The energy contributions are accounted for in self.E_core_orig
        self._C_semican = self.semicanonicalizer.C_semican[:, self.corr].copy()
        self.fock = self.semicanonicalizer.fock_semican[self.corr, self.corr].copy()
        self.eps = self.semicanonicalizer.eps_semican[self.corr].copy()
        self.Uactv = self.semicanonicalizer.Uactv.copy()
        self._space_slices = {"c": self.core, "a": self.actv, "v": self.virt}

        cumulants = dict()
        cumulants["gamma1"] = np.einsum(
            "ip,ij,jq->pq", self.Uactv, g1, self.Uactv.conj(), optimize=True
        )
        cumulants["eta1"] = (
            2 * np.eye(cumulants["gamma1"].shape[0]) - cumulants["gamma1"]
        )
        cumulants["lambda2"] = np.einsum(
            "ip,jq,ijkl,kr,ls->pqrs",
            self.Uactv,
            self.Uactv,
            l2,
            self.Uactv.conj(),
            self.Uactv.conj(),
            optimize=True,
        )
        cumulants["lambda3"] = np.einsum(
            "ip,jq,kr,ijklmn,ls,mt,nu->pqrstu",
            self.Uactv,
            self.Uactv,
            self.Uactv,
            l3,
            self.Uactv.conj(),
            self.Uactv.conj(),
            self.Uactv.conj(),
            optimize=True,
        )

        _sl = self._space_slices
        ints = dict()
        ints["E"] = cas_energy_given_RDMs(
            self.E_core_orig, self.H_orig, self.V_orig, g1, g2
        )
        ints["F"] = {
            blk: self.fock[_sl[blk[0]], _sl[blk[1]]].copy()
            for blk in self.dsrg_helper.hp_1_labels
        }
        ints["F0"] = {
            blk: self.fock[_sl[blk[0]], _sl[blk[1]]].copy()
            for blk in self.dsrg_helper.non_od_1_labels
        }
        ints["eps"] = {x: self.eps[sl].copy() for x, sl in _sl.items()}

        return ints, cumulants

    def _b_blocks(self, pairs):
        """B[Q,p,q] for each space pair xy, with B[yx] a transposed view of B[xy]."""
        C = {x: self._C_semican[:, sl] for x, sl in self._space_slices.items()}
        B = dict()
        for x, y in pairs:
            B[x + y] = self.fock_builder.B_tensor_gen_block(C[x], C[y])
            B[y + x] = B[x + y].transpose(0, 2, 1)
        return B

    @staticmethod
    def _v_blocks(B, labels):
        """Two-electron integral blocks <pq|rs> = (pr|qs) = sum_Q B[Q,p,r] B[Q,q,s]."""
        return {
            blk: np.einsum(
                "Qpr,Qqs->pqrs", B[blk[0] + blk[2]], B[blk[1] + blk[3]], optimize=True
            )
            for blk in labels
        }

    def _build_tamps(self, F, V):
        """Amplitudes from the off-diagonal one- and two-body operators F and V."""
        eps = self.ints["eps"]
        T2 = dict()
        for blk, v in V.items():
            T2[blk] = v.copy()
            compute_t2_block(T2[blk], *(eps[x] for x in blk), self.flow_param)
        S2 = self._make_s2(T2)

        T1 = dict()
        corr = self._t1_active_correction(S2)
        for blk in self.dsrg_helper.hp_1_labels:
            T1[blk] = F[blk] + corr[blk]
            compute_t1_block(T1[blk], eps[blk[0]], eps[blk[1]], self.flow_param)
        return T1, T2, S2

    @staticmethod
    def _make_s2(T2):
        """S2[ijab] = 2 T2[ijab] - T2[ijba], block by block."""
        S2 = dict()
        for blk in T2:
            swapped = blk[:2] + blk[3] + blk[2]
            if swapped in T2:
                S2[blk] = 2 * T2[blk] - T2[swapped].swapaxes(2, 3)
            else:
                # T2[ijba] = T2[jiab] by pair symmetry
                S2[blk] = 2 * T2[blk] - T2[blk].swapaxes(0, 1)
        return S2

    def _t1_active_correction(self, S2):
        """The active generalized-Fock term shared by T1 and the renormalized F.

        0.5 * sum_vw S2[ivaw] [f, gamma1][wv], with f the active Fock block.
        """
        faa = self.ints["F0"]["aa"]
        g1 = self.cumulants["gamma1"]
        D = faa @ g1 - g1 @ faa
        return {
            "ca": 0.5 * np.einsum("ivxw,wv->ix", S2["caaa"], D, optimize=True),
            "cv": 0.5 * np.einsum("ivew,wv->ie", S2["cava"], D, optimize=True),
            "av": 0.5 * np.einsum("vywe,wv->ye", S2["aaav"], D, optimize=True),
        }

    def _renormalize(self, F, V, S2, add):
        """Scale F and V by (1 + exp(-s * D^2)) if `add`, by exp(-s * D^2) otherwise.

        D is the Moller-Plesset denominator of each element. F also picks up the
        active generalized-Fock term of the amplitudes S2 before scaling.
        """
        eps = self.ints["eps"]
        for blk in V:
            bare = None if add else V[blk].copy()
            renormalize_V_block(V[blk], *(eps[x] for x in blk), self.flow_param)
            if not add:
                V[blk] -= bare
        corr = self._t1_active_correction(S2)
        for blk in self.dsrg_helper.hp_1_labels:
            d = eps[blk[0]][:, None] - eps[blk[1]][None, :]
            f = (F[blk] + corr[blk]) * np.exp(-self.flow_param * d**2)
            if add:
                F[blk] += f
            else:
                F[blk] = f

    def _accumulate_hbar(self, F, V, T1, T2, S2, scale, store_large=True):
        """Add one contribution to the active-space Hbar.

        Each contribution is half of its Hermitian completion, which
        `_build_hbar` closes.
        """
        self.dsrg_helper.H_T_C1_aa(
            self.hbar1,
            F,
            V,
            T1,
            T2,
            S2,
            self.cumulants,
            scale=scale,
            store_large=store_large,
        )
        self.dsrg_helper.H_T_C2_aaaa(
            self.hbar2, F, V, T1, T2, S2, self.cumulants, scale=scale
        )

    def _build_hbar(self):
        # hbar1/hbar2 are left untouched so this can be called more than once
        _hbar1 = self.hbar1 + self.hbar1.T + self.fock[self.actv, self.actv]
        _hbar2 = self.hbar2 + np.einsum("ijab->abij", self.hbar2)
        _hbar2 += self.ints["V"]["aaaa"]

        self._hbar0, _hbar1 = degno_active_sf(
            self.E_dsrg,
            _hbar1,
            _hbar2,
            self.cumulants["gamma1"],
            self.cumulants["lambda2"],
        )
        self._hbar1_canon, self._hbar2_canon = rotate_active_ints(
            _hbar1, _hbar2, self.Uactv
        )

    def do_reference_relaxation(self):
        self._build_hbar()
        # The integrals are rebuilt for the relaxed reference, so release them
        # before the CI solve rather than holding both sets at once.
        for key in ("F", "F0", "V", "B"):
            self.ints.pop(key)

        self.ci_solver.set_ints(self._hbar0, self._hbar1_canon, self._hbar2_canon)
        self.ci_solver.run()
        e_relaxed = self.ci_solver.compute_average_energy()
        self.relax_eigvals = self.ci_solver.evals_flat.copy()
        return e_relaxed
