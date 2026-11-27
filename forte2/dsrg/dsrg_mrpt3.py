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
class DSRG_MRPT3(DSRGBase):
    """
    Spin-adapted driven similarity renormalization group
    third-order multireference perturbation theory (DSRG-MRPT3).

    The energy is the second-order energy plus three third-order terms,

    .. math::
        E^{(3)}_1 &= -\\tfrac{1}{12} \\langle [[[\\hat{H}^{(0)}, \\hat{A}^{(1)}], \\hat{A}^{(1)}], \\hat{A}^{(1)}] \\rangle \\\\
        E^{(3)}_2 &= \\tfrac{1}{2} \\langle [\\hat{H}^{(1)} + \\bar{H}^{(1)}, \\hat{A}^{(2)}] \\rangle \\\\
        E^{(3)}_3 &= \\tfrac{1}{2} \\langle [\\bar{H}^{(2)}, \\hat{A}^{(1)}] \\rangle

    with :math:`\\bar{H}^{(1)} = \\hat{H}^{(1)} + [\\hat{H}^{(0)}, \\hat{A}^{(1)}]` and
    :math:`\\bar{H}^{(2)} = \\tfrac{1}{2} [\\hat{H}^{(1)} + \\bar{H}^{(1)}, \\hat{A}^{(1)}] + [\\hat{H}^{(0)}, \\hat{A}^{(2)}]`.

    The two-electron integrals are density fitted. Blocks with three or four
    virtual indices are never formed: every contraction that reaches one is
    evaluated directly from the three-index integrals.

    Parameters
    ----------
    flow_param : float, optional, default=0.5
        The flow parameter (in atomic units) that controls the renormalization.
    relax_reference : int | str | bool, optional, default=False
        Relax the CI reference in response to dynamical correlation.
        If an integer is given, it specifies the maximum number of relaxation iterations.
        If a string is given, it must be one of 'once', 'twice', or 'iterate':
            'once' : diagonalize the CI Hamiltonian once after computing the DSRG energy
            'twice': after the first diagonalization, recompute the DSRG energy
            'iterate': keep relaxing until convergence or reaching relax_maxiter.
        If a boolean is given, True is equivalent to relax_maxiter and False means no relaxation.
    relax_maxiter : int, optional, default=10
        The maximum number of reference relaxation iterations.
    relax_tol : float, optional, default=1e-6
        The convergence tolerance for reference relaxation (in Hartree).

    Attributes
    ----------
    E_dsrg : float
        The DSRG-MRPT3 total energy evaluated with the current reference.
    e_dsrg_mrpt2 : float
        The second-order correlation energy.
    e_dsrg_mrpt3_1, e_dsrg_mrpt3_2, e_dsrg_mrpt3_3 : float
        The three third-order correlation energy terms.
    E_relaxed_ref : float
        The DSRG-MRPT3 total energy after reference relaxation.
    relax_energies : NDArray
        The history of DSRG-MRPT3 total energies during reference relaxation.
        Given as [[Edsrg(fixed_reference), Edsrg(relaxed_reference), Eref], ...].
    relax_eigvals : np.ndarray
        The eigenvalues of the relaxed CI Hamiltonian.
    relax_eigvals_history : NDArray
        The history of eigenvalues of the relaxed CI Hamiltonian during relaxation.

    References
    ----------
    .. [1] C. Li and F. A. Evangelista, "Driven similarity renormalization group: Third-order multireference perturbation theory",
           J. Chem. Phys. 2017, 146, 124132.
    .. [2] C. Li and F. A. Evangelista, "Driven similarity renormalization group for excited states: A state-averaged perturbation theory",
           J. Chem. Phys. 2018, 148, 124106.
    """

    def __post_init__(self):
        super().__post_init__()
        self.requires_attrs |= {"two_component": False}

    def get_integrals(self):
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

        ints = dict()
        ints["E"] = cas_energy_given_RDMs(
            self.E_core_orig, self.H_orig, self.V_orig, g1, g2
        )

        _sl = {"c": self.core, "a": self.actv, "v": self.virt}
        ints["F"] = {
            blk: self.fock[_sl[blk[0]], _sl[blk[1]]].copy()
            for blk in self.dsrg_helper.hp_1_labels
        }
        ints["F0"] = {
            blk: self.fock[_sl[blk[0]], _sl[blk[1]]].copy()
            for blk in self.dsrg_helper.non_od_1_labels
        }

        # <pq|rs> = (pr|qs) = sum_Q B[Q,p,r] B[Q,q,s], so a block needs the B
        # blocks of the space pairs (p,r) and (q,s). B[yx] is a transpose of B[xy].
        C = {x: self._C_semican[:, sl] for x, sl in _sl.items()}
        B = dict()
        for xy in ["cc", "ca", "cv", "aa", "av", "vv"]:
            x, y = xy
            B[xy] = self.fock_builder.B_tensor_gen_block(C[x], C[y])
            B[y + x] = B[xy].transpose(0, 2, 1)
        ints["V"] = {
            blk: np.einsum(
                "Qpr,Qqs->pqrs", B[blk[0] + blk[2]], B[blk[1] + blk[3]], optimize=True
            )
            for blk in self.dsrg_helper.hp_2_labels + self.dsrg_helper.non_od_2_labels
        }
        # what the streamed kernels contract the many-virtual blocks from
        ints["B"] = {"vv": B["vv"], "cv": B["cv"], "av": B["av"]}

        ints["eps"] = {x: self.eps[sl].copy() for x, sl in _sl.items()}

        return ints, cumulants

    def solve_dsrg(self, form_hbar=False):
        if form_hbar:
            self.hbar1 = np.zeros((self.nact,) * 2)
            self.hbar2 = np.zeros((self.nact,) * 4)

        self.T1_1, self.T2_1, self.S2_1 = self._build_tamps(
            self.ints["F"],
            {blk: self.ints["V"][blk] for blk in self.dsrg_helper.hp_2_labels},
        )

        # order matters: each stage consumes intermediates of the previous one
        self.e_dsrg_mrpt3_1 = self._compute_energy_pt3_1(form_hbar)
        self.e_dsrg_mrpt2 = self._compute_energy_pt2(form_hbar)
        self.e_dsrg_mrpt3_2 = self._compute_energy_pt3_2(form_hbar)
        self.e_dsrg_mrpt3_3 = self._compute_energy_pt3_3(form_hbar)

        return (
            self.ints["E"]
            + self.e_dsrg_mrpt2
            + self.e_dsrg_mrpt3_1
            + self.e_dsrg_mrpt3_2
            + self.e_dsrg_mrpt3_3
        )

    def _build_tamps(self, F, V):
        """Amplitudes from the off-diagonal one- and two-body operators F and V."""
        eps = self.ints["eps"]
        T2 = dict()
        for blk in self.dsrg_helper.hp_2_labels:
            T2[blk] = V[blk].copy()
            compute_t2_block(T2[blk], *(eps[x] for x in blk), self.flow_param)
        S2 = self._make_s2(T2)

        T1 = dict()
        corr = self._t1_active_correction(S2)
        for blk in self.dsrg_helper.hp_1_labels:
            T1[blk] = F[blk] + corr[blk]
            compute_t1_block(T1[blk], eps[blk[0]], eps[blk[1]], self.flow_param)
        return T1, T2, S2

    def _make_s2(self, T2):
        """S2[ijab] = 2 T2[ijab] - T2[ijba], block by block."""
        S2 = dict()
        for blk in self.dsrg_helper.hp_2_labels:
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
        for blk in self.dsrg_helper.hp_2_labels:
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

    def _accumulate_hbar(self, F, V, T1, T2, S2, scale):
        """Add one stage's contribution to the active-space Hbar.

        Each stage contributes half of its Hermitian completion, which
        `_build_hbar` closes once all four stages have accumulated.
        """
        self.dsrg_helper.H_T_C1_aa(
            self.hbar1, F, V, T1, T2, S2, self.cumulants, scale=scale
        )
        self.dsrg_helper.H_T_C2_aaaa(
            self.hbar2, F, V, T1, T2, S2, self.cumulants, scale=scale
        )

    def _commutator_od(self, C1, C2, F, V, T1, T2, S2, scale=1.0):
        """[H, A] for an off-diagonal H = F + V."""
        h = self.dsrg_helper
        cu = self.cumulants
        h.H1_T1_C1(C1, F, T1, cu, scale=scale)
        h.H1_T2_C1(C1, F, T2, cu, scale=scale)
        h.H2_T1_C1(C1, V, T1, cu, scale=scale)
        h.H2_T2_C1(C1, V, T2, S2, cu, scale=scale)
        h.H1_T2_C2(C2, F, T2, cu, scale=scale)
        h.H2_T1_C2(C2, V, T1, cu, scale=scale)
        h.H2_T2_C2(C2, V, T2, S2, cu, scale=scale)

    def _compute_energy_pt3_1(self, form_hbar):
        """-1/12 <[[[H0th, A1st], A1st], A1st]>."""
        h = self.dsrg_helper
        cu = self.cumulants
        F0 = self.ints["F0"]

        # -[H0th, A1st]
        C1 = h.make_tensor(h.hp_1_labels)
        C2 = h.make_tensor(self.dsrg_helper.hp_2_labels)
        h.H1_T1_C1_non_od(C1, F0, self.T1_1, cu, scale=-1.0)
        h.H1_T2_C1_non_od(C1, F0, self.T2_1, cu, scale=-1.0)
        h.H1_T2_C2_non_od(C2, F0, self.T2_1, cu, scale=-1.0)

        # -[[H0th, A1st], A1st]; kept for the second stage
        self.O1 = h.make_tensor(h.hp_1_labels)
        self.O2 = h.make_tensor(self.dsrg_helper.hp_2_labels)
        self._commutator_od(self.O1, self.O2, C1, C2, self.T1_1, self.T2_1, self.S2_1)

        E = h.H_T_C0(
            self.O1, self.O2, self.T1_1, self.T2_1, self.S2_1, cu, scale=1.0 / 6.0
        )
        if form_hbar:
            self._accumulate_hbar(
                self.O1, self.O2, self.T1_1, self.T2_1, self.S2_1, 1.0 / 12.0
            )
        return E

    def _compute_energy_pt2(self, form_hbar):
        """<[H1st + Hbar1st, A1st]>, from the once-renormalized bare Hamiltonian."""
        # H1st + Hbar1st; kept for the second stage
        self.X1 = {blk: F.copy() for blk, F in self.ints["F"].items()}
        self.X2 = {
            blk: self.ints["V"][blk].copy() for blk in self.dsrg_helper.hp_2_labels
        }
        self._renormalize(self.X1, self.X2, self.S2_1, add=True)

        E = self.dsrg_helper.H_T_C0(
            self.X1, self.X2, self.T1_1, self.T2_1, self.S2_1, self.cumulants
        )
        if form_hbar:
            self._accumulate_hbar(
                self.X1, self.X2, self.T1_1, self.T2_1, self.S2_1, 0.5
            )
        return E

    def _compute_energy_pt3_2(self, form_hbar):
        """1/2 <[H1st + Hbar1st, A2nd]>, which also builds the second-order amplitudes."""
        h = self.dsrg_helper
        cu = self.cumulants
        V = self.ints["V"]
        T1, T2, S2 = self.T1_1, self.T2_1, self.S2_1

        # Y = 1/2 [H1st + Hbar1st, A1st] = [H1st, A1st] + 1/2 [[H0th, A1st], A1st]
        self.Y1 = {blk: -0.5 * O for blk, O in self.O1.items()}
        self.Y2 = {blk: -0.5 * O for blk, O in self.O2.items()}
        # last read of the first stage's commutator
        self.O1 = self.O2 = None

        # H1st is the off-diagonal Fock matrix and the whole of V. V splits into
        # its off-diagonal blocks, the stored non-off-diagonal ones, and those
        # with three or more virtual indices, streamed from the B tensors.
        V_od = {blk: V[blk] for blk in self.dsrg_helper.hp_2_labels}
        V_non_od = {blk: V[blk] for blk in h.non_od_2_labels}
        self._commutator_od(self.Y1, self.Y2, self.ints["F"], V_od, T1, T2, S2)
        h.H2_T1_C1_non_od(self.Y1, V_non_od, T1, cu)
        h.H2_T2_C1_non_od(self.Y1, V_non_od, T2, S2, cu)
        h.H2_T1_C2_non_od(self.Y2, V_non_od, T1, cu)
        h.H2_T2_C2_non_od(self.Y2, V_non_od, T2, S2, cu)
        h.H2_T2_C1_non_od_large(self.Y1, self.ints["B"], S2, cu)
        h.H2_T1_C2_non_od_large(self.Y2, self.ints["B"], T1, cu)
        h.H2_T2_C2_non_od_large(self.Y2, self.ints["B"], T2, cu)

        # The second-order amplitudes are read only here and by the third
        # stage's renormalization, which needs just S2.
        T1_2, T2_2, self.S2_2 = self._build_tamps(self.Y1, self.Y2)

        E = h.H_T_C0(self.X1, self.X2, T1_2, T2_2, self.S2_2, cu)
        if form_hbar:
            self._accumulate_hbar(self.X1, self.X2, T1_2, T2_2, self.S2_2, 0.5)
        # last read of H1st + Hbar1st
        self.X1 = self.X2 = None
        return E

    def _compute_energy_pt3_3(self, form_hbar):
        """1/2 <[Hbar2nd, A1st]>.

        The off-diagonal part of Hbar2nd = Y + [H0th, A2nd] is Y renormalized,
        with the active generalized-Fock term taken from the second-order
        amplitudes, which is what [H0th, A2nd] contracts.
        """
        self._renormalize(self.Y1, self.Y2, self.S2_2, add=False)
        self.S2_2 = None

        E = self.dsrg_helper.H_T_C0(
            self.Y1, self.Y2, self.T1_1, self.T2_1, self.S2_1, self.cumulants
        )
        if form_hbar:
            self._accumulate_hbar(
                self.Y1, self.Y2, self.T1_1, self.T2_1, self.S2_1, 0.5
            )
        self.Y1 = self.Y2 = None
        return E

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
        # The integrals and amplitudes are rebuilt for the relaxed reference, so
        # release them before the CI solve rather than holding both sets at once.
        self.T1_1 = self.T2_1 = self.S2_1 = None
        for key in ("F", "F0", "V", "B"):
            self.ints.pop(key)

        self.ci_solver.set_ints(self._hbar0, self._hbar1_canon, self._hbar2_canon)
        self.ci_solver.run()
        e_relaxed = self.ci_solver.compute_average_energy()
        self.relax_eigvals = self.ci_solver.evals_flat.copy()
        return e_relaxed
