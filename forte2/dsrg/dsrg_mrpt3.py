import itertools
from dataclasses import dataclass

import numpy as np

from .dsrg_base import DSRGBase
from .dsrg_mrpt3_kernels import (
    ALL1_LABELS,
    T1_LABELS,
    T2_LABELS,
    _DSRGBlockHelper,
)
from .utils import (
    cas_energy_given_RDMs,
    compute_t1_block,
    compute_t2_block,
    degno_active_sf,
    renormalize_V_block,
    rotate_active_ints,
)

# The running one- and two-body operators are Hbar, not commutators, so unlike
# the off-diagonal blocks they keep their all-active piece.
PH_LABELS = tuple("".join(t) for t in itertools.product("av", "ca"))
PPHH_LABELS = tuple("".join(t) for t in itertools.product("av", "av", "ca", "ca"))
# Integral blocks with three or more virtual indices are rebuilt from the
# three-index factors where they are needed, so they are never stored: at
# cc-pVQZ they are all but a tenth of what the integrals would otherwise be.
V_LABELS = tuple(
    b
    for b in ("".join(t) for t in itertools.product("cav", repeat=4))
    if b.count("v") < 3
)
B2_LABELS = tuple("".join(t) for t in itertools.product("cav", repeat=2))


def _is_pphh(label):
    """Whether a two-body block is particle-particle-hole-hole.

    Both halves must be checked: a label like "ccaa" ends in two active indices
    yet is hole-hole-particle-particle, and testing only the trailing pair
    silently folds it the wrong way round.
    """
    return all(c in "av" for c in label[:2]) and all(c in "ca" for c in label[2:])


@dataclass
class DSRG_MRPT3(DSRGBase):
    """
    Spin-adapted driven similarity renormalization group
    third-order multireference perturbation theory (DSRG-MRPT3).

    The energy is a second-order term plus three third-order terms,

    .. math::
        E^{(3)}_1 &= -\\tfrac{1}{12} [[[H^{0}, A^{(1)}], A^{(1)}], A^{(1)}] \\\\
        E^{(3)}_2 &= \\tfrac{1}{2} [H^{(1)} + \\bar{H}^{(1)}, A^{(2)}] \\\\
        E^{(3)}_3 &= \\tfrac{1}{2} [\\bar{H}^{(2)}, A^{(1)}]

    with :math:`\\bar{H}^{(1)} = H^{(1)} + [H^{0}, A^{(1)}]` and
    :math:`\\bar{H}^{(2)} = \\tfrac{1}{2} [H^{(1)} + \\bar{H}^{(1)}, A^{(1)}] + [H^{0}, A^{(2)}]`.
    They are evaluated in that order: the second stage overwrites the running
    Hamiltonian with :math:`\\bar{H}^{(2)}` and rebuilds the amplitudes to second
    order, so the first stage and the second-order term must already have
    consumed the bare quantities.

    Operators are stored dense over the whole correlated space rather than as
    the minimal block set DSRG-MRPT2 uses, because the third-order stages need
    the effective Hamiltonian everywhere, not just in the active block.

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
        The second-order correlation energy contribution.
    e_dsrg_mrpt3_1, e_dsrg_mrpt3_2, e_dsrg_mrpt3_3 : float
        The three third-order correlation energy contributions.
    E_relaxed_ref : float
        The DSRG-MRPT3 total energy after reference relaxation.
    relax_energies : NDArray
        The history of DSRG-MRPT3 total energies during reference relaxation.
        Given as [[Edsrg(fixed_reference), Edsrg(relaxed_reference), Eref], ...].
    relax_eigvals : np.ndarray
        The eigenvalues of the relaxed CI Hamiltonian.

    References
    ----------
    .. [1] C. Li and F. A. Evangelista, "Driven similarity renormalization group: Third-order
           multireference perturbation theory", J. Chem. Phys. 2017, 146, 124132.
    """

    def __post_init__(self):
        super().__post_init__()
        self.requires_attrs.update({"two_component": False})

    def get_integrals(self):
        g1, g2, l2, l3 = self.ci_solver.make_average_cumulants(
            max_order=2 if self.skip_3_cumulant else 3
        )
        self._warn_skip_3_cumulant()
        self.semicanonicalizer.semi_canonicalize(g1=g1, C_contig=self._C)
        self._C_semican = self.semicanonicalizer.C_semican[:, self.corr].copy()
        self.fock = self.semicanonicalizer.fock_semican[self.corr, self.corr].copy()
        self.eps = self.semicanonicalizer.eps_semican[self.corr].copy()
        self.Uactv = self.semicanonicalizer.Uactv.copy()

        cumulants = dict()
        cumulants["gamma1"] = np.einsum(
            "ip,ij,jq->pq", self.Uactv.conj(), g1, self.Uactv, optimize=True
        )
        cumulants["eta1"] = 2 * np.eye(self.nact) - cumulants["gamma1"]
        cumulants["lambda2"] = np.einsum(
            "ip,jq,ijkl,kr,ls->pqrs",
            self.Uactv.conj(),
            self.Uactv.conj(),
            l2,
            self.Uactv,
            self.Uactv,
            optimize=True,
        )
        if l3 is None:
            cumulants["lambda3"] = None
        else:
            cumulants["lambda3"] = np.einsum(
                "ip,jq,kr,ijklmn,ls,mt,nu->pqrstu",
                self.Uactv.conj(),
                self.Uactv.conj(),
                self.Uactv.conj(),
                l3,
                self.Uactv,
                self.Uactv,
                self.Uactv,
                optimize=True,
            )

        ints = dict()
        ints["E"] = cas_energy_given_RDMs(
            self.E_core_orig, self.H_orig, self.V_orig, g1, g2
        )

        B = self.fock_builder.B_tensor_gen_block(self._C_semican, self._C_semican)
        sp = {"c": self.core, "a": self.actv, "v": self.virt}
        ints["B"] = {
            b: np.ascontiguousarray(B[:, sp[b[0]], sp[b[1]]]) for b in B2_LABELS
        }
        ints["V"] = self._build_v_blocks(ints["B"])

        ints["eps"] = {k: self.eps[sl].copy() for k, sl in sp.items()}

        return ints, cumulants

    def _build_v_blocks(self, Bb):
        """Two-electron integrals, block by block, from the three-index factors.

        <pq|rs> = sum_Q B[Q,p,r] B[Q,q,s]. Blocks with three or more virtual
        indices are not built at all: every contraction that would reach one
        rebuilds it from the same factors instead, so the largest arrays the
        method would otherwise hold never exist.
        """
        return {
            b: np.einsum(
                "Qpr,Qqs->pqrs", Bb[b[0] + b[2]], Bb[b[1] + b[3]], optimize=True
            )
            for b in V_LABELS
        }

    # ------------------------------------------------------------------
    # adapters: the MRPT2 scalar and active-space kernels want dense operands
    # ------------------------------------------------------------------

    def _h1_dense(self, F):
        out = np.zeros((self.npart, self.nhole))
        out[self.pa, self.hc] = F["ac"]
        out[self.pa, self.ha] = F["aa"]
        out[self.pv, self.hc] = F["vc"]
        out[self.pv, self.ha] = F["va"]
        return out

    def _t1_dense(self, T1):
        out = np.zeros((self.nhole, self.npart))
        out[self.hc, self.pa] = T1["ca"]
        out[self.hc, self.pv] = T1["cv"]
        out[self.ha, self.pv] = T1["av"]
        return out

    def _evaluate_C0(self, F, V, T1, T2, S2):
        return self.dsrg_helper.evaluate_H_T_C0(
            self._t1_dense(T1),
            {"T2": T2, "S2": S2},
            self._h1_dense(F),
            V,
            self.cumulants,
            store_large=True,
        )

    def _accumulate_hbar(self, F, V, T1, T2, S2, scale):
        """Add one stage's contribution to the running active-space Hbar.

        Each stage contributes half of its Hermitian completion; `_build_hbar`
        closes the sum once, after all four stages have accumulated.
        """
        t1 = self._t1_dense(T1)
        h1 = self._h1_dense(F)
        g1, e1 = self.cumulants["gamma1"], self.cumulants["eta1"]
        self.hbar1 += scale * self.dsrg_helper.H_T_C1_active(
            t1, T2, S2, h1, V, g1, e1, self.cumulants["lambda2"], store_large=True
        )
        self.hbar2 += scale * self.dsrg_helper.H_T_C2_active(t1, T2, S2, h1, V, g1, e1)

    # ------------------------------------------------------------------
    # the running operators, held over their own block sets
    # ------------------------------------------------------------------

    def _make_ph(self):
        return {b: np.zeros(self._shape(b)) for b in PH_LABELS}

    def _make_pphh(self):
        return {b: np.zeros(self._shape(b)) for b in PPHH_LABELS}

    def _shape(self, label):
        d = {"c": self.ncore, "a": self.nact, "v": self.nvirt}
        return tuple(d[c] for c in label)

    def _fold_ph(self, dest, src):
        """Fold a commutator's two index orders into the particle-hole one."""
        for b in dest:
            if b in src:
                dest[b] += src[b]
            rev = b[1] + b[0]
            if rev in src:
                dest[b] += src[rev].T

    def _fold_pphh(self, dest, src):
        """The two-body counterpart of _fold_ph."""
        for b in dest:
            if b in src:
                dest[b] += src[b]
            rev = b[2] + b[3] + b[0] + b[1]
            if rev in src:
                dest[b] += src[rev].transpose(2, 3, 0, 1)

    # ------------------------------------------------------------------
    # amplitudes and renormalization
    # ------------------------------------------------------------------

    def _t1_correction(self, label, S2):
        """The generalized-Fock off-diagonal correction shared by T1 and F-tilde."""
        faa = self.F0th["aa"]
        g1 = self.cumulants["gamma1"]
        s2 = S2[label[0] + "a" + label[1] + "a"]
        corr = 0.5 * np.einsum("ivaw,wu,uv->ia", s2, faa, g1, optimize=True)
        corr -= 0.5 * np.einsum("iwau,vw,uv->ia", s2, faa, g1, optimize=True)
        return corr

    def _build_tamps(self, F, V):
        """First- or second-order amplitudes, depending on what F and V hold."""
        eps = self.ints["eps"]

        T2 = {}
        for b in T2_LABELS:
            src = V[b[2] + b[3] + b[0] + b[1]]
            T2[b] = np.ascontiguousarray(src.transpose(2, 3, 0, 1))
            compute_t2_block(
                T2[b], eps[b[0]], eps[b[1]], eps[b[2]], eps[b[3]], self.flow_param
            )
        S2 = {
            b: 2 * T2[b] - T2[b[0] + b[1] + b[3] + b[2]].transpose(0, 1, 3, 2)
            for b in T2_LABELS
        }

        T1 = {}
        for b in T1_LABELS:
            T1[b] = np.ascontiguousarray(F[b[1] + b[0]].T)
            T1[b] += self._t1_correction(b, S2)
            compute_t1_block(T1[b], eps[b[0]], eps[b[1]], self.flow_param)

        return T1, T2, S2

    def _renormalize(self, F, V, add):
        """Scale F and V by (1 + R) if `add`, by R otherwise.

        R is the DSRG source factor exp(-s * denominator^2); only the blocks
        that carry a denominator are touched.
        """
        eps = self.ints["eps"]
        for b, blk in V.items():
            bare = blk.copy()
            renormalize_V_block(
                blk, eps[b[0]], eps[b[1]], eps[b[2]], eps[b[3]], self.flow_param
            )
            if not add:
                blk -= bare
        for b, blk in F.items():
            rev = b[1] + b[0]
            corr = self._t1_correction(rev, self.S2).T if rev in T1_LABELS else 0.0
            d = eps[b[0]][:, None] - eps[b[1]][None, :]
            f = (blk + corr) * np.exp(-self.flow_param * d**2)
            if add:
                blk += f
            else:
                blk[:] = f

    # ------------------------------------------------------------------
    # the four energy contributions
    # ------------------------------------------------------------------

    def _compute_energy_pt3_1(self, form_hbar):
        """-1/12 [[[H0th, A1st], A1st], A1st]."""
        helper = self.helper
        cum = self.cumulants

        # -[H0th, A1st]
        C1 = helper.make_1body()
        C2 = helper.make_2body()
        helper.H1_T1_C1(C1, self.F0th, self.T1, cum, -1.0)
        helper.H1_T2_C1(C1, self.F0th, self.T2, cum, -1.0)
        helper.H1_T2_C2(C2, self.F0th, self.T2, cum, -1.0)
        self._project_od(C1, C2)

        # -[[H0th, A1st], A1st]
        D1 = helper.make_1body()
        D2 = helper.make_2body()
        helper.H1_T1_C1(D1, C1, self.T1, cum, 1.0)
        helper.H1_T2_C1(D1, C1, self.T2, cum, 1.0)
        helper.H2_T1_C1_od(D1, C2, self.T1, cum, 1.0)
        helper.H2_T2_C1_od(D1, C2, self.T2, self.S2, cum, 1.0)
        helper.H1_T2_C2(D2, C1, self.T2, cum, 1.0)
        helper.H2_T1_C2_od(D2, C2, self.T1, cum, 1.0)
        helper.H2_T2_C2_od(D2, C2, self.T2, self.S2, cum, 1.0)

        self.O1 = self._make_ph()
        self.O2 = self._make_pphh()
        self._fold_ph(self.O1, D1)
        self._fold_pphh(self.O2, D2)

        E = (1.0 / 6.0) * self._evaluate_C0(self.O1, self.O2, self.T1, self.T2, self.S2)
        if form_hbar:
            self._accumulate_hbar(
                self.O1, self.O2, self.T1, self.T2, self.S2, 1.0 / 12.0
            )
        return E

    def _project_od(self, C1, C2):
        """Fold the hole-particle direction into the particle-hole one.

        The kernels already produce nothing but off-diagonal blocks, so this is
        C1["ai"] += C1["ia"] and its two-body counterpart, block by block.
        """
        for b in ("ac", "vc", "va"):
            C1[b] += C1[b[1] + b[0]].T
        for b in C2:
            if _is_pphh(b):
                C2[b] += C2[b[2] + b[3] + b[0] + b[1]].transpose(2, 3, 0, 1)

    def _compute_energy_pt2(self, form_hbar):
        """The second-order term, from the once-renormalized bare Hamiltonian."""
        self._renormalize(self.F, self.V, add=True)
        E = self._evaluate_C0(self.F, self.V, self.T1, self.T2, self.S2)
        if form_hbar:
            self._accumulate_hbar(self.F, self.V, self.T1, self.T2, self.S2, 0.5)
        return E

    def _compute_energy_pt3_2(self, form_hbar):
        """1/2 [H1st + Hbar1st, A2nd], which also produces the second-order amplitudes."""
        helper = self.helper
        cum = self.cumulants

        # keep H1st + Hbar1st before F/V are repurposed
        X1, X2 = self.F, self.V

        # 0.5 * [H1st + Hbar1st, A1st] = [H1st, A1st] + 0.5 * [[H0th, A1st], A1st]
        self.F = {b: -0.5 * self.O1[b] for b in self.O1}
        self.V = {b: -0.5 * self.O2[b] for b in self.O2}

        D1 = helper.make_1body()
        D2 = helper.make_2body()
        helper.H1_T1_C1(D1, self.F1st, self.T1, cum, 1.0)
        helper.H1_T2_C1(D1, self.F1st, self.T2, cum, 1.0)
        helper.H1_T2_C2(D2, self.F1st, self.T2, cum, 1.0)
        self._fold_ph(self.F, D1)
        self._fold_pphh(self.V, D2)

        Bblk = self.ints["B"]
        D1 = helper.make_1body()
        D2 = helper.make_2body()
        helper.H2_T1_C1(D1, self.V_bare, self.T1, cum, 1.0)
        helper.H2_T2_C1(D1, self.V_bare, self.T2, self.S2, cum, 1.0, B=Bblk)
        helper.H2_T1_C2(D2, self.V_bare, self.T1, cum, 1.0, B=Bblk)
        helper.H2_T2_C2(D2, self.V_bare, self.T2, self.S2, cum, 1.0, B=Bblk)
        self._fold_ph(self.F, D1)
        self._fold_pphh(self.V, D2)

        # the first-order amplitudes are needed again by the third stage
        self.T1_1st, self.T2_1st = self.T1, self.T2

        self.T1, self.T2, self.S2 = self._build_tamps(self.F, self.V)

        E = self._evaluate_C0(X1, X2, self.T1, self.T2, self.S2)
        if form_hbar:
            self._accumulate_hbar(X1, X2, self.T1, self.T2, self.S2, 0.5)
        return E

    def _compute_energy_pt3_3(self, form_hbar):
        """1/2 [Hbar2nd, A1st]."""
        self._renormalize(self.F, self.V, add=False)

        # S2 must go back to first order: the previous stage left it at second
        S2 = {
            b: 2 * self.T2_1st[b]
            - self.T2_1st[b[0] + b[1] + b[3] + b[2]].transpose(0, 1, 3, 2)
            for b in T2_LABELS
        }

        E = self._evaluate_C0(self.F, self.V, self.T1_1st, self.T2_1st, S2)
        if form_hbar:
            self._accumulate_hbar(self.F, self.V, self.T1_1st, self.T2_1st, S2, 0.5)
        return E

    # ------------------------------------------------------------------
    # driver
    # ------------------------------------------------------------------

    def solve_dsrg(self, form_hbar=False):
        self.helper = _DSRGBlockHelper(self)

        if form_hbar:
            self.hbar1 = np.zeros((self.nact,) * 2)
            self.hbar2 = np.zeros((self.nact,) * 4)

        sp = {"c": self.core, "a": self.actv, "v": self.virt}
        Fb = {b: self.fock[sp[b[0]], sp[b[1]]].copy() for b in ALL1_LABELS}
        self.F0th = {
            b: (Fb[b] if b[0] == b[1] else np.zeros_like(Fb[b])) for b in ALL1_LABELS
        }
        self.F1st = {
            b: (np.zeros_like(Fb[b]) if b[0] == b[1] else Fb[b]) for b in ALL1_LABELS
        }

        self.V_bare = self.ints["V"]
        self.F = {b: Fb[b].copy() for b in PH_LABELS}
        self.V = {b: self.ints["V"][b].copy() for b in PPHH_LABELS}

        self.T1, self.T2, self.S2 = self._build_tamps(self.F, self.V)

        # order matters: each stage consumes state the next one overwrites
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

    def _release_integrals(self):
        super()._release_integrals()
        for name in (
            "V_bare",
            "V",
            "F",
            "T1",
            "T2",
            "S2",
            "T1_1st",
            "T2_1st",
            "O1",
            "O2",
        ):
            setattr(self, name, None)

    def _build_hbar(self):
        # hbar1/hbar2 are left untouched so this can be called more than once.
        _hbar1 = self.hbar1 + self.hbar1.T + self.fock[self.actv, self.actv]
        _hbar2 = self.hbar2 + np.einsum("ijab->abij", self.hbar2)
        _hbar2 = _hbar2 + self.V_bare["aaaa"]

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
        self.ci_solver.set_ints(self._hbar0, self._hbar1_canon, self._hbar2_canon)
        self.ci_solver.run()
        e_relaxed = self.ci_solver.compute_average_energy()
        self.relax_eigvals = self.ci_solver.evals_flat.copy()
        return e_relaxed
