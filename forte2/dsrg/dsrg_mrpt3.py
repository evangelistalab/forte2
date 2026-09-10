import itertools
from dataclasses import dataclass

import numpy as np

from .dsrg_base import DSRGBase
from .dsrg_common import _DSRGDenseHelper
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

        # Dense two-electron integrals over the whole correlated space:
        # <pq|rs> = (pr|qs).
        B = self.fock_builder.B_tensor_gen_block(self._C_semican, self._C_semican)
        ints["V"] = self._build_v_blocks(B)
        # kept for the contractions that would otherwise reach the four-virtual
        # integrals; a few MiB against tens for the block they replace
        c, a, v, h, p = self.core, self.actv, self.virt, self.hole, self.part
        ints["B"] = {
            "part": np.ascontiguousarray(B[:, p, p]),
            "vv": np.ascontiguousarray(B[:, v, v]),
            "vc": np.ascontiguousarray(B[:, v, c]),
            "va": np.ascontiguousarray(B[:, v, a]),
            "hv": np.ascontiguousarray(B[:, h, v]),
            "av": np.ascontiguousarray(B[:, a, v]),
        }
        # every block above is an independent copy, so the source array is dead
        # here; holding it through the rest of get_integrals doubles its cost
        del B

        ints["eps"] = dict()
        for key, sl in (
            ("c", self.core),
            ("a", self.actv),
            ("v", self.virt),
            ("h", self.hole),
            ("p", self.part),
        ):
            ints["eps"][key] = self.eps[sl].copy()

        return ints, cumulants

    def _build_v_blocks(self, B):
        """Two-electron integrals, tiled by which of the four indices are virtual.

        The correlated space is the hole space followed by the virtual space, so
        tagging each index one or the other tiles the integrals exactly, into
        sixteen rectangles rather than the eighty-one elementary blocks a
        core/active/virtual split would give.

        Two tiles are left out of the result. Any tile with three or more
        virtual indices is reached only by terms whose particle ladder is
        streamed from the three-index integrals, so it is never built -- at
        cc-pVTZ the four-virtual tile alone would be 55% of the array. And
        since the Hamiltonian is Hermitian, V[pqrs] = V[rspq]: a tile and the
        tile whose index pairs are swapped hold the same real numbers, so the
        second is stored as a transposed view of the first and costs nothing.

        Of such a pair the built member is the one leading with virtual
        indices. The running two-body operators are particle-particle-hole-hole
        shaped, so that is the member the contractions read -- by 27 reads to 2
        on the largest pair -- and it is the one worth keeping contiguous.
        """
        span = {"h": self.hole, "v": self.virt}
        tiles = {}
        for m in sorted(
            (m for m in itertools.product("hv", repeat=4) if m.count("v") < 3),
            key=lambda m: (-m[:2].count("v"), m),
        ):
            rev = m[2:] + m[:2]
            if rev in tiles:
                tiles[m] = tiles[rev].transpose(2, 3, 0, 1)
            else:
                tiles[m] = np.einsum(
                    "Bpr,Bqs->pqrs",
                    B[:, span[m[0]], span[m[2]]],
                    B[:, span[m[1]], span[m[3]]],
                    optimize=True,
                )
        return tiles

    def _v_pphh(self, vb):
        """Assemble the pphh rectangle from the block-stored integrals."""
        ha, pa, pv = self.ha, self.pa, self.pv
        out = np.zeros((self.npart, self.npart, self.nhole, self.nhole))
        out[pa, pa] = vb["h", "h", "h", "h"][ha, ha]
        out[pv, pa] = vb["v", "h", "h", "h"][:, ha]
        out[pa, pv] = vb["h", "v", "h", "h"][ha, :]
        out[pv, pv] = vb["v", "v", "h", "h"]
        return out

    # ------------------------------------------------------------------
    # block adapters: the MRPT2 kernels take dicts of individual blocks
    # ------------------------------------------------------------------

    def _h2_blocks(self, V):
        """Name the blocks the MRPT2 kernels want out of a pphh-shaped operator.

        Every block they ask for lies inside pphh, which is why the running
        two-body operators never have to be stored over the whole space.
        """
        hc, ha, pa, pv = self.hc, self.ha, self.pa, self.pv
        return {
            "vvaa": V[pv, pv, ha, ha],
            "aacc": V[pa, pa, hc, hc],
            "avca": V[pa, pv, hc, ha],
            "avac": V[pa, pv, ha, hc],
            "vaaa": V[pv, pa, ha, ha],
            "aaca": V[pa, pa, hc, ha],
            "aaaa": V[pa, pa, ha, ha],
            "vvcc": V[pv, pv, hc, hc],
            "vvac": V[pv, pv, ha, hc],
            "vacc": V[pv, pa, hc, hc],
        }

    def _t2_blocks(self, T2, S2):
        hc, ha, pa, pv = self.hc, self.ha, self.pa, self.pv
        keys = {
            "aavv": (ha, ha, pv, pv),
            "ccaa": (hc, hc, pa, pa),
            "caav": (hc, ha, pa, pv),
            "acav": (ha, hc, pa, pv),
            "aava": (ha, ha, pv, pa),
            "caaa": (hc, ha, pa, pa),
        }
        t2 = {k: T2[sl] for k, sl in keys.items()}
        s2 = {k: S2[sl] for k, sl in keys.items()}
        s2["ccvv"] = S2[hc, hc, pv, pv]
        s2["acvv"] = S2[ha, hc, pv, pv]
        s2["ccva"] = S2[hc, hc, pv, pa]
        return {"T2": t2, "S2": s2}

    def _evaluate_C0(self, F, V, T1, T2, S2):
        return self.dsrg_helper.evaluate_H_T_C0(
            T1,
            self._t2_blocks(T2, S2),
            F,
            self._h2_blocks(V),
            self.cumulants,
            store_large=True,
        )

    def _accumulate_hbar(self, F, V, T1, T2, S2, scale):
        """Add one stage's contribution to the running active-space Hbar.

        Each stage contributes half of its Hermitian completion; `_build_hbar`
        closes the sum once, after all four stages have accumulated.
        """
        t1 = T1
        blocks = self._t2_blocks(T2, S2)
        h1 = F
        h2 = self._h2_blocks(V)
        g1, e1 = self.cumulants["gamma1"], self.cumulants["eta1"]

        self.hbar1 += scale * self.dsrg_helper.H_T_C1_active(
            t1,
            blocks["T2"],
            blocks["S2"],
            h1,
            h2,
            g1,
            e1,
            self.cumulants["lambda2"],
            store_large=True,
        )
        self.hbar2 += scale * self.dsrg_helper.H_T_C2_active(
            t1, blocks["T2"], blocks["S2"], h1, h2, g1, e1
        )

    # ------------------------------------------------------------------
    # off-diagonal projections
    # ------------------------------------------------------------------

    def _make_ph(self):
        return np.zeros((self.npart, self.nhole))

    def _make_pphh(self):
        return np.zeros((self.npart, self.npart, self.nhole, self.nhole))

    def _fold_ph(self, dest, src):
        """dest[particle, hole] += src[particle, hole] + src[hole, particle].T

        The two index orders of the same commutator are summed into the
        particle-hole direction, and the all-active block is dropped: it is an
        internal excitation, which the amplitudes never carry.
        """
        dest += src[0]
        dest += src[1].T

    def _fold_pphh(self, dest, src):
        """The two-body counterpart of _fold_ph."""
        dest += src[0]
        dest += src[1].transpose(2, 3, 0, 1)

    # ------------------------------------------------------------------
    # amplitudes and renormalization
    # ------------------------------------------------------------------

    def _build_tamps(self, F, V):
        """First- or second-order amplitudes, depending on what F and V hold."""
        h, p, a = self.hole, self.part, self.actv
        eps = self.ints["eps"]

        T2 = np.ascontiguousarray(V.transpose(2, 3, 0, 1))
        compute_t2_block(T2, eps["h"], eps["h"], eps["p"], eps["p"], self.flow_param)
        T2[self.ha, self.ha, self.pa, self.pa] = 0.0
        S2 = 2 * T2 - T2.swapaxes(2, 3)

        T1 = np.ascontiguousarray(F.T)
        T1 += self._t1_active_correction(S2)
        compute_t1_block(T1, eps["h"], eps["p"], self.flow_param)
        T1[self.ha, self.pa] = 0.0

        return T1, T2, S2

    def _t1_active_correction(self, S2):
        """The generalized-Fock off-diagonal correction shared by T1 and F-tilde."""
        faa = self.F0th[self.actv, self.actv]
        g1 = self.cumulants["gamma1"]
        s2 = S2[:, self.ha, :, self.pa]
        corr = 0.5 * np.einsum("ivaw,wu,uv->ia", s2, faa, g1, optimize=True)
        corr -= 0.5 * np.einsum("iwau,vw,uv->ia", s2, faa, g1, optimize=True)
        return corr

    def _renormalize(self, F, V, S2, add):
        """Scale F and V by (1 + R) if `add`, by R otherwise.

        R is the DSRG source factor exp(-s * denominator^2). Only the
        particle-hole (and particle-particle-hole-hole) parts carry a
        denominator, so only those are touched.
        """
        h, p = self.hole, self.part
        eps = self.ints["eps"]

        bare = V.copy()
        renormalize_V_block(V, eps["p"], eps["p"], eps["h"], eps["h"], self.flow_param)
        if not add:
            V -= bare

        d = eps["p"][:, None] - eps["h"][None, :]
        f = (F + self._t1_active_correction(S2).T) * np.exp(-self.flow_param * d**2)
        if add:
            F += f
        else:
            F[:] = f

    # ------------------------------------------------------------------
    # the four energy contributions
    # ------------------------------------------------------------------

    def _compute_energy_pt3_1(self, form_hbar):
        """-1/12 [[[H0th, A1st], A1st], A1st]."""
        helper = self.dense_helper

        # -[H0th, A1st]
        C1 = helper.make_1body()
        C2 = helper.make_2body()
        helper.H1_T1_C1(C1, self.F0th, self.T1, -1.0)
        helper.H1_T2_C1(C1, self.F0th, self.T2, -1.0)
        helper.H1_T2_C2(C2, self.F0th, self.T2, -1.0)
        C1, C2 = self._project_od(C1, C2)

        # -[[H0th, A1st], A1st]
        D1 = helper.make_1body()
        D2 = helper.make_2body()
        helper.H1_T1_C1(D1, C1, self.T1, 1.0)
        helper.H1_T2_C1(D1, C1, self.T2, 1.0)
        helper.H2_T1_C1(D1, C2, self.T1, 1.0)
        helper.H2_T2_C1(D1, C2, self.T2, self.S2, 1.0)
        helper.H1_T2_C2(D2, C1, self.T2, 1.0)
        helper.H2_T1_C2(D2, C2, self.T1, 1.0)
        helper.H2_T2_C2(D2, C2, self.T2, self.S2, 1.0)

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
        """Keep the off-diagonal blocks in both index orders, drop the internal ones.

        The commutator of a block-diagonal operator with an excitation operator
        has no diagonal part, so this only removes the all-active block.

        The kernels already produce nothing but off-diagonal blocks, so this
        only drops the all-active corner -- an internal excitation, which the
        amplitudes never carry, and which would otherwise be counted twice since
        it is the one place the two directions overlap -- and folds the
        hole-particle direction into the particle-hole one:
        C1["ai"] += C1["ia"] and its two-body counterpart.
        """
        pa, ha = self.pa, self.ha
        C1[0][pa, ha] = 0.0
        C1[1][ha, pa] = 0.0
        C2[0][pa, pa, ha, ha] = 0.0
        C2[1][ha, ha, pa, pa] = 0.0

        C1[0][...] += C1[1].T
        C2[0][...] += C2[1].transpose(2, 3, 0, 1)
        return C1, C2

    def _compute_energy_pt2(self, form_hbar):
        """The second-order term, from the once-renormalized bare Hamiltonian."""
        self._renormalize(self.F, self.V, self.S2, add=True)
        E = self._evaluate_C0(self.F, self.V, self.T1, self.T2, self.S2)
        if form_hbar:
            self._accumulate_hbar(self.F, self.V, self.T1, self.T2, self.S2, 0.5)
        return E

    def _compute_energy_pt3_2(self, form_hbar):
        """1/2 [H1st + Hbar1st, A2nd], which also produces the second-order amplitudes."""
        helper = self.dense_helper
        h, p = self.hole, self.part

        # keep H1st + Hbar1st before F/V are repurposed
        X1 = self.F
        X2 = self.V

        # 0.5 * [H1st + Hbar1st, A1st] = [H1st, A1st] + 0.5 * [[H0th, A1st], A1st]
        self.F = -0.5 * self.O1
        self.V = -0.5 * self.O2
        # last read of the first stage's commutator; nothing below reaches it
        self.O1 = self.O2 = None

        D1 = helper.make_1body()
        D2 = helper.make_2body()
        helper.H1_T1_C1(D1, self.F1st, self.T1, 1.0)
        helper.H1_T2_C1(D1, self.F1st, self.T2, 1.0)
        helper.H1_T2_C2(D2, self.F1st, self.T2, 1.0)
        self._fold_ph(self.F, D1)
        self._fold_pphh(self.V, D2)

        D1 = helper.make_1body()
        D2 = helper.make_2body()
        Bblk = self.ints["B"]
        helper.H2_T1_C1(D1, self.V_bare, self.T1, 1.0)
        helper.H2_T2_C1(D1, self.V_bare, self.T2, self.S2, 1.0, B=Bblk)
        helper.H2_T1_C2(D2, self.V_bare, self.T1, 1.0, B=Bblk)
        helper.H2_T2_C2(D2, self.V_bare, self.T2, self.S2, 1.0, B=Bblk)
        self._fold_ph(self.F, D1)
        self._fold_pphh(self.V, D2)

        # the first-order amplitudes are needed again by the third stage
        self.T1_1st, self.T2_1st = self.T1, self.T2

        # Locals, not attributes: the second-order amplitudes are read only by
        # the two calls below, and the third stage works from the first-order
        # ones. Holding them on self kept them alive for the rest of the run.
        T1_2nd, T2_2nd, S2_2nd = self._build_tamps(self.F, self.V)

        E = self._evaluate_C0(X1, X2, T1_2nd, T2_2nd, S2_2nd)
        if form_hbar:
            self._accumulate_hbar(X1, X2, T1_2nd, T2_2nd, S2_2nd, 0.5)
        return E, S2_2nd

    def _compute_energy_pt3_3(self, form_hbar, S2_2nd):
        """1/2 [Hbar2nd, A1st]."""
        # Hbar2nd is renormalized against the second-order amplitudes, but the
        # commutator that follows is with A1st, so the two use different orders.
        self._renormalize(self.F, self.V, S2_2nd, add=False)

        E = self._evaluate_C0(self.F, self.V, self.T1_1st, self.T2_1st, self.S2)
        if form_hbar:
            self._accumulate_hbar(
                self.F, self.V, self.T1_1st, self.T2_1st, self.S2, 0.5
            )
        return E

    # ------------------------------------------------------------------
    # driver
    # ------------------------------------------------------------------

    def solve_dsrg(self, form_hbar=False):
        self.dense_helper = _DSRGDenseHelper(self)
        self.dense_helper.set_cumulants(self.cumulants)

        if form_hbar:
            self.hbar1 = np.zeros((self.nact,) * 2)
            self.hbar2 = np.zeros((self.nact,) * 4)

        c, a, v = self.core, self.actv, self.virt
        self.F0th = np.zeros_like(self.fock)
        for sl in (c, a, v):
            self.F0th[sl, sl] = self.fock[sl, sl]
        self.F1st = self.fock - self.F0th

        # V_bare stays dense: unlike the running operators it is contracted
        # over general index patterns by the commutator kernels.
        self.V_bare = self.ints["V"]
        self.F = np.ascontiguousarray(self.fock[self.part, self.hole])
        self.V = self._v_pphh(self.ints["V"])

        self.T1, self.T2, self.S2 = self._build_tamps(self.F, self.V)

        # order matters: each stage consumes state the next one overwrites
        self.e_dsrg_mrpt3_1 = self._compute_energy_pt3_1(form_hbar)
        self.e_dsrg_mrpt2 = self._compute_energy_pt2(form_hbar)
        self.e_dsrg_mrpt3_2, S2_2nd = self._compute_energy_pt3_2(form_hbar)
        self.e_dsrg_mrpt3_3 = self._compute_energy_pt3_3(form_hbar, S2_2nd)

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
        ha = self.ha
        _hbar2 = _hbar2 + self.V_bare["h", "h", "h", "h"][ha, ha, ha, ha]

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
