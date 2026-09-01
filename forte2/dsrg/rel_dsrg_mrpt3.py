from dataclasses import dataclass
from itertools import product

import numpy as np

from .dsrg_base import DSRGBase
from .utils import (
    hermitize_one_body,
    hermitize_and_antisymmetrize_two_body,
    hermitize_and_antisymmetrize_two_body_dense,
    cas_energy_given_RDMs,
    compute_t1_block,
    compute_t2_block,
    renormalize_V_block,
)


@dataclass
class RelDSRG_MRPT3(DSRGBase):
    """
    Two-component relativistic driven similarity renormalization group
    third-order multireference perturbation theory (2C-DSRG-MRPT3).

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

    Notes
    -----
    If the parent method publishes an hbar_shift, its energy contribution is
    added to E_dsrg and, when relaxing the reference, the rest of it is folded
    into the effective Hamiltonian handed to the CI solver, so relaxed and
    state-averaged reference states are properly perturbed by the correction
    rather than uniformly shifted after the fact. Nothing here is specific to
    where that shift came from; see RelFNO_DSRG_MRPT3, which composes this
    class with two DSRG-MRPT2 runs to produce the frozen natural orbital
    truncation correction. Both the unrelaxed and relaxed energies of that
    composition match forte's own spin-free FNO-DSRG-MRPT2/MRPT3
    implementation to ~1e-9 Eh.

    Attributes
    ----------
    E_dsrg : float
        The DSRG-MRPT3 total energy evaluated with the current reference.
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
    .. [1] F. A. Evangelista, "A driven similarity renormalization group approach to quantum many-body problems",
           J. Chem. Phys. 2014, 141, 054109.
    .. [2] C. Li and F. A. Evangelista, "Multireference driven similarity renormalization group: A second-order perturbative analysis",
           J. Chem. Theory Comput. 2015, 11, 2097-2108.
    .. [3] K. P. Hannon, C. Li, and F. A. Evangelista, "An integral-factorized implementation of the driven similarity renormalization group second-order multireference perturbation theory",
              J. Chem. Phys. 2016, 144, 204111.
    .. [4] C. Li and F. A. Evangelista, "Driven similarity renormalization group for excited states: A state-averaged perturbation theory",
           J. Chem. Phys. 2018, 148, 124106.
    .. [5] C. Li, S. Mao, R. Huang, F. A. Evangelista, "Frozen Natural Orbitals for the State-Averaged Driven Similarity Renormalization Group",
           J. Chem. Theory Comput. 2024, 20, 4170-4181.
    """

    def _release_integrals(self):
        super()._release_integrals()
        self.T1_1 = None
        self.T2_1 = None
        self.T1_2 = None
        self.T2_2 = None
        self.F_1_tilde = None
        self.V_1_tilde = None
        self.H0A1_1b = None
        self.H0A1_2b = None
        self.Htilde1A1_1b = None
        self.Htilde1A1_2b = None

    def get_integrals(self):
        g1, g2, l2, l3 = self.ci_solver.make_average_cumulants()
        # self._C are the MCSCF canonical orbitals. We always use canonical orbitals to build the generalized Fock matrix.
        self.semicanonicalizer.semi_canonicalize(g1=g1, C_contig=self._C)
        # Freeze core orbitals by removing them from the semicanonicalized quantities
        # The energy contributions are accounted for in self.E_core_orig
        self._C_semican = self.semicanonicalizer.C_semican[:, self.corr].copy()
        self.fock = self.semicanonicalizer.fock_semican[self.corr, self.corr].copy()
        self.eps = self.semicanonicalizer.eps_semican[self.corr].copy()
        self.delta_actv = self.eps[self.actv][:, None] - self.eps[self.actv][None, :]
        self.Uactv = self.semicanonicalizer.Uactv.copy()

        cumulants = dict()
        # g1 = self.ci_solver.make_average_1rdm()
        cumulants["gamma1"] = np.einsum(
            "ip,ij,jq->pq", self.Uactv, g1, self.Uactv.conj(), optimize=True
        )
        cumulants["eta1"] = (
            np.eye(cumulants["gamma1"].shape[0], dtype=complex) - cumulants["gamma1"]
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
        _sl = {"c": self.core, "a": self.actv, "v": self.virt}
        # we store the F and V tensors transposed to help with cache locality in contractions.
        F_temp = np.conj(self.fock - np.diag(np.diag(self.fock)))  # remove diagonal
        ints["F"] = self.dsrg_helper.make_tensor(self.dsrg_helper.all_1_labels)
        for blk in ints["F"].keys():
            ints["F"][blk] = F_temp[_sl[blk[0]], _sl[blk[1]]].copy()

        ints["E"] = cas_energy_given_RDMs(
            self.E_core_orig, self.H_orig, self.V_orig, g1, g2
        )
        # Build each V block directly from the 3-index B tensors, as in RelDSRG_MRPT2,
        # instead of forming the dense spinor V and slicing it: the dense tensor is
        # nspinor**4 and is unaffordable beyond the smallest systems.
        # <pq||rs> = (pr|qs) - (ps|qr), and (pr|qs) = sum_P B[P,p,r] B[P,q,s], so a block
        # only ever needs the B blocks for the space pairs (p,r), (q,s), (p,s) and (q,r).
        # The coefficients are conjugated to match the convention of the F tensor above.
        C_conj = {
            "c": self._C_semican[:, self.core].conj(),
            "a": self._C_semican[:, self.actv].conj(),
            "v": self._C_semican[:, self.virt].conj(),
        }
        B_so = dict()
        for x, y in product("cav", repeat=2):
            if x + y in B_so:
                continue
            B_so[x + y] = self.fock_builder.B_tensor_gen_block_spinor(
                C_conj[x], C_conj[y]
            )
            # B[xy] and B[yx] are related by a transpose-conjugate, so only half are built
            if y + x not in B_so:
                B_so[y + x] = B_so[x + y].transpose(0, 2, 1).conj()

        ints["V"] = self.dsrg_helper.make_tensor(self.dsrg_helper.all_2_labels)
        for blk in ints["V"].keys():
            p, q, r, s = blk
            V_blk = np.einsum("Ppr,Pqs->pqrs", B_so[p + r], B_so[q + s], optimize=True)
            # The exchange term (ps|qr) is a transpose of the direct term whenever the
            # swapped pair shares a space, which holds for 40 of the 76 blocks: with
            # D[abcd] = sum_P B[P,a,c] B[P,b,d], the exchange is D[pqsr] if r and s share
            # a space, and D[qprs] if p and q do (the two B factors commute). Only the
            # remaining blocks need a second contraction.
            if r == s:
                V_blk -= V_blk.swapaxes(2, 3)
            elif p == q:
                V_blk -= V_blk.swapaxes(0, 1)
            else:
                V_blk -= np.einsum(
                    "Pps,Pqr->pqrs", B_so[p + s], B_so[q + r], optimize=True
                )
            ints["V"][blk] = V_blk

        # store the diagonal of the Fock matrix for later use in T1 denominators
        ints["eps"] = dict()
        ints["eps"]["c"] = self.eps[self.core].copy()
        ints["eps"]["a"] = self.eps[self.actv].copy()
        ints["eps"]["v"] = self.eps[self.virt].copy()
        ints["eps"]["h"] = self.eps[self.hole].copy()
        ints["eps"]["p"] = self.eps[self.part].copy()
        ints["F0"] = {
            "cc": np.diag(ints["eps"]["c"]),
            "aa": np.diag(ints["eps"]["a"]),
            "vv": np.diag(ints["eps"]["v"]),
        }
        ints["denom_act"] = self.eps[self.actv][:, None] - self.eps[self.actv][None, :]
        ints["d1_exp"] = np.exp(
            -self.flow_param
            * (ints["eps"]["h"][:, None] - ints["eps"]["p"][None, :]) ** 2
        )

        ints["B"] = dict()
        C_core = self._C_semican[:, self.core]
        C_actv = self._C_semican[:, self.actv]
        C_virt = self._C_semican[:, self.virt]
        ints["B"]["vv"] = self.fock_builder.B_tensor_gen_block_spinor(C_virt, C_virt)
        ints["B"]["av"] = self.fock_builder.B_tensor_gen_block_spinor(C_actv, C_virt)
        ints["B"]["cv"] = self.fock_builder.B_tensor_gen_block_spinor(C_core, C_virt)
        ints["B"]["vc"] = ints["B"]["cv"].transpose(0, 2, 1).conj()
        ints["B"]["va"] = ints["B"]["av"].transpose(0, 2, 1).conj()

        return ints, cumulants

    def solve_dsrg(self, form_hbar=False):
        if form_hbar:
            self.hbar1 = np.zeros((self.nact,) * 2, dtype=complex)
            self.hbar2 = np.zeros((self.nact,) * 4, dtype=complex)
        self.e_dsrg_mrpt3_1 = self._compute_energy_pt3_1(form_hbar)
        self.e_dsrg_mrpt2 = self._compute_energy_pt2(form_hbar)
        self.e_dsrg_mrpt3_2 = self._compute_energy_pt3_2(form_hbar)
        self.e_dsrg_mrpt3_3 = self._compute_energy_pt3_3(form_hbar)
        E = (
            self.ints["E"]
            + self.e_dsrg_mrpt3_1
            + self.e_dsrg_mrpt2
            + self.e_dsrg_mrpt3_2
            + self.e_dsrg_mrpt3_3
        )
        return E

    def _build_hbar(self):
        # hbar1/hbar2 are left untouched so that this can be called more than
        # once per solve (see DSRGBase._build_hbar).
        _hbar1 = self.hbar1 + self.hbar1.T.conj() + self.fock[self.actv, self.actv]

        _hbar2 = self.hbar2.copy()
        hermitize_and_antisymmetrize_two_body_dense(_hbar2)
        _hbar2 += self.ints["V"]["aaaa"]

        # see eq 29 of Ann. Rev. Phys. Chem. Built from the bare energy: an
        # incoming shift is added below, and E_dsrg already carries its energy
        # contribution.
        self._hbar0 = (
            -np.einsum("uv,uv->", _hbar1.conj(), self.cumulants["gamma1"])
            - 0.25 * np.einsum("uvxy,uvxy->", _hbar2.conj(), self.cumulants["lambda2"])
            + 0.5
            * np.einsum(
                "uvxy,ux,vy->",
                _hbar2.conj(),
                self.cumulants["gamma1"],
                self.cumulants["gamma1"],
            )
        ) + self._E_dsrg_bare

        _hbar1 = np.conj(
            _hbar1 - np.einsum("uxvy,xy->uv", _hbar2, self.cumulants["gamma1"])
        )
        _hbar2 = np.conj(_hbar2)

        self._hbar1_canon = np.einsum(
            "ip,pq,jq->ij", self.Uactv, _hbar1, self.Uactv.conj(), optimize=True
        )
        self._hbar2_canon = np.einsum(
            "ip,jq,pqrs,kr,ls->ijkl",
            self.Uactv,
            self.Uactv,
            _hbar2,
            self.Uactv.conj(),
            self.Uactv.conj(),
            optimize=True,
        )
        # _hbar2_canon is already antisymmetric (<pq||rs>) and the CI solver
        # antisymmetrizes it again, doubling it, hence the 0.5.
        self._hbar2_canon *= 0.5

        shift = getattr(self.parent_method, "hbar_shift", None)
        if shift is not None:
            self._hbar0 += shift["hbar0"]
            self._hbar1_canon += shift["hbar1"]
            self._hbar2_canon += shift["hbar2"]

    def do_reference_relaxation(self):
        self._build_hbar()
        self.ci_solver.set_ints(self._hbar0, self._hbar1_canon, self._hbar2_canon)
        self.ci_solver.run()
        e_relaxed = self.ci_solver.compute_average_energy()
        self.relax_eigvals = self.ci_solver.evals_flat.copy()
        return e_relaxed

    def _build_tamps(self, h1, h2, conj, factor=1.0):
        t2 = self.dsrg_helper.make_tensor(self.dsrg_helper.hp_2_labels)
        for block in t2.keys():
            if conj:
                t2[block] = factor * h2[block].conj()
            else:
                t2[block] = factor * h2[block].copy()
            compute_t2_block(
                t2[block],
                self.ints["eps"][block[0]],
                self.ints["eps"][block[1]],
                self.ints["eps"][block[2]],
                self.ints["eps"][block[3]],
                self.flow_param,
            )
        _hsl = {"c": self.hc, "a": self.ha}
        _psl = {"a": self.pa, "v": self.pv}
        t1_temp = np.zeros((self.nhole, self.npart), dtype=complex)
        for block in self.dsrg_helper.hp_1_labels:
            t1_temp[_hsl[block[0]], _psl[block[1]]] = h1[block].copy()
        if conj:
            t1_temp = np.conj(t1_temp, out=t1_temp)
        t1_temp *= factor
        t2_hapa = np.zeros(
            (self.nhole, self.nact, self.npart, self.nact), dtype=complex
        )
        t2_hapa[self.hc, :, self.pa, :] = t2["caaa"].copy()
        t2_hapa[self.hc, :, self.pv, :] = -t2["caav"].swapaxes(2, 3).copy()
        t2_hapa[self.ha, :, self.pv, :] = -t2["aaav"].swapaxes(2, 3).copy()
        t1_temp += np.einsum(
            "xu,iuax,xu->ia",
            self.ints["denom_act"],
            t2_hapa,
            self.cumulants["gamma1"],
        )
        compute_t1_block(
            t1_temp,
            self.ints["eps"]["h"],
            self.ints["eps"]["p"],
            self.flow_param,
        )
        t1_temp[self.ha, self.pa] = 0.0

        t1 = self.dsrg_helper.make_tensor(self.dsrg_helper.hp_1_labels)
        for block in t1.keys():
            t1[block] = t1_temp[_hsl[block[0]], _psl[block[1]]].copy()

        return t1, t2

    def _compute_energy_pt3_1(self, form_hbar):
        self.T1_1, self.T2_1 = self._build_tamps(
            self.ints["F"], self.ints["V"], conj=False
        )

        self.H0A1_1b = self.dsrg_helper.make_tensor(self.dsrg_helper.od_1_labels)
        self.H0A1_2b = self.dsrg_helper.make_tensor(self.dsrg_helper.od_2_labels)
        self.dsrg_helper.H1_T2_C2_non_od(
            self.H0A1_2b, self.ints["F0"], self.T2_1, self.cumulants
        )
        self.dsrg_helper.H1_T1_C1_non_od(
            self.H0A1_1b, self.ints["F0"], self.T1_1, self.cumulants
        )
        hermitize_and_antisymmetrize_two_body(self.H0A1_2b)
        hermitize_one_body(self.H0A1_1b)

        H0A1A1_1b = self.dsrg_helper.make_tensor(self.dsrg_helper.od_1_labels)
        H0A1A1_2b = self.dsrg_helper.make_tensor(self.dsrg_helper.od_2_labels)

        self.dsrg_helper.H1_T1_C1(H0A1A1_1b, self.H0A1_1b, self.T1_1, self.cumulants)
        self.dsrg_helper.H1_T2_C1(H0A1A1_1b, self.H0A1_1b, self.T2_1, self.cumulants)
        self.dsrg_helper.H2_T1_C1(H0A1A1_1b, self.H0A1_2b, self.T1_1, self.cumulants)
        self.dsrg_helper.H2_T2_C1(H0A1A1_1b, self.H0A1_2b, self.T2_1, self.cumulants)
        self.dsrg_helper.H1_T2_C2(H0A1A1_2b, self.H0A1_1b, self.T2_1, self.cumulants)
        self.dsrg_helper.H2_T1_C2(H0A1A1_2b, self.H0A1_2b, self.T1_1, self.cumulants)
        self.dsrg_helper.H2_T2_C2(H0A1A1_2b, self.H0A1_2b, self.T2_1, self.cumulants)
        hermitize_and_antisymmetrize_two_body(H0A1A1_2b)
        hermitize_one_body(H0A1A1_1b)

        E = self.dsrg_helper.H_T_C0(
            H0A1A1_1b, H0A1A1_2b, self.T1_1, self.T2_1, self.cumulants, scale=-1.0 / 6
        )
        if form_hbar:
            self.dsrg_helper.H_T_C1_aa(
                self.hbar1,
                H0A1A1_1b,
                H0A1A1_2b,
                self.T1_1,
                self.T2_1,
                self.cumulants,
                scale=-1.0 / 12,
            )
            self.dsrg_helper.H_T_C2_aaaa(
                self.hbar2,
                H0A1A1_1b,
                H0A1A1_2b,
                self.T1_1,
                self.T2_1,
                self.cumulants,
                scale=-1.0 / 12,
            )
        return E

    def _compute_energy_pt2(self, form_hbar):
        F_1_tilde_tmp = np.zeros((self.nhole, self.npart), dtype=complex)
        _hsl = {"c": self.hc, "a": self.ha}
        _psl = {"a": self.pa, "v": self.pv}
        for blk in self.dsrg_helper.hp_1_labels:
            F_1_tilde_tmp[_hsl[blk[0]], _psl[blk[1]]] = self.ints["F"][blk].copy()

        F_1_tilde_tmp += F_1_tilde_tmp * self.ints["d1_exp"]
        t2_hapa = np.zeros(
            (self.nhole, self.nact, self.npart, self.nact), dtype=complex
        )
        t2_hapa[self.hc, :, self.pa, :] = self.T2_1["caaa"].copy()
        t2_hapa[self.hc, :, self.pv, :] = -self.T2_1["caav"].swapaxes(2, 3).copy()
        t2_hapa[self.ha, :, self.pv, :] = -self.T2_1["aaav"].swapaxes(2, 3).copy()
        F_1_tilde_tmp += np.multiply(
            self.ints["d1_exp"],
            np.einsum(
                "xu,iuax,xu->ia",
                self.ints["denom_act"],
                t2_hapa,
                self.cumulants["gamma1"],
            ),
        )
        np.conj(F_1_tilde_tmp, out=F_1_tilde_tmp)
        self.F_1_tilde = self.dsrg_helper.make_tensor(self.dsrg_helper.hp_1_labels)
        for blk in self.F_1_tilde.keys():
            self.F_1_tilde[blk] = F_1_tilde_tmp[_hsl[blk[0]], _psl[blk[1]]].copy()

        self.V_1_tilde = self.dsrg_helper.make_tensor(self.dsrg_helper.hp_2_labels)
        for blk in self.V_1_tilde.keys():
            self.V_1_tilde[blk] = self.ints["V"][blk].conj()
            renormalize_V_block(
                self.V_1_tilde[blk],
                self.ints["eps"][blk[0]],
                self.ints["eps"][blk[1]],
                self.ints["eps"][blk[2]],
                self.ints["eps"][blk[3]],
                self.flow_param,
            )

        E = self.dsrg_helper.H_T_C0(
            self.F_1_tilde,
            self.V_1_tilde,
            self.T1_1,
            self.T2_1,
            self.cumulants,
        )
        if form_hbar:
            self.dsrg_helper.H_T_C1_aa(
                self.hbar1,
                self.F_1_tilde,
                self.V_1_tilde,
                self.T1_1,
                self.T2_1,
                self.cumulants,
                scale=0.5,
            )
            self.dsrg_helper.H_T_C2_aaaa(
                self.hbar2,
                self.F_1_tilde,
                self.V_1_tilde,
                self.T1_1,
                self.T2_1,
                self.cumulants,
                scale=0.5,
            )
        return E

    def _compute_energy_pt3_2(self, form_hbar):
        self.Htilde1A1_1b = self.dsrg_helper.make_tensor(self.dsrg_helper.od_1_labels)
        self.Htilde1A1_2b = self.dsrg_helper.make_tensor(self.dsrg_helper.od_2_labels)

        for blk in self.H0A1_1b.keys():
            self.H0A1_1b[blk] += 2 * self.ints["F"][blk].conj()
        for blk in self.H0A1_2b.keys():
            self.H0A1_2b[blk] += 2 * self.ints["V"][blk].conj()

        self.dsrg_helper.H1_T1_C1(
            self.Htilde1A1_1b, self.H0A1_1b, self.T1_1, self.cumulants
        )
        self.dsrg_helper.H1_T2_C1(
            self.Htilde1A1_1b, self.H0A1_1b, self.T2_1, self.cumulants
        )
        self.dsrg_helper.H2_T1_C1(
            self.Htilde1A1_1b, self.H0A1_2b, self.T1_1, self.cumulants
        )
        self.dsrg_helper.H2_T2_C1(
            self.Htilde1A1_1b, self.H0A1_2b, self.T2_1, self.cumulants
        )
        self.dsrg_helper.H1_T2_C2(
            self.Htilde1A1_2b, self.H0A1_1b, self.T2_1, self.cumulants
        )
        self.dsrg_helper.H2_T1_C2(
            self.Htilde1A1_2b, self.H0A1_2b, self.T1_1, self.cumulants
        )
        self.dsrg_helper.H2_T2_C2(
            self.Htilde1A1_2b, self.H0A1_2b, self.T2_1, self.cumulants
        )

        _temp_1b = self.dsrg_helper.make_tensor(self.dsrg_helper.non_od_1_labels)
        _temp_2b = self.dsrg_helper.make_tensor(self.dsrg_helper.non_od_2_labels)
        for blk in _temp_1b.keys():
            _temp_1b[blk] = 2 * self.ints["F"][blk].conj()
        for blk in _temp_2b.keys():
            _temp_2b[blk] = 2 * self.ints["V"][blk].conj()

        self.dsrg_helper.H1_T1_C1_non_od(
            self.Htilde1A1_1b, _temp_1b, self.T1_1, self.cumulants
        )
        self.dsrg_helper.H2_T1_C1_non_od(
            self.Htilde1A1_1b, _temp_2b, self.T1_1, self.cumulants
        )
        self.dsrg_helper.H2_T2_C1_non_od(
            self.Htilde1A1_1b, _temp_2b, self.T2_1, self.cumulants
        )
        self.dsrg_helper.H1_T2_C2_non_od(
            self.Htilde1A1_2b, _temp_1b, self.T2_1, self.cumulants
        )
        self.dsrg_helper.H2_T1_C2_non_od(
            self.Htilde1A1_2b, _temp_2b, self.T1_1, self.cumulants
        )
        self.dsrg_helper.H2_T2_C2_non_od(
            self.Htilde1A1_2b, _temp_2b, self.T2_1, self.cumulants
        )
        self.dsrg_helper.H2_T2_C1_non_od_large(
            self.Htilde1A1_1b, self.ints["B"], self.T2_1, self.cumulants, scale=2.0
        )
        self.dsrg_helper.H2_T1_C2_non_od_large(
            self.Htilde1A1_2b, self.ints["B"], self.T1_1, self.cumulants, scale=2.0
        )
        self.dsrg_helper.H2_T2_C2_non_od_large(
            self.Htilde1A1_2b, self.ints["B"], self.T2_1, self.cumulants, scale=2.0
        )
        hermitize_and_antisymmetrize_two_body(self.Htilde1A1_2b)
        hermitize_one_body(self.Htilde1A1_1b)
        self.T1_2, self.T2_2 = self._build_tamps(
            self.Htilde1A1_1b, self.Htilde1A1_2b, conj=True, factor=0.5
        )
        E = self.dsrg_helper.H_T_C0(
            self.H0A1_1b, self.H0A1_2b, self.T1_2, self.T2_2, self.cumulants
        )
        if form_hbar:
            self.dsrg_helper.H_T_C1_aa(
                self.hbar1,
                self.H0A1_1b,
                self.H0A1_2b,
                self.T1_2,
                self.T2_2,
                self.cumulants,
                scale=0.5,
            )
            self.dsrg_helper.H_T_C2_aaaa(
                self.hbar2,
                self.H0A1_1b,
                self.H0A1_2b,
                self.T1_2,
                self.T2_2,
                self.cumulants,
                scale=0.5,
            )
        return E

    def _compute_energy_pt3_3(self, form_hbar):
        H0A2_1b = self.dsrg_helper.make_tensor(self.dsrg_helper.od_1_labels)
        H0A2_2b = self.dsrg_helper.make_tensor(self.dsrg_helper.od_2_labels)
        self.dsrg_helper.H1_T2_C2_non_od(
            H0A2_2b, self.ints["F0"], self.T2_2, self.cumulants
        )
        self.dsrg_helper.H1_T1_C1_non_od(
            H0A2_1b, self.ints["F0"], self.T1_2, self.cumulants
        )
        hermitize_and_antisymmetrize_two_body(H0A2_2b)
        hermitize_one_body(H0A2_1b)

        for blk, T in self.Htilde1A1_1b.items():
            T *= 0.5
            T += H0A2_1b[blk]
        for blk, T in self.Htilde1A1_2b.items():
            T *= 0.5
            T += H0A2_2b[blk]

        E = self.dsrg_helper.H_T_C0(
            self.Htilde1A1_1b,
            self.Htilde1A1_2b,
            self.T1_1,
            self.T2_1,
            self.cumulants,
        )
        if form_hbar:
            self.dsrg_helper.H_T_C1_aa(
                self.hbar1,
                self.Htilde1A1_1b,
                self.Htilde1A1_2b,
                self.T1_1,
                self.T2_1,
                self.cumulants,
                scale=0.5,
            )
            self.dsrg_helper.H_T_C2_aaaa(
                self.hbar2,
                self.Htilde1A1_1b,
                self.Htilde1A1_2b,
                self.T1_1,
                self.T2_1,
                self.cumulants,
                scale=0.5,
            )
        return E
