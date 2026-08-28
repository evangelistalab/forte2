from dataclasses import dataclass, field

import numpy as np

from forte2.helpers import logger

from .dsrg_base import DSRGBase
from .fno_utils import build_fno_virtual_space
from .utils import (
    hermitize_and_antisymmetrize_two_body_dense,
    cas_energy_given_RDMs,
    compute_t1_block,
    compute_t2_block,
    renormalize_V_block,
    renormalize_3index,
)


@dataclass
class RelDSRG_MRPT2(DSRGBase):
    """
    Two-component relativistic driven similarity renormalization group
    second-order multireference perturbation theory (2C-DSRG-MRPT2).

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
    fno_p_o : float, optional, default=None
        Truncate the virtual space to frozen natural orbitals (FNO) once the
        energy is computed, retaining the smallest set of leading virtual
        natural orbitals whose cumulative occupation is at least this fraction
        (0, 1] of the total. Mutually exclusive with fno_n_kappa.
    fno_n_kappa : float, optional, default=None
        As fno_p_o, but retaining all virtual natural orbitals with occupation
        number >= fno_n_kappa. Mutually exclusive with fno_p_o.
    fno_degeneracy_tol : float, optional, default=1e-2
        When truncating, the boundary is pushed outward (more orbitals
        retained) while the occupation numbers straddling it differ by less
        than this fraction of the larger one, so that near-degenerate natural
        orbitals (e.g. Kramers partners) are never split between the retained
        and discarded sets.
    compute_hbar_shift : bool, optional, default=False
        Publish this object's effective Hamiltonian subtracted from its
        parent's as an hbar_shift, for a method chained onto this one to apply.
        Requires the parent to expose its own effective Hamiltonian (i.e. to
        have been run with save_hbar, or with reference relaxation) and to use
        the same flow_param. Implies save_hbar.

    Attributes
    ----------
    E_dsrg : float
        The DSRG-MRPT2 total energy evaluated with the current reference.
    E_relaxed_ref : float
        The DSRG-MRPT2 total energy after reference relaxation.
    relax_energies : NDArray
        The history of DSRG-MRPT2 total energies during reference relaxation.
        Given as [[Edsrg(fixed_reference), Edsrg(relaxed_reference), Eref], ...].
    relax_eigvals : np.ndarray
        The eigenvalues of the relaxed CI Hamiltonian.
    relax_eigvals_history : NDArray
        The history of eigenvalues of the relaxed CI Hamiltonian during relaxation.
    hbar_shift : dict or None
        Set when compute_hbar_shift is given: this object's effective
        Hamiltonian subtracted from its parent's, for a method chained onto
        this one to apply. None otherwise.

    Notes
    -----
    Setting fno_p_o or fno_n_kappa truncates the virtual space *after* the
    energy is computed, so it composes with everything else this class does
    rather than replacing it; the truncated, natural-orbital-rotated space is
    exposed as this object's own mos/mo_space for a chained method to pick up.
    Setting compute_hbar_shift additionally publishes the difference between
    the parent's effective Hamiltonian and this object's.

    Chaining the two gives the FNO truncation correction of ref. [5]; see
    RelFNO_DSRG_MRPT3, which composes exactly that and is the supported way
    to run it.

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

    fno_p_o: float | None = None
    fno_n_kappa: float | None = None
    fno_degeneracy_tol: float = 1e-2
    compute_hbar_shift: bool = False

    def __post_init__(self):
        super().__post_init__()
        self.requires_attrs.update({"two_component": True})

        if self.fno_p_o is not None or self.fno_n_kappa is not None:
            assert (self.fno_p_o is None) != (
                self.fno_n_kappa is None
            ), "Specify exactly one of fno_p_o or fno_n_kappa."
        if self.compute_hbar_shift:
            # Both Hamiltonians in the difference must be available and built
            # at the same flow parameter; see _post_process.
            self.save_hbar = True

    def _post_process(self):
        """
        Truncate the virtual space to frozen natural orbitals and/or publish
        an hbar_shift, as requested. Both are optional and independent of how
        the energy itself was obtained, so neither needs to branch run().
        """
        if self.fno_p_o is not None or self.fno_n_kappa is not None:
            self._truncate_to_fno()
        if self.compute_hbar_shift:
            self._publish_hbar_shift()

    def _truncate_to_fno(self):
        """
        Diagonalize the unrelaxed virtual-virtual 1-RDM, truncate the virtual
        space, and expose the truncated, natural-orbital-rotated space as this
        object's own mos/mo_space, for a chained method to pick up.
        """
        nvirt_full = self.nvirt
        gamma_vv = self.compute_unrelaxed_gamma_vv()
        self.mos, self.mo_space = build_fno_virtual_space(
            self,
            gamma_vv,
            p_o=self.fno_p_o,
            n_kappa=self.fno_n_kappa,
            degeneracy_tol=self.fno_degeneracy_tol,
        )
        logger.log_info1(
            f"\nFrozen natural orbitals: retained {self.mo_space.nvirt} of "
            f"{nvirt_full} virtual orbitals "
            f"({100 * self.mo_space.nvirt / nvirt_full:.1f}%)."
        )
        self._release_integrals()

    def _publish_hbar_shift(self):
        """
        Expose this object's effective Hamiltonian subtracted from its
        parent's, for a method chained onto this one to apply (see the
        hbar_shift attribute on DSRGBase for the dict's keys).

        With an FNO-truncated parent this is the eq. 11 truncation correction
        [H_PT2^MO(s1) - H_PT2^FNO(s1)] of Li, Mao, Huang, Evangelista, J.
        Chem. Theory Comput. 2024, 20, 4170-4181: the parent spans the full
        virtual space and this object the truncated one. Both must therefore
        use the same flow parameter s1, and the two Hamiltonians are compared
        in the shared basis of the initial CASSCF active orbitals (the
        *_canon convention), which FNO truncation leaves untouched.
        """
        parent = self.parent_method
        assert self.flow_param == parent.flow_param, (
            "compute_hbar_shift compares this object's effective Hamiltonian "
            "against its parent's, so the two must share flow_param."
        )
        self.hbar_shift = {
            "e_dsrg": parent.E_dsrg - self.E_dsrg,
            "hbar0": parent.hbar0 - self.hbar0,
            "hbar1": parent.hbar1_canon - self.hbar1_canon,
            "hbar2": parent.hbar2_canon - self.hbar2_canon,
        }

    def _release_integrals(self):
        super()._release_integrals()
        self.T1 = None
        self.T2 = None
        self.F_tilde = None

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

        ints = dict()
        ints["F"] = self.fock - np.diag(np.diag(self.fock))  # remove diagonal

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

        ints["E"] = cas_energy_given_RDMs(
            self.E_core_orig, self.H_orig, self.V_orig, g1, g2
        )

        # Save blocks of spinorbital basis B tensor
        B_so = dict()
        C_core = self._C_semican[:, self.core]
        C_actv = self._C_semican[:, self.actv]
        C_virt = self._C_semican[:, self.virt]
        B_so["cc"] = self.fock_builder.B_tensor_gen_block_spinor(C_core, C_core)
        B_so["ca"] = self.fock_builder.B_tensor_gen_block_spinor(C_core, C_actv)
        B_so["cv"] = self.fock_builder.B_tensor_gen_block_spinor(C_core, C_virt)
        B_so["aa"] = self.fock_builder.B_tensor_gen_block_spinor(C_actv, C_actv)
        B_so["av"] = self.fock_builder.B_tensor_gen_block_spinor(C_actv, C_virt)

        ints["V"] = dict()
        ints["V"]["aaaa"] = np.einsum(
            "Bux,Bvy->uvxy",
            B_so["aa"],
            B_so["aa"],
            optimize=True,
        )
        ints["V"]["aaaa"] -= ints["V"]["aaaa"].swapaxes(2, 3)
        ints["V"]["caaa"] = np.einsum(
            "Biu,Bvw->ivuw",
            B_so["ca"],
            B_so["aa"],
            optimize=True,
        )
        ints["V"]["caaa"] -= ints["V"]["caaa"].swapaxes(2, 3)
        ints["V"]["aaav"] = np.einsum(
            "Buv,Bwa->uwva",
            B_so["aa"],
            B_so["av"],
            optimize=True,
        )
        ints["V"]["aaav"] -= ints["V"]["aaav"].swapaxes(0, 1)
        ints["V"]["ccaa"] = np.einsum(
            "Biu,Bjv->ijuv",
            B_so["ca"],
            B_so["ca"],
            optimize=True,
        )
        ints["V"]["ccaa"] -= ints["V"]["ccaa"].swapaxes(2, 3)
        ints["V"]["caav"] = np.einsum(
            "Biu,Bva->ivua",
            B_so["ca"],
            B_so["av"],
            optimize=True,
        )
        ints["V"]["caav"] -= np.einsum(
            "Bia,Bvu->ivua",
            B_so["cv"],
            B_so["aa"],
            optimize=True,
        )
        ints["V"]["aavv"] = np.einsum(
            "Bua,Bvb->uvab",
            B_so["av"],
            B_so["av"],
            optimize=True,
        )
        ints["V"]["aavv"] -= ints["V"]["aavv"].swapaxes(2, 3)

        # These are used in on-the-fly energy/Hbar computations
        ints["B"] = dict()
        ints["B"]["ca"] = B_so["ca"].transpose(1, 2, 0).copy()
        ints["B"]["cv"] = B_so["cv"].transpose(1, 2, 0).copy()
        ints["B"]["av"] = B_so["av"].transpose(1, 2, 0).copy()

        ints["eps"] = dict()
        ints["eps"]["c"] = self.eps[self.core].copy()
        ints["eps"]["a"] = self.eps[self.actv].copy()
        ints["eps"]["v"] = self.eps[self.virt].copy()
        # <Psi_0 | bare H | Psi_0>, where Psi_0 is the current (possibly relaxed) reference

        return ints, cumulants

    def solve_dsrg(self, form_hbar=False):
        self.T1, self.T2 = self._build_tamps()
        self.F_tilde = self._renormalize_F()
        # self.ints["V"] gets renormalizes to V_tilde in place for the following blocks:
        # caaa, aaav, ccaa, caav, aavv
        # The aaaa block is remains untouched, and can be safely used in reference relaxation
        self._renormalize_V_in_place()
        if form_hbar:
            self.hbar1 = np.zeros((self.nact, self.nact), dtype=complex)
        E = self._compute_pt2_energy(form_hbar=form_hbar)
        E += self.ints["E"]
        return E

    def _build_hbar(self):
        # hbar1/hbar2 are left untouched so that this can be called more than
        # once per solve (see DSRGBase._build_hbar).
        _hbar1 = self.hbar1 + self.fock[self.actv, self.actv]
        _hbar2 = self.hbar2 + self.ints["V"]["aaaa"]

        # see eq 29 of Ann. Rev. Phys. Chem. Built from the bare energy: an
        # incoming shift is added below, and E_dsrg already carries its energy
        # contribution.
        self._hbar0 = (
            -np.einsum("uv,uv->", _hbar1, self.cumulants["gamma1"])
            - 0.25 * np.einsum("uvxy,uvxy->", _hbar2, self.cumulants["lambda2"])
            + 0.5
            * np.einsum(
                "uvxy,ux,vy->",
                _hbar2,
                self.cumulants["gamma1"],
                self.cumulants["gamma1"],
            )
        ) + self._E_dsrg_bare

        _hbar1 = _hbar1 - np.einsum("uxvy,xy->uv", _hbar2, self.cumulants["gamma1"])

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

    def _build_tamps(self):
        t2 = dict()

        for key in ["caaa", "aaav", "ccaa", "caav", "aavv"]:
            t2[key] = self.ints["V"][key].conj()
            compute_t2_block(
                t2[key],
                *(self.ints["eps"][_] for _ in key),
                self.flow_param,
            )

        t1_tmp = self.ints["F"][self.hole, self.part].conj().copy()
        t2_hapa = np.zeros(
            (self.nhole, self.nact, self.npart, self.nact), dtype=complex
        )
        t2_hapa[self.hc, :, self.pa, :] = t2["caaa"].copy()
        t2_hapa[self.hc, :, self.pv, :] = -t2["caav"].swapaxes(2, 3).copy()
        t2_hapa[self.ha, :, self.pv, :] = -t2["aaav"].swapaxes(2, 3).copy()
        t1_tmp += np.einsum(
            "xu,iuax,xu->ia",
            self.delta_actv,
            t2_hapa,
            self.cumulants["gamma1"],
            optimize=True,
        )
        t1 = dict()
        t1["ca"] = t1_tmp[self.hc, self.pa].copy()
        t1["cv"] = t1_tmp[self.hc, self.pv].copy()
        t1["av"] = t1_tmp[self.ha, self.pv].copy()

        for key in ["ca", "cv", "av"]:
            compute_t1_block(
                t1[key],
                *(self.ints["eps"][_] for _ in key),
                self.flow_param,
            )

        return t1, t2

    def _renormalize_F(self):
        f_temp = np.conj(self.ints["F"][self.hole, self.part])
        delta_ia = self.eps[self.hole][:, None] - self.eps[self.part][None, :]
        exp_delta_1 = np.exp(-self.flow_param * delta_ia**2)
        t2_hapa = np.zeros(
            (self.nhole, self.nact, self.npart, self.nact), dtype=complex
        )
        t2_hapa[self.hc, :, self.pa, :] = self.T2["caaa"].copy()
        t2_hapa[self.hc, :, self.pv, :] = -self.T2["caav"].swapaxes(2, 3).copy()
        t2_hapa[self.ha, :, self.pv, :] = -self.T2["aaav"].swapaxes(2, 3).copy()
        f_temp += (
            f_temp * exp_delta_1
            + np.einsum(
                "xu,iuax,xu->ia",
                self.delta_actv,
                t2_hapa,
                self.cumulants["gamma1"],
                optimize=True,
            )
            * exp_delta_1
        )
        np.conj(f_temp, out=f_temp)
        F_tilde = dict()
        F_tilde["ca"] = f_temp[self.hc, self.pa].copy()
        F_tilde["cv"] = f_temp[self.hc, self.pv].copy()
        F_tilde["av"] = f_temp[self.ha, self.pv].copy()

        return F_tilde

    def _renormalize_V_in_place(self):
        V_tilde = self.ints["V"]
        for key in ["caaa", "aaav", "ccaa", "caav", "aavv"]:
            renormalize_V_block(
                V_tilde[key],
                *(self.ints["eps"][_] for _ in key),
                self.flow_param,
            )

    def _compute_pt2_energy(self, form_hbar=False):
        E = self.dsrg_helper.H_T_C0(
            self.F_tilde,
            self.ints["V"],
            self.T1,
            self.T2,
            self.cumulants,
            store_large=False,
        )
        E += self._compute_pt2_energy_ccvv()
        E += self._compute_pt2_energy_cavv(form_hbar=form_hbar)
        E += self._compute_pt2_energy_ccav(form_hbar=form_hbar)

        if form_hbar:
            self.hbar1 *= 0.5
            self.dsrg_helper.H_T_C1_aa(
                self.hbar1,
                self.F_tilde,
                self.ints["V"],
                self.T1,
                self.T2,
                self.cumulants,
                scale=0.5,
                store_large=False,
            )
            np.conj(self.hbar1, out=self.hbar1)
            self.hbar1 += self.hbar1.T.conj()
            self.hbar2 = np.zeros((self.nact,) * 4, dtype=complex)
            self.dsrg_helper.H_T_C2_aaaa(
                self.hbar2,
                self.F_tilde,
                self.ints["V"],
                self.T1,
                self.T2,
                self.cumulants,
                scale=0.5,
            )
            np.conj(self.hbar2, out=self.hbar2)
            hermitize_and_antisymmetrize_two_body_dense(self.hbar2)
        return E

    def _compute_pt2_energy_ccvv(self):
        # This computes the following contribution to the energy:
        # E += +0.250 * np.einsum("ijab,ijab->", T2["ccvv"], V["ccvv"], optimize=True)
        E = 0.0
        Vbare_i = np.empty((self.ncore, self.nvirt, self.nvirt), dtype=complex)
        Vtmp = np.empty((self.ncore, self.nvirt, self.nvirt), dtype=complex)
        Vr_i = np.empty((self.ncore, self.nvirt, self.nvirt), dtype=complex)
        B_cv = self.ints["B"]["cv"]
        for i in range(self.ncore):
            # T2 = conj(Vbare) * renorm
            # V = Vbare * (1 + exp)
            # So, we compute conj(Vbare) * Vr, where Vr = Vbare * renorm * (1 + exp)
            # this path is optimal because it is basically B_cv @ B_cv[i].T
            np.einsum("aB,jbB->jba", B_cv[i, :, :], B_cv, optimize=True, out=Vbare_i)
            np.copyto(Vtmp, Vbare_i.swapaxes(1, 2))
            Vbare_i -= Vtmp
            # copy to Vr_i
            Vr_i[:] = Vbare_i
            renormalize_3index(
                Vr_i,
                self.ints["eps"]["c"][i],
                self.ints["eps"]["c"],
                self.ints["eps"]["v"],
                self.ints["eps"]["v"],
                self.flow_param,
            )
            # equivalent to E += 0.250 * np.einsum("jba,jba->", Vbare_i.conj(), Vr_i, optimize=True)
            E += 0.250 * np.sum(Vbare_i.conj() * Vr_i)

        return E

    def _compute_pt2_energy_cavv(self, form_hbar=False):
        # This computes the following contribution to the energy:
        # E += +0.500 * np.einsum(
        #     "iuab,ivab,vu->",
        #     T2["cavv"],
        #     V["cavv"],
        #     gamma1,
        #     optimize=True,
        # )
        # If relaxing the reference, also compute the cavv contribution to Hbar_aa
        # _F += +0.500 * np.einsum(
        #     "iuab,ivab->uv",
        #     T2["cavv"],
        #     V["cavv"],
        #     optimize=True,
        # )
        E = 0.0
        Vbare_i = np.empty((self.nact, self.nvirt, self.nvirt), dtype=complex)
        Vtmp = np.empty((self.nact, self.nvirt, self.nvirt), dtype=complex)
        Vr_i = np.empty((self.nact, self.nvirt, self.nvirt), dtype=complex)
        B_av = self.ints["B"]["av"]
        B_cv = self.ints["B"]["cv"]
        for i in range(self.ncore):
            # T2 = conj(Vbare) * renorm
            # V = Vbare * (1 + exp)
            # So, we compute Vbare * Vr, where Vr = Vbare * renorm * (1 + exp)
            # again, this path is optimal because it is basically B_av @ B_cv[i].T
            np.einsum("aB,ubB->uba", B_cv[i, :, :], B_av, optimize=True, out=Vbare_i)
            np.copyto(Vtmp, Vbare_i.swapaxes(1, 2))
            Vbare_i -= Vtmp
            # copy to Vr_i
            Vr_i[:] = Vbare_i
            renormalize_3index(
                Vr_i,
                self.ints["eps"]["c"][i],
                self.ints["eps"]["a"],
                self.ints["eps"]["v"],
                self.ints["eps"]["v"],
                self.flow_param,
            )
            E += 0.500 * np.einsum(
                "uba,vba,vu->",
                Vbare_i.conj(),
                Vr_i,
                self.cumulants["gamma1"],
                optimize=True,
            )
            if form_hbar:
                # optimal path, fastest varying indices contracted away
                # self.hbar1 += 0.500 * np.einsum(
                #     "uba,vba->uv",
                #     Vbare_i.conj(),
                #     Vr_i,
                #     optimize=True,
                # )
                self.hbar1 += 0.500 * np.tensordot(
                    Vbare_i.conj(), Vr_i, axes=([1, 2], [1, 2])
                )

        return E

    def _compute_pt2_energy_ccav(self, form_hbar=False):
        # This computes the following contribution to the energy:
        # E += +0.500 * np.einsum(
        #     "ijua,ijva,uv->",
        #     T2["ccav"],
        #     V["ccav"],
        #     eta1,
        #     optimize=True,
        # )
        # If relaxing the reference, also compute the ccav contribution to Hbar_aa
        # _F += -0.500 * np.einsum(
        #     "ijua,ijva->vu",
        #     T2["ccav"],
        #     V["ccav"],
        #     optimize=True,
        # )

        E = 0.0
        Vbare_i = np.empty((self.ncore, self.nvirt, self.nact), dtype=complex)
        Vtmp = np.empty((self.ncore, self.nvirt, self.nact), dtype=complex)
        Vr_i = np.empty((self.ncore, self.nvirt, self.nact), dtype=complex)
        B_cv = self.ints["B"]["cv"]
        B_ca = self.ints["B"]["ca"]
        for i in range(self.ncore):
            # T2 = conj(Vbare) * renorm
            # V = Vbare * (1 + exp)
            # So, we compute conj(Vbare) * Vr, where Vr = Vbare * renorm * (1 + exp)
            np.einsum("uB,jaB->jau", B_ca[i, :, :], B_cv, optimize=True, out=Vbare_i)
            np.einsum("aB,juB->jau", B_cv[i, :, :], B_ca, optimize=True, out=Vtmp)
            Vbare_i -= Vtmp
            # copy to Vr_i
            Vr_i[:] = Vbare_i
            renormalize_3index(
                Vr_i,
                self.ints["eps"]["c"][i],
                self.ints["eps"]["c"],
                self.ints["eps"]["v"],
                self.ints["eps"]["a"],
                self.flow_param,
            )
            E += 0.500 * np.einsum(
                "jau,jav,uv->",
                Vbare_i.conj(),
                Vr_i,
                self.cumulants["eta1"],
                optimize=True,
            )
            if form_hbar:
                self.hbar1 += -0.500 * np.einsum(
                    "jau,jav->vu",
                    Vbare_i.conj(),
                    Vr_i,
                    optimize=True,
                )

        return E

    def compute_unrelaxed_gamma_vv(self):
        """
        Virtual-virtual block of the unrelaxed second-order 1-RDM,
        Gamma_ef = (1/2) <Phi0| [[E^e_f, A], A] |Phi0>
        Used to build FNOs; see eq 8-9 and Appendix A of
        Li, Mao, Huang, Evangelista, J. Chem. Theory Comput. 2024, 20, 4170-4181.
        "Unrelaxed" here is the response-theory sense of eq 7-8 of that paper (a
        plain expectation value, with no orbital or amplitude response
        contributions); it does not refer to DSRG reference relaxation.
        The reference (CASSCF) contribution to Gamma_ef is exactly zero, since
        virtual orbitals are unoccupied in every determinant of the CAS reference.
        Requires self.T1/self.T2 (i.e. solve_dsrg must have already run).
        The ccvv/cavv/ccav contributions are accumulated on the fly (mirroring
        _compute_pt2_energy_ccvv/_cavv/_ccav) since those T2 blocks aren't
        persisted by this class; the other terms use the persistently stored
        T1/T2 blocks directly.
        """
        gamma1 = self.cumulants["gamma1"]
        eta1 = self.cumulants["eta1"]
        lambda2 = self.cumulants["lambda2"]
        lambda3 = self.cumulants["lambda3"]
        T1 = self.T1
        T2 = self.T2

        Gamma = np.zeros((self.nvirt, self.nvirt), dtype=complex)
        Gamma += +1.000 * np.einsum(
            "ia,ib->ab", T1["cv"], T1["cv"].conj(), optimize=True
        )
        Gamma += +1.000 * np.einsum(
            "uv,va,ub->ab", gamma1, T1["av"], T1["av"].conj(), optimize=True
        )
        Gamma += -0.500 * np.einsum(
            "uvwx,ub,wxva->ab", lambda2, T1["av"].conj(), T2["aaav"], optimize=True
        )
        Gamma += -0.500 * np.einsum(
            "uvwx,wa,uvxb->ab", lambda2, T1["av"], T2["aaav"].conj(), optimize=True
        )
        Gamma += +1.000 * np.einsum(
            "uv,wx,ixua,iwvb->ab",
            eta1,
            gamma1,
            T2["caav"],
            T2["caav"].conj(),
            optimize=True,
        )
        Gamma += -1.000 * np.einsum(
            "uvwx,iwva,iuxb->ab", lambda2, T2["caav"], T2["caav"].conj(), optimize=True
        )
        Gamma += +0.500 * np.einsum(
            "uv,wx,yz,xzua,wyvb->ab",
            eta1,
            gamma1,
            gamma1,
            T2["aaav"],
            T2["aaav"].conj(),
            optimize=True,
        )
        Gamma += +0.250 * np.einsum(
            "uv,wxyz,yzua,wxvb->ab",
            eta1,
            lambda2,
            T2["aaav"],
            T2["aaav"].conj(),
            optimize=True,
        )
        Gamma += -1.000 * np.einsum(
            "uv,wxyz,vyxa,uwzb->ab",
            gamma1,
            lambda2,
            T2["aaav"],
            T2["aaav"].conj(),
            optimize=True,
        )
        Gamma += +0.500 * np.einsum(
            "uv,wx,vxac,uwbc->ab",
            gamma1,
            gamma1,
            T2["aavv"],
            T2["aavv"].conj(),
            optimize=True,
        )
        Gamma += +0.250 * np.einsum(
            "uvwx,wxac,uvbc->ab", lambda2, T2["aavv"], T2["aavv"].conj(), optimize=True
        )
        Gamma += -0.250000 * np.einsum(
            "uvwxyz,xywa,uvzb->ab",
            lambda3,
            T2["aaav"],
            T2["aaav"].conj(),
            optimize=True,
        )

        Gamma += self._compute_gamma_vv_ccvv()
        Gamma += self._compute_gamma_vv_cavv()
        Gamma += self._compute_gamma_vv_ccav()

        # defensive Hermitization - Gamma should already be hermitian, this removes numerical noise
        Gamma = 0.5 * (Gamma + Gamma.conj().T)
        return Gamma

    def _compute_gamma_vv_ccvv(self):
        # Gamma["vv"] += 0.500 * einsum("ijac,ijbc->ab", T2["ccvv"], T2["ccvv"].conj())
        Gamma = np.zeros((self.nvirt, self.nvirt), dtype=complex)
        Vbare_i = np.empty((self.ncore, self.nvirt, self.nvirt), dtype=complex)
        Vtmp = np.empty((self.ncore, self.nvirt, self.nvirt), dtype=complex)
        B_cv = self.ints["B"]["cv"]
        for i in range(self.ncore):
            np.einsum("aB,jbB->jba", B_cv[i, :, :], B_cv, optimize=True, out=Vbare_i)
            np.copyto(Vtmp, Vbare_i.swapaxes(1, 2))
            Vbare_i -= Vtmp
            T2_i = Vbare_i.conj()
            compute_t2_block(
                T2_i[None, :, :, :],
                self.ints["eps"]["c"][i : i + 1],
                self.ints["eps"]["c"],
                self.ints["eps"]["v"],
                self.ints["eps"]["v"],
                self.flow_param,
            )
            Gamma += 0.500 * np.einsum("jec,jfc->ef", T2_i, T2_i.conj(), optimize=True)
        return Gamma

    def _compute_gamma_vv_cavv(self):
        # Gamma["vv"] += 1.000 * einsum("uv,ivac,iubc->ab", gamma1, T2["cavv"], T2["cavv"].conj())
        Gamma = np.zeros((self.nvirt, self.nvirt), dtype=complex)
        Vbare_i = np.empty((self.nact, self.nvirt, self.nvirt), dtype=complex)
        Vtmp = np.empty((self.nact, self.nvirt, self.nvirt), dtype=complex)
        B_av = self.ints["B"]["av"]
        B_cv = self.ints["B"]["cv"]
        gamma1 = self.cumulants["gamma1"]
        for i in range(self.ncore):
            np.einsum("aB,ubB->uba", B_cv[i, :, :], B_av, optimize=True, out=Vbare_i)
            np.copyto(Vtmp, Vbare_i.swapaxes(1, 2))
            Vbare_i -= Vtmp
            T2_i = Vbare_i.conj()
            compute_t2_block(
                T2_i[None, :, :, :],
                self.ints["eps"]["c"][i : i + 1],
                self.ints["eps"]["a"],
                self.ints["eps"]["v"],
                self.ints["eps"]["v"],
                self.flow_param,
            )
            Gamma += np.einsum(
                "uv,vec,ufc->ef", gamma1, T2_i, T2_i.conj(), optimize=True
            )
        return Gamma

    def _compute_gamma_vv_ccav(self):
        # Gamma["vv"] += 0.500 * einsum("uv,ijua,ijvb->ab", eta1, T2["ccav"], T2["ccav"].conj())
        Gamma = np.zeros((self.nvirt, self.nvirt), dtype=complex)
        Vbare_i = np.empty((self.ncore, self.nvirt, self.nact), dtype=complex)
        Vtmp = np.empty((self.ncore, self.nvirt, self.nact), dtype=complex)
        B_cv = self.ints["B"]["cv"]
        B_ca = self.ints["B"]["ca"]
        eta1 = self.cumulants["eta1"]
        for i in range(self.ncore):
            np.einsum("uB,jaB->jau", B_ca[i, :, :], B_cv, optimize=True, out=Vbare_i)
            np.einsum("aB,juB->jau", B_cv[i, :, :], B_ca, optimize=True, out=Vtmp)
            Vbare_i -= Vtmp
            T2_i = Vbare_i.conj()
            compute_t2_block(
                T2_i[None, :, :, :],
                self.ints["eps"]["c"][i : i + 1],
                self.ints["eps"]["c"],
                self.ints["eps"]["v"],
                self.ints["eps"]["a"],
                self.flow_param,
            )
            Gamma += 0.500 * np.einsum(
                "uv,jeu,jfv->ef", eta1, T2_i, T2_i.conj(), optimize=True
            )
        return Gamma
