from dataclasses import dataclass

import numpy as np

from forte2.helpers import logger
from .dsrg_base import DSRGBase
from .utils import (
    antisymmetrize_2body,
    cas_energy_given_cumulants,
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
    """

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

        ints["E"] = cas_energy_given_cumulants(
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
            self.hbar_aa_df = np.zeros((self.nact, self.nact), dtype=complex)
        E = self._compute_pt2_energy(form_hbar=form_hbar)
        E += self.ints["E"]
        return E

    def do_reference_relaxation(self):
        _hbar2 = self.ints["V"]["aaaa"].copy()
        _C2 = 0.5 * self._compute_Hbar_aaaa()
        # 0.5*[H, T-T+] = 0.5*([H, T] + [H, T]+)
        _hbar2 += _C2 + np.einsum("ijab->abij", np.conj(_C2))

        _hbar1 = self.fock[self.actv, self.actv].copy()
        _C1 = 0.5 * self._compute_Hbar_aa()
        # 0.5*[H, T-T+] = 0.5*([H, T] + [H, T]+)
        _hbar1 += _C1 + _C1.conj().T

        # see eq 29 of Ann. Rev. Phys. Chem.
        _e_scalar = (
            -np.einsum("uv,uv->", _hbar1, self.cumulants["gamma1"])
            - 0.25 * np.einsum("uvxy,uvxy->", _hbar2, self.cumulants["lambda2"])
            + 0.5
            * np.einsum(
                "uvxy,ux,vy->",
                _hbar2,
                self.cumulants["gamma1"],
                self.cumulants["gamma1"],
            )
        ) + self.E_dsrg

        _hbar1 -= np.einsum("uxvy,xy->uv", _hbar2, self.cumulants["gamma1"])

        _hbar1_canon = np.einsum(
            "ip,pq,jq->ij", self.Uactv, _hbar1, self.Uactv.conj(), optimize=True
        )
        _hbar2_canon = np.einsum(
            "ip,jq,pqrs,kr,ls->ijkl",
            self.Uactv,
            self.Uactv,
            _hbar2,
            self.Uactv.conj(),
            self.Uactv.conj(),
            optimize=True,
        )

        self.ci_solver.set_ints(_e_scalar, _hbar1_canon, _hbar2_canon)
        self.ci_solver.run(use_asym_ints=True)
        self._reorder_weights()
        e_relaxed = self.ci_solver.compute_average_energy()
        self.relax_eigvals = self.ci_solver.evals_flat.copy()
        return e_relaxed

    def _reorder_weights(self):
        for i, solver in enumerate(self.ci_solver.sub_solvers):
            overlap = np.abs(solver.evecs.conj().T @ self.ci_evecs_prev[i])
            max_overlap = np.max(overlap, axis=1)
            permutation = np.argmax(overlap, axis=1)
            do_warn = len(permutation) != len(set(permutation)) or np.any(
                max_overlap <= 0.5
            )
            if do_warn:
                logger.log_warning(
                    f"DSRG reference relaxation: Relaxed states in sub-solver {i} are likely wrong due to root flipping."
                    "Please increase the number of states in the sub-solver."
                )
                logger.log_warning(f"Max overlap for sub-solver {i}: {max_overlap}")
                logger.log_warning(f"Permutation attempted for sub-solver {i}: {permutation}")
                logger.log_warning(f"Overlap matrix (<current | previous>):\n{overlap}")
            else:
                if np.allclose(permutation, np.arange(len(permutation))):
                    continue  # no need to reorder
                new_weights = solver.weights[permutation].copy()
                self.ci_solver.update_weights(i, new_weights)
                self.ci_evecs_prev[i] = solver.evecs[:, permutation].copy()
            

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
        f_temp = np.conj(self.ints["F"][self.hole, self.part]).copy()
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
        E = 0.0

        E += +1.000 * np.einsum(
            "iu,iv,vu->",
            self.F_tilde["ca"],
            self.T1["ca"],
            self.cumulants["eta1"],
            optimize=True,
        )
        E += +1.000 * np.einsum(
            "ia,ia->",
            self.F_tilde["cv"],
            self.T1["cv"],
            optimize=True,
        )
        E += +1.000 * np.einsum(
            "ua,va,uv->",
            self.F_tilde["av"],
            self.T1["av"],
            self.cumulants["gamma1"],
            optimize=True,
        )
        E += -0.500 * np.einsum(
            "iu,ixvw,vwux->",
            self.F_tilde["ca"],
            self.T2["caaa"],
            self.cumulants["lambda2"],
            optimize=True,
        )
        E += -0.500 * np.einsum(
            "ua,wxva,uvwx->",
            self.F_tilde["av"],
            self.T2["aaav"],
            self.cumulants["lambda2"],
            optimize=True,
        )
        E += -0.500 * np.einsum(
            "iu,ivwx,uvwx->",
            self.T1["ca"],
            self.ints["V"]["caaa"],
            self.cumulants["lambda2"],
            optimize=True,
        )
        E += -0.500 * np.einsum(
            "ua,vwxa,vwux->",
            self.T1["av"],
            self.ints["V"]["aaav"],
            self.cumulants["lambda2"],
            optimize=True,
        )
        E += +0.250 * np.einsum(
            "ijuv,ijwx,vx,uw->",
            self.T2["ccaa"],
            self.ints["V"]["ccaa"],
            self.cumulants["eta1"],
            self.cumulants["eta1"],
            optimize=True,
        )
        E += +0.125 * np.einsum(
            "ijuv,ijwx,uvwx->",
            self.T2["ccaa"],
            self.ints["V"]["ccaa"],
            self.cumulants["lambda2"],
            optimize=True,
        )
        E += +0.500 * np.einsum(
            "iwuv,ixyz,vz,uy,xw->",
            self.T2["caaa"],
            self.ints["V"]["caaa"],
            self.cumulants["eta1"],
            self.cumulants["eta1"],
            self.cumulants["gamma1"],
            optimize=True,
        )
        E += +1.000 * np.einsum(
            "iwuv,ixyz,vz,uxwy->",
            self.T2["caaa"],
            self.ints["V"]["caaa"],
            self.cumulants["eta1"],
            self.cumulants["lambda2"],
            optimize=True,
        )
        E += +0.250 * np.einsum(
            "iwuv,ixyz,xw,uvyz->",
            self.T2["caaa"],
            self.ints["V"]["caaa"],
            self.cumulants["gamma1"],
            self.cumulants["lambda2"],
            optimize=True,
        )
        E += +0.250 * np.einsum(
            "iwuv,ixyz,uvxwyz->",
            self.T2["caaa"],
            self.ints["V"]["caaa"],
            self.cumulants["lambda3"],
            optimize=True,
        )
        E += +1.000 * np.einsum(
            "ivua,iwxa,ux,wv->",
            self.T2["caav"],
            self.ints["V"]["caav"],
            self.cumulants["eta1"],
            self.cumulants["gamma1"],
            optimize=True,
        )
        E += +1.000 * np.einsum(
            "ivua,iwxa,uwvx->",
            self.T2["caav"],
            self.ints["V"]["caav"],
            self.cumulants["lambda2"],
            optimize=True,
        )
        E += +0.500 * np.einsum(
            "vwua,xyza,uz,yw,xv->",
            self.T2["aaav"],
            self.ints["V"]["aaav"],
            self.cumulants["eta1"],
            self.cumulants["gamma1"],
            self.cumulants["gamma1"],
            optimize=True,
        )
        E += +0.250 * np.einsum(
            "vwua,xyza,uz,xyvw->",
            self.T2["aaav"],
            self.ints["V"]["aaav"],
            self.cumulants["eta1"],
            self.cumulants["lambda2"],
            optimize=True,
        )
        E += +1.000 * np.einsum(
            "vwua,xyza,yw,uxvz->",
            self.T2["aaav"],
            self.ints["V"]["aaav"],
            self.cumulants["gamma1"],
            self.cumulants["lambda2"],
            optimize=True,
        )
        E += -0.250 * np.einsum(
            "vwua,xyza,uxyvwz->",
            self.T2["aaav"],
            self.ints["V"]["aaav"],
            self.cumulants["lambda3"],
            optimize=True,
        )
        E += +0.250 * np.einsum(
            "uvab,wxab,xv,wu->",
            self.T2["aavv"],
            self.ints["V"]["aavv"],
            self.cumulants["gamma1"],
            self.cumulants["gamma1"],
            optimize=True,
        )
        E += +0.125 * np.einsum(
            "uvab,wxab,wxuv->",
            self.T2["aavv"],
            self.ints["V"]["aavv"],
            self.cumulants["lambda2"],
            optimize=True,
        )
        E += self._compute_pt2_energy_ccvv()
        E += self._compute_pt2_energy_cavv(form_hbar=form_hbar)
        E += self._compute_pt2_energy_ccav(form_hbar=form_hbar)

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
                "uba,vba,uv->",
                Vbare_i.conj(),
                Vr_i,
                self.cumulants["gamma1"],
                optimize=True,
            )
            if form_hbar:
                # optimal path, fastest varying indices contracted away
                # self.hbar_aa_df += 0.500 * np.einsum(
                #     "uba,vba->uv",
                #     Vbare_i.conj(),
                #     Vr_i,
                #     optimize=True,
                # )
                self.hbar_aa_df += 0.500 * np.tensordot(
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
                self.hbar_aa_df += -0.500 * np.einsum(
                    "jau,jav->vu",
                    Vbare_i.conj(),
                    Vr_i,
                    optimize=True,
                )

        return E

    def _compute_Hbar_aaaa(self):
        _V = np.zeros((self.nact,) * 4, dtype=complex)
        _V += -0.500 * np.einsum(
            "ua,wxva->wxuv",
            self.F_tilde["av"],
            self.T2["aaav"],
            optimize=True,
        )
        _V += -0.500 * np.einsum(
            "iu,ixvw->uxvw",
            self.F_tilde["ca"],
            self.T2["caaa"],
            optimize=True,
        )
        _V += -0.500 * np.einsum(
            "iu,ivwx->wxuv",
            self.T1["ca"],
            self.ints["V"]["caaa"],
            optimize=True,
        )
        _V += -0.500 * np.einsum(
            "ua,vwxa->uxvw",
            self.T1["av"],
            self.ints["V"]["aaav"],
            optimize=True,
        )
        _V += +0.125 * np.einsum(
            "uvab,wxab->uvwx",
            self.T2["aavv"],
            self.ints["V"]["aavv"],
            optimize=True,
        )
        _V += +0.250 * np.einsum(
            "uvya,wxza,yz->uvwx",
            self.T2["aaav"],
            self.ints["V"]["aaav"],
            self.cumulants["eta1"],
            optimize=True,
        )
        _V += +0.125 * np.einsum(
            "ijuv,ijwx->wxuv",
            self.T2["ccaa"],
            self.ints["V"]["ccaa"],
            optimize=True,
        )
        _V += +0.250 * np.einsum(
            "iyuv,izwx,zy->wxuv",
            self.T2["caaa"],
            self.ints["V"]["caaa"],
            self.cumulants["gamma1"],
            optimize=True,
        )
        _V += +1.000 * np.einsum(
            "ivua,iwxa->vxuw",
            self.T2["caav"],
            self.ints["V"]["caav"],
            optimize=True,
        )
        _V += +1.000 * np.einsum(
            "ivuy,iwxz,yz->vxuw",
            self.T2["caaa"],
            self.ints["V"]["caaa"],
            self.cumulants["eta1"],
            optimize=True,
        )
        _V += +1.000 * np.einsum(
            "vyua,wzxa,zy->vxuw",
            self.T2["aaav"],
            self.ints["V"]["aaav"],
            self.cumulants["gamma1"],
            optimize=True,
        )

        return antisymmetrize_2body(_V.conj(), "aaaa")

    def _compute_Hbar_aa(self):
        _F = self.hbar_aa_df.copy()
        _F += -1.000 * np.einsum(
            "iu,iv->uv",
            self.F_tilde["ca"],
            self.T1["ca"],
            optimize=True,
        )
        _F += -1.000 * np.einsum(
            "iw,ivux,xw->vu",
            self.F_tilde["ca"],
            self.T2["caaa"],
            self.cumulants["eta1"],
            optimize=True,
        )
        _F += -1.000 * np.einsum(
            "ia,ivua->vu",
            self.F_tilde["cv"],
            self.T2["caav"],
            optimize=True,
        )
        _F += +1.000 * np.einsum(
            "ua,va->vu",
            self.F_tilde["av"],
            self.T1["av"],
            optimize=True,
        )
        _F += +1.000 * np.einsum(
            "wa,vxua,wx->vu",
            self.F_tilde["av"],
            self.T2["aaav"],
            self.cumulants["gamma1"],
            optimize=True,
        )
        _F += -1.000 * np.einsum(
            "iw,iuvx,wx->vu",
            self.T1["ca"],
            self.ints["V"]["caaa"],
            self.cumulants["eta1"],
            optimize=True,
        )
        _F += -1.000 * np.einsum(
            "ia,iuva->vu",
            self.T1["cv"],
            self.ints["V"]["caav"],
            optimize=True,
        )
        _F += +1.000 * np.einsum(
            "wa,uxva,xw->vu",
            self.T1["av"],
            self.ints["V"]["aaav"],
            self.cumulants["gamma1"],
            optimize=True,
        )
        _F += -0.500 * np.einsum(
            "ijuw,ijvx,wx->vu",
            self.T2["ccaa"],
            self.ints["V"]["ccaa"],
            self.cumulants["eta1"],
            optimize=True,
        )
        _F += +0.500 * np.einsum(
            "ivuw,ixyz,wxyz->vu",
            self.T2["caaa"],
            self.ints["V"]["caaa"],
            self.cumulants["lambda2"],
            optimize=True,
        )
        _F += -1.000 * np.einsum(
            "ixuw,iyvz,wz,yx->vu",
            self.T2["caaa"],
            self.ints["V"]["caaa"],
            self.cumulants["eta1"],
            self.cumulants["gamma1"],
            optimize=True,
        )
        _F += -1.000 * np.einsum(
            "ixuw,iyvz,wyxz->vu",
            self.T2["caaa"],
            self.ints["V"]["caaa"],
            self.cumulants["lambda2"],
            optimize=True,
        )
        # _F += -0.500 * np.einsum(
        #     "ijua,ijva->vu",
        #     self.T2["ccav"],
        #     self.ints["V"]["ccav"],
        #     optimize=True,
        # )
        _F += -1.000 * np.einsum(
            "iwua,ixva,xw->vu",
            self.T2["caav"],
            self.ints["V"]["caav"],
            self.cumulants["gamma1"],
            optimize=True,
        )
        _F += -0.500 * np.einsum(
            "vwua,xyza,xywz->vu",
            self.T2["aaav"],
            self.ints["V"]["aaav"],
            self.cumulants["lambda2"],
            optimize=True,
        )
        _F += -0.500 * np.einsum(
            "wxua,yzva,zx,yw->vu",
            self.T2["aaav"],
            self.ints["V"]["aaav"],
            self.cumulants["gamma1"],
            self.cumulants["gamma1"],
            optimize=True,
        )
        _F += -0.250 * np.einsum(
            "wxua,yzva,yzwx->vu",
            self.T2["aaav"],
            self.ints["V"]["aaav"],
            self.cumulants["lambda2"],
            optimize=True,
        )
        _F += +0.500 * np.einsum(
            "iuwx,ivyz,xz,wy->uv",
            self.T2["caaa"],
            self.ints["V"]["caaa"],
            self.cumulants["eta1"],
            self.cumulants["eta1"],
            optimize=True,
        )
        _F += +0.250 * np.einsum(
            "iuwx,ivyz,wxyz->uv",
            self.T2["caaa"],
            self.ints["V"]["caaa"],
            self.cumulants["lambda2"],
            optimize=True,
        )
        _F += -0.500 * np.einsum(
            "iywx,iuvz,wxyz->vu",
            self.T2["caaa"],
            self.ints["V"]["caaa"],
            self.cumulants["lambda2"],
            optimize=True,
        )
        _F += +1.000 * np.einsum(
            "iuwa,ivxa,wx->uv",
            self.T2["caav"],
            self.ints["V"]["caav"],
            self.cumulants["eta1"],
            optimize=True,
        )
        _F += +1.000 * np.einsum(
            "uxwa,vyza,wz,yx->uv",
            self.T2["aaav"],
            self.ints["V"]["aaav"],
            self.cumulants["eta1"],
            self.cumulants["gamma1"],
            optimize=True,
        )
        _F += +1.000 * np.einsum(
            "uxwa,vyza,wyxz->uv",
            self.T2["aaav"],
            self.ints["V"]["aaav"],
            self.cumulants["lambda2"],
            optimize=True,
        )
        _F += +0.500 * np.einsum(
            "xywa,uzva,wzxy->vu",
            self.T2["aaav"],
            self.ints["V"]["aaav"],
            self.cumulants["lambda2"],
            optimize=True,
        )
        # _F += +0.500 * np.einsum(
        #     "iuab,ivab->uv",
        #     self.T2["cavv"],
        #     self.ints["V"]["cavv"],
        #     optimize=True,
        # )
        _F += +0.500 * np.einsum(
            "uwab,vxab,xw->uv",
            self.T2["aavv"],
            self.ints["V"]["aavv"],
            self.cumulants["gamma1"],
            optimize=True,
        )

        return _F.conj()
