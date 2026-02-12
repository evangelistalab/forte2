from dataclasses import dataclass

import numpy as np

from .dsrg_base import DSRGBase
from .utils import (
    antisymmetrize_2body,
    cas_energy_given_RDMs,
    compute_t1_block,
    compute_t2_block,
    renormalize_V_block,
)


@dataclass
class RelDSRG_MRPT2_Slow(DSRGBase):
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

        ints["E"] = cas_energy_given_RDMs(
            self.E_core_orig, self.H_orig, self.V_orig, g1, g2
        )

        # Save blocks of spinorbital basis B tensor
        B_so = dict()
        C_core = self._C_semican[:, self.core]
        C_actv = self._C_semican[:, self.actv]
        C_virt = self._C_semican[:, self.virt]
        # B_so["cc"] = self.fock_builder.B_tensor_gen_block_spinor(C_core, C_core)
        # B_so["ca"] = self.fock_builder.B_tensor_gen_block_spinor(C_core, C_actv)
        # B_so["cv"] = self.fock_builder.B_tensor_gen_block_spinor(C_core, C_virt)
        # B_so["aa"] = self.fock_builder.B_tensor_gen_block_spinor(C_actv, C_actv)
        # B_so["av"] = self.fock_builder.B_tensor_gen_block_spinor(C_actv, C_virt)
        V_so = self.fock_builder.two_electron_integrals_block_spinor(self._C_semican)
        V_so = V_so - V_so.swapaxes(2, 3)  # antisymmetrize

        ints["V"] = dict()
        ints["V"]["aaaa"] = V_so[self.actv, self.actv, self.actv, self.actv].copy()
        ints["V"]["caaa"] = V_so[self.core, self.actv, self.actv, self.actv].copy()
        ints["V"]["aaav"] = V_so[self.actv, self.actv, self.actv, self.virt].copy()
        ints["V"]["ccaa"] = V_so[self.core, self.core, self.actv, self.actv].copy()
        ints["V"]["caav"] = V_so[self.core, self.actv, self.actv, self.virt].copy()
        ints["V"]["aavv"] = V_so[self.actv, self.actv, self.virt, self.virt].copy()
        ints["V"]["cavv"] = V_so[self.core, self.actv, self.virt, self.virt].copy()
        ints["V"]["ccvv"] = V_so[self.core, self.core, self.virt, self.virt].copy()
        ints["V"]["ccav"] = V_so[self.core, self.core, self.actv, self.virt].copy()

        # These are used in on-the-fly energy/Hbar computations
        # ints["B"] = dict()
        # ints["B"]["ca"] = B_so["ca"].transpose(1, 2, 0).copy()
        # ints["B"]["cv"] = B_so["cv"].transpose(1, 2, 0).copy()
        # ints["B"]["av"] = B_so["av"].transpose(1, 2, 0).copy()

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
        # caaa, aaav, ccaa, caav, aavv, cavv, ccvv, ccav
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
        e_relaxed = self.ci_solver.compute_average_energy()
        self.relax_eigvals = self.ci_solver.evals_flat.copy()
        return e_relaxed

    def _build_tamps(self):
        t2 = dict()

        for key in ["caaa", "aaav", "ccaa", "caav", "aavv", "cavv", "ccvv", "ccav"]:
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
        for key in ["caaa", "aaav", "ccaa", "caav", "aavv", "cavv", "ccvv", "ccav"]:
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
        E += +0.250 * np.einsum(
            "ijab,ijab->", self.T2["ccvv"], self.ints["V"]["ccvv"], optimize=True
        )
        E += +0.500 * np.einsum(
            "iuab,ivab,vu->",
            self.T2["cavv"],
            self.ints["V"]["cavv"],
            self.cumulants["gamma1"],
            optimize=True,
        )
        E += +0.500 * np.einsum(
            "ijua,ijva,uv->",
            self.T2["ccav"],
            self.ints["V"]["ccav"],
            self.cumulants["eta1"],
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
        _F += -0.500 * np.einsum(
            "ijua,ijva->vu",
            self.T2["ccav"],
            self.ints["V"]["ccav"],
            optimize=True,
        )
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
        _F += +0.500 * np.einsum(
            "iuab,ivab->uv",
            self.T2["cavv"],
            self.ints["V"]["cavv"],
            optimize=True,
        )
        _F += +0.500 * np.einsum(
            "uwab,vxab,xw->uv",
            self.T2["aavv"],
            self.ints["V"]["aavv"],
            self.cumulants["gamma1"],
            optimize=True,
        )

        return _F.conj()
