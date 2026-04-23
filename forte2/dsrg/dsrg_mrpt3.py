from dataclasses import dataclass

import numpy as np

from .dsrg_base import DSRGBase
from .utils import (
    cas_energy_given_RDMs,
    compute_t1_block,
    compute_t2_block,
    renormalize_V_block,
    renormalize_3index,
)


@dataclass
class DSRG_MRPT3(DSRGBase):
    """
    Spin-adapted driven similarity renormalization group
    third-order multireference perturbation theory (DSRG-MRPT3).

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

        ints["E"] = cas_energy_given_RDMs(
            self.E_core_orig, self.H_orig, self.V_orig, g1, g2
        )
        #  ["vvaa", "aacc", "avca", "avac", "vaaa", "aaca", "aaaa"]
        # Save blocks of spinorbital basis B tensor
        B_mo = dict()
        C_core = self._C_semican[:, self.core]
        C_actv = self._C_semican[:, self.actv]
        C_virt = self._C_semican[:, self.virt]
        B_mo["cc"] = self.fock_builder.B_tensor_gen_block(C_core, C_core)
        B_mo["ac"] = self.fock_builder.B_tensor_gen_block(C_actv, C_core)
        B_mo["vc"] = self.fock_builder.B_tensor_gen_block(C_virt, C_core)
        B_mo["aa"] = self.fock_builder.B_tensor_gen_block(C_actv, C_actv)
        B_mo["va"] = self.fock_builder.B_tensor_gen_block(C_virt, C_actv)

        ints["V"] = dict()
        ints["V"]["vvaa"] = np.einsum(
            "Bau,Bbv->abuv",
            B_mo["va"],
            B_mo["va"],
            optimize=True,
        )
        ints["V"]["aacc"] = np.einsum(
            "Bui,Bvj->uvij",
            B_mo["ac"],
            B_mo["ac"],
            optimize=True,
        )
        ints["V"]["avca"] = np.einsum(
            "Bui,Bav->uaiv",
            B_mo["ac"],
            B_mo["va"],
            optimize=True,
        )
        ints["V"]["avac"] = np.einsum(
            "Buv,Bai->uavi",
            B_mo["aa"],
            B_mo["vc"],
            optimize=True,
        )
        ints["V"]["vaaa"] = np.einsum(
            "Bav,Bux->auvx",
            B_mo["va"],
            B_mo["aa"],
            optimize=True,
        )
        ints["V"]["aaca"] = np.einsum(
            "Bui,Bvx->uvix",
            B_mo["ac"],
            B_mo["aa"],
            optimize=True,
        )
        ints["V"]["aaaa"] = np.einsum(
            "Bux,Bvy->uvxy",
            B_mo["aa"],
            B_mo["aa"],
            optimize=True,
        )
        ints["V"]["vvcc"] = np.einsum(
            "Bai,Bbj->abij",
            B_mo["vc"],
            B_mo["vc"],
            optimize=True,
        )
        ints["V"]["vvac"] = np.einsum(
            "Bau,Bbj->abuj",
            B_mo["va"],
            B_mo["vc"],
            optimize=True,
        )
        ints["V"]["vacc"] = np.einsum(
            "Bai,Buj->auij",
            B_mo["vc"],
            B_mo["ac"],
            optimize=True,
        )

        ints["eps"] = dict()
        ints["eps"]["c"] = self.eps[self.core].copy()
        ints["eps"]["a"] = self.eps[self.actv].copy()
        ints["eps"]["v"] = self.eps[self.virt].copy()
        ints["eps"]["h"] = self.eps[self.hole].copy()
        ints["eps"]["p"] = self.eps[self.part].copy()
        # <Psi_0 | bare H | Psi_0>, where Psi_0 is the current (possibly relaxed) reference

        return ints, cumulants

    def solve_dsrg(self, form_hbar=False):
        self.T1, self.T2 = self._build_tamps()
        self.F_tilde = self._renormalize_F()
        # The aaaa block is remains untouched, and can be safely used in reference relaxation
        self._renormalize_V_in_place()
        E = self._compute_pt2_energy()
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

        hbar2_temp = 2 * _hbar2 - _hbar2.swapaxes(2, 3)

        _e_scalar = self.E_dsrg
        _e_scalar -= np.einsum("vu,vu->", _hbar1, self.cumulants["gamma1"])
        _e_scalar += 0.25 * np.einsum(
            "uv,vyux,xy->",
            self.cumulants["gamma1"],
            hbar2_temp,
            self.cumulants["gamma1"],
        )
        _e_scalar -= 0.5 * np.einsum("xyuv,uvxy->", _hbar2, self.cumulants["lambda2"])

        _hbar1 -= 0.5 * np.einsum("uxvy,yx->uv", hbar2_temp, self.cumulants["gamma1"])

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
        self.ci_solver.run()
        e_relaxed = self.ci_solver.compute_average_energy()
        self.relax_eigvals = self.ci_solver.evals_flat.copy()
        return e_relaxed

    def _build_tamps(self):
        t2 = {"T2": dict(), "S2": dict()}

        for key in [
            "aavv",
            "ccaa",
            "caav",
            "acav",
            "aava",
            "caaa",
            "ccvv",
            "acvv",
            "ccva",
        ]:
            vkey = key[2:] + key[:2]  # e.g., caaa -> aaca
            t2["T2"][key] = self.ints["V"][vkey].transpose(2, 3, 0, 1).copy()
            compute_t2_block(
                t2["T2"][key],
                *(self.ints["eps"][_] for _ in key),
                self.flow_param,
            )

        t2["S2"]["aavv"] = 2 * t2["T2"]["aavv"] - t2["T2"]["aavv"].swapaxes(2, 3)
        t2["S2"]["ccaa"] = 2 * t2["T2"]["ccaa"] - t2["T2"]["ccaa"].swapaxes(2, 3)
        t2["S2"]["caav"] = 2 * t2["T2"]["caav"] - t2["T2"]["acav"].swapaxes(0, 1)
        t2["S2"]["acav"] = 2 * t2["T2"]["acav"] - t2["T2"]["caav"].swapaxes(0, 1)
        t2["S2"]["aava"] = 2 * t2["T2"]["aava"] - t2["T2"]["aava"].swapaxes(0, 1)
        t2["S2"]["caaa"] = 2 * t2["T2"]["caaa"] - t2["T2"]["caaa"].swapaxes(2, 3)
        t2["S2"]["ccvv"] = 2 * t2["T2"]["ccvv"] - t2["T2"]["ccvv"].swapaxes(2, 3)
        t2["S2"]["acvv"] = 2 * t2["T2"]["acvv"] - t2["T2"]["acvv"].swapaxes(2, 3)
        t2["S2"]["ccva"] = 2 * t2["T2"]["ccva"] - t2["T2"]["ccva"].swapaxes(0, 1)

        t1 = self.fock[self.hole, self.part].copy()
        faa = self.fock[self.actv, self.actv]
        t1[self.hc, self.pa] += 0.5 * np.einsum(
            "ivaw,wu,uv->ia",
            t2["S2"]["caaa"],
            faa,
            self.cumulants["gamma1"],
            optimize=True,
        )
        t1[self.hc, self.pv] += 0.5 * np.einsum(
            "vmwe,wu,uv->me",
            t2["S2"]["acav"],
            faa,
            self.cumulants["gamma1"],
            optimize=True,
        )
        t1[self.ha, self.pv] += 0.5 * np.einsum(
            "ivaw,wu,uv->ia",
            t2["S2"]["aava"],
            faa,
            self.cumulants["gamma1"],
            optimize=True,
        )
        t1[self.hc, self.pa] -= 0.5 * np.einsum(
            "iwau,vw,uv->ia",
            t2["S2"]["caaa"],
            faa,
            self.cumulants["gamma1"],
            optimize=True,
        )
        t1[self.hc, self.pv] -= 0.5 * np.einsum(
            "wmue,vw,uv->me",
            t2["S2"]["acav"],
            faa,
            self.cumulants["gamma1"],
            optimize=True,
        )
        t1[self.ha, self.pv] -= 0.5 * np.einsum(
            "iwau,vw,uv->ia",
            t2["S2"]["aava"],
            faa,
            self.cumulants["gamma1"],
            optimize=True,
        )

        compute_t1_block(
            t1,
            self.ints["eps"]["h"],
            self.ints["eps"]["p"],
            self.flow_param,
        )

        t1[self.ha, self.pa] = 0.0
        return t1, t2

    def _renormalize_F(self):
        faa = self.fock[self.actv, self.actv]
        F_tilde = self.fock[self.part, self.hole].copy()
        F_tilde[self.pa, self.hc] += 0.5 * np.einsum(
            "ivaw,wu,uv->ai",
            self.T2["S2"]["caaa"],
            faa,
            self.cumulants["gamma1"],
            optimize=True,
        )
        F_tilde[self.pa, self.hc] -= 0.5 * np.einsum(
            "iwau,vw,uv->ai",
            self.T2["S2"]["caaa"],
            faa,
            self.cumulants["gamma1"],
            optimize=True,
        )

        F_tilde[self.pv, self.hc] += 0.5 * np.einsum(
            "vmwe,wu,uv->em",
            self.T2["S2"]["acav"],
            faa,
            self.cumulants["gamma1"],
            optimize=True,
        )
        F_tilde[self.pv, self.hc] -= 0.5 * np.einsum(
            "wmue,vw,uv->em",
            self.T2["S2"]["acav"],
            faa,
            self.cumulants["gamma1"],
            optimize=True,
        )
        F_tilde[self.pv, self.ha] += 0.5 * np.einsum(
            "ivaw,wu,uv->ai",
            self.T2["S2"]["aava"],
            faa,
            self.cumulants["gamma1"],
            optimize=True,
        )
        F_tilde[self.pv, self.ha] -= 0.5 * np.einsum(
            "iwau,vw,uv->ai",
            self.T2["S2"]["aava"],
            faa,
            self.cumulants["gamma1"],
            optimize=True,
        )
        delta_ph = -self.eps[self.part][:, None] + self.eps[self.hole][None, :]
        F_tilde *= np.exp(-self.flow_param * delta_ph**2)
        F_tilde += self.fock[self.part, self.hole]

        return F_tilde

    def _renormalize_V_in_place(self):
        for key in [
            "vvaa",
            "aacc",
            "avca",
            "avac",
            "vaaa",
            "aaca",
            "vvcc",
            "vvac",
            "vacc",
        ]:
            renormalize_V_block(
                self.ints["V"][key],
                *(self.ints["eps"][_] for _ in key),
                self.flow_param,
            )

    def _compute_pt2_energy(self):
        return self.dsrg_helper.evaluate_H_T_C0(
            self.T1,
            self.T2,
            self.F_tilde,
            self.ints["V"],
            self.cumulants,
            store_large=True,
        )

    def _compute_Hbar_aaaa(self):
        return self.dsrg_helper.H_T_C2_active(
            self.T1,
            self.T2["T2"],
            self.T2["S2"],
            self.F_tilde,
            self.ints["V"],
            self.cumulants["gamma1"],
            self.cumulants["eta1"],
        )

    def _compute_Hbar_aa(self):
        return self.dsrg_helper.H_T_C1_active(
            self.T1,
            self.T2["T2"],
            self.T2["S2"],
            self.F_tilde,
            self.ints["V"],
            self.cumulants["gamma1"],
            self.cumulants["eta1"],
            self.cumulants["lambda2"],
            store_large=True,
        )
