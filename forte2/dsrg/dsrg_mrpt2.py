from dataclasses import dataclass

import numpy as np

from .dsrg_base import DSRGBase
from .utils import (
    cas_energy_given_cumulants,
    compute_t1_block,
    compute_t2_block,
    renormalize_V_block,
    renormalize_3index,
)


@dataclass
class DSRG_MRPT2(DSRGBase):
    """
    Spin-adapted driven similarity renormalization group
    second-order multireference perturbation theory (DSRG-MRPT2).

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

        ints["E"] = cas_energy_given_cumulants(
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

        # These are used in on-the-fly energy/Hbar computations
        ints["B"] = dict()
        ints["B"]["ac"] = B_mo["ac"].transpose(2, 1, 0).copy()
        ints["B"]["vc"] = B_mo["vc"].transpose(2, 1, 0).copy()
        ints["B"]["va"] = B_mo["va"].transpose(1, 2, 0).copy()

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
        # self.ints["V"] gets renormalizes to V_tilde in place for the following blocks:
        # caaa, aaav, ccaa, caav, aavv
        # The aaaa block is remains untouched, and can be safely used in reference relaxation
        self._renormalize_V_in_place()
        if form_hbar:
            self.hbar_aa_df = np.zeros((self.nact, self.nact))
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

        for key in ["aavv", "ccaa", "caav", "acav", "aava", "caaa"]:
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
        for key in ["vvaa", "aacc", "avca", "avac", "vaaa", "aaca"]:
            renormalize_V_block(
                self.ints["V"][key],
                *(self.ints["eps"][_] for _ in key),
                self.flow_param,
            )

    def _compute_pt2_energy(self, form_hbar=False):
        E = 0.0
        E += 2.0 * np.einsum(
            "am,ma->", self.F_tilde[:, self.hc], self.T1[self.hc, :], optimize=True
        )
        E += np.einsum(
            "ev,ue,uv->",
            self.F_tilde[self.pv, self.ha],
            self.T1[self.ha, self.pv],
            self.cumulants["gamma1"],
            optimize=True,
        )
        E -= np.einsum(
            "um,mv,uv->",
            self.F_tilde[self.pa, self.hc],
            self.T1[self.hc, self.pa],
            self.cumulants["gamma1"],
            optimize=True,
        )
        E += np.einsum(
            "ex,uvey,uvxy->",
            self.F_tilde[self.pv, self.ha],
            self.T2["T2"]["aava"],
            self.cumulants["lambda2"],
            optimize=True,
        )
        E -= np.einsum(
            "vm,muyx,uvxy->",
            self.F_tilde[self.pa, self.hc],
            self.T2["T2"]["caaa"],
            self.cumulants["lambda2"],
            optimize=True,
        )
        E += np.einsum(
            "evxy,ue,uvxy->",
            self.ints["V"]["vaaa"],
            self.T1[self.ha, self.pv],
            self.cumulants["lambda2"],
            optimize=True,
        )
        E -= np.einsum(
            "uvmy,mx,uvxy->",
            self.ints["V"]["aaca"],
            self.T1[self.hc, self.pa],
            self.cumulants["lambda2"],
            optimize=True,
        )
        E += 0.25 * np.einsum(
            "efxu,yvef,uv,xy->",
            self.ints["V"]["vvaa"],
            self.T2["S2"]["aavv"],
            self.cumulants["gamma1"],
            self.cumulants["gamma1"],
            optimize=True,
        )
        E += 0.25 * np.einsum(
            "vymn,mnux,uv,xy->",
            self.ints["V"]["aacc"],
            self.T2["S2"]["ccaa"],
            self.cumulants["eta1"],
            self.cumulants["eta1"],
            optimize=True,
        )
        E += 0.5 * np.einsum(
            "vemx,myue,uv,xy->",
            self.ints["V"]["avca"],
            self.T2["S2"]["caav"],
            self.cumulants["eta1"],
            self.cumulants["gamma1"],
            optimize=True,
        )
        E += 0.5 * np.einsum(
            "vexm,ymue,uv,xy->",
            self.ints["V"]["avac"],
            self.T2["S2"]["acav"],
            self.cumulants["eta1"],
            self.cumulants["gamma1"],
            optimize=True,
        )
        E += 0.25 * np.einsum(
            "evwx,zyeu,wz,uv,xy->",
            self.ints["V"]["vaaa"],
            self.T2["S2"]["aava"],
            self.cumulants["gamma1"],
            self.cumulants["eta1"],
            self.cumulants["gamma1"],
            optimize=True,
        )
        E += 0.25 * np.einsum(
            "vzmx,myuw,wz,uv,xy->",
            self.ints["V"]["aaca"],
            self.T2["S2"]["caaa"],
            self.cumulants["eta1"],
            self.cumulants["eta1"],
            self.cumulants["gamma1"],
            optimize=True,
        )
        E += 0.5 * np.einsum(
            "uvmn,mnxy,uvxy->",
            self.ints["V"]["aacc"],
            self.T2["T2"]["ccaa"],
            self.cumulants["lambda2"],
            optimize=True,
        )
        E += 0.5 * np.einsum(
            "uvmw,mzxy,wz,uvxy->",
            self.ints["V"]["aaca"],
            self.T2["T2"]["caaa"],
            self.cumulants["gamma1"],
            self.cumulants["lambda2"],
            optimize=True,
        )
        E += 0.5 * np.einsum(
            "efxy,uvef,uvxy->",
            self.ints["V"]["vvaa"],
            self.T2["T2"]["aavv"],
            self.cumulants["lambda2"],
            optimize=True,
        )
        E += 0.5 * np.einsum(
            "ezxy,uvew,wz,uvxy->",
            self.ints["V"]["vaaa"],
            self.T2["T2"]["aava"],
            self.cumulants["eta1"],
            self.cumulants["lambda2"],
            optimize=True,
        )
        E += np.einsum(
            "uexm,vmye,uvxy->",
            self.ints["V"]["avac"],
            self.T2["S2"]["acav"],
            self.cumulants["lambda2"],
            optimize=True,
        )
        E -= np.einsum(
            "uemx,vmye,uvxy->",
            self.ints["V"]["avca"],
            self.T2["T2"]["acav"],
            self.cumulants["lambda2"],
            optimize=True,
        )
        E -= np.einsum(
            "vemx,muye,uvxy->",
            self.ints["V"]["avca"],
            self.T2["T2"]["caav"],
            self.cumulants["lambda2"],
            optimize=True,
        )
        E += 0.5 * np.einsum(
            "euwx,zvey,wz,uvxy->",
            self.ints["V"]["vaaa"],
            self.T2["S2"]["aava"],
            self.cumulants["gamma1"],
            self.cumulants["lambda2"],
            optimize=True,
        )
        E -= 0.5 * np.einsum(
            "euxw,zvey,wz,uvxy->",
            self.ints["V"]["vaaa"],
            self.T2["T2"]["aava"],
            self.cumulants["gamma1"],
            self.cumulants["lambda2"],
            optimize=True,
        )
        E -= 0.5 * np.einsum(
            "evxw,uzey,wz,uvxy->",
            self.ints["V"]["vaaa"],
            self.T2["T2"]["aava"],
            self.cumulants["gamma1"],
            self.cumulants["lambda2"],
            optimize=True,
        )
        E += 0.5 * np.einsum(
            "wumx,mvzy,wz,uvxy->",
            self.ints["V"]["aaca"],
            self.T2["S2"]["caaa"],
            self.cumulants["eta1"],
            self.cumulants["lambda2"],
            optimize=True,
        )
        E -= 0.5 * np.einsum(
            "uwmx,mvzy,wz,uvxy->",
            self.ints["V"]["aaca"],
            self.T2["T2"]["caaa"],
            self.cumulants["eta1"],
            self.cumulants["lambda2"],
            optimize=True,
        )
        E -= 0.5 * np.einsum(
            "vwmx,muyz,wz,uvxy->",
            self.ints["V"]["aaca"],
            self.T2["T2"]["caaa"],
            self.cumulants["eta1"],
            self.cumulants["lambda2"],
            optimize=True,
        )
        E += np.einsum(
            "ewxy,uvez,xyzuwv->",
            self.ints["V"]["vaaa"],
            self.T2["T2"]["aava"],
            self.cumulants["lambda3"],
            optimize=True,
        )
        E -= np.einsum(
            "uvmz,mwxy,xyzuwv->",
            self.ints["V"]["aaca"],
            self.T2["T2"]["caaa"],
            self.cumulants["lambda3"],
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
        cvB = self.ints["B"]["vc"]

        Vbare_m = np.empty((self.ncore, self.nvirt, self.nvirt))
        Vtmp = np.empty((self.ncore, self.nvirt, self.nvirt))
        Vr_m = np.empty((self.ncore, self.nvirt, self.nvirt))

        for m in range(self.ncore):
            np.einsum(
                "eB,nfB->nfe",
                cvB[m, ...],
                cvB,
                optimize=True,
                out=Vr_m,
            )
            np.copyto(Vtmp, Vr_m.swapaxes(1, 2))
            Vbare_m[:] = 2.0 * Vr_m - Vtmp

            renormalize_3index(
                Vr_m,
                self.ints["eps"]["c"][m],
                self.ints["eps"]["c"],
                self.ints["eps"]["v"],
                self.ints["eps"]["v"],
                self.flow_param,
            )

            E += np.einsum("nfe,nfe->", Vr_m, Vbare_m, optimize=True)
        return E

    def _compute_pt2_energy_cavv(self, form_hbar=False):
        E = 0.0
        vaB = self.ints["B"]["va"]
        cvB = self.ints["B"]["vc"]
        Vbare_m = np.empty((self.nvirt, self.nvirt, self.nact))
        Vtmp = np.empty((self.nvirt, self.nvirt, self.nact))
        Vr_m = np.empty((self.nvirt, self.nvirt, self.nact))

        for m in range(self.ncore):
            np.einsum(
                "eB,fvB->efv",
                cvB[m, ...],
                vaB,
                optimize=True,
                out=Vr_m,
            )
            np.copyto(Vtmp, Vr_m.swapaxes(0, 1))
            Vbare_m[:] = 2.0 * Vr_m - Vtmp

            renormalize_3index(
                Vr_m,
                self.ints["eps"]["c"][m],
                -self.ints["eps"]["v"],
                self.ints["eps"]["v"],
                -self.ints["eps"]["a"],
                self.flow_param,
            )

            E += np.einsum(
                "efu,efv,uv->", Vr_m, Vbare_m, self.cumulants["gamma1"], optimize=True
            )
            if form_hbar:
                self.hbar_aa_df += np.einsum(
                    "efu,efv->uv", Vr_m, Vbare_m, optimize=True
                )

        return E

    def _compute_pt2_energy_ccav(self, form_hbar=False):
        E = 0.0

        cvB = self.ints["B"]["vc"]
        caB = self.ints["B"]["ac"]

        Vbare_m = np.empty((self.ncore, self.nact, self.nvirt))
        Vtmp = np.empty((self.ncore, self.nact, self.nvirt))
        Vr_m = np.empty((self.ncore, self.nact, self.nvirt))

        for m in range(self.ncore):
            # mneu
            np.einsum(
                "eB,nuB->nue",
                cvB[m, ...],
                caB,
                optimize=True,
                out=Vr_m,
            )
            # mnue
            np.einsum(
                "uB,neB->nue",
                caB[m, ...],
                cvB,
                optimize=True,
                out=Vtmp,
            )
            Vbare_m[:] = 2.0 * Vr_m - Vtmp

            renormalize_3index(
                Vr_m,
                self.ints["eps"]["c"][m],
                self.ints["eps"]["c"],
                self.ints["eps"]["a"],
                self.ints["eps"]["v"],
                self.flow_param,
            )

            E += np.einsum(
                "nue,nve,uv->", Vr_m, Vbare_m, self.cumulants["eta1"], optimize=True
            )
            if form_hbar:
                self.hbar_aa_df -= np.einsum(
                    "nue,nve->uv", Vr_m, Vbare_m, optimize=True
                )

        return E

    def _compute_Hbar_aaaa(self):
        C2 = np.zeros((self.nact,) * 4)
        C2 += np.einsum(
            "efxy,uvef->uvxy",
            self.ints["V"]["vvaa"],
            self.T2["T2"]["aavv"],
            optimize=True,
        )
        C2 += np.einsum(
            "ewxy,uvew->uvxy",
            self.ints["V"]["vaaa"],
            self.T2["T2"]["aava"],
            optimize=True,
        )
        C2 += np.einsum(
            "ewyx,vuew->uvxy",
            self.ints["V"]["vaaa"],
            self.T2["T2"]["aava"],
            optimize=True,
        )

        C2 += np.einsum(
            "uvmn,mnxy->uvxy",
            self.ints["V"]["aacc"],
            self.T2["T2"]["ccaa"],
            optimize=True,
        )
        C2 += np.einsum(
            "vumw,mwyx->uvxy",
            self.ints["V"]["aaca"],
            self.T2["T2"]["caaa"],
            optimize=True,
        )
        C2 += np.einsum(
            "uvmw,mwxy->uvxy",
            self.ints["V"]["aaca"],
            self.T2["T2"]["caaa"],
            optimize=True,
        )

        temp = np.einsum(
            "ax,uvay->uvxy",
            self.F_tilde[self.pv, self.ha],
            self.T2["T2"]["aava"],
            optimize=True,
        )
        temp -= np.einsum(
            "ui,ivxy->uvxy",
            self.F_tilde[self.pa, self.hc],
            self.T2["T2"]["caaa"],
            optimize=True,
        )
        temp += np.einsum(
            "ua,avxy->uvxy",
            self.T1[self.ha, self.pv],
            self.ints["V"]["vaaa"],
            optimize=True,
        )
        temp -= np.einsum(
            "ix,uviy->uvxy",
            self.T1[self.hc, self.pa],
            self.ints["V"]["aaca"],
            optimize=True,
        )

        temp -= 0.50 * np.einsum(
            "wz,vuaw,azyx->uvxy",
            self.cumulants["gamma1"],
            self.T2["T2"]["aava"],
            self.ints["V"]["vaaa"],
            optimize=True,
        )
        temp -= 0.50 * np.einsum(
            "wz,izyx,vuiw->uvxy",
            self.cumulants["eta1"],
            self.T2["T2"]["caaa"],
            self.ints["V"]["aaca"],
            optimize=True,
        )

        temp += np.einsum(
            "uexm,vmye->uvxy",
            self.ints["V"]["avac"],
            self.T2["S2"]["acav"],
            optimize=True,
        )
        temp += np.einsum(
            "wumx,mvwy->uvxy",
            self.ints["V"]["aaca"],
            self.T2["S2"]["caaa"],
            optimize=True,
        )

        temp += 0.50 * np.einsum(
            "wz,zvay,auwx->uvxy",
            self.cumulants["gamma1"],
            self.T2["S2"]["aava"],
            self.ints["V"]["vaaa"],
            optimize=True,
        )
        temp -= 0.50 * np.einsum(
            "wz,ivwy,zuix->uvxy",
            self.cumulants["gamma1"],
            self.T2["S2"]["caaa"],
            self.ints["V"]["aaca"],
            optimize=True,
        )

        temp -= np.einsum(
            "uemx,vmye->uvxy",
            self.ints["V"]["avca"],
            self.T2["T2"]["acav"],
            optimize=True,
        )
        temp -= np.einsum(
            "uwmx,mvwy->uvxy",
            self.ints["V"]["aaca"],
            self.T2["T2"]["caaa"],
            optimize=True,
        )

        temp -= 0.50 * np.einsum(
            "wz,zvay,auxw->uvxy",
            self.cumulants["gamma1"],
            self.T2["T2"]["aava"],
            self.ints["V"]["vaaa"],
            optimize=True,
        )
        temp += 0.50 * np.einsum(
            "wz,ivwy,uzix->uvxy",
            self.cumulants["gamma1"],
            self.T2["T2"]["caaa"],
            self.ints["V"]["aaca"],
            optimize=True,
        )

        temp -= np.einsum(
            "vemx,muye->uvxy",
            self.ints["V"]["avca"],
            self.T2["T2"]["caav"],
            optimize=True,
        )
        temp -= np.einsum(
            "vwmx,muyw->uvxy",
            self.ints["V"]["aaca"],
            self.T2["T2"]["caaa"],
            optimize=True,
        )

        temp -= 0.50 * np.einsum(
            "wz,uzay,avxw->uvxy",
            self.cumulants["gamma1"],
            self.T2["T2"]["aava"],
            self.ints["V"]["vaaa"],
            optimize=True,
        )
        temp += 0.50 * np.einsum(
            "wz,iuyw,vzix->uvxy",
            self.cumulants["gamma1"],
            self.T2["T2"]["caaa"],
            self.ints["V"]["aaca"],
            optimize=True,
        )

        C2 += temp
        C2 += np.einsum("uvxy->vuyx", temp, optimize=True)
        return C2

    def _compute_Hbar_aa(self):
        C1 = self.hbar_aa_df.copy()
        C1 += 1.00 * np.einsum(
            "ev,ue->uv",
            self.F_tilde[self.pv, self.ha],
            self.T1[self.ha, self.pv],
            optimize=True,
        )
        C1 -= 1.00 * np.einsum(
            "um,mv->uv",
            self.F_tilde[self.pa, self.hc],
            self.T1[self.hc, self.pa],
            optimize=True,
        )
        C1 += 1.00 * np.einsum(
            "em,umve->uv",
            self.F_tilde[self.pv, self.hc],
            self.T2["S2"]["acav"],
            optimize=True,
        )
        C1 += 1.00 * np.einsum(
            "xm,muxv->uv",
            self.F_tilde[self.pa, self.hc],
            self.T2["S2"]["caaa"],
            optimize=True,
        )
        C1 += 0.50 * np.einsum(
            "ex,yuev,xy->uv",
            self.F_tilde[self.pv, self.ha],
            self.T2["S2"]["aava"],
            self.cumulants["gamma1"],
            optimize=True,
        )
        C1 -= 0.50 * np.einsum(
            "ym,muxv,xy->uv",
            self.F_tilde[self.pa, self.hc],
            self.T2["S2"]["caaa"],
            self.cumulants["gamma1"],
            optimize=True,
        )
        C1 += 1.00 * np.einsum(
            "uemz,mwue->wz",
            self.ints["V"]["avca"],
            self.T2["S2"]["caav"],
            optimize=True,
        )
        C1 += 1.00 * np.einsum(
            "uezm,wmue->wz",
            self.ints["V"]["avac"],
            self.T2["S2"]["acav"],
            optimize=True,
        )
        C1 += 1.00 * np.einsum(
            "vumz,mwvu->wz",
            self.ints["V"]["aaca"],
            self.T2["S2"]["caaa"],
            optimize=True,
        )

        C1 -= 1.00 * np.einsum(
            "wemu,muze->wz",
            self.ints["V"]["avca"],
            self.T2["S2"]["caav"],
            optimize=True,
        )
        C1 -= 1.00 * np.einsum(
            "weum,umze->wz",
            self.ints["V"]["avac"],
            self.T2["S2"]["acav"],
            optimize=True,
        )
        C1 -= 1.00 * np.einsum(
            "ewvu,vuez->wz",
            self.ints["V"]["vaaa"],
            self.T2["S2"]["aava"],
            optimize=True,
        )

        temp = 0.5 * np.einsum(
            "wvef,efzu->wzuv",
            self.T2["S2"]["aavv"],
            self.ints["V"]["vvaa"],
            optimize=True,
        )
        temp += 0.5 * np.einsum(
            "wvex,exzu->wzuv",
            self.T2["S2"]["aava"],
            self.ints["V"]["vaaa"],
            optimize=True,
        )
        temp += 0.5 * np.einsum(
            "vwex,exuz->wzuv",
            self.T2["S2"]["aava"],
            self.ints["V"]["vaaa"],
            optimize=True,
        )

        temp -= 0.5 * np.einsum(
            "wmue,vezm->wzuv",
            self.T2["S2"]["acav"],
            self.ints["V"]["avac"],
            optimize=True,
        )
        temp -= 0.5 * np.einsum(
            "mwxu,xvmz->wzuv",
            self.T2["S2"]["caaa"],
            self.ints["V"]["aaca"],
            optimize=True,
        )

        temp -= 0.5 * np.einsum(
            "mwue,vemz->wzuv",
            self.T2["S2"]["caav"],
            self.ints["V"]["avca"],
            optimize=True,
        )
        temp -= 0.5 * np.einsum(
            "mwux,vxmz->wzuv",
            self.T2["S2"]["caaa"],
            self.ints["V"]["aaca"],
            optimize=True,
        )

        temp += 0.25 * np.einsum(
            "jwxu,xy,yvjz->wzuv",
            self.T2["S2"]["caaa"],
            self.cumulants["gamma1"],
            self.ints["V"]["aaca"],
            optimize=True,
        )
        temp -= 0.25 * np.einsum(
            "ywbu,xy,bvxz->wzuv",
            self.T2["S2"]["aava"],
            self.cumulants["gamma1"],
            self.ints["V"]["vaaa"],
            optimize=True,
        )
        temp -= 0.25 * np.einsum(
            "wybu,xy,bvzx->wzuv",
            self.T2["S2"]["aava"],
            self.cumulants["gamma1"],
            self.ints["V"]["vaaa"],
            optimize=True,
        )

        C1 += np.einsum("wzuv,uv->wz", temp, self.cumulants["gamma1"], optimize=True)
        temp = np.zeros((self.nact,) * 4)

        temp -= 0.5 * np.einsum(
            "mnzu,wvmn->wzuv",
            self.T2["S2"]["ccaa"],
            self.ints["V"]["aacc"],
            optimize=True,
        )
        temp -= 0.5 * np.einsum(
            "mxzu,wvmx->wzuv",
            self.T2["S2"]["caaa"],
            self.ints["V"]["aaca"],
            optimize=True,
        )
        temp -= 0.5 * np.einsum(
            "mxuz,vwmx->wzuv",
            self.T2["S2"]["caaa"],
            self.ints["V"]["aaca"],
            optimize=True,
        )

        temp += 0.5 * np.einsum(
            "vmze,weum->wzuv",
            self.T2["S2"]["acav"],
            self.ints["V"]["avac"],
            optimize=True,
        )
        temp += 0.5 * np.einsum(
            "xvez,ewxu->wzuv",
            self.T2["S2"]["aava"],
            self.ints["V"]["vaaa"],
            optimize=True,
        )

        temp += 0.5 * np.einsum(
            "mvze,wemu->wzuv",
            self.T2["S2"]["caav"],
            self.ints["V"]["avca"],
            optimize=True,
        )
        temp += 0.5 * np.einsum(
            "vxez,ewux->wzuv",
            self.T2["S2"]["aava"],
            self.ints["V"]["vaaa"],
            optimize=True,
        )

        temp -= 0.25 * np.einsum(
            "yvbz,xy,bwxu->wzuv",
            self.T2["S2"]["aava"],
            self.cumulants["eta1"],
            self.ints["V"]["vaaa"],
            optimize=True,
        )
        temp += 0.25 * np.einsum(
            "jvxz,xy,ywju->wzuv",
            self.T2["S2"]["caaa"],
            self.cumulants["eta1"],
            self.ints["V"]["aaca"],
            optimize=True,
        )
        temp += 0.25 * np.einsum(
            "jvzx,xy,wyju->wzuv",
            self.T2["S2"]["caaa"],
            self.cumulants["eta1"],
            self.ints["V"]["aaca"],
            optimize=True,
        )

        C1 += np.einsum("wzuv,uv->wz", temp, self.cumulants["eta1"], optimize=True)

        C1 += 0.50 * np.einsum(
            "vujz,jwyx,xyuv->wz",
            self.ints["V"]["aaca"],
            self.T2["T2"]["caaa"],
            self.cumulants["lambda2"],
            optimize=True,
        )
        C1 += 0.50 * np.einsum(
            "auzx,wvay,xyuv->wz",
            self.ints["V"]["vaaa"],
            self.T2["S2"]["aava"],
            self.cumulants["lambda2"],
            optimize=True,
        )
        C1 -= 0.50 * np.einsum(
            "auxz,wvay,xyuv->wz",
            self.ints["V"]["vaaa"],
            self.T2["T2"]["aava"],
            self.cumulants["lambda2"],
            optimize=True,
        )
        C1 -= 0.50 * np.einsum(
            "auxz,vway,xyvu->wz",
            self.ints["V"]["vaaa"],
            self.T2["T2"]["aava"],
            self.cumulants["lambda2"],
            optimize=True,
        )

        C1 -= 0.50 * np.einsum(
            "bwyx,vubz,xyuv->wz",
            self.ints["V"]["vaaa"],
            self.T2["T2"]["aava"],
            self.cumulants["lambda2"],
            optimize=True,
        )
        C1 -= 0.50 * np.einsum(
            "wuix,ivzy,xyuv->wz",
            self.ints["V"]["aaca"],
            self.T2["S2"]["caaa"],
            self.cumulants["lambda2"],
            optimize=True,
        )
        C1 += 0.50 * np.einsum(
            "uwix,ivzy,xyuv->wz",
            self.ints["V"]["aaca"],
            self.T2["T2"]["caaa"],
            self.cumulants["lambda2"],
            optimize=True,
        )
        C1 += 0.50 * np.einsum(
            "uwix,ivyz,xyvu->wz",
            self.ints["V"]["aaca"],
            self.T2["T2"]["caaa"],
            self.cumulants["lambda2"],
            optimize=True,
        )

        C1 += 0.50 * np.einsum(
            "avxy,uwaz,xyuv->wz",
            self.ints["V"]["vaaa"],
            self.T2["S2"]["aava"],
            self.cumulants["lambda2"],
            optimize=True,
        )
        C1 -= 0.50 * np.einsum(
            "uviy,iwxz,xyuv->wz",
            self.ints["V"]["aaca"],
            self.T2["S2"]["caaa"],
            self.cumulants["lambda2"],
            optimize=True,
        )
        G2 = dict.fromkeys(["avac", "aaac", "avaa"])
        G2["avac"] = 2.0 * self.ints["V"]["avac"] - np.einsum(
            "uemv->uevm", self.ints["V"]["avca"], optimize=True
        )
        G2["aaac"] = 2.0 * np.einsum(
            "vumw->uvwm", self.ints["V"]["aaca"], optimize=True
        ) - np.einsum("uvmw->uvwm", self.ints["V"]["aaca"], optimize=True)
        G2["avaa"] = 2.0 * np.einsum(
            "euyx->uexy", self.ints["V"]["vaaa"], optimize=True
        ) - np.einsum("euxy->uexy", self.ints["V"]["vaaa"], optimize=True)

        C1 += np.einsum(
            "ma,uavm->uv", self.T1[self.hc, self.pa], G2["aaac"], optimize=True
        )
        C1 += np.einsum(
            "ma,uavm->uv", self.T1[self.hc, self.pv], G2["avac"], optimize=True
        )
        C1 += 0.50 * np.einsum(
            "xe,yx,uevy->uv",
            self.T1[self.ha, self.pv],
            self.cumulants["gamma1"],
            G2["avaa"],
            optimize=True,
        )
        C1 -= 0.50 * np.einsum(
            "mx,xy,uyvm->uv",
            self.T1[self.hc, self.pa],
            self.cumulants["gamma1"],
            G2["aaac"],
            optimize=True,
        )

        C1 += 0.50 * np.einsum(
            "wezx,uvey,xyuv->wz",
            G2["avaa"],
            self.T2["T2"]["aava"],
            self.cumulants["lambda2"],
            optimize=True,
        )
        C1 -= 0.50 * np.einsum(
            "wuzm,mvxy,xyuv->wz",
            G2["aaac"],
            self.T2["T2"]["caaa"],
            self.cumulants["lambda2"],
            optimize=True,
        )

        return C1
