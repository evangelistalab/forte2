from dataclasses import dataclass

import numpy as np

from .sa_dsrg_base import SADSRGBase
from .utils import renormalize_3index


@dataclass
class DSRG_MRPT2(SADSRGBase):
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
        ints, cumulants = super().get_integrals()
        B = self._b_blocks(["ca", "cv", "aa", "av"])
        ints["V"] = self._v_blocks(B, self.dsrg_helper.hp_2_small_labels + ["aaaa"])
        # The streamed blocks are contracted from these one first index at a
        # time, so each is stored as [p, q, Q].
        ints["B"] = {
            "ca": B["ca"].transpose(1, 2, 0).copy(),
            "cv": B["cv"].transpose(1, 2, 0).copy(),
            "va": B["av"].transpose(2, 1, 0).copy(),
        }
        return ints, cumulants

    def solve_dsrg(self, form_hbar=False):
        h = self.dsrg_helper
        F = self.ints["F"]
        V = {blk: self.ints["V"][blk] for blk in h.hp_2_small_labels}
        T1, T2, S2 = self._build_tamps(F, V)
        # F and V become H1st + Hbar1st in place; V["aaaa"] stays bare
        self._renormalize(F, V, S2, add=True)

        E = h.H_T_C0(F, V, T1, T2, S2, self.cumulants, store_large=False)
        if form_hbar:
            self.hbar1 = np.zeros((self.nact,) * 2)
            self.hbar2 = np.zeros((self.nact,) * 4)
            self._accumulate_hbar(F, V, T1, T2, S2, 0.5, store_large=False)
        E += self._compute_pt2_energy_ccvv()
        E += self._compute_pt2_energy_cavv(form_hbar)
        E += self._compute_pt2_energy_ccav(form_hbar)
        return self.ints["E"] + E

    def _compute_pt2_energy_ccvv(self):
        # E += einsum("mnef,mnef->", V["ccvv"], S2["ccvv"]), one m at a time
        E = 0.0
        cvB = self.ints["B"]["cv"]

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
        cvB = self.ints["B"]["cv"]
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
                self.hbar1 += 0.5 * np.einsum(
                    "efu,efv->uv", Vr_m, Vbare_m, optimize=True
                )

        return E

    def _compute_pt2_energy_ccav(self, form_hbar=False):
        E = 0.0

        cvB = self.ints["B"]["cv"]
        caB = self.ints["B"]["ca"]

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
                self.hbar1 -= 0.5 * np.einsum(
                    "nue,nve->uv", Vr_m, Vbare_m, optimize=True
                )

        return E
