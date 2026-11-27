from dataclasses import dataclass

import numpy as np

from .sa_dsrg_base import SADSRGBase


@dataclass
class DSRG_MRPT3(SADSRGBase):
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

    def get_integrals(self):
        ints, cumulants = super().get_integrals()
        h = self.dsrg_helper
        B = self._b_blocks(["cc", "ca", "cv", "aa", "av", "vv"])
        ints["V"] = self._v_blocks(B, h.hp_2_labels + h.non_od_2_labels)
        # what the streamed kernels contract the many-virtual blocks from
        ints["B"] = {"vv": B["vv"], "cv": B["cv"], "av": B["av"]}
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
        # last read of the first-order amplitudes
        self.T1_1 = self.T2_1 = self.S2_1 = None

        return (
            self.ints["E"]
            + self.e_dsrg_mrpt2
            + self.e_dsrg_mrpt3_1
            + self.e_dsrg_mrpt3_2
            + self.e_dsrg_mrpt3_3
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
