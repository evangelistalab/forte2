import numpy as np
from typing import Any
from numpy.typing import NDArray
import scipy as sp

from forte2 import cpp_helpers


class MutualCorrelationAnalysis:
    """
    Performs a mutual correlation analysis.

    Parameters
    ----------
    solver : forte2 solver object of type ActiveSpaceSolver
        The solver from which to extract the RDMs.
    root : int, optional
        The root index for which to perform the analysis. Default is 0.
    sub_solver_index : int, optional
        The index of the sub-solver within the active space solver. Default is 0.

    Attributes
    ----------
    total_correlation : float
        Total correlation measure.
    M1 : NDArray
        Single orbital correlation measure.
    M2 : NDArray
        Dyad mutual correlation measure.
    M3 : NDArray
        Triad mutual correlation measure.
    M4 : NDArray
        Tetrad mutual correlation measure.
    Q : NDArray
        Orthogonal transformation matrix to form the maximally correlated orbitals.

    Notes
    ----------------------
    This analysis expects an active space solver with the following API
    on the selected sub-solver `solver.sub_solvers[sub_solver_index]`:

    - make_sd_1rdm(root) -> tuple[NDArray, NDArray]
            Returns (γa, γb) spin-dependent 1-RDMs with shape (norb, norb) each.
    - make_sd_2rdm(root) -> tuple[NDArray, NDArray, NDArray]
            Returns (γaa, γab, γbb) spin-dependent 2-RDMs.
            γaa and γbb are packed (lower-triangular pair indices) and are
            converted to full (norb, norb, norb, norb) tensors via
            `cpp_helpers.packed_tensor4_to_tensor4`. γab is already full with shape
            (norb, norb, norb, norb).

    Derived tensors and outputs:
    - Γ1 = γa + γb with shape (norb, norb).
    - Cumulants λaa, λab, λbb with shape (norb, norb, norb, norb) each.
    - total_correlation: scalar.
    - M1: shape (norb,), orbital correlation.
    - M2: shape (norb, norb), dyad mutual correlations (diagonal zeroed).
    - M3: shape (norb, norb, norb), triad mutual correlations (entries with
        any repeated indices are zeroed).
    - M4: shape (norb, norb, norb, norb), tetrad mutual correlations (entries
        with any repeated pair of indices are zeroed).
    - Q: shape (norb, norb), real orthogonal matrix from expm of a real
        antisymmetric generator. Stored after `optimize_orbitals`.

    Notes:
    - This implementation targets the non-relativistic case.
    """

    def __init__(self, solver, root=0, sub_solver_index=0):
        self.Q = None

        self.active_mo_indices = solver.mo_space.active_indices[:]

        # extract the spin-dependent 1-RDM  from the solver
        γa, γb = solver.sub_solvers[sub_solver_index].make_sd_1rdm(root)

        # compute the spin-dependent 1-RDM
        self.Γ1 = γa + γb

        # extract the spin-dependent 2-RDM from the solver
        γaa, γab, γbb = solver.sub_solvers[sub_solver_index].make_sd_2rdm(root)

        # convert packed 2-RDMs to full tensors (only the aa and bb components are packed)
        γaa = cpp_helpers.packed_tensor4_to_tensor4(γaa)
        γbb = cpp_helpers.packed_tensor4_to_tensor4(γbb)

        # convert the spin-dependent 2-RDMs to cumulants
        self.λaa = (
            γaa - np.einsum("pr,qs->pqrs", γa, γa) + np.einsum("ps,qr->pqrs", γa, γa)
        )
        self.λab = γab - np.einsum("pr,qs->pqrs", γa, γb)
        self.λbb = (
            γbb - np.einsum("pr,qs->pqrs", γb, γb) + np.einsum("ps,qr->pqrs", γb, γb)
        )

        # compute the various mutual correlation measures
        self._compute_mutual_correlation_measures(self.λaa, self.λab, self.λbb)

        # Verify that the total correlation is consistent with the sum of the mutual correlations
        total = self.M1.sum()
        total += self.M2.sum() / 2
        total += self.M3.sum() / 6
        total += self.M4.sum() / 24
        assert np.isclose(total, self.total_correlation, atol=1e-8, rtol=0)

    def _compute_mutual_correlation_measures(self, λaa, λab, λbb) -> None:
        """Recomputes the mutual correlation measures from the current cumulant RDMs."""
        C_PQRS = self._spin_free_correlation(λaa, λab, λbb)
        self.total_correlation = self._total_correlation(λaa, λab, λbb)
        self.M1 = self._orbital_correlation(C_PQRS)
        self.M2 = self._dyad_mutual_correlation(C_PQRS)
        self.M3 = self._triad_mutual_correlation(C_PQRS)
        self.M4 = self._tetrad_mutual_correlation(C_PQRS)

    def _total_correlation(self, λaa, λab, λbb) -> np.floating[Any] | np.float64:
        """Computes the total correlation from the cumulant 2-RDMs."""
        return 0.25 * (
            np.linalg.norm(λaa) ** 2
            + 4 * np.linalg.norm(λab) ** 2
            + np.linalg.norm(λbb) ** 2
        )

    def _spin_free_correlation(self, λaa, λab, λbb) -> NDArray:
        """
        Computes the spin-free correlation C_PQRS from the spin-dependent cumulant 2-RDMs.
        Here we take the absolute value squared of the cumulants so this can work with complex quantities.
        """
        C_PQRS = 0.25 * (np.abs(λaa) ** 2).copy()
        C_PQRS += 0.25 * (np.abs(λab) ** 2)
        C_PQRS += 0.25 * np.einsum("ijlk->ijkl", np.abs(λab) ** 2)
        C_PQRS += 0.25 * np.einsum("jikl->ijkl", np.abs(λab) ** 2)
        C_PQRS += 0.25 * np.einsum("jilk->ijkl", np.abs(λab) ** 2)
        C_PQRS += 0.25 * (np.abs(λbb) ** 2)
        return C_PQRS

    def _orbital_correlation(self, C_PQRS) -> NDArray:
        """Computes the orbital correlation from the spin-free correlation C_PQRS."""
        M1 = np.einsum("iiii->i", C_PQRS).copy()
        return M1

    def _dyad_mutual_correlation(self, C_PQRS) -> NDArray:
        """Computes the dyad mutual correlation M2 from the spin-free correlation C_PQRS."""
        M2 = 4 * np.einsum("iiij->ij", C_PQRS).copy()
        M2 += 2 * np.einsum("iijj->ij", C_PQRS)
        M2 += 4 * np.einsum("ijij->ij", C_PQRS)
        M2 += 4 * np.einsum("ijjj->ij", C_PQRS)
        # zero the diagonal
        idx = np.arange(M2.shape[0])
        M2[idx, idx] = 0
        return M2

    def _triad_mutual_correlation(self, C_PQRS) -> NDArray:
        """Computes the triad mutual correlation M3 from the spin-free correlation C_PQRS."""
        M3 = 4 * np.einsum("ijkk->ijk", C_PQRS).copy()
        M3 += 8 * np.einsum("ikjk->ijk", C_PQRS)
        M3 += 4 * np.einsum("ikjj->ijk", C_PQRS)
        M3 += 8 * np.einsum("ijkj->ijk", C_PQRS)
        M3 += 4 * np.einsum("jkii->ijk", C_PQRS)
        M3 += 8 * np.einsum("ijik->ijk", C_PQRS)
        # zero the terms with any two equal indices
        idx = np.arange(M3.shape[0])
        M3[idx, idx, :] = 0
        M3[idx, :, idx] = 0
        M3[:, idx, idx] = 0
        return M3

    def _tetrad_mutual_correlation(self, C_PQRS) -> NDArray:
        """Computes the tetrad mutual correlation M4 from the spin-free correlation C_PQRS."""
        M4 = 8 * C_PQRS.copy()
        M4 += 8 * np.einsum("ikjl->ijkl", C_PQRS)
        M4 += 8 * np.einsum("iljk->ijkl", C_PQRS)
        # zero the terms with any two equal indices
        idx = np.arange(M4.shape[0])
        M4[idx, idx, :, :] = 0
        M4[idx, :, idx, :] = 0
        M4[idx, :, :, idx] = 0
        M4[:, idx, idx, :] = 0
        M4[:, idx, :, idx] = 0
        M4[:, :, idx, idx] = 0
        return M4

    def mutual_correlation_matrix_summary(self, print_threshold: float = 7.5e-4) -> str:
        """
        Generates a summary of the mutual correlation matrix M2.

        Parameters
        ----------
        print_threshold : float, optional, default=7.5e-4
            Only values greater than this threshold are printed.

        Returns
        -------
        summary : str
            A formatted string summarizing the mutual correlation matrix M2.
        """

        s_lines = [
            f"Total λ2 Correlation: {self.total_correlation:8.6f}",
            f"Mutual Correlation Matrix M2 (only values > {print_threshold:.1e}):",
            "=====================",
            "    P     Q      M_PQ",
            "---------------------",
        ]

        # get the upper triangle indices and values
        M2_vals = []
        for i in range(self.M2.shape[0]):
            for j in range(i + 1, self.M2.shape[1]):
                M2_vals.append(
                    (
                        self.M2[i, j],
                        self.active_mo_indices[i],
                        self.active_mo_indices[j],
                    )
                )
        M2_vals.sort(reverse=True, key=lambda x: x[0])

        for val, i, j in M2_vals:
            if val < print_threshold:
                break
            s_lines.append(f"{i:>5} {j:>5}  {val:8.6f}")

        s_lines.append("=====================")

        return "\n".join(s_lines)

    def optimize_orbitals(
        self, k=2, random_guess_noise=0.001, method="L-BFGS-B", seed: int | None = None
    ) -> NDArray:
        """
        Optimize the orbitals by maximizing the sum of the k-th power of the mutual correlation M2.

        Parameters
        ----------
        k : int, optional, default=2
            The power to which to raise the mutual correlation values in the cost function.
        random_guess_noise : float, optional, default=0.001
            The amplitude of the random noise to add to the initial guess for the antisymmetric matrix.
        method : str, optional, default="L-BFGS-B"
            The optimization method to use.
        seed : int | None, optional
            Seed for the random initial antisymmetric matrix. If None, a
            nondeterministic seed is used (different results across runs).

        Returns
        -------
        Q : NDArray
            The optimized orthogonal transformation matrix.
        """

        # Generate a random antisymmetric matrix
        N = self.Γ1.shape[0]
        # If seed is None, default_rng uses a nondeterministic seed
        rng = np.random.default_rng(seed)
        a = rng.random(N**2) * random_guess_noise

        # define the objective function to minimize
        def objective(x):
            # construct the orthogonal matrix Q from the antisymmetric matrix A
            A = x.reshape(N, N)
            A = A - A.T
            Q = sp.linalg.expm(A)

            # apply the orthogonal transformation to the RDMs
            λaa_trans = np.einsum(
                "pqrs,pi,qj,rk,sl->ijkl", self.λaa, Q, Q, Q, Q, optimize=True
            )
            λab_trans = np.einsum(
                "pqrs,pi,qj,rk,sl->ijkl", self.λab, Q, Q, Q, Q, optimize=True
            )
            λbb_trans = np.einsum(
                "pqrs,pi,qj,rk,sl->ijkl", self.λbb, Q, Q, Q, Q, optimize=True
            )

            # compute the new mutual correlation matrix
            C_PQRS_trans = self._spin_free_correlation(λaa_trans, λab_trans, λbb_trans)
            M2_trans = self._dyad_mutual_correlation(C_PQRS_trans)

            # compute the objective function value
            obj = -np.sum(np.abs(M2_trans) ** k)

            return obj

        # minimize the objective function as a function of the antisymmetric matrix A
        res = sp.optimize.minimize(objective, a, method=method)

        A = res["x"].reshape(N, N)
        A = A - A.T
        Q = sp.linalg.expm(A)
        self.Q = Q

        # Apply the orthogonal transformation to the 1-RDM and 2-RDM cumulants
        self.Γ1 = np.einsum("pq,pi,qj->ij", self.Γ1, Q, Q)
        self.λaa = np.einsum("pqrs,pi,qj,rk,sl->ijkl", self.λaa, Q, Q, Q, Q)
        self.λab = np.einsum("pqrs,pi,qj,rk,sl->ijkl", self.λab, Q, Q, Q, Q)
        self.λbb = np.einsum("pqrs,pi,qj,rk,sl->ijkl", self.λbb, Q, Q, Q, Q)

        # Recompute the mutual correlation measures
        self._compute_mutual_correlation_measures(self.λaa, self.λab, self.λbb)

        return self.Q


class UMP2MPQFast:
    """
    Exact dyad mutual correlation M2 for UMP2 in spin-resolved form.

    Indices p,q,r,s are spatial MO indices in [0, nmo).
    """

    def __init__(self, mp2):
        self.mp2 = mp2
        self.nmo = mp2.nmo
        self.naocc = mp2.naocc
        self.nbocc = mp2.nbocc
        self.navir = mp2.navir
        self.nbvir = mp2.nbvir

        if any(getattr(mp2, name, None) is None for name in ("t2_a", "t2_b", "t2_ab")):
            raise ValueError("UMP2 t2 amplitudes must be stored.")

        self.t2_a = mp2.t2_a
        self.t2_b = mp2.t2_b
        self.t2_ab = mp2.t2_ab

        self.M1 = None
        self.M2 = None
        self.Γ1 = mp2.make_1rdm_sf()

    def _oa(self, p):
        return 0 <= p < self.naocc

    def _va(self, p):
        return self.naocc <= p < self.nmo

    def _ob(self, p):
        return 0 <= p < self.nbocc

    def _vb(self, p):
        return self.nbocc <= p < self.nmo

    def _a_vir(self, p):
        return p - self.naocc

    def _b_vir(self, p):
        return p - self.nbocc

    def lambda2_aa_elem(self, p, q, r, s):
        sign = 1

        if p > q:
            p, q = q, p
            sign *= -1
        if r > s:
            r, s = s, r
            sign *= -1

        if self._oa(p) and self._oa(q) and self._va(r) and self._va(s):
            a = self._a_vir(r)
            b = self._a_vir(s)

            val = self.t2_a[p, q, a, b] - self.t2_a[p, q, b, a]
            return sign * val

        if self._va(p) and self._va(q) and self._oa(r) and self._oa(s):
            a = self._a_vir(p)
            b = self._a_vir(q)

            val = self.t2_a[r, s, a, b] - self.t2_a[r, s, b, a]
            return sign * val

        return 0.0

    def lambda2_bb_elem(self, p, q, r, s):
        if self._ob(p) and self._ob(q) and self._vb(r) and self._vb(s):
            a = self._b_vir(r)
            b = self._b_vir(s)
            return self.t2_b[p, q, a, b]

        if self._vb(p) and self._vb(q) and self._ob(r) and self._ob(s):
            a = self._b_vir(p)
            b = self._b_vir(q)
            return self.t2_b[r, s, a, b]

        return 0.0

    def lambda2_ab_elem(self, p, q, r, s):
        # alpha index positions are 1st and 3rd; beta positions are 2nd and 4th
        if self._oa(p) and self._ob(q) and self._va(r) and self._vb(s):
            a = self._a_vir(r)
            b = self._b_vir(s)
            return self.t2_ab[p, q, a, b]

        if self._va(p) and self._vb(q) and self._oa(r) and self._ob(s):
            a = self._a_vir(p)
            b = self._b_vir(q)
            return self.t2_ab[r, s, a, b]

        return 0.0

    def C_elem(self, p, q, r, s):
        aa = self.lambda2_aa_elem(p, q, r, s)
        bb = self.lambda2_bb_elem(p, q, r, s)
        ab1 = self.lambda2_ab_elem(p, q, r, s)
        ab2 = self.lambda2_ab_elem(p, q, s, r)
        ab3 = self.lambda2_ab_elem(q, p, r, s)
        ab4 = self.lambda2_ab_elem(q, p, s, r)

        return 0.25 * (
            abs(aa) ** 2
            + abs(bb) ** 2
            + abs(ab1) ** 2
            + abs(ab2) ** 2
            + abs(ab3) ** 2
            + abs(ab4) ** 2
        )

    def make_measures(self):
        M1 = np.zeros(self.nmo)
        M2 = np.zeros((self.nmo, self.nmo))

        for p in range(self.nmo):
            M1[p] = self.C_elem(p, p, p, p)

        for p in range(self.nmo):
            for q in range(p + 1, self.nmo):
                val = (
                    4.0 * self.C_elem(p, p, p, q)
                    + 2.0 * self.C_elem(p, p, q, q)
                    + 4.0 * self.C_elem(p, q, p, q)
                    + 4.0 * self.C_elem(p, q, q, q)
                )
                M2[p, q] = M2[q, p] = val
        self.M1 = M1
        self.M2 = M2
        return self.M1, self.M2

    def make_M1(self):
        if self.M1 is None:
            self.make_measures()
        return self.M1

    def make_M2(self):
        if self.M2 is None:
            self.make_measures()
        return self.M2

    def MPQ_matrix_summary(self, print_threshold: float = 7.5e-4) -> str:
        """
        Generates a summary of the mutual correlation matrix M2.

        Parameters
        ----------
        print_threshold : float, optional, default=7.5e-4
            Only values greater than this threshold are printed.

        Returns
        -------
        summary : str
            A formatted string summarizing the mutual correlation matrix M2.
        """
        if self.M2 is None:
            self.make_M2()
        s_lines = [
            f"Mutual Correlation Matrix M2 (only values > {print_threshold:.1e}):",
            "=====================",
            "    P     Q      M_PQ",
            "---------------------",
        ]

        # get the upper triangle indices and values
        M2_vals = []
        for i in range(self.M2.shape[0]):
            for j in range(i + 1, self.M2.shape[1]):
                M2_vals.append(
                    (
                        self.M2[i, j],
                        list(range(self.nmo))[i],
                        list(range(self.nmo))[j],
                    )
                )
        M2_vals.sort(reverse=True, key=lambda x: x[0])

        for val, i, j in M2_vals:
            if val < print_threshold:
                break
            s_lines.append(f"{i:>5} {j:>5}  {val:8.6f}")

        s_lines.append("=====================")

        return "\n".join(s_lines)


class RMP2MPQFast:
    def __init__(self, mp2, U=None):
        self.mp2 = mp2
        self.nmo = mp2.nocc + mp2.nvir
        self.nocc = mp2.nocc
        self.nvir = mp2.nvir

        if mp2.t2 is None:
            raise ValueError("RMP2 t2 amplitudes must be stored.")

        # --- 1-RDM ---
        self.Γ1 = mp2.make_1rdm()

        # --- choose rotation ---
        if U is not None:
            self.U = U
            self.occs = np.diag(U.T @ self.Γ1 @ U)
        else:
            self.U, self.occs = self._build_block_no_rotation(self.Γ1)

        Uo = self.U[: self.nocc, : self.nocc]  # occupied block
        Uv = self.U[self.nocc :, self.nocc :]  # virtual block

        # --- fast masks (critical for performance) ---
        self.occ_mask = np.zeros(self.nmo, dtype=bool)
        self.occ_mask[: self.nocc] = True

        self.vir_mask = ~self.occ_mask

        # --- rotate t2 ---
        self.t2_no = self._rotate_t2(mp2.t2, Uo, Uv)

        self.gamma_oooo = 0.5 * np.einsum(
            "ijab,klab->ijkl", self.t2_no, self.t2_no, optimize=True
        )
        self.gamma_vvvv = 0.5 * np.einsum(
            "ijab,ijcd->abcd", self.t2_no, self.t2_no, optimize=True
        )
        self.gamma_ovov = np.einsum(
            "imac,jmbc->iajb", self.t2_no, self.t2_no, optimize=True
        )

        # --- rotated orbitals ---
        self.C_no = mp2.C[0] @ self.U

        self.M1 = None
        self.M2 = None

    # =========================
    # Partition helpers (FAST)
    # =========================
    def _o(self, p):
        return self.occ_mask[p]

    def _v(self, p):
        return self.vir_mask[p]

    # =========================
    # Full NO construction
    # =========================
    def _build_block_no_rotation(self, Gamma1):
        nocc = self.nocc
        nmo = self.nmo

        # --- split blocks ---
        Goo = Gamma1[:nocc, :nocc]
        Gvv = Gamma1[nocc:, nocc:]

        # --- diagonalize each block ---
        occ_vals, Uo = np.linalg.eigh(Goo)
        vir_vals, Uv = np.linalg.eigh(Gvv)

        # --- sort descending within each block ---
        occ_order = np.argsort(occ_vals)[::-1]
        vir_order = np.argsort(vir_vals)[::-1]

        occ_vals = occ_vals[occ_order]
        vir_vals = vir_vals[vir_order]

        Uo = Uo[:, occ_order]
        Uv = Uv[:, vir_order]

        # --- assemble full block-diagonal rotation ---
        U = np.eye(nmo)
        U[:nocc, :nocc] = Uo
        U[nocc:, nocc:] = Uv

        # --- combined occupations ---
        occs = np.diag(U.T @ Gamma1 @ U)

        return U, occs

    # =========================
    # Rotate t2 → full space
    # =========================
    def _rotate_t2(self, t2, Uo, Uv):
        t2_no = np.einsum(
            "pi,qj,ijab,ar,bs->pqrs", Uo.T, Uo.T, t2, Uv, Uv, optimize=True
        )
        return t2_no

    def occ_index(self, p):
        return p

    def vir_index(self, p):
        return p - self.nocc

    # =========================
    # Lambda elements
    # =========================

    def lambda2_aa_elem(self, p, q, r, s):

        val = 0.0

        # ---- FIRST ORDER (existing) ----
        if self._o(p) and self._o(q) and self._v(r) and self._v(s):
            i, j = p, q
            a, b = r - self.nocc, s - self.nocc

            val += (
                self.t2_no[i, j, a, b]
                - self.t2_no[i, j, b, a]
                - self.t2_no[j, i, a, b]
                + self.t2_no[j, i, b, a]
            )

        if self._v(p) and self._v(q) and self._o(r) and self._o(s):
            i, j = r, s
            a, b = p - self.nocc, q - self.nocc

            val += (
                self.t2_no[i, j, a, b]
                - self.t2_no[i, j, b, a]
                - self.t2_no[j, i, a, b]
                + self.t2_no[j, i, b, a]
            )

        # ---- SECOND ORDER (NEW) ----
        if self._o(p) and self._o(q) and self._o(r) and self._o(s):
            val += self.gamma_oooo[p, q, r, s]

        if self._v(p) and self._v(q) and self._v(r) and self._v(s):
            a, b, c, d = p - self.nocc, q - self.nocc, r - self.nocc, s - self.nocc
            val += self.gamma_vvvv[a, b, c, d]

        return val

    # def lambda2_aa_elem(self, p, q, r, s):
    #     if self._o(p) and self._o(q) and self._v(r) and self._v(s):
    #         i = p
    #         j = q
    #         a = r - self.nocc
    #         b = s - self.nocc

    #         return (
    #             self.t2_no[i, j, a, b]
    #             - self.t2_no[i, j, b, a]
    #             - self.t2_no[j, i, a, b]
    #             + self.t2_no[j, i, b, a]
    #         )

    #     if self._v(p) and self._v(q) and self._o(r) and self._o(s):
    #         i = r
    #         j = s
    #         a = p - self.nocc
    #         b = q - self.nocc

    #         return (
    #             self.t2_no[i, j, a, b]
    #             - self.t2_no[i, j, b, a]
    #             - self.t2_no[j, i, a, b]
    #             + self.t2_no[j, i, b, a]
    #         )

    #     return 0.0

    def lambda2_bb_elem(self, p, q, r, s):
        return self.lambda2_aa_elem(p, q, r, s)

    def lambda2_ab_elem(self, p, q, r, s):

        val = 0.0

        # ---- FIRST ORDER (existing) ----
        if self._o(p) and self._o(q) and self._v(r) and self._v(s):
            i, j = p, q
            a, b = r - self.nocc, s - self.nocc
            val += self.t2_no[i, j, a, b]

        if self._v(p) and self._v(q) and self._o(r) and self._o(s):
            i, j = r, s
            a, b = p - self.nocc, q - self.nocc
            val += self.t2_no[i, j, a, b]

        # ---- SECOND ORDER (NEW) ----
        if self._o(p) and self._v(q) and self._o(r) and self._v(s):
            i, a = p, q - self.nocc
            j, b = r, s - self.nocc
            val += self.gamma_ovov[i, a, j, b]

        if self._v(p) and self._o(q) and self._v(r) and self._o(s):
            j, b = q, p - self.nocc
            i, a = s, r - self.nocc
            val += self.gamma_ovov[i, a, j, b]

        return val

    # =========================
    # Cumulant contribution
    # =========================
    def C_elem(self, p, q, r, s):

        aa = self.lambda2_aa_elem(p, q, r, s)
        bb = self.lambda2_bb_elem(p, q, r, s)

        ab1 = self.lambda2_ab_elem(p, q, r, s)
        ab2 = self.lambda2_ab_elem(p, q, s, r)
        ab3 = self.lambda2_ab_elem(q, p, r, s)
        ab4 = self.lambda2_ab_elem(q, p, s, r)

        return 0.25 * (
            abs(aa) ** 2
            + abs(bb) ** 2
            + abs(ab1) ** 2
            + abs(ab2) ** 2
            + abs(ab3) ** 2
            + abs(ab4) ** 2
        )

    # =========================
    # Measures
    # =========================
    def make_measures(self):
        M1 = np.zeros(self.nmo)
        M2 = np.zeros((self.nmo, self.nmo))

        for p in range(self.nmo):
            M1[p] = self.C_elem(p, p, p, p)

        for p in range(self.nmo):
            for q in range(p + 1, self.nmo):
                val = (
                    4.0 * self.C_elem(p, p, p, q)
                    + 2.0 * self.C_elem(p, p, q, q)
                    + 4.0 * self.C_elem(p, q, p, q)
                    + 4.0 * self.C_elem(p, q, q, q)
                )
                M2[p, q] = M2[q, p] = val

        self.M1 = M1
        self.M2 = M2
        return self.M1, self.M2

    def make_M1(self):
        if self.M1 is None:
            self.make_measures()
        return self.M1

    def make_M2(self):
        if self.M2 is None:
            self.make_measures()
        return self.M2

    def MPQ_matrix_summary(self, print_threshold: float = 7.5e-4) -> str:
        if self.M2 is None:
            self.make_M2()

        s_lines = [
            f"Mutual Correlation Matrix M2 (only values > {print_threshold:.1e}):",
            "=====================",
            "    P     Q      M_PQ",
            "---------------------",
        ]

        M2_vals = []
        for i in range(self.nmo):
            for j in range(i + 1, self.nmo):
                M2_vals.append((self.M2[i, j], i, j))

        M2_vals.sort(reverse=True, key=lambda x: x[0])

        for val, i, j in M2_vals:
            if val < print_threshold:
                break
            s_lines.append(f"{i:>5} {j:>5}  {val:8.6f}")

        s_lines.append("=====================")
        return "\n".join(s_lines)

    @staticmethod
    def mpq_memory_bytes(nmo, dtype_bytes=8):
        return dtype_bytes * (nmo**2)

    @staticmethod
    def full_T_memory_bytes(nmo, dtype_bytes=8):
        return dtype_bytes * (nmo**4)

    @staticmethod
    def format_bytes(n):
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if n < 1024:
                return f"{n:.2f} {unit}"
            n /= 1024
