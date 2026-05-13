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

    def __init__(self, mp2, Ua=None, Ub=None):
        self.mp2 = mp2

        self.nmo = mp2.nmo
        self.naocc = mp2.naocc
        self.nbocc = mp2.nbocc
        self.navir = mp2.navir
        self.nbvir = mp2.nbvir

        if any(getattr(mp2, name, None) is None for name in ("t2_a", "t2_b", "t2_ab")):
            raise ValueError("UMP2 t2 amplitudes must be stored.")

        # --- 1-RDMs (spin-resolved) ---
        self.γa, self.γb = mp2.make_1rdm_sd()

        # --- rotations ---
        if Ua is None:
            self.Ua, self.occs_a = self._build_block_no_rotation(self.γa, self.naocc)
        else:
            self.Ua = Ua
            self.occs_a = np.diag(Ua.T @ self.γa @ Ua)

        if Ub is None:
            self.Ub, self.occs_b = self._build_block_no_rotation(self.γb, self.nbocc)
        else:
            self.Ub = Ub
            self.occs_b = np.diag(Ub.T @ self.γb @ Ub)

        # --- split rotations ---
        Uoa = self.Ua[: self.naocc, : self.naocc]
        Uva = self.Ua[self.naocc :, self.naocc :]

        Uob = self.Ub[: self.nbocc, : self.nbocc]
        Uvb = self.Ub[self.nbocc :, self.nbocc :]

        # --- rotate amplitudes ---
        self.t2_a = self._rotate_t2(self.mp2.t2_a, Uoa, Uva)
        self.t2_b = self._rotate_t2(self.mp2.t2_b, Uob, Uvb)

        self.t2_ab = np.einsum(
            "pi,qj,ijab,ar,bs->pqrs",
            Uoa.T,
            Uob.T,
            self.mp2.t2_ab,
            Uva,
            Uvb,
            optimize=True,
        )

        # --- SECOND ORDER TERMS ---

        # alpha-alpha
        self.gamma_oooo_aa = 0.5 * np.einsum(
            "ijab,klab->ijkl", self.t2_a, self.t2_a, optimize=True
        )
        self.gamma_vvvv_aa = 0.5 * np.einsum(
            "ijab,ijcd->abcd", self.t2_a, self.t2_a, optimize=True
        )

        # beta-beta
        self.gamma_oooo_bb = 0.5 * np.einsum(
            "ijab,klab->ijkl", self.t2_b, self.t2_b, optimize=True
        )
        self.gamma_vvvv_bb = 0.5 * np.einsum(
            "ijab,ijcd->abcd", self.t2_b, self.t2_b, optimize=True
        )

        # alpha-beta (ONLY ovov block exists)
        self.gamma_ovov_ab = np.einsum(
            "imac,jmbc->iajb", self.t2_ab, self.t2_ab, optimize=True
        )

        self.M1 = None
        self.M2 = None
        gamma_sf = self.γa + self.γb
        gamma_sf_no = self.Ua.T @ gamma_sf @ self.Ua
        self.Γ1 = gamma_sf_no

    @property
    def occs(self):
        # spin-summed occupations for plotting
        return np.diag(self.Γ1)

    def _build_block_no_rotation(self, Gamma1, nocc):
        nmo = Gamma1.shape[0]

        Goo = Gamma1[:nocc, :nocc]
        Gvv = Gamma1[nocc:, nocc:]

        occ_vals, Uo = np.linalg.eigh(Goo)
        vir_vals, Uv = np.linalg.eigh(Gvv)

        occ_order = np.argsort(occ_vals)[::-1]
        vir_order = np.argsort(vir_vals)[::-1]

        Uo = Uo[:, occ_order]
        Uv = Uv[:, vir_order]

        U = np.eye(nmo)
        U[:nocc, :nocc] = Uo
        U[nocc:, nocc:] = Uv

        occs = np.zeros(nmo)
        occs[:nocc] = occ_vals
        occs[nocc:] = vir_vals

        return U, occs

    def _rotate_t2(self, t2, Uo, Uv):
        return np.einsum(
            "pi,qj,ijab,ar,bs->pqrs",
            Uo.T,
            Uo.T,
            t2,
            Uv,
            Uv,
            optimize=True,
        )

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
        val = 0.0

        if self._oa(p) and self._oa(q) and self._va(r) and self._va(s):
            a, b = self._a_vir(r), self._a_vir(s)
            val += (
                self.t2_a[p, q, a, b]
                - self.t2_a[p, q, b, a]
                - self.t2_a[q, p, a, b]
                + self.t2_a[q, p, b, a]
            )

        if self._va(p) and self._va(q) and self._oa(r) and self._oa(s):
            a, b = self._a_vir(p), self._a_vir(q)
            val += (
                self.t2_a[r, s, a, b]
                - self.t2_a[r, s, b, a]
                - self.t2_a[s, r, a, b]
                + self.t2_a[s, r, b, a]
            )

        # second order
        if self._oa(p) and self._oa(q) and self._oa(r) and self._oa(s):
            val += self.gamma_oooo_aa[p, q, r, s]

        if self._va(p) and self._va(q) and self._va(r) and self._va(s):
            a, b, c, d = (
                self._a_vir(p),
                self._a_vir(q),
                self._a_vir(r),
                self._a_vir(s),
            )
            val += self.gamma_vvvv_aa[a, b, c, d]

        return val

    def lambda2_bb_elem(self, p, q, r, s):
        val = 0.0

        if self._ob(p) and self._ob(q) and self._vb(r) and self._vb(s):
            a, b = self._b_vir(r), self._b_vir(s)
            val += (
                self.t2_b[p, q, a, b]
                - self.t2_b[p, q, b, a]
                - self.t2_b[q, p, a, b]
                + self.t2_b[q, p, b, a]
            )

        if self._vb(p) and self._vb(q) and self._ob(r) and self._ob(s):
            a, b = self._b_vir(p), self._b_vir(q)
            val += (
                self.t2_b[r, s, a, b]
                - self.t2_b[r, s, b, a]
                - self.t2_b[s, r, a, b]
                + self.t2_b[s, r, b, a]
            )

        if self._ob(p) and self._ob(q) and self._ob(r) and self._ob(s):
            val += self.gamma_oooo_bb[p, q, r, s]

        if self._vb(p) and self._vb(q) and self._vb(r) and self._vb(s):
            a, b, c, d = (
                self._b_vir(p),
                self._b_vir(q),
                self._b_vir(r),
                self._b_vir(s),
            )
            val += self.gamma_vvvv_bb[a, b, c, d]

        return val

    def lambda2_ab_elem(self, p, q, r, s):
        val = 0.0

        # ---------- FIRST ORDER ----------
        if self._oa(p) and self._ob(q) and self._va(r) and self._vb(s):
            val += self.t2_ab[p, q, self._a_vir(r), self._b_vir(s)]

        if self._va(p) and self._vb(q) and self._oa(r) and self._ob(s):
            val += self.t2_ab[r, s, self._a_vir(p), self._b_vir(q)]
        # ---------- SECOND ORDER (corrected) ----------

        # (oα, vβ, oα, vβ)
        if self._oa(p) and self._vb(q) and self._oa(r) and self._vb(s):
            i = p
            j = r
            b1 = self._b_vir(q)
            b2 = self._b_vir(s)

            val += self.gamma_ovov_ab[i, b1, j, b2]

        # symmetric partner (IMPORTANT: do NOT reuse virtual as occupied)
        if self._oa(r) and self._vb(s) and self._oa(p) and self._vb(q):
            i = r
            j = p
            b1 = self._b_vir(s)
            b2 = self._b_vir(q)

            val += self.gamma_ovov_ab[i, b1, j, b2]

        return val

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

        # ---- SECOND ORDER ----
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


class RMP2MPQOnTheFly:
    """
    On-the-fly dyad mutual correlation M1/M2 for RMP2.

    This mirrors :class:`RMP2MPQFast` without requiring stored full MP2
    amplitudes. It generates and rotates occupied-pair amplitude blocks from
    the DF factors as needed.
    """

    def __init__(self, mp2, U=None, cache_pair_blocks=True):
        self.mp2 = mp2
        self.nmo = mp2.nocc + mp2.nvir
        self.nocc = mp2.nocc
        self.nvir = mp2.nvir
        self.cache_pair_blocks = cache_pair_blocks

        if getattr(mp2, "B_iaQ", None) is None:
            raise ValueError("mp2.B_iaQ is missing. Run mp2.run() first.")

        self.Γ1 = mp2.make_1rdm()

        if U is not None:
            self.U = U
            self.occs = np.diag(U.T @ self.Γ1 @ U)
        else:
            self.U, self.occs = self._build_block_no_rotation(self.Γ1)

        self.Uo = self.U[: self.nocc, : self.nocc]
        self.Uv = self.U[self.nocc :, self.nocc :]

        self.occ_mask = np.zeros(self.nmo, dtype=bool)
        self.occ_mask[: self.nocc] = True
        self.vir_mask = ~self.occ_mask

        # CHANGE:
        # For restricted MP2, mp2.C is usually a 2D MO coefficient matrix.
        # For safety, keep compatibility if C is accidentally spin-indexed.
        if isinstance(mp2.C, (tuple, list)):
            self.C_no = mp2.C[0] @ self.U
        elif getattr(mp2.C, "ndim", None) == 3:
            self.C_no = mp2.C[0] @ self.U
        else:
            self.C_no = mp2.C @ self.U

        self.M1 = None
        self.M2 = None

        self._cache_pair = {}
        self._cache_pair_as = {}
        self._cache_fixed = {}

    def _o(self, p):
        return self.occ_mask[p]

    def _v(self, p):
        return self.vir_mask[p]

    def _build_block_no_rotation(self, Gamma1):
        nocc = self.nocc

        Goo = Gamma1[:nocc, :nocc]
        Gvv = Gamma1[nocc:, nocc:]

        occ_vals, Uo = np.linalg.eigh(Goo)
        vir_vals, Uv = np.linalg.eigh(Gvv)

        occ_order = np.argsort(occ_vals)[::-1]
        vir_order = np.argsort(vir_vals)[::-1]

        Uo = Uo[:, occ_order]
        Uv = Uv[:, vir_order]

        U = np.eye(self.nmo)
        U[:nocc, :nocc] = Uo
        U[nocc:, nocc:] = Uv

        occs = np.diag(U.T @ Gamma1 @ U)
        return U, occs

    def _t2_fixed_j_canonical(self, j):
        """
        Canonical RMP2 amplitudes t2[i,j,a,b] for fixed occupied index j.

        Returns
        -------
        T : ndarray, shape (nocc, nvir, nvir)
        """
        if j in self._cache_fixed:
            return self._cache_fixed[j]

        B = self.mp2.B_iaQ
        eps_i = self.mp2.eps[: self.nocc]
        eps_a = self.mp2.eps[self.nocc :]
        eps_vv = eps_a[:, None] + eps_a[None, :]

        g = np.einsum("iaQ,bQ->iab", B, B[j], optimize=True)
        denom = eps_i[:, None, None] + self.mp2.eps[j] - eps_vv[None, :, :]

        T = self.mp2._safe_divide(g, denom, label="RMP2 denom")
        self._cache_fixed[j] = T
        return T

    def _t2_pair_no(self, i, j):
        """
        Rotated RMP2 amplitude block t2_NO[i,j,:,:].

        Returns
        -------
        Tij : ndarray, shape (nvir, nvir)
        """
        key = (i, j)
        if self.cache_pair_blocks and key in self._cache_pair:
            return self._cache_pair[key]

        T_ab = np.zeros((self.nvir, self.nvir))

        for J in range(self.nocc):
            T_fixed_J = self._t2_fixed_j_canonical(J)
            T_i_ab = np.einsum("I,Iab->ab", self.Uo[:, i], T_fixed_J, optimize=True)
            T_ab += self.Uo[J, j] * T_i_ab

        T_no = np.einsum("aA,ab,bB->AB", self.Uv, T_ab, self.Uv, optimize=True)

        if self.cache_pair_blocks:
            self._cache_pair[key] = T_no

        return T_no

    def _t2_pair_no_as(self, i, j):
        """
        Antisymmetrized same-spin rotated RMP2 amplitude block:

            t_as[i,j,a,b] = t[i,j,a,b] - t[i,j,b,a]

        Since _t2_pair_no returns the non-antisymmetrized spatial Coulomb
        amplitude, this helper should be used for same-spin aa/bb blocks.
        """
        key = (i, j)
        if self.cache_pair_blocks and key in self._cache_pair_as:
            return self._cache_pair_as[key]

        Tij = self._t2_pair_no(i, j)
        Tij_as = Tij - Tij.T

        if self.cache_pair_blocks:
            self._cache_pair_as[key] = Tij_as

        return Tij_as

    def _t2_elem(self, i, j, a, b):
        return self._t2_pair_no(i, j)[a, b]

    def _gamma_oooo_elem(self, i, j, k, l):
        """
        Same-spin second-order oooo block:

            gamma_oooo[i,j,k,l] = 1/2 sum_ab t_as[i,j,a,b] t_as[k,l,a,b]

        where t_as is the same-spin antisymmetrized amplitude.
        """
        Tij = self._t2_pair_no_as(i, j)
        Tkl = self._t2_pair_no_as(k, l)
        return 0.5 * np.einsum("ab,ab->", Tij, Tkl, optimize=True)

    def _gamma_vvvv_elem(self, a, b, c, d):
        """
        Same-spin second-order vvvv block:

            gamma_vvvv[a,b,c,d] = 1/2 sum_ij t_as[i,j,a,b] t_as[i,j,c,d]
        """
        val = 0.0
        for i in range(self.nocc):
            for j in range(self.nocc):
                Tij = self._t2_pair_no_as(i, j)
                val += Tij[a, b] * Tij[c, d]
        return 0.5 * val

    def _gamma_ovov_elem(self, i, a, j, b):
        val = 0.0
        for m in range(self.nocc):
            Tim = self._t2_pair_no(i, m)
            Tjm = self._t2_pair_no(j, m)
            val += np.dot(Tim[a, :], Tjm[b, :])
        return val

    def lambda2_aa_elem(self, p, q, r, s):
        if p == q or r == s:
            return 0.0

        val = 0.0

        # First-order oo/vv block.
        # CHANGE:
        # _t2_pair_no is non-antisymmetrized spatial t_ij^ab.
        # Same-spin needs only t_ij^ab - t_ij^ba, not the four-term expression.
        if self._o(p) and self._o(q) and self._v(r) and self._v(s):
            i, j = p, q
            a, b = r - self.nocc, s - self.nocc
            val += self._t2_elem(i, j, a, b) - self._t2_elem(i, j, b, a)

        # Hermitian vv/oo partner.
        if self._v(p) and self._v(q) and self._o(r) and self._o(s):
            i, j = r, s
            a, b = p - self.nocc, q - self.nocc
            val += self._t2_elem(i, j, a, b) - self._t2_elem(i, j, b, a)

        # Second-order oooo block.
        if self._o(p) and self._o(q) and self._o(r) and self._o(s):
            val += self._gamma_oooo_elem(p, q, r, s)

        # Second-order vvvv block.
        if self._v(p) and self._v(q) and self._v(r) and self._v(s):
            a, b, c, d = p - self.nocc, q - self.nocc, r - self.nocc, s - self.nocc
            val += self._gamma_vvvv_elem(a, b, c, d)

        return val

    def lambda2_bb_elem(self, p, q, r, s):
        return self.lambda2_aa_elem(p, q, r, s)

    def lambda2_ab_elem(self, p, q, r, s):
        val = 0.0

        # First-order oo/vv block: opposite-spin uses non-antisymmetrized amplitude.
        if self._o(p) and self._o(q) and self._v(r) and self._v(s):
            i, j = p, q
            a, b = r - self.nocc, s - self.nocc
            val += self._t2_elem(i, j, a, b)

        # Hermitian vv/oo partner.
        if self._v(p) and self._v(q) and self._o(r) and self._o(s):
            i, j = r, s
            a, b = p - self.nocc, q - self.nocc
            val += self._t2_elem(i, j, a, b)

        # Second-order o/v/o/v block.
        if self._o(p) and self._v(q) and self._o(r) and self._v(s):
            i, a = p, q - self.nocc
            j, b = r, s - self.nocc
            val += self._gamma_ovov_elem(i, a, j, b)

        # Permuted partner.
        if self._v(p) and self._o(q) and self._v(r) and self._o(s):
            j, b = q, p - self.nocc
            i, a = s, r - self.nocc
            val += self._gamma_ovov_elem(i, a, j, b)

        return val

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

    def make_measures(self, indices=None):
        if indices is None:
            indices = list(range(self.nmo))
        else:
            indices = list(indices)

        M1 = np.zeros(self.nmo)
        M2 = np.zeros((self.nmo, self.nmo))

        for p in indices:
            M1[p] = self.C_elem(p, p, p, p)

        for ii, p in enumerate(indices):
            for q in indices[ii + 1 :]:
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

    def make_M1(self, indices=None):
        if self.M1 is not None and indices is None:
            return self.M1

        if indices is None:
            indices = list(range(self.nmo))
        else:
            indices = list(indices)

        M1 = np.zeros(self.nmo)
        for p in indices:
            M1[p] = self.C_elem(p, p, p, p)

        self.M1 = M1
        return self.M1

    def make_M2(self, indices=None):
        if self.M2 is not None and indices is None:
            return self.M2

        self.make_measures(indices=indices)
        return self.M2

    def MPQ_matrix_summary(self, print_threshold: float = 7.5e-4, indices=None) -> str:
        if self.M2 is None or indices is not None:
            self.make_M2(indices=indices)

        if indices is None:
            print_indices = list(range(self.nmo))
        else:
            print_indices = list(indices)

        s_lines = [
            f"Mutual Correlation Matrix M2 (only values > {print_threshold:.1e}):",
            "=====================",
            "    P     Q      M_PQ",
            "---------------------",
        ]

        M2_vals = []
        for ii, i in enumerate(print_indices):
            for j in print_indices[ii + 1 :]:
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
        return f"{n:.2f} PB"


class UMP2MPQOnTheFly:
    """
    On-the-fly dyad mutual correlation M1/M2 for UMP2.

    This class avoids storing full MP2 amplitudes and avoids building full
    2-RDM / 2-cumulant tensors.

    It computes the cumulant elements needed by C_elem(p,q,r,s) by generating
    and rotating occupied-pair MP2 amplitude blocks on demand.

    Notes
    -----
    - Spatial MO indices p,q,r,s are in [0, nmo).
    - Same interface goal as UMP2MPQFast:
          make_M1()
          make_M2()
          make_measures()
          MPQ_matrix_summary()
    - Designed to compare against UMP2MPQFast.
    """

    def __init__(self, mp2, Ua=None, Ub=None, cache_pair_blocks=True):
        self.mp2 = mp2

        self.nmo = mp2.nmo
        self.naocc = mp2.naocc
        self.nbocc = mp2.nbocc
        self.navir = mp2.navir
        self.nbvir = mp2.nbvir

        self.cache_pair_blocks = cache_pair_blocks

        # Require only DF integrals, not stored t2.
        if getattr(mp2, "B_iaQ", None) is None:
            raise ValueError("mp2.B_iaQ is missing. Run mp2.run() first.")

        # ------------------------------------------------------------
        # 1-RDMs
        # ------------------------------------------------------------
        # This should use your streamed make_1rdm_sd() implementation.
        # If your current make_1rdm_sd() still calls _get_t2_all(), patch that first.
        self.γa, self.γb = mp2.make_1rdm_sd()

        # ------------------------------------------------------------
        # Rotations
        # ------------------------------------------------------------
        if Ua is None:
            self.Ua, self.occs_a = self._build_block_no_rotation(self.γa, self.naocc)
        else:
            self.Ua = Ua
            self.occs_a = np.diag(Ua.T @ self.γa @ Ua)

        if Ub is None:
            self.Ub, self.occs_b = self._build_block_no_rotation(self.γb, self.nbocc)
        else:
            self.Ub = Ub
            self.occs_b = np.diag(Ub.T @ self.γb @ Ub)

        # Split rotations.
        self.Uoa = self.Ua[: self.naocc, : self.naocc]
        self.Uva = self.Ua[self.naocc :, self.naocc :]

        self.Uob = self.Ub[: self.nbocc, : self.nbocc]
        self.Uvb = self.Ub[self.nbocc :, self.nbocc :]

        # For plotting compatibility with UMP2MPQFast.
        gamma_sf = self.γa + self.γb
        self.Γ1 = self.Ua.T @ gamma_sf @ self.Ua

        self.M1 = None
        self.M2 = None

        # Optional pair-block caches.
        self._cache_aa = {}
        self._cache_bb = {}
        self._cache_ab = {}

        # Optional canonical fixed-occupied-slab caches.
        # These reduce repeated DF contractions in the on-the-fly path.
        self._cache_fixed_aa = {}
        self._cache_fixed_bb = {}
        self._cache_fixed_ab_beta = {}

    @property
    def occs(self):
        return np.diag(self.Γ1)

    # ============================================================
    # Rotation helper
    # ============================================================

    def _build_block_no_rotation(self, Gamma1, nocc):
        nmo = Gamma1.shape[0]

        Goo = Gamma1[:nocc, :nocc]
        Gvv = Gamma1[nocc:, nocc:]

        occ_vals, Uo = np.linalg.eigh(Goo)
        vir_vals, Uv = np.linalg.eigh(Gvv)

        occ_order = np.argsort(occ_vals)[::-1]
        vir_order = np.argsort(vir_vals)[::-1]

        occ_vals = occ_vals[occ_order]
        vir_vals = vir_vals[vir_order]

        Uo = Uo[:, occ_order]
        Uv = Uv[:, vir_order]

        U = np.eye(nmo)
        U[:nocc, :nocc] = Uo
        U[nocc:, nocc:] = Uv

        occs = np.zeros(nmo)
        occs[:nocc] = occ_vals
        occs[nocc:] = vir_vals

        return U, occs

    # ============================================================
    # Index helpers
    # ============================================================

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

    # ============================================================
    # Canonical on-the-fly MP2 amplitude slabs
    # ============================================================

    def _t2_aa_fixed_j_canonical(self, j):
        """
        Canonical alpha-alpha amplitudes t2_a[i,j,a,b] for fixed j.

        Returns
        -------
        T : ndarray, shape (naocc, navir, navir)
        """

        if j in self._cache_fixed_aa:
            return self._cache_fixed_aa[j]

        Ba, _ = self.mp2.B_iaQ

        eps_i = self.mp2.eps_a[: self.naocc]
        eps_a = self.mp2.eps_a[self.naocc :]

        eps_vv = eps_a[:, None] + eps_a[None, :]

        g = np.einsum("iaQ,bQ->iab", Ba, Ba[j], optimize=True)
        g_as = g - g.transpose(0, 2, 1)

        denom = eps_i[:, None, None] + self.mp2.eps_a[j] - eps_vv[None, :, :]

        T = self.mp2._safe_divide(g_as, denom, label="UMP2 aa denom")

        self._cache_fixed_aa[j] = T
        return T

    def _t2_bb_fixed_j_canonical(self, j):
        """
        Canonical beta-beta amplitudes t2_b[i,j,a,b] for fixed j.

        Returns
        -------
        T : ndarray, shape (nbocc, nbvir, nbvir)
        """

        if j in self._cache_fixed_bb:
            return self._cache_fixed_bb[j]

        _, Bb = self.mp2.B_iaQ

        eps_i = self.mp2.eps_b[: self.nbocc]
        eps_a = self.mp2.eps_b[self.nbocc :]

        eps_vv = eps_a[:, None] + eps_a[None, :]

        g = np.einsum("iaQ,bQ->iab", Bb, Bb[j], optimize=True)
        g_as = g - g.transpose(0, 2, 1)

        denom = eps_i[:, None, None] + self.mp2.eps_b[j] - eps_vv[None, :, :]

        T = self.mp2._safe_divide(g_as, denom, label="UMP2 bb denom")

        self._cache_fixed_bb[j] = T
        return T

    def _t2_ab_fixed_beta_j_canonical(self, j):
        """
        Canonical alpha-beta amplitudes t2_ab[i,j,a,b] for fixed beta occupied j.

        Returns
        -------
        T : ndarray, shape (naocc, navir, nbvir)
        """

        if j in self._cache_fixed_ab_beta:
            return self._cache_fixed_ab_beta[j]

        Ba, Bb = self.mp2.B_iaQ

        eps_ai = self.mp2.eps_a[: self.naocc]
        eps_av = self.mp2.eps_a[self.naocc :]
        eps_bv = self.mp2.eps_b[self.nbocc :]

        eps_vv = eps_av[:, None] + eps_bv[None, :]

        g = np.einsum("iaQ,bQ->iab", Ba, Bb[j], optimize=True)

        denom = eps_ai[:, None, None] + self.mp2.eps_b[j] - eps_vv[None, :, :]

        T = self.mp2._safe_divide(g, denom, label="UMP2 ab denom")

        self._cache_fixed_ab_beta[j] = T
        return T

    # ============================================================
    # Rotated occupied-pair amplitude blocks in NO basis
    # ============================================================

    def _t2_aa_pair_no(self, i, j):
        """
        Rotated alpha-alpha amplitude block t2_a_NO[i,j,:,:].

        Returns
        -------
        Tij : ndarray, shape (navir, navir)
        """
        key = (i, j)
        if self.cache_pair_blocks and key in self._cache_aa:
            return self._cache_aa[key]

        T_ab = np.zeros((self.navir, self.navir))

        # t_NO[i,j,A,B] =
        # sum_IJab Uoa[I,i] Uoa[J,j] t_CAN[I,J,a,b] Uva[a,A] Uva[b,B]
        for J in range(self.naocc):
            T_fixed_J = self._t2_aa_fixed_j_canonical(J)  # I,a,b

            # Contract occupied I into NO occupied i.
            T_i_ab = np.einsum(
                "I,Iab->ab",
                self.Uoa[:, i],
                T_fixed_J,
                optimize=True,
            )

            T_ab += self.Uoa[J, j] * T_i_ab

        # Rotate virtuals.
        T_no = np.einsum(
            "aA,ab,bB->AB",
            self.Uva,
            T_ab,
            self.Uva,
            optimize=True,
        )

        if self.cache_pair_blocks:
            self._cache_aa[key] = T_no

        return T_no

    def _t2_bb_pair_no(self, i, j):
        """
        Rotated beta-beta amplitude block t2_b_NO[i,j,:,:].

        Returns
        -------
        Tij : ndarray, shape (nbvir, nbvir)
        """
        key = (i, j)
        if self.cache_pair_blocks and key in self._cache_bb:
            return self._cache_bb[key]

        T_ab = np.zeros((self.nbvir, self.nbvir))

        for J in range(self.nbocc):
            T_fixed_J = self._t2_bb_fixed_j_canonical(J)  # I,a,b

            T_i_ab = np.einsum(
                "I,Iab->ab",
                self.Uob[:, i],
                T_fixed_J,
                optimize=True,
            )

            T_ab += self.Uob[J, j] * T_i_ab

        T_no = np.einsum(
            "aA,ab,bB->AB",
            self.Uvb,
            T_ab,
            self.Uvb,
            optimize=True,
        )

        if self.cache_pair_blocks:
            self._cache_bb[key] = T_no

        return T_no

    def _t2_ab_pair_no(self, i, j):
        """
        Rotated alpha-beta amplitude block t2_ab_NO[i,j,:,:].

        Here i is alpha occupied NO index and j is beta occupied NO index.

        Returns
        -------
        Tij : ndarray, shape (navir, nbvir)
        """
        key = (i, j)
        if self.cache_pair_blocks and key in self._cache_ab:
            return self._cache_ab[key]

        T_ab = np.zeros((self.navir, self.nbvir))

        for J in range(self.nbocc):
            T_fixed_J = self._t2_ab_fixed_beta_j_canonical(J)  # I,a,b

            T_i_ab = np.einsum(
                "I,Iab->ab",
                self.Uoa[:, i],
                T_fixed_J,
                optimize=True,
            )

            T_ab += self.Uob[J, j] * T_i_ab

        T_no = np.einsum(
            "aA,ab,bB->AB",
            self.Uva,
            T_ab,
            self.Uvb,
            optimize=True,
        )

        if self.cache_pair_blocks:
            self._cache_ab[key] = T_no

        return T_no

    # ============================================================
    # Rotated t2 elements
    # ============================================================

    def _t2_aa_elem(self, i, j, a, b):
        return self._t2_aa_pair_no(i, j)[a, b]

    def _t2_bb_elem(self, i, j, a, b):
        return self._t2_bb_pair_no(i, j)[a, b]

    def _t2_ab_elem(self, i, j, a, b):
        return self._t2_ab_pair_no(i, j)[a, b]

    # ============================================================
    # Second-order contractions, evaluated on demand
    # ============================================================

    def _gamma_oooo_aa_elem(self, i, j, k, l):
        Tij = self._t2_aa_pair_no(i, j)
        Tkl = self._t2_aa_pair_no(k, l)
        return 0.5 * np.einsum("ab,ab->", Tij, Tkl, optimize=True)

    def _gamma_oooo_bb_elem(self, i, j, k, l):
        Tij = self._t2_bb_pair_no(i, j)
        Tkl = self._t2_bb_pair_no(k, l)
        return 0.5 * np.einsum("ab,ab->", Tij, Tkl, optimize=True)

    def _gamma_vvvv_aa_elem(self, a, b, c, d):
        val = 0.0
        for i in range(self.naocc):
            for j in range(self.naocc):
                Tij = self._t2_aa_pair_no(i, j)
                val += Tij[a, b] * Tij[c, d]
        return 0.5 * val

    def _gamma_vvvv_bb_elem(self, a, b, c, d):
        val = 0.0
        for i in range(self.nbocc):
            for j in range(self.nbocc):
                Tij = self._t2_bb_pair_no(i, j)
                val += Tij[a, b] * Tij[c, d]
        return 0.5 * val

    def _gamma_ovov_ab_elem(self, i, a, j, b):
        """
        Opposite-spin second-order ovov-like contraction.

        Matches the structure used in UMP2MPQFast:

            gamma_ovov_ab[i,a,j,b] = sum_{m,c} t_ab[i,m,a,c] t_ab[j,m,b,c]

        where a,b are alpha-virtual indices and c is beta-virtual.
        """
        val = 0.0
        for m in range(self.nbocc):
            Tim = self._t2_ab_pair_no(i, m)
            Tjm = self._t2_ab_pair_no(j, m)
            val += np.dot(Tim[a, :], Tjm[b, :])
        return val

    # ============================================================
    # Cumulant elements
    # ============================================================
    def lambda2_aa_elem(self, p, q, r, s):
        """
        Alpha-alpha cumulant element in the rotated NO basis.

        Same-spin cumulants are antisymmetric with respect to exchange of the
        first pair and the second pair:

            lambda[p,q,r,s] = -lambda[q,p,r,s]
            lambda[p,q,r,s] = -lambda[p,q,s,r]

        Therefore, if p == q or r == s, the element is exactly zero.
        This pruning is essential for fast M1 evaluation.
        """

        if p == q or r == s:
            return 0.0

        val = 0.0

        # First-order doubles block: oo-vv.
        if self._oa(p) and self._oa(q) and self._va(r) and self._va(s):
            a = self._a_vir(r)
            b = self._a_vir(s)

            val += (
                self._t2_aa_elem(p, q, a, b)
                - self._t2_aa_elem(p, q, b, a)
                - self._t2_aa_elem(q, p, a, b)
                + self._t2_aa_elem(q, p, b, a)
            )

        # Hermitian partner: vv-oo.
        if self._va(p) and self._va(q) and self._oa(r) and self._oa(s):
            a = self._a_vir(p)
            b = self._a_vir(q)

            val += (
                self._t2_aa_elem(r, s, a, b)
                - self._t2_aa_elem(r, s, b, a)
                - self._t2_aa_elem(s, r, a, b)
                + self._t2_aa_elem(s, r, b, a)
            )

        # Second-order oooo.
        if self._oa(p) and self._oa(q) and self._oa(r) and self._oa(s):
            val += self._gamma_oooo_aa_elem(p, q, r, s)

        # Second-order vvvv.
        if self._va(p) and self._va(q) and self._va(r) and self._va(s):
            a = self._a_vir(p)
            b = self._a_vir(q)
            c = self._a_vir(r)
            d = self._a_vir(s)

            val += self._gamma_vvvv_aa_elem(a, b, c, d)

        return val

    def lambda2_bb_elem(self, p, q, r, s):
        """
        Beta-beta cumulant element in the rotated NO basis.

        Same-spin cumulants are antisymmetric with respect to exchange of the
        first pair and the second pair. Therefore p == q or r == s gives an
        exactly zero element.
        """

        if p == q or r == s:
            return 0.0

        val = 0.0

        # First-order doubles block: oo-vv.
        if self._ob(p) and self._ob(q) and self._vb(r) and self._vb(s):
            a = self._b_vir(r)
            b = self._b_vir(s)

            val += (
                self._t2_bb_elem(p, q, a, b)
                - self._t2_bb_elem(p, q, b, a)
                - self._t2_bb_elem(q, p, a, b)
                + self._t2_bb_elem(q, p, b, a)
            )

        # Hermitian partner: vv-oo.
        if self._vb(p) and self._vb(q) and self._ob(r) and self._ob(s):
            a = self._b_vir(p)
            b = self._b_vir(q)

            val += (
                self._t2_bb_elem(r, s, a, b)
                - self._t2_bb_elem(r, s, b, a)
                - self._t2_bb_elem(s, r, a, b)
                + self._t2_bb_elem(s, r, b, a)
            )

        # Second-order oooo.
        if self._ob(p) and self._ob(q) and self._ob(r) and self._ob(s):
            val += self._gamma_oooo_bb_elem(p, q, r, s)

        # Second-order vvvv.
        if self._vb(p) and self._vb(q) and self._vb(r) and self._vb(s):
            a = self._b_vir(p)
            b = self._b_vir(q)
            c = self._b_vir(r)
            d = self._b_vir(s)

            val += self._gamma_vvvv_bb_elem(a, b, c, d)

        return val

    def lambda2_ab_elem(self, p, q, r, s):
        """
        Opposite-spin cumulant element.

        This follows the same index logic as the existing UMP2MPQFast class:
        first index / third index are alpha-space;
        second index / fourth index are beta-space.
        """
        val = 0.0

        # First-order oo-vv block.
        if self._oa(p) and self._ob(q) and self._va(r) and self._vb(s):
            val += self._t2_ab_elem(
                p,
                q,
                self._a_vir(r),
                self._b_vir(s),
            )

        # First-order vv-oo partner.
        if self._va(p) and self._vb(q) and self._oa(r) and self._ob(s):
            val += self._t2_ab_elem(
                r,
                s,
                self._a_vir(p),
                self._b_vir(q),
            )

        # ------------------------------------------------------------
        # Second-order ovov-like block.
        #
        # This mirrors the contraction shape used in your UMP2MPQFast.
        # Important: this assumes q and s map to alpha-virtual-like slots
        # after the spin-free C_elem permutations. If this disagrees with
        # the full cumulant test, this is the first place to inspect.
        # ------------------------------------------------------------
        if self._oa(p) and self._va(q) and self._oa(r) and self._va(s):
            val += self._gamma_ovov_ab_elem(
                p,
                self._a_vir(q),
                r,
                self._a_vir(s),
            )

        if self._oa(r) and self._va(s) and self._oa(p) and self._va(q):
            val += self._gamma_ovov_ab_elem(
                r,
                self._a_vir(s),
                p,
                self._a_vir(q),
            )

        return val

    # ============================================================
    # Mutual correlation element
    # ============================================================

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

    # ============================================================
    # M1 / M2
    # ============================================================
    def make_measures(self, indices=None):
        """
        Compute M1 and M2.

        Parameters
        ----------
        indices : iterable[int] or None
            If None, compute M1/M2 over all orbitals.
            If provided, compute M1 and M2 only for this orbital subset,
            but return full-size arrays with zeros elsewhere.
        """

        if indices is None:
            indices = list(range(self.nmo))
        else:
            indices = list(indices)

        M1 = np.zeros(self.nmo)
        M2 = np.zeros((self.nmo, self.nmo))

        for p in indices:
            M1[p] = self.C_elem(p, p, p, p)

        for ii, p in enumerate(indices):
            for q in indices[ii + 1 :]:
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

    def make_M1(self, indices=None):
        """
        Compute M1 only.

        This is useful because M1 should not require full M2 construction.
        """

        if self.M1 is not None and indices is None:
            return self.M1

        if indices is None:
            indices = list(range(self.nmo))
        else:
            indices = list(indices)

        M1 = np.zeros(self.nmo)

        for p in indices:
            M1[p] = self.C_elem(p, p, p, p)

        self.M1 = M1
        return self.M1

    def make_M2(self, indices=None):
        """
        Compute M2.

        If M2 has already been computed over all orbitals and indices is None,
        return the cached M2. Otherwise recompute for the requested subset.
        """

        if self.M2 is not None and indices is None:
            return self.M2

        self.make_measures(indices=indices)
        return self.M2

    def MPQ_matrix_summary(self, print_threshold: float = 7.5e-4, indices=None) -> str:
        """
        Generates a summary of the mutual correlation matrix M2.

        Parameters
        ----------
        print_threshold : float, optional, default=7.5e-4
            Only values greater than this threshold are printed.

        indices : iterable[int] or None
            Optional orbital subset for M2 construction and printing.
        """

        if self.M2 is None or indices is not None:
            self.make_M2(indices=indices)

        if indices is None:
            print_indices = list(range(self.nmo))
        else:
            print_indices = list(indices)

        s_lines = [
            f"Mutual Correlation Matrix M2 (only values > {print_threshold:.1e}):",
            "=====================",
            "    P     Q      M_PQ",
            "---------------------",
        ]

        M2_vals = []
        for ii, i in enumerate(print_indices):
            for j in print_indices[ii + 1 :]:
                M2_vals.append((self.M2[i, j], i, j))

        M2_vals.sort(reverse=True, key=lambda x: x[0])

        for val, i, j in M2_vals:
            if val < print_threshold:
                break
            s_lines.append(f"{i:>5} {j:>5}  {val:8.6f}")

        s_lines.append("=====================")

        return "\n".join(s_lines)
