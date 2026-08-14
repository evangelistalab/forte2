import numpy as np
from typing import Any
from numpy.typing import NDArray
from forte2.helpers import logger
import scipy as sp

from forte2.lib import cpp_helpers


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

        # Extract the spin-dependent 1-RDM from the solver.
        γa, γb = solver.sub_solvers[sub_solver_index].make_sd_1rdm(root)

        # Form the spin-free 1-RDM.
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


class RMP2MPQOnTheFly:
    """Evaluate dyad mutual correlation measures on the fly for RMP2.

    Occupied-pair amplitude blocks are generated from density-fitting factors
    as needed, so the complete rank-four amplitude tensor is not allocated up
    front.  Pair-block caching scales with the requested orbital subset;
    canonical fixed-occupied slab caching is opt-in because it can grow to the
    size of the complete amplitude tensor.
    Set ``include_quadratic=False`` to retain only cumulant terms linear in the
    MP2 doubles amplitudes.

    ``Gamma1`` and ``Γ1`` are stored in the block natural-orbital basis;
    the original canonical-MO density is retained as ``Gamma1_mo``.
    """

    def __init__(
        self,
        mp2,
        U=None,
        cache_pair_blocks=True,
        cache_fixed_slabs=False,
        include_quadratic=True,
        orbital_indices=None,
    ):
        self.mp2 = mp2
        self.nmo = mp2.nocc + mp2.nvir
        self.nocc = mp2.nocc
        self.nvir = mp2.nvir
        self.cache_pair_blocks = cache_pair_blocks
        self.cache_fixed_slabs = cache_fixed_slabs
        self.include_quadratic = bool(include_quadratic)

        if getattr(mp2, "B_iaQ", None) is None:
            raise ValueError("mp2.B_iaQ is missing. Run mp2.run() first.")

        gamma1_mo = mp2.make_1rdm()
        self.Gamma1_mo = 0.5 * (gamma1_mo + gamma1_mo.T.conj())
        self.Γ1_mo = self.Gamma1_mo

        if U is not None:
            self.U = np.asarray(U)
            if self.U.shape != (self.nmo, self.nmo):
                raise ValueError(
                    f"U has shape {self.U.shape}; expected "
                    f"({self.nmo}, {self.nmo})."
                )
        else:
            _, _, self.U = mp2.make_natural_orbital_transform(self.Gamma1_mo)

        self.Gamma1_no = self.U.T.conj() @ self.Gamma1_mo @ self.U
        self.Gamma1_no = 0.5 * (self.Gamma1_no + self.Gamma1_no.T.conj())
        self.Gamma1 = self.Gamma1_no
        self.Γ1 = self.Gamma1_no
        self.occs = np.diag(self.Gamma1_no).real

        self.Uo = self.U[: self.nocc, : self.nocc]
        self.Uv = self.U[self.nocc :, self.nocc :]

        self.occ_mask = np.zeros(self.nmo, dtype=bool)
        self.occ_mask[: self.nocc] = True
        self.vir_mask = ~self.occ_mask

        # RMP2 normally stores one 2D coefficient matrix.  Accept the legacy
        # spin-indexed representation as well.
        if isinstance(mp2.C, (tuple, list)):
            self.C_no = mp2.C[0] @ self.U
        elif getattr(mp2.C, "ndim", None) == 3:
            self.C_no = mp2.C[0] @ self.U
        else:
            self.C_no = mp2.C @ self.U
        self.no_occs = self.occs
        self.no_transform = (self.C_no, self.no_occs, self.U)

        self.M1 = None
        self.M2 = None
        self._M1_indices = None
        self._M2_indices = None
        self.rdm_info_indices = self._normalize_indices(orbital_indices)

        self._cache_pair = {}
        self._cache_pair_as = {}
        self._cache_fixed = {}

    def _o(self, p):
        return self.occ_mask[p]

    def _v(self, p):
        return self.vir_mask[p]

    def _normalize_indices(self, indices):
        if indices is None:
            requested = tuple(range(self.nmo))
        else:
            requested = tuple(dict.fromkeys(int(p) for p in indices))
        if any(p < 0 or p >= self.nmo for p in requested):
            raise IndexError("Every requested orbital index must be in [0, nmo).")
        if not requested:
            raise ValueError("At least one orbital index must be requested.")
        return requested

    def _requested_indices(self, indices):
        if indices is None:
            return self.rdm_info_indices
        return self._normalize_indices(indices)

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
        if self.cache_fixed_slabs:
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
        Tij = self._t2_pair_no_as(i, j)
        Tkl = self._t2_pair_no_as(k, l)
        return 0.5 * np.einsum("ab,ab->", Tij, Tkl, optimize=True)

    def _gamma_vvvv_elem(self, a, b, c, d):
        val = 0.0
        for i in range(self.nocc):
            for j in range(self.nocc):
                Tij = self._t2_pair_no_as(i, j)
                val += Tij[a, b] * Tij[c, d]
        return 0.5 * val

    def _gamma_ovov_elem(self, i, a, j, b):
        """Return a quadratic opposite-spin particle-hole contraction."""
        val = 0.0

        for m in range(self.nocc):
            Tim = self._t2_pair_no(i, m)
            Tjm = self._t2_pair_no(j, m)

            # -sum_c T_im^{c b} T_jm^{c a}
            val -= np.dot(Tim[:, b], Tjm[:, a])

        return val

    def lambda2_aa_linear_elem(self, p, q, r, s):
        """Return the same-spin cumulant contribution linear in ``t2``."""
        if p == q or r == s:
            return 0.0

        if self._o(p) and self._o(q) and self._v(r) and self._v(s):
            i, j = p, q
            a, b = r - self.nocc, s - self.nocc
            return self._t2_elem(i, j, a, b) - self._t2_elem(i, j, b, a)

        if self._v(p) and self._v(q) and self._o(r) and self._o(s):
            i, j = r, s
            a, b = p - self.nocc, q - self.nocc
            return self._t2_elem(i, j, a, b) - self._t2_elem(i, j, b, a)

        return 0.0

    def lambda2_aa_quadratic_elem(self, p, q, r, s):
        """Return the same-spin cumulant contribution quadratic in ``t2``."""
        if p == q or r == s:
            return 0.0

        if self._o(p) and self._o(q) and self._o(r) and self._o(s):
            return self._gamma_oooo_elem(p, q, r, s)

        if self._v(p) and self._v(q) and self._v(r) and self._v(s):
            a, b, c, d = p - self.nocc, q - self.nocc, r - self.nocc, s - self.nocc
            return self._gamma_vvvv_elem(a, b, c, d)

        return 0.0

    def lambda2_aa_elem(self, p, q, r, s):
        value = self.lambda2_aa_linear_elem(p, q, r, s)
        if self.include_quadratic:
            value += self.lambda2_aa_quadratic_elem(p, q, r, s)
        return value

    def lambda2_bb_elem(self, p, q, r, s):
        return self.lambda2_aa_elem(p, q, r, s)

    def lambda2_ab_linear_elem(self, p, q, r, s):
        """Return the opposite-spin cumulant contribution linear in ``t2``."""
        if self._o(p) and self._o(q) and self._v(r) and self._v(s):
            i, j = p, q
            a, b = r - self.nocc, s - self.nocc
            return self._t2_elem(i, j, a, b)

        if self._v(p) and self._v(q) and self._o(r) and self._o(s):
            i, j = r, s
            a, b = p - self.nocc, q - self.nocc
            return self._t2_elem(i, j, a, b)

        return 0.0

    def lambda2_ab_quadratic_elem(self, p, q, r, s):
        """Return the opposite-spin cumulant contribution quadratic in ``t2``."""
        if self._o(p) and self._v(q) and self._o(r) and self._v(s):
            i, a = p, q - self.nocc
            j, b = r, s - self.nocc
            return self._gamma_ovov_elem(i, a, j, b)

        if self._v(p) and self._o(q) and self._v(r) and self._o(s):
            j, b = q, p - self.nocc
            i, a = s, r - self.nocc
            return self._gamma_ovov_elem(i, a, j, b)

        return 0.0

    def lambda2_ab_elem(self, p, q, r, s):
        value = self.lambda2_ab_linear_elem(p, q, r, s)
        if self.include_quadratic:
            value += self.lambda2_ab_quadratic_elem(p, q, r, s)
        return value

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

    def make_M1(self, indices=None):
        """Compute the retained-tensor one-orbital correlation measure."""
        requested = self._requested_indices(indices)
        if self.M1 is not None and requested == self._M1_indices:
            return self.M1

        M1 = np.zeros(self.nmo)
        for p in requested:
            M1[p] = self.C_elem(p, p, p, p)

        self.M1 = M1
        self._M1_indices = requested
        return self.M1

    def make_measures(self, indices=None):
        """Compute and return ``(M1, M2)`` for a common orbital subset."""
        return self.make_M1(indices=indices), self.make_M2(indices=indices)

    def make_M2(self, indices=None):
        requested = self._requested_indices(indices)
        if self.M2 is not None and requested == self._M2_indices:
            return self.M2

        M2 = np.zeros((self.nmo, self.nmo))
        for position, p in enumerate(requested):
            for q in requested[position + 1 :]:
                value = (
                    4.0 * self.C_elem(p, p, p, q)
                    + 2.0 * self.C_elem(p, p, q, q)
                    + 4.0 * self.C_elem(p, q, p, q)
                    + 4.0 * self.C_elem(p, q, q, q)
                )
                M2[p, q] = M2[q, p] = value

        self.M2 = M2
        self._M2_indices = requested
        return self.M2

    def MPQ_matrix_summary(self, print_threshold: float = 7.5e-4, indices=None) -> str:
        requested = self._requested_indices(indices)
        if self.M2 is None or requested != self._M2_indices:
            self.make_M2(indices=requested)

        s_lines = [
            f"Mutual Correlation Matrix M2 (only values > {print_threshold:.1e}):",
            "=====================",
            "    P     Q      M_PQ",
            "---------------------",
        ]

        values = []
        for position, p in enumerate(requested):
            for q in requested[position + 1 :]:
                values.append((self.M2[p, q], p, q))

        values.sort(reverse=True, key=lambda item: item[0])

        for value, p, q in values:
            if value < print_threshold:
                break
            s_lines.append(f"{p:>5} {q:>5}  {value:8.6f}")

        s_lines.append("=====================")
        return "\n".join(s_lines)

    def clear_amplitude_caches(self):
        """Release cached rotated pairs and canonical amplitude slabs."""
        self._cache_pair.clear()
        self._cache_pair_as.clear()
        self._cache_fixed.clear()


class UMP2MPQOnTheFly:
    """Evaluate a low-cost, block-rotated UMP2 dyad correlation matrix.

    The retained approximate cumulant blocks are

    * first-order ``oovv`` and ``vvoo`` blocks for alpha-alpha, beta-beta,
      and alpha-beta amplitudes;
    * quadratic same-spin ``oooo`` and ``vvvv`` blocks, when enabled; and
    * the two pure opposite-spin particle-hole orientations, when enabled.

    Other quadratic particle-hole and opposite-spin ``oooo``/``vvvv``
    contributions are not included.  Consequently, ``M2`` is a screening
    diagnostic and must not be described as the complete MP2 cumulant norm.

    When ``Ua`` and ``Ub`` contain occupied-virtual mixing, only their
    occupied-occupied and virtual-virtual blocks are applied.  This is the
    block-projected common-NO approximation used by this low-cost class.
    The public spin densities ``gamma1_a``/``gamma1_b`` and the spin-free
    ``Gamma1``/``Γ1`` are stored in the target NO basis; canonical-MO inputs
    are retained as ``gamma1_mo_a`` and ``gamma1_mo_b``.

    Parameters
    ----------
    mp2
        Executed UMP2 object containing ``B_iaQ``, orbital energies, and the
        spin-resolved MP2 one-particle density matrices.
    Ua, Ub : ndarray, optional
        Alpha and beta canonical-MO to target-orbital transformations.  If
        omitted, separate occupied/virtual block natural-orbital rotations
        are built for each spin.
    gamma1 : tuple(ndarray, ndarray), optional
        Previously computed alpha and beta MP2 1-RDMs.  Passing this from the
        common-NO wrapper avoids computing the 1-RDM twice.
    cache_pair_blocks : bool, optional
        Cache rotated occupied-pair amplitude blocks.  This minimizes wall
        time at amplitude-scale memory cost.
    cache_fixed_slabs : bool, optional
        Cache canonical fixed-occupied amplitude slabs.  This is fastest, but
        the cache can eventually hold the complete canonical amplitudes.
        Disabled by default to preserve the low-memory behavior.
    orbital_indices : iterable[int], optional
        Default common-NO subset used by ``make_M1``, ``make_M2``, and
        ``make_measures``.  Full-space arrays are returned with zeros outside
        this RDM-info subset.
    include_quadratic : bool, optional
        Include retained cumulant contractions quadratic in the MP2 doubles
        amplitudes.  If false, only the linear ``oovv`` and ``vvoo`` terms are
        used.
    common_no_mixing_tolerance : float, optional
        Threshold above which occupied-virtual mixing produces a warning.
    """

    def __init__(
        self,
        mp2,
        Ua=None,
        Ub=None,
        gamma1=None,
        orbital_indices=None,
        include_quadratic=True,
        cache_pair_blocks=True,
        cache_fixed_slabs=False,
        common_no_mixing_tolerance=1.0e-10,
    ):
        self.mp2 = mp2

        self.nmo = mp2.nmo
        self.naocc = mp2.naocc
        self.nbocc = mp2.nbocc
        self.navir = mp2.navir
        self.nbvir = mp2.nbvir

        self.cache_pair_blocks = cache_pair_blocks
        self.cache_fixed_slabs = cache_fixed_slabs
        self.include_quadratic = bool(include_quadratic)
        self.common_no_mixing_tolerance = common_no_mixing_tolerance
        self.rdm_info_indices = self._normalize_indices(orbital_indices)

        if getattr(mp2, "B_iaQ", None) is None:
            raise ValueError("mp2.B_iaQ is missing. Run mp2.run() first.")

        if gamma1 is None:
            gamma1 = mp2.make_1rdm_sd()
        gamma1_mo_a, gamma1_mo_b = gamma1
        self.gamma1_mo_a = 0.5 * (gamma1_mo_a + gamma1_mo_a.T.conj())
        self.gamma1_mo_b = 0.5 * (gamma1_mo_b + gamma1_mo_b.T.conj())

        if Ua is None:
            self.Ua, _ = self._build_block_no_rotation(
                self.gamma1_mo_a, self.naocc
            )
        else:
            self.Ua = np.asarray(Ua)
            self._validate_rotation(self.Ua, "Ua")

        if Ub is None:
            self.Ub, _ = self._build_block_no_rotation(
                self.gamma1_mo_b, self.nbocc
            )
        else:
            self.Ub = np.asarray(Ub)
            self._validate_rotation(self.Ub, "Ub")

        self.gamma1_no_a = self.Ua.T.conj() @ self.gamma1_mo_a @ self.Ua
        self.gamma1_no_b = self.Ub.T.conj() @ self.gamma1_mo_b @ self.Ub
        self.gamma1_no_a = 0.5 * (
            self.gamma1_no_a + self.gamma1_no_a.T.conj()
        )
        self.gamma1_no_b = 0.5 * (
            self.gamma1_no_b + self.gamma1_no_b.T.conj()
        )
        self.gamma1_a = self.gamma1_no_a
        self.gamma1_b = self.gamma1_no_b
        self.γa = self.gamma1_no_a
        self.γb = self.gamma1_no_b
        self.occs_a = np.diag(self.gamma1_no_a).real
        self.occs_b = np.diag(self.gamma1_no_b).real

        mix_a = np.hypot(
            np.linalg.norm(self.Ua[: self.naocc, self.naocc :]),
            np.linalg.norm(self.Ua[self.naocc :, : self.naocc]),
        )
        mix_b = np.hypot(
            np.linalg.norm(self.Ub[: self.nbocc, self.nbocc :]),
            np.linalg.norm(self.Ub[self.nbocc :, : self.nbocc]),
        )
        self.common_no_ov_mixing = max(mix_a, mix_b)
        if self.common_no_ov_mixing > self.common_no_mixing_tolerance:
            logger.log_info1(
                "Using the occupied/virtual-block approximation to a "
                "common-NO transformation; discarded ov/vo mixing norm is "
                f"{self.common_no_ov_mixing:.3e}.",
            )

        self.Uoa = self.Ua[: self.naocc, : self.naocc]
        self.Uva = self.Ua[self.naocc :, self.naocc :]
        self.Uob = self.Ub[: self.nbocc, : self.nbocc]
        self.Uvb = self.Ub[self.nbocc :, self.nbocc :]

        # Both spin densities now use the same target NO index convention.
        self.Gamma1_no = self.gamma1_no_a + self.gamma1_no_b
        self.Gamma1_no = 0.5 * (self.Gamma1_no + self.Gamma1_no.T.conj())
        self.Gamma1 = self.Gamma1_no
        self.Γ1 = self.Gamma1_no

        self.M1 = None
        self.M2 = None
        self._M1_indices = None
        self._M2_indices = None

        self._cache_aa = {}
        self._cache_bb = {}
        self._cache_ab = {}
        self._cache_fixed_aa = {}
        self._cache_fixed_bb = {}
        self._cache_fixed_ab_beta = {}

        self._zero_aa = np.zeros((self.navir, self.navir))
        self._zero_bb = np.zeros((self.nbvir, self.nbvir))

    @property
    def occs(self):
        return np.diag(self.Gamma1_no).real

    def _normalize_indices(self, indices):
        if indices is None:
            requested = tuple(range(self.nmo))
        else:
            requested = tuple(dict.fromkeys(int(p) for p in indices))
        if any(p < 0 or p >= self.nmo for p in requested):
            raise IndexError("Every requested orbital index must be in [0, nmo).")
        if not requested:
            raise ValueError("At least one orbital index must be requested.")
        return requested

    def _requested_indices(self, indices):
        if indices is None:
            return self.rdm_info_indices
        return self._normalize_indices(indices)

    def _validate_rotation(self, U, name):
        if U.shape != (self.nmo, self.nmo):
            raise ValueError(
                f"{name} has shape {U.shape}; expected "
                f"({self.nmo}, {self.nmo})."
            )

    def _build_block_no_rotation(self, gamma1, nocc):
        gamma1 = 0.5 * (gamma1 + gamma1.T.conj())
        Goo = gamma1[:nocc, :nocc]
        Gvv = gamma1[nocc:, nocc:]

        occ_vals, Uo = np.linalg.eigh(Goo)
        vir_vals, Uv = np.linalg.eigh(Gvv)

        occ_order = np.argsort(occ_vals)[::-1]
        vir_order = np.argsort(vir_vals)[::-1]

        U = np.eye(self.nmo)
        U[:nocc, :nocc] = Uo[:, occ_order]
        U[nocc:, nocc:] = Uv[:, vir_order]

        occs = np.empty(self.nmo)
        occs[:nocc] = occ_vals[occ_order]
        occs[nocc:] = vir_vals[vir_order]
        return U, occs

    # ------------------------------------------------------------------
    # Spin-space index helpers
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Canonical fixed-occupied amplitude slabs
    # ------------------------------------------------------------------

    def _t2_aa_fixed_j_canonical(self, j):
        if j in self._cache_fixed_aa:
            return self._cache_fixed_aa[j]

        Ba, _ = self.mp2.B_iaQ
        eps_i = self.mp2.eps_a[: self.naocc]
        eps_a = self.mp2.eps_a[self.naocc :]

        g = np.einsum("iaQ,bQ->iab", Ba, Ba[j], optimize=True)
        g_as = g - g.transpose(0, 2, 1)
        denominator = (
            eps_i[:, None, None]
            + self.mp2.eps_a[j]
            - eps_a[None, :, None]
            - eps_a[None, None, :]
        )
        T = self.mp2._safe_divide(g_as, denominator, label="UMP2 aa denom")

        if self.cache_fixed_slabs:
            self._cache_fixed_aa[j] = T
        return T

    def _t2_bb_fixed_j_canonical(self, j):
        if j in self._cache_fixed_bb:
            return self._cache_fixed_bb[j]

        _, Bb = self.mp2.B_iaQ
        eps_i = self.mp2.eps_b[: self.nbocc]
        eps_a = self.mp2.eps_b[self.nbocc :]

        g = np.einsum("iaQ,bQ->iab", Bb, Bb[j], optimize=True)
        g_as = g - g.transpose(0, 2, 1)
        denominator = (
            eps_i[:, None, None]
            + self.mp2.eps_b[j]
            - eps_a[None, :, None]
            - eps_a[None, None, :]
        )
        T = self.mp2._safe_divide(g_as, denominator, label="UMP2 bb denom")

        if self.cache_fixed_slabs:
            self._cache_fixed_bb[j] = T
        return T

    def _t2_ab_fixed_beta_j_canonical(self, j):
        if j in self._cache_fixed_ab_beta:
            return self._cache_fixed_ab_beta[j]

        Ba, Bb = self.mp2.B_iaQ
        eps_ai = self.mp2.eps_a[: self.naocc]
        eps_av = self.mp2.eps_a[self.naocc :]
        eps_bv = self.mp2.eps_b[self.nbocc :]

        g = np.einsum("iaQ,bQ->iab", Ba, Bb[j], optimize=True)
        denominator = (
            eps_ai[:, None, None]
            + self.mp2.eps_b[j]
            - eps_av[None, :, None]
            - eps_bv[None, None, :]
        )
        T = self.mp2._safe_divide(g, denominator, label="UMP2 ab denom")

        if self.cache_fixed_slabs:
            self._cache_fixed_ab_beta[j] = T
        return T

    # ------------------------------------------------------------------
    # Block-rotated occupied-pair amplitudes
    # ------------------------------------------------------------------

    def _t2_aa_pair_no(self, i, j):
        if i == j:
            return self._zero_aa
        if i > j:
            return -self._t2_aa_pair_no(j, i)

        key = (i, j)
        if key in self._cache_aa:
            return self._cache_aa[key]

        T_ab = np.zeros((self.navir, self.navir))
        for J in range(self.naocc):
            fixed_J = self._t2_aa_fixed_j_canonical(J)
            T_i_ab = np.einsum(
                "I,Iab->ab", self.Uoa[:, i], fixed_J, optimize=True
            )
            T_ab += self.Uoa[J, j] * T_i_ab

        T_no = self.Uva.T @ T_ab @ self.Uva
        if self.cache_pair_blocks:
            self._cache_aa[key] = T_no
        return T_no

    def _t2_bb_pair_no(self, i, j):
        if i == j:
            return self._zero_bb
        if i > j:
            return -self._t2_bb_pair_no(j, i)

        key = (i, j)
        if key in self._cache_bb:
            return self._cache_bb[key]

        T_ab = np.zeros((self.nbvir, self.nbvir))
        for J in range(self.nbocc):
            fixed_J = self._t2_bb_fixed_j_canonical(J)
            T_i_ab = np.einsum(
                "I,Iab->ab", self.Uob[:, i], fixed_J, optimize=True
            )
            T_ab += self.Uob[J, j] * T_i_ab

        T_no = self.Uvb.T @ T_ab @ self.Uvb
        if self.cache_pair_blocks:
            self._cache_bb[key] = T_no
        return T_no

    def _t2_ab_pair_no(self, i, j):
        key = (i, j)
        if key in self._cache_ab:
            return self._cache_ab[key]

        T_ab = np.zeros((self.navir, self.nbvir))
        for J in range(self.nbocc):
            fixed_J = self._t2_ab_fixed_beta_j_canonical(J)
            T_i_ab = np.einsum(
                "I,Iab->ab", self.Uoa[:, i], fixed_J, optimize=True
            )
            T_ab += self.Uob[J, j] * T_i_ab

        T_no = self.Uva.T @ T_ab @ self.Uvb
        if self.cache_pair_blocks:
            self._cache_ab[key] = T_no
        return T_no

    def _t2_aa_elem(self, i, j, a, b):
        return self._t2_aa_pair_no(i, j)[a, b]

    def _t2_bb_elem(self, i, j, a, b):
        return self._t2_bb_pair_no(i, j)[a, b]

    def _t2_ab_elem(self, i, j, a, b):
        return self._t2_ab_pair_no(i, j)[a, b]

    # ------------------------------------------------------------------
    # Retained quadratic contractions
    # ------------------------------------------------------------------

    def _gamma_oooo_aa_elem(self, i, j, k, l):
        return 0.5 * np.vdot(
            self._t2_aa_pair_no(i, j), self._t2_aa_pair_no(k, l)
        ).real

    def _gamma_oooo_bb_elem(self, i, j, k, l):
        return 0.5 * np.vdot(
            self._t2_bb_pair_no(i, j), self._t2_bb_pair_no(k, l)
        ).real

    def _gamma_vvvv_aa_elem(self, a, b, c, d):
        value = 0.0
        for i in range(self.naocc):
            for j in range(i + 1, self.naocc):
                Tij = self._t2_aa_pair_no(i, j)
                # The original 1/2 sum over all ordered (i,j) pairs reduces
                # to one sum over i < j by occupied-index antisymmetry.
                value += Tij[a, b] * Tij[c, d]
        return value

    def _gamma_vvvv_bb_elem(self, a, b, c, d):
        value = 0.0
        for i in range(self.nbocc):
            for j in range(i + 1, self.nbocc):
                Tij = self._t2_bb_pair_no(i, j)
                value += Tij[a, b] * Tij[c, d]
        return value

    def _gamma_ovov_ab_elem(self, i, a, j, b):
        """Alpha-occupied/beta-virtual particle-hole orientation."""
        value = 0.0
        for m in range(self.nbocc):
            Tim = self._t2_ab_pair_no(i, m)
            Tjm = self._t2_ab_pair_no(j, m)
            value -= np.dot(Tim[:, b], Tjm[:, a])
        return value

    def _gamma_vovo_ab_elem(self, a, i, b, j):
        """Alpha-virtual/beta-occupied particle-hole orientation."""
        value = 0.0
        for m in range(self.naocc):
            Tmi = self._t2_ab_pair_no(m, i)
            Tmj = self._t2_ab_pair_no(m, j)
            value -= np.dot(Tmi[b, :], Tmj[a, :])
        return value

    # ------------------------------------------------------------------
    # Retained cumulant elements, separated by amplitude degree
    # ------------------------------------------------------------------

    def lambda2_aa_linear_elem(self, p, q, r, s):
        """Return the alpha-alpha cumulant contribution linear in ``t2``."""
        if p == q or r == s:
            return 0.0

        if self._oa(p) and self._oa(q) and self._va(r) and self._va(s):
            return self._t2_aa_elem(
                p, q, self._a_vir(r), self._a_vir(s)
            )

        if self._va(p) and self._va(q) and self._oa(r) and self._oa(s):
            return self._t2_aa_elem(
                r, s, self._a_vir(p), self._a_vir(q)
            )

        return 0.0

    def lambda2_aa_quadratic_elem(self, p, q, r, s):
        """Return the retained alpha-alpha contribution quadratic in ``t2``."""
        if p == q or r == s:
            return 0.0

        if self._oa(p) and self._oa(q) and self._oa(r) and self._oa(s):
            return self._gamma_oooo_aa_elem(p, q, r, s)

        if self._va(p) and self._va(q) and self._va(r) and self._va(s):
            return self._gamma_vvvv_aa_elem(
                self._a_vir(p),
                self._a_vir(q),
                self._a_vir(r),
                self._a_vir(s),
            )

        return 0.0

    def lambda2_aa_elem(self, p, q, r, s):
        value = self.lambda2_aa_linear_elem(p, q, r, s)
        if self.include_quadratic:
            value += self.lambda2_aa_quadratic_elem(p, q, r, s)
        return value

    def lambda2_bb_linear_elem(self, p, q, r, s):
        """Return the beta-beta cumulant contribution linear in ``t2``."""
        if p == q or r == s:
            return 0.0

        if self._ob(p) and self._ob(q) and self._vb(r) and self._vb(s):
            return self._t2_bb_elem(
                p, q, self._b_vir(r), self._b_vir(s)
            )

        if self._vb(p) and self._vb(q) and self._ob(r) and self._ob(s):
            return self._t2_bb_elem(
                r, s, self._b_vir(p), self._b_vir(q)
            )

        return 0.0

    def lambda2_bb_quadratic_elem(self, p, q, r, s):
        """Return the retained beta-beta contribution quadratic in ``t2``."""
        if p == q or r == s:
            return 0.0

        if self._ob(p) and self._ob(q) and self._ob(r) and self._ob(s):
            return self._gamma_oooo_bb_elem(p, q, r, s)

        if self._vb(p) and self._vb(q) and self._vb(r) and self._vb(s):
            return self._gamma_vvvv_bb_elem(
                self._b_vir(p),
                self._b_vir(q),
                self._b_vir(r),
                self._b_vir(s),
            )

        return 0.0

    def lambda2_bb_elem(self, p, q, r, s):
        value = self.lambda2_bb_linear_elem(p, q, r, s)
        if self.include_quadratic:
            value += self.lambda2_bb_quadratic_elem(p, q, r, s)
        return value

    def lambda2_ab_linear_elem(self, p, q, r, s):
        """Return the alpha-beta cumulant contribution linear in ``t2``."""
        if self._oa(p) and self._ob(q) and self._va(r) and self._vb(s):
            return self._t2_ab_elem(
                p, q, self._a_vir(r), self._b_vir(s)
            )

        if self._va(p) and self._vb(q) and self._oa(r) and self._ob(s):
            return self._t2_ab_elem(
                r, s, self._a_vir(p), self._b_vir(q)
            )

        return 0.0

    def lambda2_ab_quadratic_elem(self, p, q, r, s):
        """Return the retained alpha-beta contribution quadratic in ``t2``."""
        if self._oa(p) and self._vb(q) and self._oa(r) and self._vb(s):
            return self._gamma_ovov_ab_elem(
                p, self._b_vir(q), r, self._b_vir(s)
            )

        if self._va(p) and self._ob(q) and self._va(r) and self._ob(s):
            return self._gamma_vovo_ab_elem(
                self._a_vir(p), q, self._a_vir(r), s
            )

        return 0.0

    def lambda2_ab_elem(self, p, q, r, s):
        value = self.lambda2_ab_linear_elem(p, q, r, s)
        if self.include_quadratic:
            value += self.lambda2_ab_quadratic_elem(p, q, r, s)
        return value

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

    # ------------------------------------------------------------------
    # M1/M2 public interface
    # ------------------------------------------------------------------

    def make_M1(self, indices=None):
        """Compute the retained-tensor one-orbital correlation measure.

        If ``indices`` is supplied, only those entries are evaluated; the
        returned array retains its full ``(nmo,)`` shape and is zero elsewhere.
        """
        requested = self._requested_indices(indices)

        if self.M1 is not None and requested == self._M1_indices:
            return self.M1

        M1 = np.zeros(self.nmo)
        for p in requested:
            M1[p] = self.C_elem(p, p, p, p)

        self.M1 = M1
        self._M1_indices = requested
        return self.M1

    def make_measures(self, indices=None):
        """Compute and return ``(M1, M2)`` for a common orbital subset."""
        return self.make_M1(indices=indices), self.make_M2(indices=indices)

    def make_M2(self, indices=None):
        requested = self._requested_indices(indices)

        if self.M2 is not None and requested == self._M2_indices:
            return self.M2

        M2 = np.zeros((self.nmo, self.nmo))
        for position, p in enumerate(requested):
            for q in requested[position + 1 :]:
                value = (
                    4.0 * self.C_elem(p, p, p, q)
                    + 2.0 * self.C_elem(p, p, q, q)
                    + 4.0 * self.C_elem(p, q, p, q)
                    + 4.0 * self.C_elem(p, q, q, q)
                )
                M2[p, q] = M2[q, p] = value

        self.M2 = M2
        self._M2_indices = requested
        return self.M2

    def MPQ_matrix_summary(self, print_threshold=7.5e-4, indices=None):
        requested = self._requested_indices(indices)
        if self.M2 is None or requested != self._M2_indices:
            self.make_M2(indices=requested)

        lines = [
            f"Mutual Correlation Matrix M2 (only values > {print_threshold:.1e}):",
            "=====================",
            "    P     Q      M_PQ",
            "---------------------",
        ]

        values = []
        for position, p in enumerate(requested):
            for q in requested[position + 1 :]:
                values.append((self.M2[p, q], p, q))
        values.sort(reverse=True, key=lambda item: item[0])

        for value, p, q in values:
            if value < print_threshold:
                break
            lines.append(f"{p:>5} {q:>5}  {value:8.6f}")

        lines.append("=====================")
        return "\n".join(lines)

    def clear_amplitude_caches(self):
        """Release cached amplitude slabs and pair blocks."""
        self._cache_aa.clear()
        self._cache_bb.clear()
        self._cache_ab.clear()
        self._cache_fixed_aa.clear()
        self._cache_fixed_bb.clear()
        self._cache_fixed_ab_beta.clear()
