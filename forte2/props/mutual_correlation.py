import numpy as np
from numpy.typing import NDArray, Any
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

    def _compute_mutual_correlation_measures(self, λaa, λab, λbb):
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
        """Computes the spin-free correlation C_PQRS from the spin-dependent cumulant 2-RDMs."""
        C_PQRS = 0.25 * (λaa**2).copy()
        C_PQRS += 0.25 * (λab**2)
        C_PQRS += 0.25 * np.einsum("ijlk->ijkl", λab**2)
        C_PQRS += 0.25 * np.einsum("jikl->ijkl", λab**2)
        C_PQRS += 0.25 * np.einsum("jilk->ijkl", λab**2)
        C_PQRS += 0.25 * (λbb**2)
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
                M2_vals.append((self.M2[i, j], i, j))
        M2_vals.sort(reverse=True, key=lambda x: x[0])

        for val, i, j in M2_vals:
            if val < threshold:
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

            # compute the cost function
            def cost(x):
                return np.sum(x**k)

            return -cost(M2_trans)

        # minimize the objective function as a function of the antisymmetric matrix A
        res = sp.optimize.minimize(objective, a, method=method)

        A = res["x"].reshape(N, N)
        A = A - A.T
        Q = sp.linalg.expm(A)
        self.Q = Q

        # Apply the orthogonal transformation to the RDMs
        self.λaa = np.einsum("pqrs,pi,qj,rk,sl->ijkl", self.λaa, Q, Q, Q, Q)
        self.λab = np.einsum("pqrs,pi,qj,rk,sl->ijkl", self.λab, Q, Q, Q, Q)
        self.λbb = np.einsum("pqrs,pi,qj,rk,sl->ijkl", self.λbb, Q, Q, Q, Q)

        # Recompute the mutual correlation measures
        self._compute_mutual_correlation_measures(self.λaa, self.λab, self.λbb)

        return self.Q
