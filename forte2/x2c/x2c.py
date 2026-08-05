import numpy as np
import scipy

from forte2 import integrals
from forte2.helpers import (
    logger,
    block_diag_2x2,
    i_sigma_dot,
    canonical_orth,
    invsqrt_matrix,
    print_metric_info,
)
from forte2.system.build_basis import decontract_basis

LIGHT_SPEED = 137.035999177
ROW_Z_START = np.array([1, 3, 11, 19, 37, 55, 87])


def _row_given_Z(Z):
    return np.searchsorted(ROW_Z_START, Z, side="right")


def _inverse_sqrt_deriv(matrix, matrix_deriv):
    """Frechet derivative of the inverse square root of a positive matrix."""
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    if np.min(eigenvalues) <= 0.0:
        raise ValueError("The X2C renormalization metric is not positive definite.")

    function_values = eigenvalues**-0.5
    delta = eigenvalues[:, None] - eigenvalues[None, :]
    scale = max(float(np.max(eigenvalues)), 1.0)
    close = np.abs(delta) < 1.0e-12 * scale
    divided_difference = np.empty_like(delta)
    np.divide(
        function_values[:, None] - function_values[None, :],
        delta,
        out=divided_difference,
        where=~close,
    )
    average = 0.5 * (eigenvalues[:, None] + eigenvalues[None, :])
    divided_difference[close] = -0.5 * average[close] ** -1.5

    transformed_deriv = eigenvectors.conj().T @ matrix_deriv @ eigenvectors
    result = (
        eigenvectors @ (divided_difference * transformed_deriv) @ eigenvectors.conj().T
    )
    return 0.5 * (result + result.conj().T)


class X2CHelper:
    """
    Helper class to compute the X2C one-electron Hamiltonian for a given system.

    Parameters
    ----------
    system : System
        The molecular system for which to compute the X2C Hamiltonian.

    Attributes
    ----------
    X : NDArray
        The decoupling matrix used in the X2C transformation.
    R : NDArray
        The renormalization matrix used in the X2C transformation.
    nbf : int
        The number of basis functions in the decontracted basis.

    Notes
    -----
    Implementation follows the general algorithm of J. Chem. Phys. 135, 084114 (2011),
    but adopts some numerical tricks from J. Chem. Phys. 131, 031104 (2009), especially
    for the spin-orbit case. See also PySCF's x2c module for reference.
    """

    def __init__(self, system):
        self.system = system
        self.overlap_ortho_rtol = system.overlap_ortho_rtol
        assert self.system.x2c_type in [
            "sf",
            "so",
        ], f"Invalid x2c_type: {self.system.x2c_type}. Must be 'sf' or 'so'."
        _snso_type = system.snso_type.lower() if system.snso_type else None
        if _snso_type is not None:
            assert _snso_type in [
                "boettger",
                "dc",
                "dcb",
                "row-dependent",
            ], f"Invalid snso_type: {_snso_type}. Must be 'boettger', 'dc', 'dcb', or 'row-dependent'."

        logger.log_info1(f"Number of contracted basis functions: {self.system.nbf}")

        self.xbasis = decontract_basis(system.basis)

        self.proj = scipy.linalg.solve(
            integrals.overlap(self.system, self.xbasis),
            integrals.overlap(self.system, self.xbasis, self.system.basis),
            assume_a="pos",
        )

        nbf_decon = len(self.xbasis)
        logger.log_info1(f"Number of decontracted basis functions: {nbf_decon}")

        self.S = integrals.overlap(self.system, self.xbasis)
        self.T = integrals.kinetic(self.system, self.xbasis)
        # the V and W integrals know about Gaussian nuclear charges
        self.V = integrals.nuclear(self.system, self.xbasis)
        self.W = integrals.opVop(self.system, self.xbasis)

        # Get orthonormal transformation for X2C
        self.Xorth_l, self.Xorthm1_l, self.orth_info = canonical_orth(
            self.S, self.overlap_ortho_rtol
        )
        print_metric_info(self.orth_info)
        logger.log_info1(
            f"Number of orthogonalized decontracted basis functions: {self.orth_info['n_kept']}"
        )

    @property
    def x2c_type(self):
        """Current X2C mode, including a SpinorUpcaster override."""
        return self.system.x2c_type.lower()

    def hcore_x2c(self):
        """
        Return the one-electron X2C core Hamiltonian matrix for the given system.

        Returns
        -------
        NDArray
            The X2C core Hamiltonian matrix in the contracted basis.
        """
        S, T, V, W = self._get_integrals()

        # build and solve the one-electron matrix Dirac equation
        _, c_dirac = self._solve_dirac_eq(S, T, V, W)

        # build the decoupling matrix X
        self.X = self._get_decoupling_matrix(c_dirac)

        # build the transformation matrix R
        self.R = self._get_transformation_matrix(S, T)

        # build the Foldy-Wouthuysen Hamiltonian
        h_fw = self._build_foldy_wouthuysen_hamiltonian(T, V, W)

        # return to original non-orthogonal AO basis
        _, Xorthm1 = self._get_Xorth()
        h_fw = Xorthm1.conj().T @ h_fw @ Xorthm1

        h_fw = self._apply_snso_to_hcore(h_fw)

        # project back to the contracted basis
        proj = self._get_projection_matrix()
        h_fw = proj.conj().T @ h_fw @ proj

        return h_fw

    def hcore_deriv(self):
        r"""Return analytic nuclear derivatives of the contracted X2C Hamiltonian.

        The derivative includes the response of the decoupling and
        renormalization matrices, the decontracted-to-contracted projection,
        and SNSO scaling. The result is ordered atom-major Cartesian and has
        shape ``(3 * natoms, nbf, nbf)`` for SF-X2C or
        ``(3 * natoms, 2 * nbf, 2 * nbf)`` for SO-X2C.
        """
        # Refresh X, R, and the reference Hamiltonian from exactly the same
        # path used by the SCF calculation.
        self.hcore_x2c()

        S_deriv = integrals.overlap_deriv_matrices(self.system, self.xbasis)
        T_deriv = integrals.kinetic_deriv_matrices(self.system, self.xbasis)
        V_deriv = integrals.nuclear_deriv_matrices(self.system, self.xbasis)
        W_deriv = integrals.opVop_deriv_matrices(self.system, self.xbasis)
        S_cross_deriv = integrals.overlap_deriv_matrices(
            self.system, self.xbasis, self.system.basis
        )

        S_xbasis = self.S
        projection = self.proj
        Xorth_l = self.Xorth_l
        Xorthm1_l = self.Xorthm1_l
        overlap_eigenvalues, overlap_eigenvectors = np.linalg.eigh(S_xbasis)
        ndiscard = self.orth_info["n_discarded"]
        discarded_vectors = overlap_eigenvectors[:, :ndiscard]
        discarded_values = overlap_eigenvalues[:ndiscard]
        kept_vectors = overlap_eigenvectors[:, ndiscard:]
        kept_values = overlap_eigenvalues[ndiscard:]

        S, T, V, W = self._get_integrals()
        eigenvalues, c_dirac = self._solve_dirac_eq(S, T, V, W)
        X = self._get_decoupling_matrix(c_dirac)
        R = self._get_transformation_matrix(S, T)
        L = self._build_nesc_matrix(T, V, W, X)
        h_orth = R.conj().T @ L @ R

        ncoord = 3 * self.system.natoms
        nbf_decontracted = len(self.xbasis)
        nbf_output = self.system.nbf if self.x2c_type == "sf" else 2 * self.system.nbf
        dtype = np.float64 if self.x2c_type == "sf" else np.complex128
        result = np.zeros((ncoord, nbf_output, nbf_output), dtype=dtype)

        for coord in range(ncoord):
            S_orth_deriv = Xorth_l.conj().T @ S_deriv[coord] @ Xorth_l
            # Parallel-transport gauge. It satisfies d(X^+ S X)/dR = 0 and
            # avoids singular eigenvector derivatives inside degenerate overlap
            # eigenspaces.
            Xorth_l_deriv = -0.5 * Xorth_l @ S_orth_deriv
            if ndiscard:
                denominator = kept_values[None, :] - discarded_values[:, None]
                if np.min(np.abs(denominator)) < 1.0e-14:
                    raise RuntimeError(
                        "The retained and discarded X2C overlap spaces are degenerate."
                    )
                coupling = (
                    discarded_vectors.conj().T @ S_deriv[coord] @ kept_vectors
                ) / denominator
                Xorth_l_deriv += (
                    discarded_vectors @ coupling / np.sqrt(kept_values)[None, :]
                )
            Xorthm1_l_deriv = (
                Xorth_l_deriv.conj().T @ S_xbasis + Xorth_l.conj().T @ S_deriv[coord]
            )

            if self.x2c_type == "sf":
                Xorth = Xorth_l
                Xorth_deriv = Xorth_l_deriv
                Xorthm1 = Xorthm1_l
                Xorthm1_deriv = Xorthm1_l_deriv
                T_ao = self.T
                T_ao_deriv = T_deriv[coord]
                V_ao = self.V
                V_ao_deriv = V_deriv[coord]
                W_ao = self.W[0]
                W_ao_deriv = W_deriv[0, coord]
            else:
                Xorth = block_diag_2x2(Xorth_l)
                Xorth_deriv = block_diag_2x2(Xorth_l_deriv)
                Xorthm1 = block_diag_2x2(Xorthm1_l)
                Xorthm1_deriv = block_diag_2x2(Xorthm1_l_deriv)
                T_ao = block_diag_2x2(self.T)
                T_ao_deriv = block_diag_2x2(T_deriv[coord])
                V_ao = block_diag_2x2(self.V)
                V_ao_deriv = block_diag_2x2(V_deriv[coord])
                W_ao = i_sigma_dot(*self.W)
                W_ao_deriv = i_sigma_dot(*W_deriv[:, coord])

            T_prime = self._orthogonal_basis_deriv(Xorth, Xorth_deriv, T_ao, T_ao_deriv)
            V_prime = self._orthogonal_basis_deriv(Xorth, Xorth_deriv, V_ao, V_ao_deriv)
            W_prime = self._orthogonal_basis_deriv(Xorth, Xorth_deriv, W_ao, W_ao_deriv)

            D_prime, M_prime = self._dirac_matrix_deriv(T_prime, V_prime, W_prime)
            X_prime = self._decoupling_matrix_deriv(
                eigenvalues, c_dirac, D_prime, M_prime, X
            )

            renorm_metric = np.eye(X.shape[0], dtype=dtype)
            renorm_metric += (0.5 / LIGHT_SPEED**2) * X.conj().T @ T @ X
            renorm_metric_prime = (0.5 / LIGHT_SPEED**2) * (
                X_prime.conj().T @ T @ X
                + X.conj().T @ T_prime @ X
                + X.conj().T @ T @ X_prime
            )
            R_prime = _inverse_sqrt_deriv(renorm_metric, renorm_metric_prime)

            L_prime = self._build_nesc_matrix_deriv(
                T, T_prime, V_prime, W, W_prime, X, X_prime
            )
            h_orth_prime = (
                R_prime.conj().T @ L @ R
                + R.conj().T @ L_prime @ R
                + R.conj().T @ L @ R_prime
            )
            h_xbasis_prime = (
                Xorthm1_deriv.conj().T @ h_orth @ Xorthm1
                + Xorthm1.conj().T @ h_orth_prime @ Xorthm1
                + Xorthm1.conj().T @ h_orth @ Xorthm1_deriv
            )
            h_xbasis = Xorthm1.conj().T @ h_orth @ Xorthm1
            h_xbasis = self._apply_snso_to_hcore(h_xbasis)
            h_xbasis_prime = self._apply_snso_to_hcore(h_xbasis_prime)

            projection_prime = scipy.linalg.solve(
                S_xbasis,
                S_cross_deriv[coord] - S_deriv[coord] @ projection,
                assume_a="pos",
            )
            if self.x2c_type == "so":
                projection_full = block_diag_2x2(projection)
                projection_prime = block_diag_2x2(projection_prime)
            else:
                projection_full = projection

            contracted_prime = (
                projection_prime.conj().T @ h_xbasis @ projection_full
                + projection_full.conj().T @ h_xbasis_prime @ projection_full
                + projection_full.conj().T @ h_xbasis @ projection_prime
            )
            result[coord] = 0.5 * (contracted_prime + contracted_prime.conj().T)

        return result

    def hcore_gradient(self, density):
        r"""Contract the analytic X2C Hamiltonian derivative with ``density``."""
        from .x2c_grad import compute_hcore_gradient

        return compute_hcore_gradient(self, density)

    @staticmethod
    def _orthogonal_basis_deriv(Xorth, Xorth_deriv, matrix, matrix_deriv):
        return (
            Xorth_deriv.conj().T @ matrix @ Xorth
            + Xorth.conj().T @ matrix_deriv @ Xorth
            + Xorth.conj().T @ matrix @ Xorth_deriv
        )

    def _dirac_matrix_deriv(self, T_deriv, V_deriv, W_deriv):
        north = self._get_northo()
        dtype = np.float64 if self.x2c_type == "sf" else np.complex128
        D_deriv = np.zeros((2 * north, 2 * north), dtype=dtype)
        M_deriv = np.zeros_like(D_deriv)
        D_deriv[:north, :north] = V_deriv
        D_deriv[:north, north:] = T_deriv
        D_deriv[north:, :north] = T_deriv
        D_deriv[north:, north:] = (0.25 / LIGHT_SPEED**2) * W_deriv - T_deriv
        M_deriv[north:, north:] = (0.5 / LIGHT_SPEED**2) * T_deriv
        return D_deriv, M_deriv

    def _decoupling_matrix_deriv(self, eigenvalues, eigenvectors, D_deriv, M_deriv, X):
        north = self._get_northo()
        negative = eigenvectors[:, :north]
        positive = eigenvectors[:, north:]
        eps_negative = eigenvalues[:north]
        eps_positive = eigenvalues[north:]

        residual_deriv = (
            D_deriv @ positive - (M_deriv @ positive) * eps_positive[None, :]
        )
        coupling = negative.conj().T @ residual_deriv
        denominator = eps_positive[None, :] - eps_negative[:, None]
        if np.min(np.abs(denominator)) < 1.0e-10:
            raise RuntimeError("The X2C electronic-positronic energy gap is singular.")
        positive_deriv = negative @ (coupling / denominator)

        large = positive[:north]
        large_deriv = positive_deriv[:north]
        small_deriv = positive_deriv[north:]
        return (small_deriv - X @ large_deriv) @ scipy.linalg.pinv(large)

    @staticmethod
    def _build_nesc_matrix(T, V, W, X):
        return (
            T @ X
            + X.conj().T @ T
            - X.conj().T @ T @ X
            + V
            + (0.25 / LIGHT_SPEED**2) * X.conj().T @ W @ X
        )

    @staticmethod
    def _build_nesc_matrix_deriv(T, T_deriv, V_deriv, W, W_deriv, X, X_deriv):
        return (
            T_deriv @ X
            + T @ X_deriv
            + X_deriv.conj().T @ T
            + X.conj().T @ T_deriv
            - X_deriv.conj().T @ T @ X
            - X.conj().T @ T_deriv @ X
            - X.conj().T @ T @ X_deriv
            + V_deriv
            + (0.25 / LIGHT_SPEED**2)
            * (
                X_deriv.conj().T @ W @ X
                + X.conj().T @ W_deriv @ X
                + X.conj().T @ W @ X_deriv
            )
        )

    def _get_projection_matrix(self):
        return self.proj if self.system.x2c_type == "sf" else block_diag_2x2(self.proj)

    def _get_Xorth(self):
        if self.system.x2c_type == "sf":
            return self.Xorth_l, self.Xorthm1_l
        elif self.system.x2c_type == "so":
            return block_diag_2x2(self.Xorth_l), block_diag_2x2(self.Xorthm1_l)

    def _get_northo(self):
        if self.system.x2c_type == "sf":
            return self.orth_info["n_kept"]
        elif self.system.x2c_type == "so":
            return self.orth_info["n_kept"] * 2

    def _get_integrals(self):
        Xorth, _ = self._get_Xorth()
        if self.system.x2c_type == "sf":
            S = np.eye(Xorth.shape[1])
            T = Xorth.conj().T @ self.T @ Xorth
            V = Xorth.conj().T @ self.V @ Xorth
            W = Xorth.conj().T @ self.W[0] @ Xorth
        elif self.system.x2c_type == "so":
            S = np.eye(Xorth.shape[1], dtype=complex)
            T = Xorth.conj().T @ block_diag_2x2(self.T) @ Xorth
            V = Xorth.conj().T @ block_diag_2x2(self.V) @ Xorth
            W = Xorth.conj().T @ i_sigma_dot(*self.W) @ Xorth

        return S, T, V, W

    def _solve_dirac_eq(self, S, T, V, W):
        dtype = np.float64 if self.system.x2c_type == "sf" else np.complex128
        north = self._get_northo()
        D = np.zeros((north * 2,) * 2, dtype=dtype)
        M = np.zeros((north * 2,) * 2, dtype=dtype)
        D[:north, :north] = V
        D[north:, north:] = (0.25 / LIGHT_SPEED**2) * W - T
        D[:north, north:] = T
        D[north:, :north] = T
        M[:north, :north] = S
        M[north:, north:] = (0.5 / LIGHT_SPEED**2) * T
        return scipy.linalg.eigh(D, M)

    def _get_decoupling_matrix(self, c_dirac):
        north = self._get_northo()
        clpos = c_dirac[:north, north:]
        cspos = c_dirac[north:, north:]
        return cspos @ scipy.linalg.pinv(clpos)

    def _get_transformation_matrix(self, S, T):
        """
        This implementation follows eqs 26-34 of J. Chem. Phys. 131, 031104 (2009),
        which avoids doing matrix inversions and leads to a more numerically stable transformation.
        """
        S_tilde = S + (0.5 / LIGHT_SPEED**2) * self.X.conj().T @ T @ self.X
        # S is guaranteed to be identity in the orthonormal basis
        # so we just need to compute the inverse square root of S_tilde
        # the tolerance used here isn't self.overlap_ortho_rtol because we're already in the
        # orthonormal basis, it's just an additional numerical guard against division by zero.
        S_tilde_m12, *_ = invsqrt_matrix(S_tilde, rtol=1e-12)
        return S_tilde_m12 @ S
        # This was the old way (Cheng and Gauss), worked fine for sfx2c1e, but seems unusable for sox2c1e
        # S_tilde = S + (0.5 / c0**2) * X.conj().T @ T @ X
        # Ssqrt = scipy.linalg.sqrtm(S)
        # S12 = forte2.helpers.invsqrt_matrix(S, tol=tol)
        # SSS = S12 @ S_tilde @ S12
        # SSS12 = forte2.helpers.invsqrt_matrix(SSS, tol=tol)
        # return S12 @ SSS12 @ Ssqrt

    def _build_foldy_wouthuysen_hamiltonian(self, T, V, W):
        L = self._build_nesc_matrix(T, V, W, self.X)
        return self.R.conj().T @ L @ self.R

    def _apply_snso_to_hcore(self, hcore):
        if self.x2c_type != "so" or self.system.snso_type is None:
            return hcore

        nbf = len(self.xbasis)
        haa = hcore[:nbf, :nbf]
        hab = hcore[:nbf, nbf:]
        hba = hcore[nbf:, :nbf]
        hbb = hcore[nbf:, nbf:]
        h0 = (haa + hbb) / 2
        h1 = self._apply_snso_scaling((hab + hba) / 2)
        h2 = self._apply_snso_scaling((hab - hba) / (-2j))
        h3 = self._apply_snso_scaling((haa - hbb) / 2)
        return np.block([[h0 + h3, h1 - 1j * h2], [h1 + 1j * h2, h0 - h3]])

    def _apply_snso_scaling(self, ints):
        """
        Apply the 'screened-nuclear-spin-orbit' (SNSO) scaling to the core Hamiltonian.
        Original paper ('Boettger'): Phys. Rev. B 62, 7809 (2000)
        Re-parameterized schemes ('DC'/'DCB'/'Row-dependent'): J. Chem. Theory Comput. 19, 5785 (2023)
        """
        # applied in the decontracted basis before recontraction (if requested)
        basis = self.xbasis
        atoms = self.system.atoms

        if self.system.snso_type is None:
            return ints
        if basis.max_l > 7:
            raise RuntimeError(
                "SNSO scaling is not implemented for basis sets with l > 7."
            )
        match self.system.snso_type.lower():
            case "boettger":
                Ql = np.array([0.0, 2.0, 10.0, 28.0, 60.0, 110.0, 182.0, 280.0])
            case "dc":
                Ql = np.array([0.0, 2.32, 10.64, 28.38, 60.0, 110.0, 182.0, 280.0])
            case "dcb":
                Ql = np.array([0.0, 2.97, 11.93, 29.84, 64.0, 115.0, 188.0, 287.0])
            case "row-dependent":
                Ql = {
                    1: np.array([0.0, 2.97, 11.93, 29.84, 64.0, 115.0, 188.0, 287.0]),
                    2: np.array([0.0, 2.80, 11.93, 29.84, 64.0, 115.0, 188.0, 287.0]),
                    3: np.array([0.0, 2.95, 11.93, 29.84, 64.0, 115.0, 188.0, 287.0]),
                    4: np.array([0.0, 3.09, 11.49, 29.84, 64.0, 115.0, 188.0, 287.0]),
                    5: np.array([0.0, 3.02, 11.91, 29.84, 64.0, 115.0, 188.0, 287.0]),
                    6: np.array([0.0, 2.85, 12.31, 30.61, 64.0, 115.0, 188.0, 287.0]),
                    7: np.array([0.0, 2.85, 12.31, 30.61, 64.0, 115.0, 188.0, 287.0]),
                }
            case _:
                raise ValueError(
                    f"Invalid SNSO type: {self.system.snso_type}. Must be 'boettger', 'dc', 'dcb', or 'row-dependent'."
                )

        center_first = np.array([_[0] for _ in basis.center_first_and_last])
        center_given_shell = (
            lambda ishell: np.searchsorted(center_first, ishell, side="right") - 1
        )

        iptr = jptr = 0
        for ishell in range(basis.nshells):
            isize = basis[ishell].size
            li = int(basis[ishell].l)
            if li == 0:
                iptr += isize
                jptr = 0
                continue
            Zi = atoms[center_given_shell(ishell)][0]
            if isinstance(Ql, dict):
                Ql_i = Ql[_row_given_Z(Zi)][li]
            else:
                Ql_i = Ql[li]
            for jshell in range(basis.nshells):
                jsize = basis[jshell].size
                lj = int(basis[jshell].l)
                if lj == 0:
                    jptr += jsize
                    continue
                Zj = atoms[center_given_shell(jshell)][0]
                if isinstance(Ql, dict):
                    Ql_j = Ql[_row_given_Z(Zj)][lj]
                else:
                    Ql_j = Ql[lj]
                snso_factor = 1 - np.sqrt(Ql_i * Ql_j / (Zi * Zj))
                ints[iptr : iptr + isize, jptr : jptr + jsize] *= snso_factor
                jptr += jsize
            iptr += isize
            jptr = 0

        return ints
