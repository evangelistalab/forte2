import numpy as np
import scipy

from forte2 import integrals
from forte2.lib import ints
from forte2.helpers import block_diag_2x2, i_sigma_dot

from .x2c import LIGHT_SPEED


def _hermitize(matrix):
    return 0.5 * (matrix + matrix.conj().T)


def _inverse_sqrt_kernel(matrix):
    """Return the inverse square root and its Frechet divided differences."""
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
    inverse_sqrt = eigenvectors @ np.diag(function_values) @ eigenvectors.conj().T
    return inverse_sqrt, eigenvectors, divided_difference


def _inverse_sqrt_frechet(eigenvectors, divided_difference, perturbation):
    """Apply the self-adjoint Frechet derivative of the inverse square root."""
    transformed = eigenvectors.conj().T @ perturbation @ eigenvectors
    result = eigenvectors @ (divided_difference * transformed) @ eigenvectors.conj().T
    return _hermitize(result)


class _X2CAdjoint:
    """Reverse-mode X2C response for one density contraction."""

    def __init__(self, helper):
        self.helper = helper
        self.mode = helper.x2c_type
        self.Xorth, self.Xorthm1 = helper._get_Xorth()
        S, self.T, self.V, self.W = helper._get_integrals()

        self.eigenvalues, eigenvectors = helper._solve_dirac_eq(
            S, self.T, self.V, self.W
        )
        north = self.T.shape[0]
        self.negative = eigenvectors[:, :north]
        self.positive = eigenvectors[:, north:]
        self.gap = self.eigenvalues[north:][None, :] - self.eigenvalues[:north][:, None]
        if np.min(np.abs(self.gap)) < 1.0e-10:
            raise RuntimeError("The X2C electronic-positronic energy gap is singular.")

        self.large_pinv = scipy.linalg.pinv(self.positive[:north])
        self.X = self.positive[north:] @ self.large_pinv
        self.renorm_metric = (
            S + (0.5 / LIGHT_SPEED**2) * self.X.conj().T @ self.T @ self.X
        )
        (
            self.R,
            self.renorm_eigenvectors,
            self.renorm_divided_difference,
        ) = _inverse_sqrt_kernel(self.renorm_metric)
        self.L = helper._build_nesc_matrix(self.T, self.V, self.W, self.X)
        self.h_orth = self.R.conj().T @ self.L @ self.R
        self.h_xbasis = self.Xorthm1.conj().T @ self.h_orth @ self.Xorthm1
        self.projection = helper._get_projection_matrix()

        if self.mode == "sf":
            self.T_ao = helper.T
            self.V_ao = helper.V
            self.W_ao = helper.W[0]
        else:
            self.T_ao = block_diag_2x2(helper.T)
            self.V_ao = block_diag_2x2(helper.V)
            self.W_ao = i_sigma_dot(*helper.W)

        self.overlap_eigenvalues, self.overlap_eigenvectors = np.linalg.eigh(helper.S)

    def primitive_weights(self, density):
        """Return adjoints of S, T, V, W, and the cross overlap."""
        helper = self.helper
        density = np.asarray(density)
        expected = self.projection.shape[1]
        if density.shape != (expected, expected):
            raise ValueError(
                f"Expected X2C density shape {(expected, expected)}, "
                f"got {density.shape}."
            )
        density = _hermitize(density)

        h_xbasis = helper._apply_snso_to_hcore(self.h_xbasis)
        h_xbasis_bar = self.projection @ density @ self.projection.conj().T
        projection_bar = 2.0 * h_xbasis @ self.projection @ density
        h_xbasis_bar = helper._apply_snso_to_hcore(h_xbasis_bar)

        if self.mode == "so":
            nx = len(helper.xbasis)
            nb = helper.system.nbf
            projection_bar_l = projection_bar[:nx, :nb] + projection_bar[nx:, nb:]
        else:
            projection_bar_l = projection_bar

        cross_bar = scipy.linalg.solve(helper.S, projection_bar_l, assume_a="pos")
        S_bar = -cross_bar @ helper.proj.conj().T

        h_orth_bar = self.Xorthm1 @ h_xbasis_bar @ self.Xorthm1.conj().T
        Xorthm1_bar = 2.0 * self.h_orth @ self.Xorthm1 @ h_xbasis_bar
        if self.mode == "so":
            nx = len(helper.xbasis)
            nk = helper.Xorth_l.shape[1]
            Xorthm1_bar_l = Xorthm1_bar[:nk, :nx] + Xorthm1_bar[nk:, nx:]
        else:
            Xorthm1_bar_l = Xorthm1_bar

        Xorth_l_bar = helper.S @ Xorthm1_bar_l.conj().T
        S_bar += helper.Xorth_l @ Xorthm1_bar_l

        L_bar = self.R @ h_orth_bar @ self.R.conj().T
        R_bar = _hermitize(2.0 * self.L @ self.R @ h_orth_bar)
        renorm_bar = _inverse_sqrt_frechet(
            self.renorm_eigenvectors,
            self.renorm_divided_difference,
            R_bar,
        )

        metric_factor = 0.5 / LIGHT_SPEED**2
        w_factor = 0.25 / LIGHT_SPEED**2
        T_bar = metric_factor * self.X @ renorm_bar @ self.X.conj().T
        X_bar = 2.0 * metric_factor * self.T @ self.X @ renorm_bar

        T_bar += (
            L_bar @ self.X.conj().T + self.X @ L_bar - self.X @ L_bar @ self.X.conj().T
        )
        V_bar = L_bar.copy()
        W_bar = w_factor * self.X @ L_bar @ self.X.conj().T
        X_bar += (
            2.0 * self.T @ L_bar
            - 2.0 * self.T @ self.X @ L_bar
            + 2.0 * w_factor * self.W @ self.X @ L_bar
        )

        north = self.X.shape[0]
        electronic_response_bar = (
            np.hstack((-self.X, np.eye(north, dtype=self.X.dtype))).conj().T
            @ X_bar
            @ self.large_pinv.conj().T
        )
        coupling_bar = (self.negative.conj().T @ electronic_response_bar) / self.gap
        residual_bar = self.negative @ coupling_bar
        eps_positive = self.eigenvalues[north:]
        D_bar = _hermitize(residual_bar @ self.positive.conj().T)
        M_bar = _hermitize(
            -residual_bar @ (self.positive * eps_positive[None, :]).conj().T
        )

        V_bar += D_bar[:north, :north]
        T_bar += (
            D_bar[:north, north:]
            + D_bar[north:, :north]
            - D_bar[north:, north:]
            + metric_factor * M_bar[north:, north:]
        )
        W_bar += w_factor * D_bar[north:, north:]
        T_bar = _hermitize(T_bar)
        V_bar = _hermitize(V_bar)
        W_bar = _hermitize(W_bar)

        T_ao_bar = self.Xorth @ T_bar @ self.Xorth.conj().T
        V_ao_bar = self.Xorth @ V_bar @ self.Xorth.conj().T
        W_ao_bar = self.Xorth @ W_bar @ self.Xorth.conj().T
        Xorth_bar = (
            2.0 * self.T_ao @ self.Xorth @ T_bar
            + 2.0 * self.V_ao @ self.Xorth @ V_bar
            + 2.0 * self.W_ao @ self.Xorth @ W_bar
        )

        nx = len(helper.xbasis)
        if self.mode == "so":
            nk = helper.Xorth_l.shape[1]
            Xorth_l_bar += Xorth_bar[:nx, :nk] + Xorth_bar[nx:, nk:]
            T_ao_bar = T_ao_bar[:nx, :nx] + T_ao_bar[nx:, nx:]
            V_ao_bar = V_ao_bar[:nx, :nx] + V_ao_bar[nx:, nx:]
            W_components_bar = self._i_sigma_dot_adjoint(W_ao_bar)
        else:
            Xorth_l_bar += Xorth_bar
            W_components_bar = np.zeros((4, nx, nx))
            W_components_bar[0] = W_ao_bar.real

        S_bar += self._canonical_orth_adjoint(Xorth_l_bar)
        return {
            "S": _hermitize(S_bar).real,
            "T": _hermitize(T_ao_bar).real,
            "V": _hermitize(V_ao_bar).real,
            "W": W_components_bar,
            "S_cross": cross_bar.real,
        }

    def _canonical_orth_adjoint(self, Xorth_bar):
        helper = self.helper
        Xorth = helper.Xorth_l
        result = -0.5 * (Xorth @ Xorth.conj().T) @ Xorth_bar @ Xorth.conj().T

        ndiscard = helper.orth_info["n_discarded"]
        if ndiscard:
            discarded = self.overlap_eigenvectors[:, :ndiscard]
            kept = self.overlap_eigenvectors[:, ndiscard:]
            discarded_values = self.overlap_eigenvalues[:ndiscard]
            kept_values = self.overlap_eigenvalues[ndiscard:]
            gap = kept_values[None, :] - discarded_values[:, None]
            if np.min(np.abs(gap)) < 1.0e-14:
                raise RuntimeError(
                    "The retained and discarded X2C overlap spaces are degenerate."
                )
            coupling_bar = discarded.conj().T @ Xorth_bar
            coupling_bar /= gap * np.sqrt(kept_values)[None, :]
            result += discarded @ coupling_bar @ kept.conj().T

        return _hermitize(result)

    @staticmethod
    def _i_sigma_dot_adjoint(matrix):
        nbf = matrix.shape[0] // 2
        aa = matrix[:nbf, :nbf]
        ab = matrix[:nbf, nbf:]
        ba = matrix[nbf:, :nbf]
        bb = matrix[nbf:, nbf:]
        return np.asarray(
            [
                (aa + bb).real,
                (ab + ba).imag,
                (ab - ba).real,
                (aa - bb).imag,
            ]
        )


def compute_hcore_gradient(helper, density):
    """Contract the analytic X2C Hamiltonian derivative with ``density``."""
    weights = _X2CAdjoint(helper).primitive_weights(density)
    atoms = helper.system.atoms
    gradient = np.zeros(3 * helper.system.natoms)
    gradient += np.asarray(
        ints.overlap_deriv(
            helper.xbasis,
            helper.xbasis,
            np.ascontiguousarray(weights["S"]),
            atoms,
        )
    )
    gradient += np.asarray(
        ints.kinetic_deriv(
            helper.xbasis,
            helper.xbasis,
            np.ascontiguousarray(weights["T"]),
            atoms,
        )
    )
    gradient += np.asarray(
        integrals.nuclear_deriv(
            helper.system,
            np.ascontiguousarray(weights["V"]),
            helper.xbasis,
        )
    )
    gradient += np.asarray(
        integrals.opVop_deriv(
            helper.system,
            np.ascontiguousarray(weights["W"]),
            helper.xbasis,
        )
    )
    gradient += np.asarray(
        ints.overlap_deriv(
            helper.xbasis,
            helper.system.basis,
            np.ascontiguousarray(weights["S_cross"]),
            atoms,
        )
    )
    return gradient.reshape(helper.system.natoms, 3)
