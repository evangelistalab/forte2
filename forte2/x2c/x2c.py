from functools import cached_property

import numpy as np
import scipy

from forte2 import integrals
from forte2.helpers import logger, block_diag_2x2, i_sigma_dot, canonical_orth
from forte2.system.build_basis import build_basis

X2C_LINDEP_TOL = 5e-8
LIGHT_SPEED = 137.035999177
ROW_Z_START = np.array([1, 3, 11, 19, 37, 55, 87])


def _row_given_Z(Z):
    return np.searchsorted(ROW_Z_START, Z, side="right")


class X2CHelper:
    """
    Helper class to compute the X2C one-electron Hamiltonian for a given system.

    Parameters
    ----------
    system : System
        The molecular system for which to compute the X2C Hamiltonian.

    Attributes
    ----------
    hcore_x2c : NDArray
        The one-electron X2C core Hamiltonian matrix in the (re-)contracted basis.
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
        self.x2c_type = system.x2c_type.lower()
        assert self.x2c_type in [
            "sf",
            "so",
        ], f"Invalid x2c_type: {self.x2c_type}. Must be 'sf' or 'so'."
        self.snso_type = system.snso_type.lower() if system.snso_type else None
        if self.snso_type is not None:
            assert self.snso_type in [
                "boettger",
                "dc",
                "dcb",
                "row-dependent",
            ], f"Invalid snso_type: {self.snso_type}. Must be 'boettger', 'dc', 'dcb', or 'row-dependent'."

        logger.log_info1(f"Number of contracted basis functions: {self.system.nbf}")

        self.xbasis = build_basis(
            system.basis_set,
            system.geom_helper,
            decontract=True,
        )
        self.proj = self._get_projection_matrix()

        nbf_decon = len(self.xbasis)
        logger.log_info1(f"Number of decontracted basis functions: {nbf_decon}")
        self.nbf = nbf_decon if self.system.x2c_type == "sf" else nbf_decon * 2
        self.executed = False

    @cached_property
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
        self.R = self._get_transformation_matrix(S, T, tol=X2C_LINDEP_TOL)

        # build the Foldy-Wouthuysen Hamiltonian
        h_fw = self._build_foldy_wouthuysen_hamiltonian(T, V, W)

        # return to original non-orthogonal AO basis
        h_fw = self.Xorthm1_l.conj().T @ h_fw @ self.Xorthm1_l

        if self.x2c_type.lower() == "so" and self.snso_type is not None:
            nbf = self.nbf // 2
            haa = h_fw[:nbf, :nbf]
            hab = h_fw[:nbf, nbf:]
            hba = h_fw[nbf:, :nbf]
            hbb = h_fw[nbf:, nbf:]
            # the pauli representation of a spin-dependent operator.
            # h0 is spin-free, h1-3 are spin-dependent
            # SNSO is applied to the spin-dependent parts only.
            # see for example eq 4-6 of 10.1002/wcms.1436
            h0 = (haa + hbb) / 2
            h1 = (hab + hba) / 2
            h2 = (hab - hba) / (-2j)
            h3 = (haa - hbb) / 2
            h1 = self._apply_snso_scaling(h1)
            h2 = self._apply_snso_scaling(h2)
            h3 = self._apply_snso_scaling(h3)
            h_fw = np.block([[h0 + h3, h1 - 1j * h2], [h1 + 1j * h2, h0 - h3]])

        # project back to the contracted basis
        h_fw = self.proj.conj().T @ h_fw @ self.proj

        self.executed = True
        return h_fw

    def _get_projection_matrix(self):
        proj = scipy.linalg.solve(
            integrals.overlap(self.system, self.xbasis),
            integrals.overlap(self.system, self.xbasis, self.system.basis),
            assume_a="pos",
        )
        return proj if self.system.x2c_type == "sf" else block_diag_2x2(proj)

    def _get_integrals(self):
        S = integrals.overlap(self.system, self.xbasis)
        T = integrals.kinetic(self.system, self.xbasis)
        # the V and W integrals know about Gaussian nuclear charges
        V = integrals.nuclear(self.system, self.xbasis)
        W = integrals.opVop(self.system, self.xbasis)

        # Get orthonormal transformation for X2C
        Xorth_l, Xorthm1_l = canonical_orth(S, tol=X2C_LINDEP_TOL, return_inverse=True)
        northo = Xorth_l.shape[1]
        logger.log_info1(
            f"Number of orthogonalized decontracted basis functions: {northo}"
        )

        if self.system.x2c_type == "sf":
            S = np.eye(Xorth_l.shape[1])
            T = Xorth_l.conj().T @ T @ Xorth_l
            V = Xorth_l.conj().T @ V @ Xorth_l
            W = Xorth_l.conj().T @ W[0] @ Xorth_l
        elif self.system.x2c_type == "so":
            Xorth_l = block_diag_2x2(Xorth_l)
            Xorthm1_l = block_diag_2x2(Xorthm1_l)
            S = np.eye(Xorth_l.shape[1], dtype=complex)
            T = Xorth_l.conj().T @ block_diag_2x2(T) @ Xorth_l
            V = Xorth_l.conj().T @ block_diag_2x2(V) @ Xorth_l
            W = Xorth_l.conj().T @ i_sigma_dot(*W) @ Xorth_l
            northo *= 2

        self.Xorth_l = Xorth_l
        self.Xorthm1_l = Xorthm1_l
        self.northo = northo
        return S, T, V, W

    def _solve_dirac_eq(self, S, T, V, W):
        dtype = np.float64 if self.x2c_type == "sf" else np.complex128
        north = self.northo
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
        north = self.northo
        clpos = c_dirac[:north, north:]
        cspos = c_dirac[north:, north:]
        return cspos @ scipy.linalg.pinv(clpos)

    def _get_transformation_matrix(self, S, T, tol=1e-9):
        """
        This implementation follows eqs 26-34 of J. Chem. Phys. 131, 031104 (2009),
        which avoids doing matrix inversions and leads to a more numerically stable transformation.
        """
        S_tilde = S + (0.5 / LIGHT_SPEED**2) * self.X.conj().T @ T @ self.X
        # S is guaranteed to be identity in the orthonormal basis
        lam, z = np.linalg.eigh(S_tilde)
        idx = lam > 1e-14
        R = (z[:, idx] / np.sqrt(lam[idx])) @ z[:, idx].T.conj() @ S
        return R
        # This was the old way (Cheng and Gauss), worked fine for sfx2c1e, but seems unusable for sox2c1e
        # S_tilde = S + (0.5 / c0**2) * X.conj().T @ T @ X
        # Ssqrt = scipy.linalg.sqrtm(S)
        # S12 = forte2.helpers.invsqrt_matrix(S, tol=tol)
        # SSS = S12 @ S_tilde @ S12
        # SSS12 = forte2.helpers.invsqrt_matrix(SSS, tol=tol)
        # return S12 @ SSS12 @ Ssqrt

    def _build_foldy_wouthuysen_hamiltonian(self, T, V, W):
        L = (
            T @ self.X
            + self.X.conj().T @ T
            - self.X.conj().T @ T @ self.X
            + V
            + (0.25 / LIGHT_SPEED**2) * self.X.conj().T @ W @ self.X
        )
        return self.R.conj().T @ L @ self.R

    def _apply_snso_scaling(self, ints):
        """
        Apply the 'screened-nuclear-spin-orbit' (SNSO) scaling to the core Hamiltonian.
        Original paper ('Boettger'): Phys. Rev. B 62, 7809 (2000)
        Re-parameterized schemes ('DC'/'DCB'/'Row-dependent'): J. Chem. Theory Comput. 19, 5785 (2023)
        """
        # applied in the decontracted basis before recontraction (if requested)
        basis = self.xbasis
        atoms = self.system.atoms

        if self.snso_type is None:
            return ints
        if basis.max_l > 7:
            raise RuntimeError(
                "SNSO scaling is not implemented for basis sets with l > 7."
            )
        match self.snso_type.lower():
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
                    f"Invalid SNSO type: {self.snso_type}. Must be 'boettger', 'dc', 'dcb', or 'row-dependent'."
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
