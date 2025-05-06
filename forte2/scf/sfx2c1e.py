import forte2
import numpy as np
import scipy, scipy.constants


LIGHT_SPEED = scipy.constants.physical_constants["inverse fine-structure constant"][0]


class X2C1E:
    def __init__(self, system):
        self.system = system
        print("Number of contracted basis functions: ", system.nao())
        self.xbasis = self.system.decontract()
        self.nao = len(self.xbasis)
        print(f"Number of decontracted basis functions: {self.nao}")
        # expensive way to get this for now but works for all types of contraction schemes
        self.contr_coeff = scipy.linalg.pinv(
            forte2.ints.overlap(self.xbasis)
        ) @ forte2.ints.overlap(self.xbasis, self.system.basis)
        self.c0 = LIGHT_SPEED
        self._get_ints()
        self._build_dirac_eq()
        self._eigh()
        self._build_X()
        self._build_R()
        self._build_h_fw()

    def _get_ints(self):
        self.S = forte2.ints.overlap(self.xbasis)
        self.T = forte2.ints.kinetic(self.xbasis)
        self.V = forte2.ints.nuclear(self.xbasis, self.system.atoms)
        self.W = forte2.ints.opVop(self.xbasis, self.system.atoms)[0]

    def _build_dirac_eq(self):
        # DC = MCE
        self.D = np.zeros((self.nao * 2,) * 2)
        self.M = np.zeros((self.nao * 2,) * 2)
        self.D[: self.nao, : self.nao] = self.V
        self.D[self.nao :, self.nao :] = (0.25 / self.c0**2) * self.W - self.T
        self.D[: self.nao, self.nao :] = self.T
        self.D[self.nao :, : self.nao] = self.T
        self.M[: self.nao, : self.nao] = self.S
        self.M[self.nao :, self.nao :] = (0.5 / self.c0**2) * self.T

    def _eigh(self):
        try:
            e, c = scipy.linalg.eigh(self.D, self.M)
            self.e_dirac, self.c_dirac = e, c
        except scipy.linalg.LinAlgError:
            print(
                "Linear dependency detected in the Dirac equation! Using symmetric orthogonalization."
            )
            S12 = forte2.helpers.invsqrt_matrix(self.M, tol=1e-9)
            Morth = S12.conj().T @ self.M @ S12
            e, c = np.linalg.eigh(Morth)
            self.e_dirac, self.c_dirac = e, S12 @ c

    def _build_X(self):
        clpos = self.c_dirac[: self.nao, self.nao :]
        cspos = self.c_dirac[self.nao :, self.nao :]
        # the two ways are equivalent
        # self.X = scipy.linalg.solve(clpos.T.conj(), cspos.T.conj()).T.conj()
        self.X = cspos @ scipy.linalg.pinv(clpos)

    def _build_R(self):
        S_tilde = self.S + (0.5 / self.c0**2) * self.X.conj().T @ self.T @ self.X
        Ssqrt = scipy.linalg.sqrtm(self.S)
        S12 = forte2.helpers.invsqrt_matrix(self.S, tol=1e-9)
        SSS = S12 @ S_tilde @ S12
        SSS12 = forte2.helpers.invsqrt_matrix(SSS, tol=1e-9)
        self.R = S12 @ SSS12 @ Ssqrt

    def _build_h_fw(self):
        L = (
            self.T @ self.X
            + self.X.conj().T @ self.T
            - self.X.conj().T @ self.T @ self.X
            + self.V
            + (0.25 / self.c0**2) * self.X.conj().T @ self.W @ self.X
        )
        self.h_fw = self.R.conj().T @ L @ self.R
        # project back to the contracted basis
        self.h_fw = self.contr_coeff.conj().T @ self.h_fw @ self.contr_coeff

    def _get_hcore(self, system):
        return self.h_fw
