from collections import deque
import copy
import numpy as np
from numpy.typing import NDArray


class DIIS:
    """A class that implements DIIS

    Parameters
    ----------
    diis_start : int
        Start the iterations when the DIIS dimension is greater than this parameter (default = 3)
        A value less than 1 means no DIIS
    diis_nvec : int
        The number of vectors to keep in the DIIS (default = 8)
    """

    def __init__(self, diis_start: int = 4, diis_nvec: int = 8):
        self.do_diis = not ((diis_start < 1) or (diis_nvec < 1))
        if self.do_diis:
            # Initialize the parameter and error deques
            self.p_diis = deque(maxlen=diis_nvec)
            self.e_diis = deque(maxlen=diis_nvec)
            self.diis_start = diis_start
            self.diis_nvec = diis_nvec
            self.iter = -1

    def update(self, p: NDArray, e: NDArray) -> NDArray:
        """Update the DIIS object and return extrapolated parameters

        Parameters
        ----------
        p : ndarray
            The updated parameters
        p_old : list
            The previous set of parameters

        Returns
        -------
        list
            The extrapolated parameters
        """

        if not self.do_diis:
            return p

        self.iter += 1

        if self.iter < self.diis_start:
            return p

        self.p_diis.append(p)
        self.e_diis.append(e)

        diis_dim = len(self.p_diis)
        # construct diis B matrix (following Crawford Group github tutorial)
        B = np.ones((diis_dim + 1, diis_dim + 1), dtype=p.dtype) * -1.0
        B[-1, -1] = 0.0
        bsol = np.zeros(diis_dim + 1, dtype=p.dtype)
        bsol[-1] = -1.0
        for i in range(diis_dim):
            for j in range(i, diis_dim):
                B[i, j] = np.dot(
                    self.e_diis[i].flatten().conj(), self.e_diis[j].flatten()
                )
                if i != j:
                    B[j, i] = B[i, j].conj()

        B[:-1, :-1] /= np.abs(B[:-1, :-1]).max()
        x = np.linalg.solve(B, bsol)
        p_new = np.zeros_like(p)
        for l in range(diis_dim):
            p_new += x[l] * self.p_diis[l]
        return p_new
