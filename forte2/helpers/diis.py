from collections import deque
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

from forte2.helpers import logger


@dataclass
class DIIS:
    """
    A class that implements the direct inversion in the iterative subspace (DIIS) method.

    Parameters
    ----------
    diis_start : int, optional, default=3
        Start saving DIIS vectors after this many iterations.
        A value less than 1 means no DIIS
    diis_nvec : int, optional, default=8
        The number of vectors to keep in the DIIS.
    diis_min : int, optional, default=3
        The minimum number of vectors to perform extrapolation.
    do_diis : bool, optional
        If True, DIIS is performed. If False, DIIS is not performed.
        If None, DIIS is performed unless `diis_start` or `diis_nvec` is less than 1.

    Attributes
    ----------
    status : str
        Status of the last DIIS update. It could be one of the following:
        - "": No DIIS update was performed.
        - "S": DIIS vectors are saved.
        - "S/E": DIIS vectors are saved and extrapolation is performed.

    Notes
    -----
    The currently implemented DIIS method is Pulay's original method (https://doi.org/10.1016/0009-2614(80)80396-4), which is also known as CDIIS.
    The error vector is the AO-basis (we use the orthonormal basis in actual implementation) orbital gradient: FDS-SDF (https://doi.org/10.1080/00268976900100941).
    """

    diis_start: int = 4
    diis_nvec: int = 8
    diis_min: int = 3
    do_diis: bool = None

    def __post_init__(self):
        if self.diis_min > self.diis_nvec:
            raise ValueError("diis_min must be less than or equal to diis_nvec")

        turn_off = (self.diis_start < 1) or (self.diis_nvec < 1)
        if self.do_diis is True and turn_off:
            logger.log_warning(
                "DIIS is turned off due to diis_start < 1 or diis_nvec < 1."
            )
            self.do_diis = False
        if self.do_diis is None:
            self.do_diis = not turn_off

        self.status = ""
        if self.do_diis:
            # Initialize the parameter and error double-ended queues (deques)
            self.p_diis = deque(maxlen=self.diis_nvec)
            self.e_diis = deque(maxlen=self.diis_nvec)
            self.iter = -1

    def update(self, p: NDArray, e: NDArray) -> NDArray:
        """
        Update the DIIS object and return extrapolated parameters

        Parameters
        ----------
        p : NDArray
            The current set of parameters
        e : NDArray
            The current error vector

        Returns
        -------
        NDArray
            The extrapolated parameters

        """
        self.status = ""

        if not self.do_diis:
            return p

        self.iter += 1

        if self.iter < self.diis_start:
            return p

        self.p_diis.append(p.copy())
        self.e_diis.append(e.copy())

        self.status += "S"
        diis_dim = len(self.p_diis)
        if diis_dim < self.diis_min:
            return p
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
        try:
            x = np.linalg.solve(B, bsol)
        except np.linalg.LinAlgError:
            logger.log_warning("DIIS matrix is singular, skipping DIIS update.")
            return p

        self.status += "/E"
        p_new = np.zeros_like(p)
        for l in range(diis_dim):
            p_new += x[l] * self.p_diis[l]
        return p_new
