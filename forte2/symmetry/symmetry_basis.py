from dataclasses import dataclass

import numpy as np

from .mo_sym_detect import MOSymmetryDetector

# The geometry is exactly symmetric, so the AO representations are exact to roundoff.
_AO_SYMMETRY_TOL = 1e-8


@dataclass
class SymmetryBasis:
    """An orthonormal AO basis partitioned into Abelian irreps."""

    C: np.ndarray
    irreps: np.ndarray
    U_ops: dict
    point_group: str
    _system: object
    _info: object
    _S: np.ndarray

    @classmethod
    def build(cls, system, info, S, X):
        C = X.copy()
        detector = MOSymmetryDetector(
            system, info, S, C, np.zeros(C.shape[1]), tol=_AO_SYMMETRY_TOL
        )
        detector.run()
        return cls(
            C,
            np.array(detector.irrep_indices),
            detector.U_ops or {},
            system.point_group,
            system,
            info,
            S,
        )

    def adapt(self, C, eps):
        """Resolve a supplied guess with the cached AO symmetry operations."""
        C, eps = C.copy(), eps.copy()
        detector = MOSymmetryDetector(
            self._system,
            self._info,
            self._S,
            C,
            eps,
            U_ops=self.U_ops,
        )
        detector.run()
        return eps, C, np.array(detector.irrep_indices)

    def eigh(self, F):
        """Diagonalize each irrep and return energies, coefficients, and irreps."""
        eps = np.empty(self.C.shape[1])
        C = np.empty(self.C.shape, dtype=np.result_type(self.C, F))
        for irrep in np.unique(self.irreps):
            indices = np.flatnonzero(self.irreps == irrep)
            X = self.C[:, indices]
            eps[indices], c = np.linalg.eigh(X.T.conj() @ F @ X)
            C[:, indices] = X @ c
        order = np.argsort(eps, kind="stable")
        return eps[order], C[:, order], self.irreps[order]
