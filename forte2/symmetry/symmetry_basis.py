from dataclasses import dataclass

import numpy as np

from .sym_utils import CHARACTER_TABLE, COTTON_LABELS, SYMMETRY_OPS, local_sign

# The geometry is exactly symmetric, so the AO representations are exact to roundoff.
_AO_SYMMETRY_TOL = 1e-8


def irrep_coupling(M, irreps):
    """
    Measure how strongly a matrix couples functions of different irreps.

    Parameters
    ----------
    M : NDArray
        A matrix in a basis of functions that each transform as one irrep.
    irreps : ArrayLike
        The irrep index of each basis function.

    Returns
    -------
    float
        The largest element of ``M`` between functions of different irreps, relative to
        the largest element of ``M``.
    """
    irreps = np.asarray(irreps)
    scale = np.abs(M).max(initial=0.0)
    if scale == 0.0:
        return 0.0
    off_block = irreps[:, None] != irreps[None, :]
    return np.abs(M[off_block]).max(initial=0.0) / scale


def couples_irreps(M, irreps):
    """Return whether ``M`` couples different irreps by more than roundoff."""
    return irrep_coupling(M, irreps) > _AO_SYMMETRY_TOL


def eigh_by_irrep(M, irreps):
    """
    Diagonalize a Hermitian matrix within each irrep.

    Parameters
    ----------
    M : NDArray
        A Hermitian matrix in a basis of functions that each transform as one irrep.
    irreps : ArrayLike
        The irrep index of each basis function.

    Returns
    -------
    tuple[NDArray, NDArray]
        The eigenvalues and eigenvectors. The eigenvectors of each irrep occupy that
        irrep's positions in ascending order of eigenvalue, so each keeps the irrep of
        its position.
    """
    irreps = np.asarray(irreps)
    values = np.empty(len(irreps))
    U = np.zeros(M.shape, dtype=M.dtype)
    for h in np.unique(irreps):
        idx = np.flatnonzero(irreps == h)
        values[idx], U[np.ix_(idx, idx)] = np.linalg.eigh(M[np.ix_(idx, idx)])
    return values, U


def ao_symmetry_operations(system, info):
    r"""
    Compute how AO coefficient vectors transform under each symmetry operation.

    Each basis function maps to the matching function on the partner atom given by
    ``system.atom_permutations``, multiplied by a phase describing how its real spherical
    harmonic transforms under the operation.

    Parameters
    ----------
    system : forte2.System
        System with a detected point group.
    info : forte2.BasisInfo
        Basis set information for ``system``.

    Returns
    -------
    dict[str, NDArray]
        For each operation :math:`g`, the matrix :math:`U(g)` such that the orbital with
        coefficients :math:`c` is mapped to the orbital with coefficients :math:`U(g) c`.
    """
    U_ops = {}
    for op, permutation in system.atom_permutations.items():
        U = np.zeros((system.nbf, system.nbf))
        for i, j in enumerate(permutation):
            partner = {
                (bas.n, bas.l, bas.ml): bas.abs_idx
                for bas in info.basis_labels
                if bas.iatom == j
            }
            for bas in info.basis_labels:
                if bas.iatom == i:
                    U[bas.abs_idx, partner[(bas.n, bas.l, bas.ml)]] = local_sign(
                        bas.l, bas.ml, op
                    )
        U_ops[op] = U
    return U_ops


@dataclass
class SymmetryBasis:
    r"""
    An orthonormal AO basis whose vectors each transform as one irrep of the point group.

    The vectors of irrep :math:`\Gamma` span the range of the projector
    :math:`P_\Gamma = |G|^{-1} \sum_g \chi_\Gamma(g) U(g)` within the orthonormal orbital space.

    Attributes
    ----------
    point_group : str
        The Abelian point group.
    S : NDArray
        The AO overlap matrix.
    vectors : NDArray
        The orthonormal symmetry-adapted vectors, as AO coefficients.
    irreps : NDArray
        The irrep index of each vector.
    """

    point_group: str
    S: np.ndarray
    vectors: np.ndarray
    irreps: np.ndarray

    @classmethod
    def build(cls, system, info, S, X):
        """
        Build the symmetry basis spanning the orthonormal orbitals ``X``.

        Raises
        ------
        RuntimeError
            If the space spanned by ``X`` is not closed under the symmetry operations.
        """
        pg = system.point_group
        U_ops = ao_symmetry_operations(system, info)
        reps = np.array([X.conj().T @ S @ U_ops[op] @ X for op in SYMMETRY_OPS[pg]])
        identity = np.eye(X.shape[1])
        if any(
            not np.allclose(R.conj().T @ R, identity, atol=_AO_SYMMETRY_TOL, rtol=0)
            for R in reps
        ):
            raise RuntimeError(f"The orbital space is not closed under {pg} symmetry.")
        vectors, irreps = [], []
        for label, irrep in COTTON_LABELS[pg].items():
            P = np.einsum("g,gij->ij", CHARACTER_TABLE[pg][label], reps) / len(reps)
            values, v = np.linalg.eigh(0.5 * (P + P.conj().T))
            if not np.allclose(values, np.round(values), atol=_AO_SYMMETRY_TOL, rtol=0):
                raise RuntimeError(f"The {pg} projector for {label} is not idempotent.")
            keep = values > 0.5
            vectors.append(X @ v[:, keep])
            irreps.append(np.full(np.count_nonzero(keep), irrep))
        vectors, irreps = np.hstack(vectors), np.concatenate(irreps)
        if vectors.shape[1] != X.shape[1]:
            raise RuntimeError(f"The {pg} projectors do not span the orbital space.")
        return cls(pg, S, vectors, irreps)

    def eigh(self, F):
        """Diagonalize each irrep and return energies, coefficients, and irreps."""
        eps = np.empty(self.vectors.shape[1])
        C = np.empty(self.vectors.shape, dtype=np.result_type(self.vectors, F))
        for irrep in np.unique(self.irreps):
            indices = np.flatnonzero(self.irreps == irrep)
            X = self.vectors[:, indices]
            eps[indices], c = np.linalg.eigh(X.conj().T @ F @ X)
            C[:, indices] = X @ c
        order = np.argsort(eps, kind="stable")
        return eps[order], C[:, order], self.irreps[order]

    def adapt(self, C, eps):
        """
        Symmetry-adapt orbitals by diagonalizing S C diag(eps) C^dagger S in each irrep.

        Orbitals that already transform as irreps are returned unchanged, up to rotations
        among equal values of ``eps``. Otherwise the result is the set of symmetry-adapted
        orbitals that best follows the ranking given by ``eps``.

        Returns
        -------
        tuple[NDArray, NDArray, NDArray]
            The values of ``eps`` for the adapted orbitals in ascending order, the
            orbitals, and their irreps.
        """
        SC = self.S @ C
        return self.eigh((SC * eps) @ SC.conj().T)

    def check_symmetric(self, M, name):
        """
        Check that an AO operator does not couple different irreps.

        Parameters
        ----------
        M : NDArray
            The AO operator.
        name : str
            The name of the operator, used in the error message.

        Raises
        ------
        ValueError
            If the largest coupling between different irreps, relative to the largest
            element of ``M`` in the symmetry basis, exceeds roundoff.
        """
        coupling = irrep_coupling(self.vectors.conj().T @ M @ self.vectors, self.irreps)
        if coupling > _AO_SYMMETRY_TOL:
            raise ValueError(
                f"The {name} breaks {self.point_group} symmetry (relative coupling "
                f"between irreps {coupling:.1e}). Run with symmetry=False."
            )

    def orbital_irreps(self, C, tol=1e-6):
        """
        Return the irrep index of each orbital.

        Parameters
        ----------
        C : NDArray
            Orbital coefficients, one orbital per column.
        tol : float, optional, default=1e-6
            Largest weight an orbital may have outside its irrep.

        Raises
        ------
        RuntimeError
            If an orbital does not transform as a single irrep.
        """
        weights = np.abs(self.vectors.conj().T @ self.S @ C) ** 2
        nirrep = len(COTTON_LABELS[self.point_group])
        irrep_weights = np.array(
            [weights[self.irreps == h].sum(axis=0) for h in range(nirrep)]
        )
        if np.any(irrep_weights.max(axis=0) < 1 - tol):
            raise RuntimeError(
                f"Some orbitals do not transform as a single {self.point_group} irrep."
            )
        return np.argmax(irrep_weights, axis=0)
