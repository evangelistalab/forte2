import numpy as np

from forte2.helpers.matrix_functions import block_eigh
from .sym_utils import (
    CHARACTER_TABLE,
    COTTON_LABELS,
    SYMMETRY_OPS,
    get_symmetry_ops,
    local_sign,
)

# Atoms are matched to their images, and the AO representations checked, to this tolerance.
_SYMMETRY_TOL = 1e-6


def ao_symmetry_operations(system, info):
    r"""
    Compute how AO coefficient vectors transform under each symmetry operation.

    Each basis function maps to the matching function on the atom at its image, which
    must lie within 1e-6 bohr, multiplied by a phase describing how its real spherical
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
    charges = np.array([atom[0] for atom in system.atoms])
    positions = system.prin_atomic_positions
    U_ops = {}
    for op, R in get_symmetry_ops(system.point_group).items():
        U = np.zeros((system.nbf, system.nbf))
        for i, image in enumerate(positions @ R.T):
            distances = np.linalg.norm(positions - image, axis=1)
            partners = np.flatnonzero(
                (charges == charges[i]) & (distances < _SYMMETRY_TOL)
            )
            if len(partners) == 0:
                # Left as zero; the closure check of the symmetry basis reports it.
                continue
            partner = {
                (bas.n, bas.l, bas.ml): bas.abs_idx
                for bas in info.basis_labels
                if bas.iatom == partners[0]
            }
            for bas in info.basis_labels:
                if bas.iatom == i:
                    U[bas.abs_idx, partner[(bas.n, bas.l, bas.ml)]] = local_sign(
                        bas.l, bas.ml, op
                    )
        U_ops[op] = U
    return U_ops


class SymmetryBasis:
    r"""
    An orthonormal AO basis whose vectors each transform as one irrep of the point group.

    The vectors of irrep :math:`\Gamma` span the range of the projector
    :math:`P_\Gamma = |G|^{-1} \sum_g \chi_\Gamma(g) U(g)` within the orthonormal orbital space.

    Parameters
    ----------
    system : forte2.System
        System with a detected point group.
    info : forte2.BasisInfo
        Basis set information for ``system``.
    S : NDArray
        The AO overlap matrix.
    X : NDArray
        Orthonormal orbitals, as AO coefficients, that the symmetry basis spans.

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

    Raises
    ------
    RuntimeError
        If the space spanned by ``X`` is not closed under the symmetry operations.
    """

    def __init__(self, system, info, S, X):
        pg = system.point_group
        U_ops = ao_symmetry_operations(system, info)
        # < i | R(g) | j > where i and j are orthonormal AO functions
        reps = np.array([X.conj().T @ S @ U_ops[op] @ X for op in SYMMETRY_OPS[pg]])
        identity = np.eye(X.shape[1])
        if any(
            not np.allclose(R.conj().T @ R, identity, atol=_SYMMETRY_TOL, rtol=0)
            for R in reps
        ):
            raise RuntimeError(
                f"The orbital space is not closed under {pg} symmetry. The geometry may "
                f"not be symmetric to within {_SYMMETRY_TOL} bohr; run with symmetry=False."
            )
        vectors, irreps = [], []
        for label, irrep in COTTON_LABELS[pg].items():
            # projector into irrep, if P v = v then v transforms as the irrep
            P = np.einsum("g,gij->ij", CHARACTER_TABLE[pg][label], reps) / len(reps)
            # symmetrization to eliminate numerical noise
            # v is the symmetry adapted basis
            values, v = np.linalg.eigh(0.5 * (P + P.conj().T))
            # A projector's eigenvalues are 0 or 1; an irrep absent from the basis has all 0's
            rounded = np.round(values)
            if not (
                np.allclose(values, rounded, atol=_SYMMETRY_TOL, rtol=0)
                and np.isin(rounded, (0, 1)).all()
            ):
                raise RuntimeError(f"The {pg} projector for {label} is not idempotent.")
            keep = rounded == 1
            vectors.append(X @ v[:, keep])
            irreps.append(np.full(np.count_nonzero(keep), irrep))
        if sum(v.shape[1] for v in vectors) != X.shape[1]:
            raise RuntimeError(f"The {pg} projectors do not span the orbital space.")
        self.point_group = pg
        self.S = S
        self.vectors = np.hstack(vectors)
        self.irreps = np.concatenate(irreps)

    def eigh(self, F):
        """
        Diagonalize an AO operator within each irrep.

        Returns
        -------
        tuple[NDArray, NDArray, NDArray]
            The eigenvalues in ascending order, the eigenvectors as AO coefficients,
            and their irreps.

        Raises
        ------
        ValueError
            If ``F`` couples different irreps by more than the symmetry tolerance.
        """
        eps, c, irreps = block_eigh(
            self.vectors.conj().T @ F @ self.vectors,
            self.irreps,
            atol=0.0,
            rtol=_SYMMETRY_TOL,
            sort=True,
        )
        return eps, self.vectors @ c, irreps

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
