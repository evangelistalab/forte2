import numpy as np

from forte2.helpers.matrix_functions import block_eigh
from .sym_utils import CHARACTER_TABLE, COTTON_LABELS, SYMMETRY_OPS, local_sign


def ao_symmetry_operations(system, info):
    r"""
    Compute how AO coefficient vectors transform under each symmetry operation.

    Each basis function maps to the matching function on the partner atom given by
    ``system.atom_map``, multiplied by a phase describing how its real spherical
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
    for op, image_atom in system.atom_map.items():
        U = np.zeros((system.nbf, system.nbf))
        for i, j in enumerate(image_atom):
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


class SymmetryBasis:
    r"""
    An orthonormal basis of symmetry-adapted linear combinations (SALCs) of AOs, each
    transforming as one irrep of the point group.

    The SALCs of irrep :math:`\Gamma` span the range of the projector
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
    tol : float, optional, default=1e-6
        Largest error allowed in the symmetry of the orbital space:

        1. in the orthogonality of the representation matrices,
        2. in the projector eigenvalues,
        3. (relative) couplings between irreps that the blocked eigensolver discards.

    Attributes
    ----------
    point_group : str
        The Abelian point group.
    S : NDArray
        The AO overlap matrix.
    salcs : NDArray
        The orthonormal SALCs, as AO coefficients, one per column.
    irreps : NDArray
        The irrep index of each SALC.
    tol : float
        The symmetry tolerance.

    Raises
    ------
    RuntimeError
        If the space spanned by ``X`` is not closed under the symmetry operations.
    """

    def __init__(self, system, info, S, X, tol=1e-6):
        pg = system.point_group
        U_ops = ao_symmetry_operations(system, info)
        # < i | R(g) | j > where i and j are orthonormal AO functions
        reps = np.array([X.conj().T @ S @ U_ops[op] @ X for op in SYMMETRY_OPS[pg]])
        identity = np.eye(X.shape[1])
        if any(
            not np.allclose(R.conj().T @ R, identity, atol=tol, rtol=0) for R in reps
        ):
            raise RuntimeError(f"The orbital space is not closed under {pg} symmetry.")
        salcs, irreps = [], []
        for label, irrep in COTTON_LABELS[pg].items():
            # projector into irrep, if P v = v then v transforms as the irrep
            P = np.einsum("g,gij->ij", CHARACTER_TABLE[pg][label], reps) / len(reps)
            # symmetrization to eliminate numerical noise
            # v is the symmetry adapted basis
            values, v = np.linalg.eigh(0.5 * (P + P.conj().T))
            # A projector's eigenvalues are 0 or 1; an irrep absent from the basis has all 0's
            rounded = np.round(values)
            if not (
                np.allclose(values, rounded, atol=tol, rtol=0)
                and np.isin(rounded, (0, 1)).all()
            ):
                raise RuntimeError(f"The {pg} projector for {label} is not idempotent.")
            keep = rounded == 1
            salcs.append(X @ v[:, keep])
            irreps.append(np.full(np.count_nonzero(keep), irrep))
        if sum(c.shape[1] for c in salcs) != X.shape[1]:
            raise RuntimeError(f"The {pg} projectors do not span the orbital space.")
        self.point_group = pg
        self.S = S
        self.tol = tol
        self.salcs = np.hstack(salcs)
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
            If ``F`` couples different irreps by more than ``tol`` times its largest
            element.
        """
        try:
            eps, c, irreps = block_eigh(
                self.salcs.conj().T @ F @ self.salcs,
                self.irreps,
                atol=0.0,
                rtol=self.tol,
                sort=True,
            )
        except ValueError as e:
            raise ValueError(
                f"The operator breaks {self.point_group} symmetry, for example through an "
                "external field that lowers the symmetry. Run with symmetry=False."
            ) from e
        return eps, self.salcs @ c, irreps

    def adapt(self, C, occupations):
        """
        Symmetry-adapt orbitals, keeping as much of their occupied space as possible.

        The adapted orbitals are the natural orbitals of the totally symmetric part of the
        density built from ``C`` and ``occupations``. Orbitals that already transform as
        irreps keep the space spanned by each distinct occupation.

        Parameters
        ----------
        C : NDArray
            Orbital coefficients, one orbital per column.
        occupations : NDArray
            The occupation of each orbital.

        Returns
        -------
        NDArray
            The adapted orbitals, in order of decreasing occupation.
        """
        # the orbitals in the SALC basis
        c = self.salcs.conj().T @ self.S @ C
        D = (c * occupations) @ c.conj().T
        # For an Abelian group, the totally symmetric part of an operator is its blocks
        # within each irrep.
        D[self.irreps[:, None] != self.irreps[None, :]] = 0.0
        _, U, _ = block_eigh(-D, self.irreps, atol=0.0, rtol=0.0, sort=True)
        return self.salcs @ U

    def orbital_irreps(self, C, tol=1e-6):
        """
        Return the irrep index of each orbital.

        Parameters
        ----------
        C : NDArray
            Orbital coefficients, one orbital per column.
        tol : float, optional, default=1e-6
            Largest total weight (squared overlap) an orbital may have outside its
            irrep, and largest deviation of its total weight over the SALCs from 1.

        Raises
        ------
        RuntimeError
            If an orbital is not normalized, does not lie in the span of the SALCs, or
            does not transform as a single irrep.
        """
        # SALC-to-MO weights
        weights = np.abs(self.salcs.conj().T @ self.S @ C) ** 2
        # A normalized orbital within the span of the SALCs has weights summing to 1.
        if not np.allclose(weights.sum(axis=0), 1, atol=tol, rtol=0):
            raise RuntimeError(
                "The orbitals must be normalized and lie in the span of the SALCs."
            )
        nirrep = len(COTTON_LABELS[self.point_group])
        # nirreps x norbs, total projection of each MO onto each of the irreps
        irrep_weights = np.array(
            [weights[self.irreps == h, :].sum(axis=0) for h in range(nirrep)]
        )
        if np.any(irrep_weights.max(axis=0) < 1 - tol):
            raise RuntimeError(
                f"Some orbitals do not transform as a single {self.point_group} irrep."
            )
        return np.argmax(irrep_weights, axis=0)
