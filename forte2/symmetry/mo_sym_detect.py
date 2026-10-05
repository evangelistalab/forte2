import numpy as np

from forte2.helpers import logger
from .sym_utils import (
    SYMMETRY_OPS,
    CHARACTER_TABLE,
    COTTON_LABELS,
    rotation_mat,
    reflection_mat,
)


def local_sign(l, m, op):
    """
    Return phase describing how the spherical harmonic Y_lm transforms under Abelian symmetry
    operations `op`, which is one of [E, C2z, C2x, C2y, σ_xy, σ_xz, σ_yz].
    """
    ma = abs(m)
    if op == "E":
        return 1
    if op == "i":
        return (-1) ** l
    if op == "C2z":
        return (-1) ** ma
    if op == "C2x":
        if m == 0:
            return (-1) ** l
        return (-1) ** (l + ma) if m > 0 else (-1) ** (l + ma + 1)
    if op == "C2y":
        if m == 0:
            return (-1) ** l
        return (-1) ** l if m > 0 else (-1) ** (l + 1)
    if op == "σ_xz":
        return 1 if m >= 0 else -1
    if op == "σ_yz":
        if m == 0:
            return 1
        return (-1) ** ma if m > 0 else (-1) ** (ma + 1)
    if op == "σ_xy":
        return (-1) ** (l + ma)
    raise ValueError(f"Unknown op {op}")


def get_symmetry_ops(point_group):
    """
    Compute 3x3 matrix representations for the symmetry operators in `point_group`.
    These representation perform reflections/rotations in the molecular principal frame.
    """
    symmetry_ops = {}

    axes = {"x": 0, "y": 1, "z": 2}
    I = np.eye(3)

    ops = SYMMETRY_OPS[point_group]
    for op in ops:
        if op == "E":
            symmetry_ops[op] = I
        elif "C2" in op:
            symmetry_ops[op] = rotation_mat(I[:, axes[op[-1]]], np.deg2rad(180.0))
        elif op == "i":
            symmetry_ops[op] = -I
        elif "σ_" in op:
            symmetry_ops[op] = reflection_mat((axes[op[-2]], axes[op[-1]]))
    return symmetry_ops


def build_ao_symmetry_operations(system, info, point_group, tol=1e-6):
    """Build spatial AO permutation/phase matrices in the principal frame."""
    U_ops = {}
    for op, R in get_symmetry_ops(point_group).items():
        U = np.zeros((system.nbf, system.nbf))
        for i, a in enumerate(system.atoms):
            position = R @ system.prin_atomic_positions[i]
            basis_a = [bas for bas in info.basis_labels if bas.iatom == i]
            for j, b in enumerate(system.atoms):
                if (
                    a[0] == b[0]
                    and np.linalg.norm(position - system.prin_atomic_positions[j]) < tol
                ):
                    basis_b = [bas for bas in info.basis_labels if bas.iatom == j]
                    for bas1 in basis_a:
                        for bas2 in basis_b:
                            if (bas1.n, bas1.l, bas1.ml) == (bas2.n, bas2.l, bas2.ml):
                                U[bas1.abs_idx, bas2.abs_idx] = local_sign(
                                    bas1.l, bas1.ml, op
                                )
                    break
        U_ops[op] = U
    return U_ops


class MOSymmetryDetector:
    r"""
    Class to detect the irreducible representation (irrep) labels of molecular orbitals.

    Parameters
    ----------
    system : forte2.System
        Forte2 System object with symmetry information
    info : forte2.BasisInfo
        Forte2 BasisInfo object with basis set information
    S : ndarray
        AO overlap matrix
    C : ndarray
        MO coefficient matrix (columns are MOs), symmetrized in place.
    eps : ndarray
        MO energies, updated in place after symmetry projection.
    tol : float, optional, default=1e-6
        Tolerance for matching atomic positions and symmetry representations.
    point_group : str | None, optional
        Override the detected point group, for example to use inversion parity.
    U_ops : dict | None, optional
        Precomputed AO symmetry matrices for the same system and point group.

    Attributes
    ----------
    irrep_indices : list of int
        List of irrep indices for each MO according to COTTON_LABELS
    labels : list of str
        List of irrep labels for each MO (e.g. 'a1', 'b2', etc.)

    Raises
    ------
    RuntimeError
        If the MO space cannot be resolved into irreps of the point group.

    Notes
    -----
    We compute symmetry irreps for MOs a posteriori by computing the character of each MO
    under each symmetry operation of the point group. In a nutshell, what we want is the character

    .. math::

        \chi(g)_{p} = \langle p | \hat{R}(g) | p \rangle = \sum_{uv} c_{pu}^* c_{pv} \langle u | \hat{R}(g) | v \rangle

    Each AO function :math:`|u\rangle \sim R_{nl}(r) Y_{lm}(\theta,\phi)`.
    For Abelian point groups, we only need to consider
    C2 rotations and mirror planes. None of these affect the radial part, but they transform the
    angular part to a symmetric partner on the same or a different atom with a phase.

    If :math:`\hat{R}(g)|v\rangle = \sum_{w} U_{vw} |w\rangle`, then :math:`\langle u | \hat{R}(g) | v \rangle = \sum_{w} U_{vw} \langle u | w \rangle`,
    and :math:`\chi(g)_{p} = \sum_{uvw} c_{pu}^* c_{pv} U_{vw} S_{uw}.`

    The above procedure works if the symmetry operations do not mix MOs (i.e., the true molecular point group is Abelian).
    If some MOs are mixed by the symmetry operations, then we need diagonalize subsets of the MO space
    that are mixed together to obtain MOs that purely transform as irreps of the Abelian subgroup.
    """

    def __init__(
        self, system, info, S, C, eps, tol=1e-6, *, point_group=None, U_ops=None
    ):
        self.system = system
        self.info = info
        self.S = S
        self.C = C
        self.eps = eps
        self.tol = tol
        self.two_component = self.system.two_component
        self.point_group = point_group or self.system.point_group
        self.U_ops = U_ops

    def run(self):
        if self.point_group == "C1":
            self.labels = ["a" for _ in range(self.C.shape[1])]
            self.irrep_indices = [0 for _ in range(self.C.shape[1])]
        else:
            # step 1: build symmetry transformation matrices
            symmetry_ops = get_symmetry_ops(self.point_group)

            # step 2: build U matrices (permutation * phase)
            if self.U_ops is None:
                self.U_ops = self._build_U_matrices(symmetry_ops)

            # step 3: assign irrep labels
            self.labels, chars = self._assign_irrep_labels()

            for i, c in enumerate(chars):
                logger.log_debug(f"orbital {i}, character = {c}")

            self.irrep_indices = [
                COTTON_LABELS[self.point_group][label] for label in self.labels
            ]

    def _compute_characters(self):
        """
        Compute the characters of all MO vectors across all symmetry operators in the point group.
        """
        X = self.C.T.conj() @ self.S
        reps = {op: X @ U @ self.C for op, U in self.U_ops.items()}

        # Connected components of all representations identify invariant blocks,
        # including noncontiguous MOs and pairs with resolvable energy splittings.
        coupled = np.logical_or.reduce(
            [np.abs(rep) > self.tol for rep in reps.values()]
        )
        coupled |= coupled.T
        remaining = set(range(self.C.shape[1]))
        C = self.C.copy()
        eps = self.eps.copy()
        while remaining:
            indices = {min(remaining)}
            remaining -= indices
            pending = list(indices)
            while pending:
                i = pending.pop()
                neighbors = set(np.flatnonzero(coupled[i])) & remaining
                indices |= neighbors
                remaining -= neighbors
                pending.extend(neighbors)
            indices = [int(i) for i in sorted(indices)]
            if len(indices) > 1:
                logger.log_debug(
                    f"Symmetrizing MOs {indices}: orbital-energy span "
                    f"{np.ptp(self.eps[indices]):.6e} Eh"
                )
                rotation, energies = self._project_onto_irrep(reps, indices)
                C[:, indices] = C[:, indices] @ rotation
                eps[indices] = energies

        X = C.T.conj() @ self.S
        chars = []
        for op, U in self.U_ops.items():
            rep = X @ U @ C
            if not np.allclose(
                np.abs(rep), np.eye(rep.shape[0]), atol=self.tol, rtol=0
            ):
                raise RuntimeError(
                    f"MO symmetry projection failed for {self.point_group} "
                    f"operation {op}: the MO space is not fully symmetrized."
                )
            chars.append(np.diag(rep))

        self.C[:] = C
        self.eps[:] = eps
        return np.column_stack(chars)

    def _project_onto_irrep(self, reps, indices):
        """Simultaneously resolve all operations within a coupled MO block."""
        rotation = np.eye(len(indices), dtype=self.C.dtype)
        subspaces = [np.arange(len(indices))]
        for rep in reps.values():
            rep = rep[np.ix_(indices, indices)]
            rep = (rep + rep.T.conj()) * 0.5
            next_subspaces = []
            for subspace in subspaces:
                vectors = rotation[:, subspace]
                values, c = np.linalg.eigh(vectors.T.conj() @ rep @ vectors)
                rotation[:, subspace] = vectors @ c
                # Each Abelian operation has eigenvalues +/-1. Subsequent
                # operations act only within these eigenspaces, so they cannot
                # undo the symmetry resolved by earlier operations.
                for mask in (values < 0, values >= 0):
                    if np.any(mask):
                        next_subspaces.append(subspace[mask])
            subspaces = next_subspaces

        # Canonicalize within each irrep to avoid arbitrary rotations between
        # same-symmetry MOs, then retain energy ordering within the block.
        for subspace in subspaces:
            vectors = rotation[:, subspace]
            F = vectors.T.conj() @ (self.eps[indices, None] * vectors)
            _, c = np.linalg.eigh(F)
            rotation[:, subspace] = vectors @ c
        energies = np.sum(np.abs(rotation) ** 2 * self.eps[indices, None], axis=0)
        order = np.argsort(energies, kind="stable")
        return rotation[:, order], energies[order]

    def _assign_irrep_labels(self):
        """
        Assigns the MO irrep labels in `point_group` by matching the character vectors to their expected values.
        """
        # Compute character vector for each orbital in all symmetry ops
        chars = self._compute_characters()

        # Compare the character vector to the expected results and pick the closest match
        table = CHARACTER_TABLE[self.point_group]
        T = np.array([table[name] for name in table])  # (n_irrep, |G|)
        names = list(table.keys())

        # Distance to each irrep vector
        dists = np.sum(
            np.abs(chars[:, None, :] - T[None, :, :]) ** 2, axis=2
        )  # (M, n_irrep)
        best = np.argmin(dists, axis=1)
        if not np.allclose(chars, T[best], atol=self.tol, rtol=0):
            raise RuntimeError(
                f"MO characters do not match the {self.point_group} character table."
            )
        labels = [names[k] for k in best]
        return labels, chars

    def _build_U_matrices(self, symmetry_operations):
        r"""
        Compute the matrices :math:`U(g)_{\mu\nu}= \langle \mu | R(g) | \nu \rangle`
        that describes how the AO basis functions transform under each symmetry operation R(g).
        This involves finding the symmetric partner atom for each basis function,
        and then mutiplying that with a local phase describing how the spherical
        harmonic transforms under the symmetry operation.
        """
        U_ops = build_ao_symmetry_operations(
            self.system, self.info, self.point_group, self.tol
        )
        if self.C.shape[0] == 2 * self.system.nbf:
            U_ops = {op: np.kron(np.eye(2), U) for op, U in U_ops.items()}
        return U_ops
