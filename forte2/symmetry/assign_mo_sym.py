r"""
This module computes symmetry irreps for MOs a posteriori by computing the character of each MO
under each symmetry operation of the point group. In a nutshell, what we want is the character

.. math::

    \chi(g)_{p} = \langle p | \hat{R}(g) | p \rangle = \sum_{uv} c_{pu}^* c_{pv} \langle u | \hat{R}(g) | v \rangle

Each AO function :math:`|u\rangle \sim R_{nl}(r) Y_{lm}(\theta,\phi)`.
For Abelian point groups, we only need to consider
C2 rotations and mirror planes. None of these affect the radial part, but they transform the
angular part to a symmetric partner on the same or a different atom with a phase.

If :math:`\hat{R}(g)|v\rangle = \sum_{w} U_{vw} |w\rangle`, then :math:`\langle u | \hat{R}(g) | v \rangle = \sum_{w} U_{vw} \langle u | w \rangle`,
and :math:`\chi(g)_{p} = \sum_{uvw} c_{pu}^* c_{pv} U_{vw} S_{uw}.`
"""

from dataclasses import dataclass
import numpy as np

import forte2
from forte2.helpers import logger, block_diag_2x2
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


class MOSymmetryDetector:
    def __init__(self, system, info, S, C, eps, tol=1e-6):
        """
        Parameters
        ----------
        system
            Forte2 System object with symmetry information
        info
            Forte2 BasisInfo object with basis set information
        S
            AO overlap matrix
        C
            MO coefficient matrix (columns are MOs)
        eps
            MO energies
        tol
            tolerance for matching atomic positions under symmetry operations
        """
        self.system = system
        self.info = info
        self.S = S
        self.C = C
        self.eps = eps
        self.tol = tol

    def __post_init__(self):
        self.two_component = self.system.two_component

    def run(self):
        if self.system.point_group == "C1":
            self.labels = ["a" for _ in range(self.C.shape[1])]
            self.irrep_indices = [0 for _ in range(self.C.shape[1])]
        else:
            # step 1: build symmetry transformation matrices
            symmetry_ops = get_symmetry_ops(self.system.point_group)

            # step 2: build U matrices (permutation * phase)
            self.U_ops = self.build_U_matrices(symmetry_ops)

            # step 3: assign irrep labels
            self.labels, chars = self.assign_irrep_labels()

            for i, c in enumerate(chars):
                logger.log_debug(f"orbital {i + 1}, character = {c}")

            self.irrep_indices = [
                COTTON_LABELS[self.system.point_group][label] for label in self.labels
            ]

    def compute_characters(self):
        """
        Compute the characters of all MO vectors across all symmetry operators in the point group.
        """
        X = self.C.T.conj() @ self.S
        off_diag = []
        nondiag_rep = None
        for op, U in self.U_ops.items():
            if self.two_component:
                U = block_diag_2x2(U)
            rep = X @ U @ self.C
            if np.allclose(np.abs(rep), np.eye(rep.shape[0]), atol=1e-6):
                continue
            else:
                for i in range(rep.shape[0]):
                    if (np.abs(rep[:, i]) > 1e-6).sum() > 1:
                        off_diag.append(i)
                nondiag_rep = rep
                break

        if nondiag_rep is not None:
            self.C = self.project_onto_irrep(nondiag_rep, off_diag)
            X = self.C.T.conj() @ self.S

        return np.column_stack(
            [np.diag(X @ U @ self.C) for op, U in self.U_ops.items()]
        )

    def project_onto_irrep(self, nondiag_rep, off_diag):
        # group the MOs that mix together into degenerate subsets and diagonalize each subset
        eps_nondiag = self.eps[off_diag]
        pass

        for i in range(0, len(off_diag), 2):
            sl = slice(off_diag[i], off_diag[i] + 2)
            rep_sub = nondiag_rep[sl, sl]
            _, c = np.linalg.eigh(rep_sub)
            self.C[:, sl] = self.C[:, sl] @ c

        X = self.C.T.conj() @ self.S
        for op, U in self.U_ops.items():
            rep = X @ U @ self.C
            if np.allclose(np.abs(rep), np.eye(rep.shape[0]), atol=1e-6):
                continue
            else:
                print(f"representation for op {op} is not diagonal!")

    def assign_irrep_labels(self):
        """
        Assigns the MO irrep labels in `point_group` by matching the character vectors to their expected values.
        """
        # Compute character vector for each orbital in all symmetry ops
        chars = self.compute_characters()

        # Compare the character vector to the expected results and pick the closest match
        table = CHARACTER_TABLE[self.system.point_group]
        T = np.array([table[name] for name in table])  # (n_irrep, |G|)
        names = list(table.keys())

        # Distance to each irrep vector
        dists = np.sum((chars[:, None, :] - T[None, :, :]) ** 2, axis=2)  # (M, n_irrep)
        best = np.argmin(dists, axis=1)
        labels = [names[k] for k in best]
        return labels, chars

    def build_U_matrices(self, symmetry_operations):
        r"""
        Compute the matrices :math:`U(g)_{\mu\nu}= \langle \mu | R(g) | \nu \rangle`
        that describes how the AO basis functions transform under each symmetry operation R(g).
        This involves finding the symmetric partner atom for each basis function,
        and then mutiplying that with a local phase describing how the spherical
        harmonic transforms under the symmetry operation.
        """
        U_ops = {}
        for op_label, R in symmetry_operations.items():
            U = np.zeros((self.system.nbf, self.system.nbf))
            for i, a in enumerate(self.system.atoms):
                v = (
                    R @ self.system.prin_atomic_positions[i]
                )  # apply symmetry operation in principal axis frame
                # get basis fcns centered on atom a
                basis_a = [bas for bas in self.info.basis_labels if bas.iatom == i]
                for j, b in enumerate(self.system.atoms):
                    if (a[0] == b[0]) and (
                        np.linalg.norm(v - self.system.prin_atomic_positions[j])
                        < self.tol
                    ):
                        # get basis fcns centered on atom b
                        basis_b = [
                            bas for bas in self.info.basis_labels if bas.iatom == j
                        ]
                        for bas1 in basis_a:
                            sgn = local_sign(bas1.l, bas1.ml, op_label)
                            for bas2 in basis_b:
                                if (
                                    bas1.n == bas2.n
                                    and bas1.l == bas2.l
                                    and bas1.ml == bas2.ml
                                ):
                                    U[bas1.abs_idx, bas2.abs_idx] = sgn
                        break
            U_ops[op_label] = U
        return U_ops
