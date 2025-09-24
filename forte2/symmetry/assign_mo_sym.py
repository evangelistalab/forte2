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

import numpy as np
from forte2.helpers import logger, block_diag_2x2
from .sym_utils import SYMMETRY_OPS, CHARACTER_TABLE, COTTON_LABELS, rotation_mat, reflection_mat


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


def characters(S, C, U_ops):
    """
    Compute the characters of all MO vectors across all symmetry operators in the point group.
    """
    X = C.T.conj() @ S

    # pull out first entry in U_ops dictionary to see whether we are in a spatial or spinor basis
    (_, first_U), *_ = U_ops.items()

    if first_U[0].shape[0] == S.shape[0] // 2:
        return np.column_stack(
            [np.diag(X @ block_diag_2x2(U) @ C) for op, U in U_ops.items()]
        )
    else:
        return np.column_stack([np.diag(X @ U @ C) for op, U in U_ops.items()])


def assign_irrep_labels(point_group, U_ops, S, C):
    """
    Assigns the MO irrep labels in `point_group` by matching the character vectors to their expected values.
    """
    # Compute character vector for each orbital in all symmetry ops
    chars = characters(S, C, U_ops)

    # Compare the character vector to the expected results and pick the closest match
    table = CHARACTER_TABLE[point_group]
    T = np.array([table[name] for name in table])  # (n_irrep, |G|)
    names = list(table.keys())

    # Distance to each irrep vector
    dists = np.sum((chars[:, None, :] - T[None, :, :]) ** 2, axis=2)  # (M, n_irrep)
    best = np.argmin(dists, axis=1)
    labels = [names[k] for k in best]
    return labels, chars


def build_U_matrices(symmetry_operations, system, info, tol=1e-6):
    """
    Compute the matrices U(g)_{uv} < u | R(g) | v > that describes how the
    AO basis functions transform under each symmetry operation R(g). This involves
    finding the symmetric partner atom for each basis function and then mutiplying
    that with a local phase describing how the spherical harmonic transforms under
    the symmetry operation.
    """
    U_ops = {}
    for op_label, R in symmetry_operations.items():
        U = np.zeros((system.nbf, system.nbf))
        for i, a in enumerate(system.atoms):
            v = (
                R @ system.prin_atomic_positions[i]
            )  # apply symmetry operation in principal axis frame
            # get basis fcns centered on atom a
            basis_a = [bas for bas in info.basis_labels if bas.iatom == i]
            for j, b in enumerate(system.atoms):
                if (a[0] == b[0]) and (
                    np.linalg.norm(v - system.prin_atomic_positions[j]) < tol
                ):
                    # get basis fcns centered on atom b
                    basis_b = [bas for bas in info.basis_labels if bas.iatom == j]
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


def assign_mo_symmetries(system, info, S, C):
    if system.point_group == "C1":
        labels = ["a" for _ in range(C.shape[1])]
        irrep_indices = [0 for _ in range(C.shape[1])]
    else:
        # step 1: build symmetry transformation matrices
        symmetry_ops = get_symmetry_ops(system.point_group)

        # step 2: build U matrices (permutation * phase)
        U = build_U_matrices(symmetry_ops, system, info, tol=1e-6)

        # step 3: assign irrep labels
        labels, chars = assign_irrep_labels(system.point_group, U, S, C)

        for i, c in enumerate(chars):
            logger.log_debug(f"orbital {i + 1}, character = {c}")

        irrep_indices = [COTTON_LABELS[system.point_group][label] for label in labels]

    return labels, irrep_indices
