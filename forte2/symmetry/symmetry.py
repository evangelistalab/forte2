import numpy as np
from forte2.system.basis_utils import BasisInfo, get_shell_label
from forte2.system.parse_geometry import rotation_mat, reflection_mat

_SYMMETRY_OPS = {
    "C2v": ["E","C2z","σ_xz","σ_yz"],            
    "C2h": ["E","C2z","i","σ_xy"],
    "D2":  ["E","C2z","C2y","C2x"],
    "D2h": ["E","C2z","C2y","C2x","i","σ_xy","σ_xz","σ_yz"],  
    "Cs":  ["E","σ_xy"],
    "Ci":  ["E","i"],
    "C2":  ["E","C2z"],
    "C1":  ["E"],
}

# Full 1-D character tables (±1 per operation, in the same order as _SYMMETRY_OPS[group])
_CHARACTER_TABLE = {
    "C2v": {
        "a1": [+1, +1, +1, +1],
        "a2": [+1, +1, -1, -1],
        "b1": [+1, -1, +1, -1],
        "b2": [+1, -1, -1, +1],
    },
    "C2h": {
        "ag": [+1, +1, +1, +1],
        "au": [+1, +1, -1, -1],
        "bg": [+1, -1, +1, -1],
        "bu": [+1, -1, -1, +1],
    },
    "D2": {
        "a":  [+1, +1, +1, +1],
        "b1": [+1, +1, -1, -1],  
        "b2": [+1, -1, +1, -1],  
        "b3": [+1, -1, -1, +1],  
    },
    "D2h": {
        "ag":  [+1,+1,+1,+1,+1,+1,+1,+1],
        "au":  [+1,+1,+1,+1,-1,-1,-1,-1],
        "b1g": [+1,+1,-1,-1,+1,+1,-1,-1],
        "b1u": [+1,+1,-1,-1,-1,-1,+1,+1],
        "b2g": [+1,-1,+1,-1,+1,-1,+1,-1],
        "b2u": [+1,-1,+1,-1,-1,+1,-1,+1],
        "b3g": [+1,-1,-1,+1,+1,-1,-1,+1],
        "b3u": [+1,-1,-1,+1,-1,+1,+1,-1],
    },
    "Cs": {"a'":[+1,+1], "a''":[+1,-1]},
    "Ci": {"g":[+1,+1],  "u":[+1,-1]},
    "C2": {"a":[+1,+1],  "b":[+1,-1]},
    "C1": {"a":[+1]},
}


def sph_parity_cca(l, m):
    if abs(m) > l:
        raise ValueError(f'Something wrong - |m| cannot exceed l')
    ma = abs(m)
    pz = (1 - ma) & 1
    if m == 0:
        return 0, 0, pz
    if m > 0: # cos-type
        px = ma & 1
        py = 0
    else: # sin-type
        px = (ma - 1) & 1
        py = 1
    return px, py, pz

# def local_sign(l, m, op):
#     if op == 'E':
#         return +1
#     elif op == 'i':
#         return (-1)**l
#     else:
#         px, py, pz = sph_parity_cca(l, m)
#         if op == 'C2x': return (-1)**(py + pz)
#         elif op == 'C2y': return (-1)**(px + pz)
#         elif op == 'C2z': return (-1)**(px + py)
#         elif op == 'σ_xz': return (-1)**(py)
#         elif op == 'σ_yz': return (-1)**(px)
#         elif op == 'σ_xy': return (-1)**(pz)

def local_sign(l, m, op):
    ma = abs(m)
    if op == 'E':
        return 1
    if op == 'i':
        return (-1)**l
    if op == 'C2z':
        return (-1)**ma
    if op == 'C2x':
        if m == 0:
            return (-1)**l
        return (-1)**(l + ma) if m > 0 else (-1)**(l + ma + 1)
    if op == 'C2y':
        if m == 0:
            return (-1)**l
        return (-1)**l if m > 0 else (-1)**(l + 1)
    if op == 'σ_xz':
        return 1 if m >= 0 else -1
    if op == 'σ_yz':
        if m == 0:
            return 1
        return (-1)**ma if m > 0 else (-1)**(ma + 1)
    if op == 'σ_xy':
        return (-1)**(l + ma)
    raise ValueError(f"Unknown op {op}")


def get_symmetry_ops(point_group, prinaxis):
    symmetry_ops = {}

    axes = {'x': 0, 'y': 1, 'z': 2}
    I = np.eye(3)

    ops = _SYMMETRY_OPS[point_group]
    for op in ops:
        if op == 'E':
            symmetry_ops[op] = I
        elif 'C2' in op:
            symmetry_ops[op] = rotation_mat(I[:, axes[op[-1]]], np.deg2rad(180.))
        elif op == 'i':
            symmetry_ops[op] = -I
        elif 'σ_' in op:
            symmetry_ops[op] = reflection_mat((axes[op[-2]], axes[op[-1]]))
    return symmetry_ops

def characters(S, C, U_ops):
    X = C.T.conj() @ S

    # pull out first entry in U_ops dictionary to see whether we are in a spatial or spinor basis
    (_, first_U), *_ = U_ops.items()

    if first_U[0].shape[0] == S.shape[0] // 2:
        return np.column_stack([np.diag(X @ to_spinor(U) @ C) for op, U in U_ops.items()]) 
    else:
        return np.column_stack([np.diag(X @ U @ C) for op, U in U_ops.items()])  

def to_spinor(X):
    return np.block([[X, np.zeros_like(X)], [np.zeros_like(X), X]])

def assign_irrep_labels(group, U_ops, S, C):
    ops_order = _SYMMETRY_OPS[group]

    # Compute character vector for each orbital in all symmetry ops
    chars = characters(S, C, U_ops)

    # Compare the character vector to the expected results and pick the closest match
    table = _CHARACTER_TABLE[group]
    T = np.array([table[name] for name in table])     # (n_irrep, |G|)
    names = list(table.keys())

    # Distance to each irrep vector
    dists = np.sum((chars[:, None, :] - T[None, :, :])**2, axis=2)   # (M, n_irrep)
    best = np.argmin(dists, axis=1)
    labels = [names[k] for k in best]
    return labels, chars

def build_U_matrices(symmetry_operations, system, info, tol=1e-6):

    U_ops = {}
    for op_label, R in symmetry_operations.items():
        U = np.zeros((system.nbf, system.nbf))
        for i, a in enumerate(system.atoms):
            v = R @ a[1]
            # get basis fcns centered on atom a
            basis_a = [bas for bas in info.basis_labels if bas.iatom == i]
            for j, b in enumerate(system.atoms):
                if (a[0] == b[0]) and (np.linalg.norm(v - b[1]) < tol):
                    # get basis fcns centered on atom b
                    basis_b = [bas for bas in info.basis_labels if bas.iatom == j]
                    for bas1 in basis_a:
                        sgn = local_sign(bas1.l, bas1.ml, op_label)
                        for bas2 in basis_b:
                            if bas1.n == bas2.n and bas1.l == bas2.l and bas1.ml == bas2.ml:
                                U[bas1.abs_idx, bas2.abs_idx] = sgn
                    break
        U_ops[op_label] = U
    return U_ops
    def _mix_block_real_CCA(l, mabs, op_label):
        """Return the 2x2 block (cos,sin) for a π rotation about x or y in the real CCA basis.
           Basis order for the block is [Y_c(l,m), Y_s(l,m)] with m>0.
        """
        if op_label == 'C2x':
            # [[0, (-1)^(l+m)], [(-1)^(l+m+1), 0]]
            return np.array([[0,               (-1)**(l + mabs)],
                             [(-1)**(l + mabs + 1), 0           ]], dtype=int)
        elif op_label == 'C2y':
            # [[0, (-1)^(l+1)], [(-1)^l, 0]]
            return np.array([[0,          (-1)**(l + 1)],
                             [(-1)**(l),  0           ]], dtype=int)
        else:
            raise ValueError("mix block only defined for C2x/C2y")

    U_ops = {}
    for op_label, R in symmetry_operations.items():
        U = np.zeros((system.nbf, system.nbf))
        for i, a in enumerate(system.atoms):
            v = R @ a[1]
            # get basis fcns centered on atom a
            basis_a = [bas for bas in info.basis_labels if bas.iatom == i]
            for j, b in enumerate(system.atoms):
                if (a[0] == b[0]) and (np.linalg.norm(v - b[1]) < tol):
                    # get basis fcns centered on atom b
                    basis_b = [bas for bas in info.basis_labels if bas.iatom == j]

                    # Pre-index the +/-m destinations on atom b for quick lookup
                    # dest[(n,l,+m)] = abs_idx of cos component; dest[(n,l,-m)] = abs_idx of sin component
                    dest = {}
                    for bas2 in basis_b:
                        dest[(bas2.n, bas2.l, bas2.ml)] = bas2.abs_idx

                    for bas1 in basis_a:
                        l = bas1.l
                        m = bas1.ml

                        # Handle the only cases that require mixing: C2x / C2y with |m|>0
                        if op_label in ('C2x', 'C2y') and m != 0:
                            mabs = abs(m)

                            # Find destination indices for the (+m) cos and (-m) sin partners on atom b
                            key_c = (bas1.n, l, +mabs)   # cos component (m>0)
                            key_s = (bas1.n, l, -mabs)   # sin component (m<0)
                            if key_c not in dest or key_s not in dest:
                                # If the partner AOs are not present (e.g., truncated basis), skip safely
                                continue
                            j_c = dest[key_c]
                            j_s = dest[key_s]

                            M = _mix_block_real_CCA(l, mabs, op_label)
                            row = bas1.abs_idx
                            if m > 0:  # row corresponds to cos
                                U[row, j_c] = M[0, 0]
                                U[row, j_s] = M[0, 1]
                            else:      # row corresponds to sin
                                U[row, j_c] = M[1, 0]
                                U[row, j_s] = M[1, 1]
                            continue  # done with this bas1

                        # Default: original scalar-sign path (covers E, i, C2z, mirrors, and m=0 for C2x/C2y)
                        sgn = local_sign(l, m, op_label)
                        for bas2 in basis_b:
                            if bas1.n == bas2.n and bas1.l == bas2.l and bas1.ml == bas2.ml:
                                U[bas1.abs_idx, bas2.abs_idx] = sgn
                    break
        U_ops[op_label] = U
    return U_ops


def assign_mo_symmetries(system, S, C, verbose=False):

    if system.point_group == 'C1':
        return ['a' for _ in range(C.shape[1])]

    info = BasisInfo(system, system.basis)

    # step 1: build symmetry transformation matrices
    symmetry_ops = get_symmetry_ops(system.point_group, system.prinaxis)

    # step 2: build U matrices (permutation * phase)
    U = build_U_matrices(symmetry_ops, 
                         system, 
                         info, 
                         tol=1e-6)

    # step 3: assign irrep labels
    labels, chars = assign_irrep_labels(system.point_group, U, S, C)

    if verbose:
        for i, c in enumerate(chars):
            print(f"orbital {i + 1}, character = {c}")

    return labels