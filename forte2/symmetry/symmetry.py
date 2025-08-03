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
        "b1": [+1, -1, +1, -1],
        "b2": [+1, +1, -1, -1],
        "b3": [+1, -1, -1, +1],
    },
    "D2h": {
        "ag":  [+1,+1,+1,+1, +1,+1,+1,+1],
        "au":  [+1,+1,+1,+1, -1,-1,-1,-1],
        "b1g": [+1,-1,-1,+1, +1,-1,-1,+1],
        "b1u": [+1,-1,-1,+1, -1,+1,+1,-1],
        "b2g": [+1,+1,-1,-1, +1,+1,-1,-1],
        "b2u": [+1,+1,-1,-1, -1,-1,+1,+1],
        "b3g": [+1,-1,+1,-1, +1,-1,+1,-1],
        "b3u": [+1,-1,+1,-1, -1,+1,-1,+1],
    },
    "Cs": {"a'":[+1,+1], "a''":[+1,-1]},
    "Ci": {"g":[+1,+1],  "u":[+1,-1]},
    "C2": {"a":[+1,+1],  "b":[+1,-1]},
    "C1": {"a":[+1]},
}

# labels: s ; px,py,pz ; dz2, dx2-y2, dxy, dxz, dyz
_P_SIGNS = {
    'px':  {'C2x': +1, 'C2y': -1, 'C2z': -1, 'σ_xy': +1, 'σ_xz': +1, 'σ_yz': -1, 'i': -1},
    'py':  {'C2x': -1, 'C2y': +1, 'C2z': -1, 'σ_xy': +1, 'σ_xz': -1, 'σ_yz': +1, 'i': -1},
    'pz':  {'C2x': -1, 'C2y': -1, 'C2z': +1, 'σ_xy': -1, 'σ_xz': +1, 'σ_yz': +1, 'i': -1},
}
_D_SIGNS = {
    'dz2':     {'C2x': +1, 'C2y': +1, 'C2z': +1, 'σ_xy': +1, 'σ_xz': +1, 'σ_yz': +1, 'i': +1},
    'dx2-y2':  {'C2x': +1, 'C2y': +1, 'C2z': +1, 'σ_xy': +1, 'σ_xz': +1, 'σ_yz': +1, 'i': +1},
    'dxy':     {'C2x': -1, 'C2y': -1, 'C2z': +1, 'σ_xy': +1, 'σ_xz': -1, 'σ_yz': -1, 'i': +1},
    'dxz':     {'C2x': -1, 'C2y': +1, 'C2z': -1, 'σ_xy': -1, 'σ_xz': +1, 'σ_yz': -1, 'i': +1},
    'dyz':     {'C2x': +1, 'C2y': -1, 'C2z': -1, 'σ_xy': -1, 'σ_xz': -1, 'σ_yz': +1, 'i': +1},
}
# F (l=3): 7 real cubic harmonics
# Conventions:
#   fz3      ~ z*(2z^2 - x^2 - y^2)
#   fxz2     ~ x*(2z^2 - x^2 - y^2)
#   fyz2     ~ y*(2z^2 - x^2 - y^2)
#   fxyz     ~ x*y*z
#   fx(x2-3y2) ~ x*(x^2 - 3y^2)
#   fy(3x2-y2) ~ y*(3x^2 - y^2)
#   fz(x2-y2)  ~ z*(x^2 - y^2)

_F_SIGNS = {
    'fz3':        {'C2x': -1, 'C2y': -1, 'C2z': +1, 'σ_xy': -1, 'σ_xz': +1, 'σ_yz': +1, 'i': -1},
    'fxz2':       {'C2x': +1, 'C2y': -1, 'C2z': -1, 'σ_xy': +1, 'σ_xz': +1, 'σ_yz': -1, 'i': -1},
    'fyz2':       {'C2x': -1, 'C2y': +1, 'C2z': -1, 'σ_xy': +1, 'σ_xz': -1, 'σ_yz': +1, 'i': -1},
    'fxyz':       {'C2x': +1, 'C2y': +1, 'C2z': +1, 'σ_xy': -1, 'σ_xz': -1, 'σ_yz': -1, 'i': -1},
    'fx(x2-3y2)': {'C2x': +1, 'C2y': -1, 'C2z': -1, 'σ_xy': +1, 'σ_xz': +1, 'σ_yz': -1, 'i': -1},
    'fy(3x2-y2)': {'C2x': -1, 'C2y': +1, 'C2z': -1, 'σ_xy': +1, 'σ_xz': -1, 'σ_yz': +1, 'i': -1},
    'fz(x2-y2)':  {'C2x': -1, 'C2y': -1, 'C2z': +1, 'σ_xy': -1, 'σ_xz': +1, 'σ_yz': +1, 'i': -1},
}

# G (l=4): 9 real quartic harmonics
# Conventions:
#   gz4         ~ 35 z^4 - 30 z^2 r^2 + 3 r^4
#   gxz3        ~ x*z*(7 z^2 - 3 r^2)
#   gyz3        ~ y*z*(7 z^2 - 3 r^2)
#   g(z2)(x2-y2)~ (x^2 - y^2)*(7 z^2 - r^2)
#   g(z2)xy     ~ x*y*(7 z^2 - r^2)
#   gz(x3-3y2x) ~ z*(x^3 - 3 x y^2)
#   gz(3x2y-y3) ~ z*(3 x^2 y - y^3)
#   g(x4-6x2y2+y4) ~ x^4 - 6 x^2 y^2 + y^4
#   gxy(x2-y2)  ~ x*y*(x^2 - y^2)
_G_SIGNS = {
    'gz4':                 {'C2x': +1, 'C2y': +1, 'C2z': +1, 'σ_xy': +1, 'σ_xz': +1, 'σ_yz': +1, 'i': +1},
    'gxz3':                {'C2x': -1, 'C2y': +1, 'C2z': -1, 'σ_xy': -1, 'σ_xz': +1, 'σ_yz': -1, 'i': +1},
    'gyz3':                {'C2x': +1, 'C2y': -1, 'C2z': -1, 'σ_xy': -1, 'σ_xz': -1, 'σ_yz': +1, 'i': +1},
    'gz2(x2-y2)':          {'C2x': +1, 'C2y': +1, 'C2z': +1, 'σ_xy': +1, 'σ_xz': +1, 'σ_yz': +1, 'i': +1},
    'gz2xy':               {'C2x': -1, 'C2y': -1, 'C2z': +1, 'σ_xy': +1, 'σ_xz': -1, 'σ_yz': -1, 'i': +1},
    'gz(x3-3y2x)':         {'C2x': -1, 'C2y': +1, 'C2z': -1, 'σ_xy': -1, 'σ_xz': +1, 'σ_yz': -1, 'i': +1},
    'gz(3x2y-y3)':         {'C2x': +1, 'C2y': -1, 'C2z': -1, 'σ_xy': -1, 'σ_xz': -1, 'σ_yz': +1, 'i': +1},
    'g(x4-6x2y2+y4)':      {'C2x': +1, 'C2y': +1, 'C2z': +1, 'σ_xy': +1, 'σ_xz': +1, 'σ_yz': +1, 'i': +1},
    'gxy(x2-y2)':          {'C2x': -1, 'C2y': -1, 'C2z': +1, 'σ_xy': +1, 'σ_xz': -1, 'σ_yz': -1, 'i': +1},
}


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
    X = C.T @ S
    return np.column_stack([np.diag(X @ U @ C) for op, U in U_ops.items()])  # shape (M, |G|)

def local_sign(l, label, op):
    if op == 'E': return +1
    if l == 0:    return +1
    lab = label.replace(' ','').lower()
    if l == 1:    return _P_SIGNS[lab][op]
    if l == 2:    return _D_SIGNS[lab][op]
    if l == 3:    return _F_SIGNS[lab][op]
    if l == 4:    return _G_SIGNS[lab][op]
    # if op == 'i': return +1 if (l % 2 == 0) else -1  # generic parity fallback
    raise NotImplementedError(f"Local phases for AO basis functions with l = {l} are not implemented!")

def assign_irrep_labels(group, U_ops, S, C, tol):
    ops_order = _SYMMETRY_OPS[group]
    # 1) Compute character vector for each MO across ALL operations
    chi = characters(S, C, U_ops)          # (M, |G|)
    M = chi.shape[0]

    # 2) Snap to nearest ±1 sign pattern (elementwise)
    # sgn = np.where(chi >=  tol, +1,
    #       np.where(chi <= -tol, -1, np.sign(chi)))    # (M, |G|)
    sgn = chi

    # 3) Compare to each irrep’s exact ±1 vector and pick the closest (Hamming or L2)
    table = _CHARACTER_TABLE[group]
    T = np.array([table[name] for name in table])     # (n_irrep, |G|)
    names = list(table.keys())

    # L2 distance to each irrep vector
    # (optionally weight by |chi| to penalize uncertain entries less)
    dists = np.sum((chi[:,None,:] - T[None,:,:])**2, axis=2)   # (M, n_irrep)
    best = np.argmin(dists, axis=1)
    labels = [names[k] for k in best]

    # 4) Confidence: closeness to ±1 across ops (min |chi|), and match margin
    conf_minabs = np.min(np.clip(np.abs(chi), 0, 1), axis=1)   # ∈ [0,1]
    # margin between best and second-best Hamming distance (bigger is better)
    sorted_d = np.sort(dists, axis=1)
    margin = sorted_d[:,1] - sorted_d[:,0]

    return labels, chi, conf_minabs, margin

def build_U_matrices(symmetry_operations, system, info, tol=1e-6, verbose=False):

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
                        sgn = local_sign(bas1.l, get_shell_label(bas1.l, bas1.m), op_label)
                        for bas2 in basis_b:
                            if bas1.n == bas2.n and bas1.l == bas2.l and bas1.m == bas2.m:
                                U[bas1.abs_idx, bas2.abs_idx] = sgn
                    break

        U_ops[op_label] = U
        if verbose: 
            for p in range(U.shape[0]):
                for q in range(U.shape[1]):
                    if p == q and U[p, q] != 1.:
                        print(f"U({p, q}) = {U[p, q]}")
    return U_ops

def assign_mo_symmetries(system, C, verbose=True):

    if system.symgroup_assign == 'C1':
        return ['a' for _ in range(C.shape[1])]

    info = BasisInfo(system, system.basis)

    # step 1: build symmetry transformation matrices
    symmetry_ops = get_symmetry_ops(system.symgroup_assign, system.prinaxis)

    # step 2: build U matrices (permutation * phase)
    U = build_U_matrices(symmetry_ops, 
                         system, 
                         info, 
                         tol=1e-6, 
                         verbose=False)

    # step 3: assign irrep labels
    labels, chi, conf, conf_minabs = assign_irrep_labels(system.symgroup_assign, 
                                                         U, 
                                                         system.ints_overlap(), 
                                                         C, 
                                                         tol=1.0)

    if verbose:
        for i, c in enumerate(chi):
            print(f"orbital {i + 1}, character = {c}")

    return labels