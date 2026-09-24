import numpy as np
from scipy.linalg import logm
from scipy.optimize import linear_sum_assignment

from forte2.helpers.matrix_functions import real_orthogonal_logm
from forte2.lib.det import Determinant

from .orbital_blocks import OrbitalBlockBuilder


def rotate_ci_vectors(ci_solver, U_actv):
    r"""
    Re-expresses the CI vectors of a solved CI in rotated active orbitals.

    Rotating the vectors, instead of solving the CI again in the rotated
    orbitals, keeps the identity, order, and phase of every root.

    Parameters
    ----------
    ci_solver : CISolver or RelCISolver
        A solver that has run.
    U_actv : NDArray
        The orthogonal, or unitary if two-component, rotation of the active
        orbitals, :math:`C_\text{new} = C_\text{old} U`, shape
        ``(nactv, nactv)``. It must not couple orbitals in different GAS spaces
        or irreps, which would move the vectors out of the CI space.

    Returns
    -------
    list[NDArray]
        The rotated CI vectors of each sub-solver, shape ``(nbasis, nroot)``,
        in the basis of the solver: CSFs if nonrelativistic, determinants if
        two-component.

    Raises
    ------
    TypeError
        If ``ci_solver`` isn't a `CISolver` or `RelCISolver`.
    RuntimeError
        If ``ci_solver`` hasn't run.
    ValueError
        If ``U_actv`` has the wrong shape or couples GAS spaces or irreps.
    """
    from forte2.ci import CISolver, RelCISolver

    if not isinstance(ci_solver, (CISolver, RelCISolver)):
        raise TypeError(
            "rotate_ci_vectors requires a CISolver or RelCISolver, "
            f"got {type(ci_solver).__name__}."
        )
    if not ci_solver.executed:
        raise RuntimeError("CI solver has not been executed yet.")
    two_component = isinstance(ci_solver, RelCISolver)

    space = ci_solver.mo_space
    if U_actv.shape != (space.nactv, space.nactv):
        raise ValueError(
            f"U_actv must have shape ({space.nactv}, {space.nactv}), "
            f"got {U_actv.shape}."
        )
    irrep_indices = np.asarray(ci_solver.mos.irrep_indices[0])[space.orig_to_contig]
    blocks = OrbitalBlockBuilder(space, irrep_indices).active_blocks(
        relative_index=True
    )
    in_block = np.zeros(U_actv.shape, dtype=bool)
    for block in blocks:
        in_block[np.ix_(block, block)] = True
    if np.any(np.abs(U_actv[~in_block]) > 1e-10):
        raise ValueError(
            "U_actv couples orbitals in different GAS spaces or irreps, which "
            "moves the CI vectors out of the CI space."
        )

    rotated = []
    for solver in ci_solver.sub_solvers:
        C = solver.evecs
        if not two_component:
            C = np.column_stack([solver.csf_C_to_det_C(c) for c in C.T])
        C = transform_ci_vectors(solver, C, U_actv, two_component, blocks)
        if not two_component:
            C = np.column_stack([_det_C_to_csf_C(solver, c) for c in C.T])
        rotated.append(C)
    return rotated


def transform_ci_vectors(solver, C, U, two_component, blocks=None, d=None):
    r"""
    Re-expresses determinant-basis CI vectors after the active orbitals are
    transformed by the unitary ``U``, then, if ``d`` is given, by
    ``diag(1 / d)``.

    ``U`` is split into a signed permutation, applied exactly by relabeling
    determinants, and a rotation close to the identity, applied as
    :math:`\exp(-\hat{T})`.

    Parameters
    ----------
    solver : _CISingleStateSolver or _RelCISingleStateSolver
        The sub-solver whose determinant space holds the vectors.
    C : NDArray
        The CI vectors, shape ``(ndet, nvec)``.
    U : NDArray
        The unitary transform of the active orbitals, shape ``(nactv, nactv)``.
    two_component : bool
        Whether the orbitals are spinors.
    blocks : list[NDArray], optional
        Index blocks that ``U`` is block diagonal in. Defaults to one block.
    d : NDArray, optional
        The positive diagonal rescale applied after ``U``, shape ``(nactv,)``.

    Returns
    -------
    NDArray
        The transformed CI vectors, shape ``(ndet, nvec)``.
    """
    from forte2.lib.ci_helpers import CISigmaBuilder, RelCISigmaBuilder

    nactv = U.shape[0]
    if nactv == 0:
        return C
    if blocks is None:
        blocks = [np.arange(nactv)]
    dtype = complex if two_component else float

    P, log_R = _split_rotation(U, blocks, two_component)
    target, factor = _permutation_map(solver, P, two_component)
    builder_cls = RelCISigmaBuilder if two_component else CISigmaBuilder
    # Harrison-Zarrabian, unlike Knowles-Handy, can swap in a new H without V
    builder = builder_cls(
        solver.ci_strings,
        0.0,
        np.zeros((nactv, nactv), dtype=dtype),
        np.zeros((nactv,) * 4, dtype=dtype),
        solver.log_level,
        algorithm="hz",
    )
    nel = solver.ci_strings.na + solver.ci_strings.nb

    # orbitals transform by right multiplication, so the CI-vector actions of
    # U = P R compose in reverse: P first, then R
    C = np.asarray(C, dtype=dtype)
    transformed = np.empty_like(C)
    for k in range(C.shape[1]):
        c = np.zeros(C.shape[0], dtype=dtype)
        c[target] = factor * C[:, k]
        transformed[:, k] = _apply_generator(builder, c, log_R, nel)
    if d is not None:
        # scaling active orbital u by 1 / d_u multiplies each coefficient by
        # prod_u d_u^omega_u
        exponent = _occupation_sum(builder, np.log(d), C.shape[0], dtype)
        transformed *= np.exp(exponent)[:, None]
    return transformed


def _split_rotation(U, blocks, two_component):
    """
    Splits U = P R into a signed permutation P, which matches each new orbital
    to the old orbital it overlaps most, and a rotation R close to the
    identity, and returns P and log R. Both are block diagonal in blocks. In
    the real case, the signs of P make each block of R proper, since an
    improper rotation has no real logarithm.
    """
    P = np.zeros_like(U)
    log_R = np.zeros_like(U)
    for block in blocks:
        if len(block) == 0:
            continue
        ix = np.ix_(block, block)
        U_block = U[ix]
        rows, cols = linear_sum_assignment(-np.abs(U_block))
        matched = U_block[rows, cols]
        P_block = np.zeros_like(U_block)
        P_block[rows, cols] = matched / np.abs(matched)
        if not two_component and np.linalg.det(P_block.T @ U_block) < 0:
            weakest = np.argmin(np.abs(matched))
            P_block[rows[weakest], cols[weakest]] *= -1
        R_block = P_block.conj().T @ U_block
        P[ix] = P_block
        log_R[ix] = logm(R_block) if two_component else real_orthogonal_logm(R_block)
    return P, log_R


def _permutation_map(solver, P, two_component):
    """
    Gets, for each determinant, the index of its image under the signed
    permutation ``P`` and the factor its coefficient picks up.
    """
    old, new = np.nonzero(P)
    new_of_old = np.empty(len(old), dtype=int)
    new_of_old[old] = new
    # new orbital j is P_ij times old orbital i, so old orbital i is
    # conj(P_ij) times new orbital j
    phase = np.empty(len(old), dtype=P.dtype)
    phase[old] = P[old, new].conj()

    nactv = P.shape[0]
    target = np.empty(len(solver.dets), dtype=int)
    factor = np.empty(len(solver.dets), dtype=P.dtype)
    for I, det in enumerate(solver.dets):
        image = Determinant.zero()
        f = 1.0
        # apply the relabeled creation operators right to left: beta, then
        # alpha, each in descending order; each returns its fermionic sign
        if not two_component:
            for p in reversed([p for p in range(nactv) if det.nb(p)]):
                f *= phase[p] * image.create_beta(int(new_of_old[p]))
        for p in reversed([p for p in range(nactv) if det.na(p)]):
            f *= phase[p] * image.create_alpha(int(new_of_old[p]))
        target[I] = solver.ci_strings.determinant_index(image)
        factor[I] = f
    return target, factor


def _det_C_to_csf_C(solver, c):
    csf = np.zeros(solver.spin_adapter.ncsf)
    solver.spin_adapter.det_C_to_csf_C(np.ascontiguousarray(c), csf)
    return csf


def _occupation_sum(builder, h, ndet, dtype):
    """
    Gets sum_u h_u omega_u(I) for every determinant I, where omega_u(I) is the
    occupation of active orbital u, as a sigma build of the diagonal
    one-electron operator diag(h) on a vector of ones.
    """
    builder.set_Hamiltonian(H=np.diag(h).astype(dtype))
    ones = np.ones(ndet, dtype=dtype)
    occupation_sum = np.empty_like(ones)
    builder.sigma_one_electron(ones, occupation_sum)
    return occupation_sum.real


def _apply_generator(builder, c, log_R, nel, tol=1e-13, max_order=40):
    """
    Computes exp(-T) c, with T = sum_uv log_R[u, v] E_uv for a rotation R of
    the active orbitals.

    The Taylor series converges quickly only if T is small. T applies log_R to
    each of nel electrons in turn, so its norm is at most nel times the norm
    of log_R. T is halved s times until that bound is at most 1, and the
    series of exp(-T / 2^s) is applied 2^s times in succession.
    """
    bound = nel * np.linalg.norm(log_R, 2)
    s = int(np.ceil(np.log2(bound))) if bound > 1.0 else 0
    builder.set_Hamiltonian(H=np.ascontiguousarray(log_R / 2**s))
    sigma = np.empty_like(c)
    for _ in range(2**s):
        term = c
        for k in range(1, max_order + 1):
            builder.sigma_one_electron(term, sigma)
            term = -sigma / k
            c = c + term
            if np.linalg.norm(term) <= tol * np.linalg.norm(c):
                break
        else:
            raise RuntimeError(
                f"The Taylor series didn't converge in {max_order} terms."
            )
    return c
