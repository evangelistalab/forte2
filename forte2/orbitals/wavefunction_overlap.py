from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import logm, polar, schur

from .orbital_overlap import mo_overlap


def biorthogonalize_casscf_orbitals(
    S: NDArray, ndocc: int, nactv: int
) -> tuple[NDArray, NDArray]:
    r"""
    Build a biorthonormalizing pair of orbital transformations for two CASSCF
    orbital sets with matching inactive/active partitions.

    Implements the "pseudo-corresponding orbitals" construction of Malmqvist
    [Int. J. Quantum Chem. 30, 479 (1986)], Appendix, specialized to CASSCF
    wavefunctions (one inactive block, fully doubly occupied, and one active
    block with unrestricted occupation; no restricted/virtual block).

    Given the mixed docc+active MO overlap ``S`` between orbital sets X and Y,
    this returns transformation matrices ``C_XA`` and ``C_YB`` such that the
    new orbital sets :math:`\varphi^A = \varphi^X C^{XA}` and
    :math:`\varphi^B = \varphi^Y C^{YB}` are biorthonormal,
    :math:`(C^{XA})^\dagger S C^{YB} = 1`.

    ``C_XA`` is block upper-triangular in the (docc, active) ordering (its
    active-row/docc-column block is exactly zero), so the new docc orbitals of
    A are pure recombinations of the old docc orbitals of X; the new active
    orbitals of A may pick up admixture of the old docc orbitals of X. This is
    the block structure required for the transformation to close within the
    CAS CI expansion.

    Parameters
    ----------
    S : NDArray
        Mixed MO overlap ``S^{XY} = (C_X)^\dagger S_AO C_Y``, restricted to the
        docc+active columns of each orbital set, shape
        ``(ndocc + nactv, ndocc + nactv)``.
    ndocc : int
        Number of (always doubly occupied) inactive orbitals. Must match
        between the two orbital sets.
    nactv : int
        Number of active orbitals. Must match between the two orbital sets.

    Returns
    -------
    C_XA : NDArray
        Transformation from X to the biorthonormal basis A, shape
        ``(ndocc + nactv, ndocc + nactv)``.
    C_YB : NDArray
        Transformation from Y to the biorthonormal basis B, shape
        ``(ndocc + nactv, ndocc + nactv)``.

    Raises
    ------
    numpy.linalg.LinAlgError
        If the inactive-inactive or (modified) active-active overlap block is
        singular, i.e. the two orbital sets do not satisfy Malmqvist's
        nonsingularity criterion for the existence of a biorthonormal basis.

    Notes
    -----
    The SVDs of the inactive-inactive and modified active-active blocks are
    unique only up to an arbitrary joint rotation of degenerate singular
    subspaces. When a block has (near-)degenerate singular values, ``C_XA``
    and ``C_YB`` may come back far from the identity even if the two orbital
    sets are nearly the same; the pair remains a valid biorthonormalizing
    solution, but callers exponentiating the corresponding generator (e.g.
    :func:`transform_ci_vector_direct`) handle this via scaling-and-squaring
    rather than requiring a small angle.
    """
    n = ndocc + nactv
    S = np.asarray(S)
    if S.shape != (n, n):
        raise ValueError(f"S must have shape ({n}, {n}), got {S.shape}.")

    docc = slice(0, ndocc)
    actv = slice(ndocc, n)

    S_II = S[docc, docc]
    S_IT = S[docc, actv]
    S_TI = S[actv, docc]
    S_TT = S[actv, actv]

    # Step 1: corresponding orbitals within the inactive block only.
    U1I, D_I, U2I_h = np.linalg.svd(S_II)
    U2I = U2I_h.conj().T
    _check_nonsingular(D_I, "inactive-inactive overlap block")

    # Step 2: modified active-active block, correcting for inactive admixture.
    S_TT_mod = S_TT - S_TI @ U2I @ np.diag(1.0 / D_I) @ U1I.conj().T @ S_IT

    # Step 3: corresponding orbitals for the modified active-active block.
    U1T, D_T, U2T_h = np.linalg.svd(S_TT_mod)
    U2T = U2T_h.conj().T
    _check_nonsingular(D_T, "modified active-active overlap block")

    # Step 4: pseudo-corresponding (still orthonormal) orbitals, and the
    # mixed inactive-active overlaps in that basis. These are two independent
    # blocks (P1 and P2 are different orbital sets, so S^{P1P2} need not be
    # Hermitian): S_IT_P1P2 = <P1-docc|P2-active>, S_TI_P1P2 = <P1-active|P2-docc>.
    S_IT_P1P2 = U1I.conj().T @ S_IT @ U2T  # (ndocc, nactv)
    S_TI_P1P2 = U1T.conj().T @ S_TI @ U2I  # (nactv, ndocc)

    # Step 5: shift to the nonorthogonal basis. New docc orbitals are left
    # untouched (pure old-docc combinations); new active orbitals of each side
    # pick up an admixture of the (unchanged) new-docc orbitals of that same
    # side, chosen to cancel the inactive-active overlap blocks.
    invD_I = 1.0 / D_I
    M_A = -np.diag(invD_I) @ S_TI_P1P2.conj().T  # (ndocc, nactv), A side
    M_B = -np.diag(invD_I) @ S_IT_P1P2  # (ndocc, nactv), B side

    C_XA = np.zeros((n, n), dtype=S.dtype)
    C_XA[docc, docc] = U1I
    C_XA[docc, actv] = U1I @ M_A
    C_XA[actv, actv] = U1T

    C_YB = np.zeros((n, n), dtype=S.dtype)
    C_YB[docc, docc] = U2I
    C_YB[docc, actv] = U2I @ M_B
    C_YB[actv, actv] = U2T

    # Step 6: rescale the B side by the diagonal so that S^{AB} = 1 exactly
    # (after step 5, S^{AB} = diag(D_I, D_T)).
    d = np.concatenate([D_I, D_T])
    C_YB = C_YB @ np.diag(1.0 / d)

    return C_XA, C_YB


def _check_nonsingular(singular_values: NDArray, name: str, tol: float = 1e-10) -> None:
    if np.any(np.abs(singular_values) < tol):
        raise np.linalg.LinAlgError(
            f"The {name} is singular (smallest singular value "
            f"{np.min(np.abs(singular_values)):.3e}); the two orbital sets "
            "cannot be biorthogonalized within the CASSCF-closed transformation "
            "group. This typically means the two active spaces describe "
            "qualitatively different orbital character."
        )


def _real_generator(t_actv: NDArray, tol: float = 1e-10) -> NDArray:
    """
    Return ``t_actv`` as a real array, raising if it has a non-negligible
    imaginary part. ``logm`` of a real orthogonal matrix with an eigenvalue
    near -1 can return a spuriously complex result (a branch-cut artifact);
    callers restricted to real CI vectors need this caught rather than
    silently truncated.
    """
    t_actv = np.asarray(t_actv)
    if np.iscomplexobj(t_actv):
        imag_norm = np.max(np.abs(t_actv.imag)) if t_actv.size else 0.0
        if imag_norm > tol:
            raise ValueError(
                "t_actv has a non-negligible imaginary part "
                f"(max|Im| = {imag_norm:.3e}); this backend only supports real "
                "CI vectors/orbitals. This can happen when logm(C_XA) hits a "
                "branch-cut ambiguity (an eigenvalue near -1)."
            )
        t_actv = t_actv.real
    return t_actv


def _real_orthogonal_logm(Q: NDArray, tol: float = 1e-10) -> NDArray:
    r"""
    Real antisymmetric logarithm of a real orthogonal matrix ``Q`` (i.e.
    :math:`\kappa` such that :math:`\exp(\kappa) = Q`), via the real Schur
    decomposition rather than ``scipy.linalg.logm``.

    ``logm`` computes eigenvalue-wise logs using the principal branch of the
    complex logarithm, which is discontinuous on the negative real axis; a
    proper rotation with an eigenvalue near -1 (a rotation by close to
    :math:`\pi`) can therefore come back with a spurious, numerically large
    imaginary part even though a real antisymmetric logarithm always exists
    for :math:`\det Q = +1`. Working in the real Schur form sidesteps this:
    each 2x2 rotation block's angle is read off directly via ``arctan2``
    (branch-free over the full range), and any standalone real eigenvalues at
    -1 (unpaired by the block structure, but guaranteed to occur an even
    number of times when :math:`\det Q = +1`) are paired up into synthetic
    rotation-by-:math:`\pi` blocks.

    Parameters
    ----------
    Q : NDArray
        A real orthogonal matrix with determinant +1.
    tol : float, optional
        Tolerance for detecting off-diagonal Schur entries (2x2 blocks) and
        eigenvalues near -1.

    Returns
    -------
    NDArray
        A real antisymmetric matrix :math:`\kappa` with :math:`\exp(\kappa) = Q`.

    Raises
    ------
    ValueError
        If :math:`\det Q < 0` (an improper rotation/reflection), which has no
        real antisymmetric logarithm.
    """
    Q = np.asarray(Q)
    n = Q.shape[0]
    if n == 0:
        return np.zeros((0, 0))
    if np.linalg.det(Q) < 0:
        raise ValueError(
            "Q has determinant < 0 (an improper rotation/reflection); no real "
            "antisymmetric logarithm exists."
        )

    T, Z = schur(Q, output="real")
    K = np.zeros((n, n))
    unpaired_minus_ones = []
    i = 0
    while i < n:
        if i + 1 < n and abs(T[i + 1, i]) > tol:
            # 2x2 rotation block [[c, -s], [s, c]]; arctan2 is branch-free.
            theta = np.arctan2(T[i + 1, i], T[i, i])
            K[i, i + 1] = -theta
            K[i + 1, i] = theta
            i += 2
        else:
            if T[i, i] < 0:
                unpaired_minus_ones.append(i)
            i += 1

    # det(Q) = +1 forces an even count of unpaired -1's; pair them up into
    # synthetic pi-rotation blocks (the pairing itself is a gauge choice,
    # same in spirit as the SVD gauge freedom in biorthogonalize_casscf_orbitals).
    for a, b in zip(unpaired_minus_ones[::2], unpaired_minus_ones[1::2]):
        K[a, b] = -np.pi
        K[b, a] = np.pi

    return Z @ K @ Z.T


def _real_symmetric_logm(P: NDArray, tol: float = 1e-12) -> NDArray:
    """
    Real logarithm of a symmetric positive-definite matrix ``P``, via
    eigendecomposition. Always real and well-defined (eigenvalues of a
    positive-definite matrix are strictly positive, so this never touches the
    negative-real-axis branch cut that affects the general/orthogonal cases).
    """
    P = np.asarray(P)
    w, V = np.linalg.eigh(P)
    if np.any(w <= tol):
        raise np.linalg.LinAlgError(
            f"P is not positive definite (smallest eigenvalue {np.min(w):.3e})."
        )
    return V @ np.diag(np.log(w)) @ V.T


def _robust_orthogonal_steps(Q: NDArray, tol: float = 1e-10) -> list[tuple]:
    """
    Decompose an orthogonal matrix ``Q`` into an ordered sequence of CI-vector
    steps -- ``("reflect",)`` and/or ``("generator", t)`` -- whose composed
    application (in the returned order) represents ``Q``.

    Handles both the genuine absence of a real antisymmetric logarithm for an
    improper ``Q`` (:math:`\\det Q < 0`: no real matrix exponentiates to a
    negative determinant) and ``scipy.linalg.logm``'s branch-cut artifacts for
    a proper ``Q`` with an eigenvalue near -1 (see
    :func:`_real_orthogonal_logm`), by factoring an improper ``Q`` as
    ``Q = F @ R`` (``F`` flips the sign of active orbital 0, chosen
    arbitrarily among equally valid reflections; ``R`` proper) and logging
    ``R`` via :func:`_real_orthogonal_logm`. Because orbitals transform via
    right-multiplication (:math:`\\varphi^{new} = \\varphi^{old} M`), the
    CI-vector representation of a product ``M1 @ M2`` applies as
    ``rho[M2] . rho[M1]`` (an anti-homomorphism), so for ``Q = F @ R`` the
    reflection is applied to the CI vector *before* ``R``'s generator.
    """
    if np.linalg.det(Q) < 0:
        F0 = np.eye(Q.shape[0])
        F0[0, 0] = -1.0
        return [("reflect",), ("generator", _real_orthogonal_logm(F0 @ Q, tol=tol))]
    return [("generator", _real_orthogonal_logm(Q, tol=tol))]


def _robust_real_logm(M: NDArray, tol: float = 1e-8) -> list[tuple]:
    """
    Decompose a real, invertible matrix ``M`` into an ordered sequence of
    CI-vector steps -- ``("reflect",)`` and/or ``("generator", t)`` -- whose
    composed application (in the returned order) represents ``M``, working
    around ``scipy.linalg.logm``'s branch-cut artifacts and the genuine
    absence of a real logarithm for matrices with a component like an
    improper rotation (see :func:`_robust_orthogonal_steps`).

    ``scipy.linalg.logm(M)`` is tried first (fast path, correct for almost
    all inputs). If that comes back non-negligibly complex and ``M`` is
    orthogonal, :func:`_robust_orthogonal_steps` handles it directly -- the
    case actually produced by :func:`biorthogonalize_casscf_orbitals` for
    ``C_XA``'s active-active block (always exactly an SVD orthogonal factor)
    and for ``C_YB``'s whenever the corresponding singular values are all 1
    (e.g. comparing two identical or near-identical orbital sets). Otherwise,
    ``M`` is factored via the polar decomposition ``M = Q @ P`` (``Q``
    orthogonal, ``P`` symmetric positive-definite -- always possible for
    invertible ``M``, and exactly recovers ``C_YB``'s actual construction, an
    SVD orthogonal factor times a positive diagonal rescale, when that block's
    naive ``logm`` also hits a branch cut). ``P``'s generator is always
    well-behaved (:func:`_real_symmetric_logm`); ``Q``'s is handled by
    :func:`_robust_orthogonal_steps`. Using the same right-multiplication
    composition rule as there, for ``M = Q @ P`` the returned step order
    applies ``Q``'s steps first, then ``P``'s generator.

    Parameters
    ----------
    M : NDArray
        A real, invertible matrix.
    tol : float, optional
        Tolerance for judging ``logm(M)``'s imaginary part negligible, and
        for judging ``M`` numerically orthogonal.

    Returns
    -------
    list[tuple]
        An ordered sequence of ``("reflect",)`` / ``("generator", t)`` steps.
    """
    result = logm(M)
    if not np.iscomplexobj(result) or np.max(np.abs(result.imag)) <= tol:
        return [("generator", np.asarray(np.real(result)))]
    if np.allclose(M @ M.T, np.eye(M.shape[0]), atol=tol):
        return _robust_orthogonal_steps(M, tol=tol)

    Q, P = polar(M)
    return _robust_orthogonal_steps(Q, tol=tol) + [
        ("generator", _real_symmetric_logm(P, tol=tol))
    ]


def _apply_orbital0_reflection(ci_strings, C: NDArray) -> NDArray:
    """
    The CI-vector action of flipping the sign of active orbital 0: multiplies
    each determinant's coefficient by :math:`(-1)^{n_0}`, ``n_0`` its
    occupation of orbital 0. This is the exact, closed-form representation of
    a single-orbital reflection (:math:`\\det = -1`, an improper
    transformation with no real antisymmetric generator), needed as a
    companion to :func:`_robust_real_logm`'s improper-matrix branch.
    """
    dets = ci_strings.make_determinants()
    sign = np.array([(-1) ** (int(d.na(0)) + int(d.nb(0))) for d in dets], dtype=float)
    return np.asarray(C) * sign


# -- Ground-truth backend (generic sparse-operator machinery) --------------
# Kept as a correctness reference; not optimized for large active spaces.


def _one_body_sparse_operator(t: NDArray, two_component: bool = False):
    r"""
    Build the one-body second-quantized operator :math:`\sum_{pq} t_{pq}
    E_{pq}` as a ``SparseOperator``.

    For the nonrelativistic case, :math:`E_{pq} = p_\alpha^\dagger q_\alpha +
    p_\beta^\dagger q_\beta` (spin-summed). For the two-component case, each
    index already refers to a full spinor (no separate alpha/beta channels;
    the beta string is always empty, see :func:`biorthogonalize_casscf_orbitals`
    callers in the two-component path), so only the single spinor-channel
    term :math:`p^\dagger q` is added.

    Parameters
    ----------
    t : NDArray
        The (active-space) one-body generator matrix, shape ``(nactv, nactv)``.
    two_component : bool, optional
        If True, build the single-channel (spinor) operator instead of the
        spin-summed one.

    Returns
    -------
    forte2.lib.sparse_ops.SparseOperator
        The corresponding one-body operator.
    """
    from forte2.lib.sparse_ops import SparseOperator

    n = t.shape[0]
    sop = SparseOperator()
    for p in range(n):
        for q in range(n):
            coeff = complex(t[p, q])
            if coeff == 0:
                continue
            sop.add([p], [], [q], [], coeff)
            if not two_component:
                sop.add([], [p], [], [q], coeff)
    return sop


def _make_sparse_state(ci_strings, C: NDArray, threshold: float = 1e-12):
    r"""
    Convert a dense CI vector to a ``SparseState``, screening out coefficients
    at or below ``threshold``. Equivalent to ``CISigmaBuilder.make_sparse_state``,
    reimplemented here in Python since that C++ method is only bound for
    real (float64) CI vectors, and this needs to work for the two-component
    (complex) case too.

    Parameters
    ----------
    ci_strings : forte2.lib.ci_helpers.CIStrings
        The determinant list ``C`` is expressed in.
    C : NDArray
        The CI vector, shape ``(ndet,)``.
    threshold : float, optional
        Coefficients with magnitude at or below this value are omitted.

    Returns
    -------
    forte2.lib.sparse_ops.SparseState
        The CI vector as a sparse state.
    """
    from forte2.lib.sparse_ops import SparseState

    dets = ci_strings.make_determinants()
    return SparseState({d: c for d, c in zip(dets, C) if abs(c) > threshold})


def transform_ci_vector_sparse_ops(
    ci_strings,
    C: NDArray,
    t_actv: NDArray,
    docc_scale: complex = 1.0,
    maxk: int = 32,
    screen_thresh: float = 1e-14,
    two_component: bool = False,
):
    r"""
    Transform an active-space CI vector by a (possibly nonunitary) active-space
    orbital transformation, via the generic sparse-operator infrastructure.

    Applies :math:`\exp(-\sum_{pq} t_{pq} E_{pq})` to the CI vector using
    :class:`forte2.lib.sparse_ops.SparseExp`'s Taylor-series operator
    exponential, then rescales by ``docc_scale``. Inactive orbitals are not
    part of the active-space determinant representation, so their
    contribution to the transformation is a single overall scalar factor
    rather than a generator term; see :func:`biorthogonalize_casscf_orbitals`.

    This is the "ground truth" backend: correct and simple, but not optimized
    for large active spaces (see :func:`transform_ci_vector_direct` for that).

    Parameters
    ----------
    ci_strings : forte2.lib.ci_helpers.CIStrings
        The determinant list ``C`` is expressed in.
    C : NDArray
        The active-space CI vector, shape ``(ndet,)``.
    t_actv : NDArray
        The active-active block of the one-body transformation generator
        (:math:`\log C^{XA}` restricted to the active-active block), shape
        ``(nactv, nactv)``.
    docc_scale : complex, optional
        The scalar factor from the inactive-space part of the transformation.
        Default is 1.0 (no inactive space, or an untouched inactive block).
    maxk : int, optional
        Maximum Taylor expansion order, forwarded to ``SparseExp``.
    screen_thresh : float, optional
        Screening threshold, forwarded to ``SparseExp`` and ``_make_sparse_state``.
    two_component : bool, optional
        If True, build the single-channel (spinor) generator operator instead
        of the spin-summed one; see :func:`_one_body_sparse_operator`.

    Returns
    -------
    forte2.lib.sparse_ops.SparseState
        The transformed CI vector, as a sparse state in the biorthonormal basis.
    """
    from forte2.lib.sparse_ops import SparseExp

    state = _make_sparse_state(ci_strings, np.asarray(C), screen_thresh)
    T_op = _one_body_sparse_operator(t_actv, two_component=two_component)
    new_state = SparseExp(maxk, screen_thresh).apply_op(
        T_op, state, scaling_factor=-1.0
    )
    if docc_scale != 1.0:
        # cast explicitly: multiplying by a bare numpy scalar makes numpy try to
        # broadcast SparseState as an array-like rather than dispatching to
        # SparseState.__rmul__.
        new_state = new_state * complex(docc_scale)
    return new_state


# -- Efficient backend (direct-CI, string-addressed machinery) -------------
# The default. Built on CISigmaBuilder.sigma_one_electron/set_Hamiltonian
# (forte2/ci/ci_sigma_builder.{h,cc}), which is what every CASSCF/CI run
# already uses for its own sigma-vector builds.


def transform_ci_vector_direct(
    ci_strings,
    C: NDArray,
    t_actv: NDArray,
    docc_scale: complex = 1.0,
    tol: float = 1e-13,
    max_taylor_order: int = 25,
    max_squarings: int = 10,
    scale_threshold: float = 0.5,
    two_component: bool = False,
) -> NDArray:
    r"""
    Transform an active-space CI vector by a (possibly nonunitary) active-space
    orbital transformation, via forte2's string-addressed direct-CI machinery.

    Applies :math:`\exp(-\sum_{pq} t_{pq} E_{pq})` to the CI vector as a dense
    array (same determinant ordering as ``C``), then rescales by
    ``docc_scale``; see :func:`transform_ci_vector_sparse_ops` for the
    equivalent computation via the generic sparse-operator path and
    :func:`biorthogonalize_casscf_orbitals` for why only the active-active
    block of the generator is needed here.

    Uses scaling-and-squaring around a Taylor series (the standard technique
    for ``exp(A)v`` when ``A`` may have a large norm): the generator is
    divided by ``2**m`` for the smallest ``m`` that brings its spectral norm
    under ``scale_threshold``, a Taylor series is used for that well-converged
    small-angle exponential, and the result is applied to the vector
    ``2**m`` times in sequence (exact, since
    :math:`\exp(-T) = (\exp(-T/2^m))^{2^m}`). This also means large rotation
    angles (e.g. from :func:`biorthogonalize_casscf_orbitals`'s SVD gauge
    freedom on near-degenerate singular values, or genuinely large orbital
    differences between two independently-optimized states) do not require
    tuning ``max_taylor_order`` up front the way the sparse-ops backend's
    fixed-order Taylor series does.

    Parameters
    ----------
    ci_strings : forte2.lib.ci_helpers.CIStrings
        The determinant list ``C`` is expressed in.
    C : NDArray
        The active-space CI vector, shape ``(ndet,)``.
    t_actv : NDArray
        The active-active block of the one-body transformation generator,
        shape ``(nactv, nactv)``. Must be real (or have a negligible
        imaginary part; see :func:`_real_generator`), unless
        ``two_component`` is True.
    docc_scale : complex, optional
        The scalar factor from the inactive-space part of the transformation.
    tol : float, optional
        Relative convergence tolerance for each small-angle Taylor series.
    max_taylor_order : int, optional
        Maximum Taylor expansion order per squaring step.
    max_squarings : int, optional
        Maximum number of times the generator may be halved.
    scale_threshold : float, optional
        Target spectral norm for the scaled generator before the Taylor
        series is applied.
    two_component : bool, optional
        If True, treat ``C`` and ``t_actv`` as genuinely complex (two-component
        spinor case) and build on ``RelCISigmaBuilder`` instead of
        ``CISigmaBuilder``. Skips the real-only :func:`_real_generator` check.

    Returns
    -------
    NDArray
        The transformed CI vector, a dense array in the same determinant
        ordering as ``C``.

    Raises
    ------
    RuntimeError
        If a small-angle Taylor series fails to converge within
        ``max_taylor_order`` terms even after scaling.
    """
    from forte2.lib.ci_helpers import CISigmaBuilder, RelCISigmaBuilder

    if two_component:
        dtype = complex
        builder_cls = RelCISigmaBuilder
        t_actv = np.asarray(t_actv, dtype=complex)
    else:
        dtype = float
        builder_cls = CISigmaBuilder
        t_actv = _real_generator(t_actv)
    nactv = t_actv.shape[0]
    V_zero = np.zeros((nactv, nactv, nactv, nactv), dtype=dtype)

    norm_t = np.linalg.norm(t_actv, ord=2) if nactv else 0.0
    if norm_t > scale_threshold:
        m = int(np.ceil(np.log2(norm_t / scale_threshold)))
        m = min(max(m, 0), max_squarings)
    else:
        m = 0
    t_scaled = t_actv / (2**m)

    builder = builder_cls(ci_strings, 0.0, t_scaled, V_zero)

    def apply_T(vec: NDArray) -> NDArray:
        sigma = np.empty_like(vec)
        builder.sigma_one_electron(vec, sigma)
        return sigma

    def taylor_exp_neg_T(vec: NDArray) -> NDArray:
        term = vec.copy()
        total = vec.copy()
        for k in range(1, max_taylor_order + 1):
            term = -apply_T(term) / k
            total += term
            if np.linalg.norm(term) < tol * max(np.linalg.norm(total), 1.0):
                break
        else:
            raise RuntimeError(
                f"Taylor series for exp(-T/2^{m}) did not converge within "
                f"{max_taylor_order} terms (tol={tol}). Try increasing "
                "max_squarings or max_taylor_order."
            )
        return total

    vec = np.asarray(C, dtype=dtype)
    for _ in range(2**m):
        vec = taylor_exp_neg_T(vec)

    return complex(docc_scale) * vec if docc_scale != 1.0 else vec


def _apply_generator_steps(
    ci_strings,
    C: NDArray,
    M: NDArray,
    docc_scale: complex = 1.0,
    two_component: bool = False,
    **transform_kwargs,
) -> NDArray:
    """
    Apply the active-active orbital transformation ``M`` (invertible, not
    necessarily orthogonal/unitary) to the CI vector ``C`` via
    :func:`transform_ci_vector_direct`.

    For the real (nonrelativistic) case, ``M`` is decomposed into a sequence
    of real, well-behaved steps (see :func:`_robust_real_logm`) rather than
    exponentiating a single (possibly spuriously or genuinely complex)
    ``logm(M)``. For the two-component case, no such decomposition is needed:
    a complex matrix logarithm always exists for any invertible complex ``M``
    (unlike the real case, where an improper rotation has no real logarithm
    at all), so ``logm(M)`` is applied directly.

    Parameters
    ----------
    ci_strings : forte2.lib.ci_helpers.CIStrings
        The determinant list ``C`` is expressed in.
    C : NDArray
        The active-space CI vector, shape ``(ndet,)``.
    M : NDArray
        The active-active orbital transformation matrix, shape
        ``(nactv, nactv)``.
    docc_scale : complex, optional
        The scalar factor from the inactive-space part of the transformation,
        applied once at the end.
    two_component : bool, optional
        If True, treat ``C`` and ``M`` as genuinely complex (two-component
        spinor case); see :func:`transform_ci_vector_direct`.
    **transform_kwargs
        Forwarded to each :func:`transform_ci_vector_direct` call.

    Returns
    -------
    NDArray
        The transformed CI vector, a dense array in the same determinant
        ordering as ``C``.
    """
    if two_component:
        t = logm(np.asarray(M, dtype=complex))
        vec = transform_ci_vector_direct(
            ci_strings,
            np.asarray(C, dtype=complex),
            t,
            two_component=True,
            **transform_kwargs,
        )
        return complex(docc_scale) * vec

    vec = np.asarray(C, dtype=float)
    for step in _robust_real_logm(M):
        if step[0] == "reflect":
            vec = _apply_orbital0_reflection(ci_strings, vec)
        else:
            vec = transform_ci_vector_direct(
                ci_strings, vec, step[1], **transform_kwargs
            )
    return complex(docc_scale) * vec if docc_scale != 1.0 else vec


# -- Dispatcher --------------------------------------------------------------


def casscf_wavefunction_overlap(
    ci_strings_1,
    C1: NDArray,
    C_docc_actv_1: NDArray,
    system_1,
    ci_strings_2,
    C2: NDArray,
    C_docc_actv_2: NDArray,
    system_2,
    ndocc: int,
    nactv: int,
    backend: Literal["direct", "sparse_ops"] = "direct",
    **backend_kwargs,
) -> complex:
    r"""
    Compute the overlap :math:`\langle \Psi_1 | \Psi_2 \rangle` between two
    CASSCF wavefunctions with independent (possibly non-orthogonal) orbital
    sets, via Malmqvist's nonunitary biorthogonalization scheme [Int. J.
    Quantum Chem. 30, 479 (1986)].

    See :func:`biorthogonalize_casscf_orbitals` for the orbital-space
    construction, and :func:`transform_ci_vector_direct` /
    :func:`transform_ci_vector_sparse_ops` for the two available CI-vector
    transform backends.

    Parameters
    ----------
    ci_strings_1, ci_strings_2 : forte2.lib.ci_helpers.CIStrings
        The determinant lists for states 1 and 2, respectively.
    C1, C2 : NDArray
        The active-space CI vectors for states 1 and 2.
    C_docc_actv_1, C_docc_actv_2 : NDArray
        The docc+active MO coefficients (in that column order) for states 1
        and 2, each of shape ``(nbf, ndocc + nactv)``.
    system_1, system_2 : forte2.System
        The systems that ``C_docc_actv_1`` and ``C_docc_actv_2`` are expressed
        in the AO basis of. May be the same object. Whether this is a
        two-component (spinor, e.g. GHF/X2C) or nonrelativistic overlap is
        read off ``system_1.two_component``; ``system_2`` must agree.
    ndocc, nactv : int
        The number of inactive and active orbitals, shared between the two
        states.
    backend : {"direct", "sparse_ops"}, optional
        Which CI-vector transform to use. ``"direct"`` (default) is the
        efficient, string-addressed backend; ``"sparse_ops"`` is the
        ground-truth reference.
    **backend_kwargs
        Forwarded to the selected backend's ``transform_ci_vector_*`` function.

    Returns
    -------
    complex
        The wavefunction overlap :math:`\langle \Psi_1 | \Psi_2 \rangle`.
    """
    two_component = system_1.two_component
    S_XY = mo_overlap(C_docc_actv_1, system_1, C_docc_actv_2, system_2)
    C_XA, C_YB = biorthogonalize_casscf_orbitals(S_XY, ndocc, nactv)

    # Only the docc-docc and active-active diagonal blocks of the (ndocc+nactv)
    # transformation act nontrivially on a CAS (docc-always-fully-occupied) CI
    # vector: any one-body term with a docc creation index p != q is Pauli-blocked
    # (p is already fully occupied), and the docc-active coupling block is
    # structurally zero (see biorthogonalize_casscf_orbitals). What remains from
    # the docc-docc block collapses to a single scalar, since
    # exp(-sum_i t_ii) = exp(-trace(logm(M))) = det(M)^-1 for the docc-docc block
    # M, per occupied channel. Nonrelativistic docc orbitals are doubly occupied
    # (alpha and beta strings both hold them, hence the square); two-component
    # docc spinors are singly occupied (a single, spin-summed string; see
    # rel_ci.py's "all active electrons live in the alpha (spinor) string"),
    # so only one power applies.
    docc_power = 1 if two_component else 2
    docc_scale_1 = (
        1.0 / np.linalg.det(C_XA[:ndocc, :ndocc]) ** docc_power if ndocc else 1.0
    )
    docc_scale_2 = (
        1.0 / np.linalg.det(C_YB[:ndocc, :ndocc]) ** docc_power if ndocc else 1.0
    )

    if backend == "direct":
        # Real-valued (nonrelativistic) backend: logm(M) may be spuriously
        # (or genuinely) complex for a real M, so route through the robust
        # step-sequence decomposition (see _robust_real_logm). The
        # two-component path skips this; see _apply_generator_steps.
        state_A = _apply_generator_steps(
            ci_strings_1,
            C1,
            C_XA[ndocc:, ndocc:],
            docc_scale_1,
            two_component=two_component,
            **backend_kwargs,
        )
        state_B = _apply_generator_steps(
            ci_strings_2,
            C2,
            C_YB[ndocc:, ndocc:],
            docc_scale_2,
            two_component=two_component,
            **backend_kwargs,
        )
        return complex(np.vdot(state_A, state_B))
    elif backend == "sparse_ops":
        # A plain logm (possibly complex) is already correct here, since
        # SparseOperator/SparseExp handle complex generators natively.
        t_actv_1 = logm(C_XA[ndocc:, ndocc:])
        t_actv_2 = logm(C_YB[ndocc:, ndocc:])
        state_A = transform_ci_vector_sparse_ops(
            ci_strings_1,
            C1,
            t_actv_1,
            docc_scale_1,
            two_component=two_component,
            **backend_kwargs,
        )
        state_B = transform_ci_vector_sparse_ops(
            ci_strings_2,
            C2,
            t_actv_2,
            docc_scale_2,
            two_component=two_component,
            **backend_kwargs,
        )
        return state_A.overlap(state_B)
    else:
        raise ValueError(
            f"Unknown backend {backend!r}; expected 'direct' or 'sparse_ops'."
        )
