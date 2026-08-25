import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import logm, schur

from .orbital_overlap import mo_overlap


@dataclass
class BiorthogonalOrbitals:
    r"""
    The output of :func:`biorthogonalize_casscf_orbitals`: the assembled
    transformations plus the active-space factors they were built from.

    The active-active block of ``C_XA`` is exactly ``U_actv_A``, and that of
    ``C_YB`` is exactly ``U_actv_B @ diag(1 / d_actv)``. Keeping the factors
    is what lets :func:`casscf_wavefunction_overlap` apply an orthogonal
    rotation and a diagonal rescale as two separate, well-conditioned steps
    (Malmqvist's Appendix, steps 4-6) instead of taking a single matrix
    logarithm of their product.

    Attributes
    ----------
    C_XA, C_YB : NDArray
        Transformations from orbital sets X and Y to the biorthonormal bases
        A and B, each of shape ``(ndocc + nactv, ndocc + nactv)``.
    U_actv_A, U_actv_B : NDArray
        The orthogonal (unitary) active-space factors, shape
        ``(nactv, nactv)``. Either may be improper (determinant -1).
    d_actv : NDArray
        The strictly positive singular values of the modified active-active
        overlap block, shape ``(nactv,)``. Only the B side is rescaled by
        these, following Malmqvist.
    """

    C_XA: NDArray
    C_YB: NDArray
    U_actv_A: NDArray
    U_actv_B: NDArray
    d_actv: NDArray


def biorthogonalize_casscf_orbitals(
    S: NDArray, ndocc: int, nactv: int
) -> BiorthogonalOrbitals:
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
    BiorthogonalOrbitals
        The two transformations and the active-space orthogonal/singular-value
        factors they were assembled from.

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

    References
    ----------
    P.-A. Malmqvist, Int. J. Quantum Chem. 30, 479 (1986); the numbered steps
    below follow the Appendix, Eqs. (A.1)-(A.7).
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

    # Step 1 (A.2): corresponding orbitals within the inactive block only.
    U1I, D_I, U2I_h = np.linalg.svd(S_II)
    U2I = U2I_h.conj().T
    _check_nonsingular(D_I, "inactive-inactive overlap block")

    # Step 2 (A.3): modified active-active block, correcting for inactive admixture.
    S_TT_mod = S_TT - S_TI @ U2I @ np.diag(1.0 / D_I) @ U1I.conj().T @ S_IT

    # Step 3 (A.4): corresponding orbitals for the modified active-active block.
    U1T, D_T, U2T_h = np.linalg.svd(S_TT_mod)
    U2T = U2T_h.conj().T
    _check_nonsingular(D_T, "modified active-active overlap block")

    # Step 4 (A.5): pseudo-corresponding (still orthonormal) orbitals, and the
    # mixed inactive-active overlaps in that basis. These are two independent
    # blocks (P1 and P2 are different orbital sets, so S^{P1P2} need not be
    # Hermitian): S_IT_P1P2 = <P1-docc|P2-active>, S_TI_P1P2 = <P1-active|P2-docc>.
    S_IT_P1P2 = U1I.conj().T @ S_IT @ U2T  # (ndocc, nactv)
    S_TI_P1P2 = U1T.conj().T @ S_TI @ U2I  # (nactv, ndocc)

    # Step 5 (A.6): shift to the nonorthogonal basis. New docc orbitals are left
    # untouched (pure old-docc combinations); new active orbitals of each side
    # pick up an admixture of the (unchanged) new-docc orbitals of that same
    # side, chosen to cancel the inactive-active overlap blocks. This shift
    # leaves the CI expansion itself unaltered: adding a multiple of an
    # always-fully-occupied orbital to another occupied orbital does not change
    # a Slater determinant.
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

    # Step 6 (A.7): rescale the B side by the diagonal so that S^{AB} = 1 exactly
    # (after step 5, S^{AB} = diag(D_I, D_T)).
    d = np.concatenate([D_I, D_T])
    C_YB = C_YB @ np.diag(1.0 / d)

    return BiorthogonalOrbitals(
        C_XA=C_XA, C_YB=C_YB, U_actv_A=U1T, U_actv_B=U2T, d_actv=D_T
    )


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


def _robust_orthogonal_steps(Q: NDArray, tol: float = 1e-10) -> list[tuple]:
    """
    Decompose an orthogonal matrix ``Q`` into an ordered sequence of CI-vector
    steps -- ``("reflect",)`` and/or ``("generator", t)`` -- whose composed
    application (in the returned order) represents ``Q``.

    An improper ``Q`` (:math:`\\det Q < 0`) has no real antisymmetric
    logarithm at all, since :math:`\\det(\\exp \\kappa) = \\exp(\\Tr \\kappa) > 0`
    for real :math:`\\kappa`. Such a ``Q`` is factored as ``Q = F @ R``
    (``F`` flips the sign of active orbital 0, chosen arbitrarily among
    equally valid reflections; ``R`` proper), and ``R`` is logged via
    :func:`_real_orthogonal_logm`, which is branch-cut free. Because orbitals
    transform via right-multiplication
    (:math:`\\varphi^{new} = \\varphi^{old} M`), the CI-vector representation
    of a product ``M1 @ M2`` applies as ``rho[M2] . rho[M1]`` (an
    anti-homomorphism), so for ``Q = F @ R`` the reflection is applied to the
    CI vector *before* ``R``'s generator.
    """
    if np.linalg.det(Q) < 0:
        F0 = np.eye(Q.shape[0])
        F0[0, 0] = -1.0
        return [("reflect",), ("generator", _real_orthogonal_logm(F0 @ Q, tol=tol))]
    return [("generator", _real_orthogonal_logm(Q, tol=tol))]


def _apply_orbital0_reflection(ci_strings, C: NDArray) -> NDArray:
    """
    The CI-vector action of flipping the sign of active orbital 0: multiplies
    each determinant's coefficient by :math:`(-1)^{n_0}`, ``n_0`` its
    occupation of orbital 0. This is the exact, closed-form representation of
    a single-orbital reflection (:math:`\\det = -1`, an improper
    transformation with no real antisymmetric generator), needed as a
    companion to :func:`_robust_orthogonal_steps`'s improper-matrix branch.
    """
    dets = ci_strings.make_determinants()
    sign = np.array([(-1) ** (int(d.na(0)) + int(d.nb(0))) for d in dets], dtype=float)
    return np.asarray(C) * sign


def _apply_active_scaling(ci_strings, C: NDArray, d: NDArray) -> NDArray:
    r"""
    The CI-vector action of the diagonal active-orbital rescale
    :math:`\varphi_t \to \varphi_t / d_t`: multiplies each determinant's
    coefficient by :math:`\prod_t d_t^{n_t}`, with :math:`n_t` its occupation
    of active orbital ``t``.

    This is Malmqvist's Appendix step 6 in closed form. The equivalent
    generator route -- exponentiating :math:`\log \operatorname{diag}(1/d)` --
    needs a Taylor series whose cost grows with the spread of ``d``, and is
    the single worst-conditioned piece of the whole transformation when the
    two active spaces differ substantially. Here it costs one pass over the
    determinant list.

    Parameters
    ----------
    ci_strings : forte2.lib.ci_helpers.CIStrings
        The determinant list ``C`` is expressed in.
    C : NDArray
        The active-space CI vector, shape ``(ndet,)``.
    d : NDArray
        The strictly positive active-space singular values, shape ``(nactv,)``.

    Returns
    -------
    NDArray
        The rescaled CI vector.
    """
    dets = ci_strings.make_determinants()
    log_d = np.log(d)
    exponents = np.array(
        [[int(det.na(t)) + int(det.nb(t)) for t in range(len(d))] for det in dets],
        dtype=float,
    )
    return np.asarray(C) * np.exp(exponents @ log_d)


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


def _transform_side_direct(
    ci_strings,
    C: NDArray,
    U_actv: NDArray,
    d_actv: NDArray | None,
    docc_scale: complex,
    two_component: bool,
    **transform_kwargs,
) -> NDArray:
    r"""
    Re-express one side's CI vector in its biorthonormal basis, via the
    string-addressed direct-CI backend.

    The active-space transformation is applied as its two natural factors
    rather than as a single matrix logarithm of their product: first the
    orthogonal (unitary) rotation ``U_actv``, then -- for the B side only --
    the diagonal rescale :math:`\varphi_t \to \varphi_t / d_t`. Because
    orbitals transform by right multiplication, the CI-vector operators
    compose in reverse (:math:`\rho[M_1 M_2] = \rho[M_2]\rho[M_1]`), so
    ``U_actv`` is applied to the vector first and the rescale second.

    Splitting the two factors keeps every exponentiated generator a bounded
    rotation and turns the (arbitrarily ill-conditioned) diagonal part into
    the closed form of :func:`_apply_active_scaling`.

    Parameters
    ----------
    ci_strings : forte2.lib.ci_helpers.CIStrings
        The determinant list ``C`` is expressed in.
    C : NDArray
        The active-space CI vector, shape ``(ndet,)``.
    U_actv : NDArray
        The orthogonal (unitary) active-space factor, shape ``(nactv, nactv)``.
        May be improper in the real case; see :func:`_robust_orthogonal_steps`.
    d_actv : NDArray or None
        The active-space singular values to rescale by, or None to skip the
        rescale (the A side is not rescaled).
    docc_scale : complex
        The scalar factor from the inactive-space part of the transformation,
        applied once at the end.
    two_component : bool
        Whether ``C`` and ``U_actv`` are complex two-component quantities.
    **transform_kwargs
        Forwarded to each :func:`transform_ci_vector_direct` call.

    Returns
    -------
    NDArray
        The transformed CI vector, a dense array in the same determinant
        ordering as ``C``.
    """
    if two_component:
        # A complex logarithm exists for any invertible complex matrix, so the
        # unitary factor needs no reflection/branch-cut handling here.
        vec = transform_ci_vector_direct(
            ci_strings,
            np.asarray(C, dtype=complex),
            logm(np.asarray(U_actv, dtype=complex)),
            two_component=True,
            **transform_kwargs,
        )
    else:
        vec = np.asarray(C, dtype=float)
        for step in _robust_orthogonal_steps(U_actv):
            if step[0] == "reflect":
                vec = _apply_orbital0_reflection(ci_strings, vec)
            else:
                vec = transform_ci_vector_direct(
                    ci_strings, vec, step[1], **transform_kwargs
                )

    if d_actv is not None:
        vec = _apply_active_scaling(ci_strings, vec, d_actv)
    return complex(docc_scale) * vec if docc_scale != 1.0 else vec


# -- Dispatcher --------------------------------------------------------------


def _validate_overlap_inputs(
    ci_strings_1,
    C1,
    C_docc_actv_1,
    system_1,
    ci_strings_2,
    C2,
    C_docc_actv_2,
    system_2,
    ndocc,
    nactv,
    n_frozen_docc,
) -> bool:
    """
    Check the inputs to :func:`casscf_wavefunction_overlap` and return whether
    this is a two-component calculation.

    Every mismatch caught here would otherwise either raise deep inside numpy
    with an opaque shape error or, worse, return a plausible-looking but
    meaningless number: the final step is a dot product of two CI vectors, which
    is silently well-defined whenever the determinant counts happen to agree.
    """
    if bool(system_1.two_component) != bool(system_2.two_component):
        raise ValueError(
            "system_1 and system_2 disagree on two_component "
            f"({system_1.two_component} vs {system_2.two_component}); a "
            "two-component wavefunction cannot be compared with a "
            "nonrelativistic one."
        )
    if not 0 <= n_frozen_docc <= ndocc:
        raise ValueError(
            f"n_frozen_docc must be between 0 and ndocc={ndocc}, "
            f"got {n_frozen_docc}."
        )
    if (ci_strings_1.na, ci_strings_1.nb) != (ci_strings_2.na, ci_strings_2.nb):
        raise ValueError(
            "The two determinant lists have different alpha/beta electron "
            f"counts ({ci_strings_1.na}, {ci_strings_1.nb}) vs "
            f"({ci_strings_2.na}, {ci_strings_2.nb}); their CI vectors span "
            "different Fock-space sectors and their overlap is not defined."
        )
    if ci_strings_1.ndet != ci_strings_2.ndet:
        raise ValueError(
            f"The two determinant lists differ in size ({ci_strings_1.ndet} vs "
            f"{ci_strings_2.ndet}); the CI vectors are not comparable."
        )
    for label, C, strings in (
        ("C1", C1, ci_strings_1),
        ("C2", C2, ci_strings_2),
    ):
        if np.asarray(C).shape != (strings.ndet,):
            raise ValueError(
                f"{label} has shape {np.asarray(C).shape}, expected "
                f"({strings.ndet},) to match its determinant list."
            )
    for label, C_mo in (
        ("C_docc_actv_1", C_docc_actv_1),
        ("C_docc_actv_2", C_docc_actv_2),
    ):
        if np.asarray(C_mo).shape[1] != ndocc + nactv:
            raise ValueError(
                f"{label} has {np.asarray(C_mo).shape[1]} columns, expected "
                f"ndocc + nactv = {ndocc + nactv}."
            )
    return bool(system_1.two_component)


def _warn_if_frozen_docc_coupled(
    S_full: NDArray, n_frozen_docc: int, tol: float = 0.3
) -> None:
    r"""
    Warn if the orbitals about to be discarded are substantially coupled to
    the orbitals that are kept.

    Plasser's Eq. (21) makes two assumptions about frozen orbitals: that they
    are orthonormal between the two states, and that they are orthogonal to
    every retained orbital. Only the second is checked here. The first
    *deliberately* fails whenever the nuclei have moved -- a displaced core
    orbital's self-overlap decays, and removing precisely that decaying factor
    is the reason to freeze it at all -- so testing it would flag the intended
    use case.

    Coupling to the retained space is the assumption whose failure makes
    freezing invalid: an orbital that partly lives in the retained space cannot
    be eliminated without changing the retained problem. Tight cores stay well
    below the threshold; valence orbitals do not. For FH/STO-3G displaced by
    0.1 bohr, freezing the 1s gives a coupling of 0.09 (and improves the
    overlap), while also freezing the valence docc orbitals gives 0.97 (and
    destroys it).

    Only a warning is raised, so a caller who knows what they are doing can
    still proceed.

    Parameters
    ----------
    S_full : NDArray
        The full mixed MO overlap, before any frozen orbitals are dropped.
        Both coupling blocks are read off it, so no extra AO integrals are
        needed.
    n_frozen_docc : int
        Number of leading inactive orbitals about to be discarded.
    tol : float, optional
        Largest tolerated frozen-retained overlap.
    """
    frozen, retained = slice(0, n_frozen_docc), slice(n_frozen_docc, None)
    blocks = [S_full[frozen, retained], S_full[retained, frozen]]
    coupling = max((float(np.max(np.abs(b))) for b in blocks if b.size), default=0.0)
    if coupling > tol:
        warnings.warn(
            f"n_frozen_docc={n_frozen_docc} discards orbitals that are "
            "substantially coupled to the retained orbitals (max overlap "
            f"{coupling:.3g}). The overlap will be unreliable; freeze only "
            "tight core orbitals.",
            UserWarning,
            stacklevel=3,
        )


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
    n_frozen_docc: int = 0,
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
        The determinant lists for states 1 and 2, respectively. Must describe
        the same number of electrons in the same number of active orbitals.
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
    n_frozen_docc : int, optional
        Discard this many leading inactive orbitals from the overlap, assuming
        they are orthonormal between the two states and orthogonal to all
        remaining orbitals. Strongly recommended when the two states sit at
        different geometries: tight core orbitals move with their nuclei, so
        their mutual overlap decays steeply with displacement and multiplies
        into the result, which can destroy an overlap that should be near 1.
        The effect worsens with nuclear charge: for FH/STO-3G, a rigid 0.1 bohr
        translation of an *identical* state returns 0.75 with all cores
        included, versus 0.94 with the 1s frozen. Freeze only tight core
        orbitals -- the justifying assumption fails for valence orbitals, and
        over-freezing degrades the result badly (0.05 for the same case with
        all four docc orbitals frozen). A warning is issued when the discarded
        orbitals visibly violate the assumption. See Plasser et al., J. Chem.
        Theory Comput. 12, 1207 (2016), Sec. 3.3 and Eq. (21).
    **backend_kwargs
        Forwarded to the selected backend's ``transform_ci_vector_*`` function.

    Returns
    -------
    complex
        The wavefunction overlap :math:`\langle \Psi_1 | \Psi_2 \rangle`.

    Raises
    ------
    ValueError
        If the two systems disagree on ``two_component``, if the determinant
        lists are incompatible, if a CI vector length does not match its
        determinant list, or if the orbital counts are inconsistent.
    """
    two_component = _validate_overlap_inputs(
        ci_strings_1,
        C1,
        C_docc_actv_1,
        system_1,
        ci_strings_2,
        C2,
        C_docc_actv_2,
        system_2,
        ndocc,
        nactv,
        n_frozen_docc,
    )

    # Plasser Eq. (21): frozen cores are taken to be orthonormal between the
    # two states and orthogonal to everything else, so they drop out of the
    # overlap entirely rather than contributing a steeply decaying factor.
    ndocc_eff = ndocc - n_frozen_docc
    S_full = mo_overlap(C_docc_actv_1, system_1, C_docc_actv_2, system_2)
    if n_frozen_docc:
        _warn_if_frozen_docc_coupled(S_full, n_frozen_docc)
    keep = slice(n_frozen_docc, ndocc + nactv)
    bio = biorthogonalize_casscf_orbitals(S_full[keep, keep], ndocc_eff, nactv)

    # Only the docc-docc and active-active diagonal blocks of the transformation
    # act nontrivially on a CAS (docc-always-fully-occupied) CI vector: any
    # one-body term with a docc creation index p != q is Pauli-blocked (p is
    # already fully occupied), and the docc-active coupling block is structurally
    # zero (see biorthogonalize_casscf_orbitals). What remains from the docc-docc
    # block collapses to a single scalar, since exp(-sum_i t_ii) =
    # exp(-trace(logm(M))) = det(M)^-1 for the docc-docc block M, per occupied
    # channel. Nonrelativistic docc orbitals are doubly occupied (alpha and beta
    # strings both hold them, hence the square); two-component docc spinors are
    # singly occupied (a single, spin-summed string; see rel_ci.py's "all active
    # electrons live in the alpha (spinor) string"), so only one power applies.
    # This is Malmqvist's factor prod_i (t_ii)^2, p. 489 and Eq. (10) on p. 492.
    docc_power = 1 if two_component else 2
    docc_scale_1 = (
        1.0 / np.linalg.det(bio.C_XA[:ndocc_eff, :ndocc_eff]) ** docc_power
        if ndocc_eff
        else 1.0
    )
    docc_scale_2 = (
        1.0 / np.linalg.det(bio.C_YB[:ndocc_eff, :ndocc_eff]) ** docc_power
        if ndocc_eff
        else 1.0
    )

    if backend == "direct":
        # Apply the active-space transformation as its two exact factors: the
        # orthogonal rotation, then (B side only) the diagonal rescale.
        state_A = _transform_side_direct(
            ci_strings_1,
            C1,
            bio.U_actv_A,
            None,
            docc_scale_1,
            two_component,
            **backend_kwargs,
        )
        state_B = _transform_side_direct(
            ci_strings_2,
            C2,
            bio.U_actv_B,
            bio.d_actv,
            docc_scale_2,
            two_component,
            **backend_kwargs,
        )
        return complex(np.vdot(state_A, state_B))
    elif backend == "sparse_ops":
        # Ground-truth reference: deliberately kept on the combined
        # active-block matrix and a single logm, so that it cross-checks the
        # factored route above rather than sharing its assumptions. A plain
        # (possibly complex) logm is correct here, since SparseOperator/
        # SparseExp handle complex generators natively.
        state_A = transform_ci_vector_sparse_ops(
            ci_strings_1,
            C1,
            logm(bio.C_XA[ndocc_eff:, ndocc_eff:]),
            docc_scale_1,
            two_component=two_component,
            **backend_kwargs,
        )
        state_B = transform_ci_vector_sparse_ops(
            ci_strings_2,
            C2,
            logm(bio.C_YB[ndocc_eff:, ndocc_eff:]),
            docc_scale_2,
            two_component=two_component,
            **backend_kwargs,
        )
        return state_A.overlap(state_B)
    else:
        raise ValueError(
            f"Unknown backend {backend!r}; expected 'direct' or 'sparse_ops'."
        )
