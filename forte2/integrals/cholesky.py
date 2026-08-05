"""
On-the-fly Cholesky decomposition (CD) of the four-center two-electron integrals.

This implements the one-step, pivoted Cholesky algorithm of Koch, Sánchez de Merás, and
Pedersen [J. Chem. Phys. 118, 9481 (2003)], producing the same ``B_Pmn`` tensor of shape
``(naux, nbf, nbf)`` that :class:`~forte2.jkbuilder.jkbuilder.FockBuilder` consumes, such that

.. math::
    (\\mu\\nu|\\rho\\sigma) \\approx \\sum_J B^J_{\\mu\\nu} B^J_{\\rho\\sigma}.

Here ``B^J`` plays the role of Koch's Cholesky vector ``L^J`` in Eq. (1), and ``naux`` is Koch's
number of vectors ``M``. It is the same object produced by the density-fitting path (Koch's
resolution-of-identity alternative, Eq. (2)), so ``FockBuilder`` consumes either interchangeably.

Unlike the dense reference path (``FockBuilder._build_B_cholesky_dense``), the full ``N^4`` ERI
tensor is never formed: only the diagonal ``(mn|mn)`` and, per pivot, one shell-pair block of
columns are computed on demand via the four-center primitives in :mod:`forte2.integrals.integrals`.

Layout note
-----------
The primitives return AO pairs in *packed shell-pair* order (shell-pair ``(A, B)`` in the
canonical order below, AO pair ``iA * nB + iB`` within it), whereas the reshaped ``(naux, nbf,
nbf)`` output and the dense oracle use *global* row-major order ``p = m * nbf + n``. The whole
decomposition is carried out in packed order (the natural order of the primitive output, which
avoids a per-column scatter) and the resulting factor is permuted back to global order once at the
end. A symmetric permutation of the ERI matrix leaves the Cholesky reconstruction valid.

Two-step decomposition (Folkestad 2019)
---------------------------------------
:func:`cholesky_pivoted` implements the two-step algorithm of Folkestad, Kjønstad, and Koch
[J. Chem. Phys. 150, 194112 (2019)], routed through ``cholesky_tei="pivoted"``. Its central idea
(Folkestad Sec. II) is to split the decomposition into

* **Step I** (:func:`cholesky_pivots`) -- determine *only* the pivot indices ``B = {J}``, the
  "Cholesky basis". Because the Cholesky vectors are not the deliverable of this step, the diagonal
  decreases monotonically (Folkestad Eq. 12) and "a diagonal ``D_p = M_pp`` below ``τ`` will never
  be selected as a pivot", so we may screen **both rows and columns** of ``M`` by the same
  threshold ``τ`` -- a stronger screening than Koch's one-step algorithm, which builds the vectors
  during pivoting and can therefore only screen rows via Cauchy-Schwarz.
* **Step II** (:func:`cholesky_vectors_ri`) -- construct the Cholesky vectors from ``B`` using the
  resolution-of-identity (RI) formulation (Folkestad Eqs. 2, 3, 14): each pivot ``J = γδ ∈ B``
  defines a Cholesky-basis product function ``ρ_J(r) = χ_γ(r) χ_δ(r)``, and

  .. math::
      L^J_{αβ} = \\sum_{K ∈ B} (αβ|K)\\, Q^{-T}_{KJ}, \\qquad S_{JK} = (γδ|γ'δ') = (Q Q^T)_{JK}.

  This is exactly a density fit whose auxiliary basis is the Cholesky basis (Folkestad: "a Cholesky
  decomposition is equivalent to an RI approximation"), which is why the result drops directly into
  the same ``B_Pmn`` tensor that :func:`~forte2.jkbuilder.jkbuilder.FockBuilder._build_B_density_fitting`
  produces.

Relative to Folkestad, this reference implementation keeps the transient Step-I vectors in a dense
array rather than exploiting the "memory drops to zero" behaviour of their Fig. 3 (we validate on
small systems against the dense oracle), and defers the paper's further variants -- partitioned
CD (PCD, Eqs. 22-24), one-center CD (1C-CD), and method-specific/active-space screening
(Eqs. 18-21) -- to future work. These deviations are noted at each site below.
"""

import numpy as np
import scipy as sp

from forte2 import integrals
from forte2.helpers import logger
from forte2.helpers.matrix_functions import invsqrt_matrix


def _build_shell_pair_layout(basis):
    """Build the canonical shell-pair list and the packed<->global AO-pair maps.

    The shell pair ``(A, B)`` is the granularity at which the four-center primitive returns columns,
    and corresponds to Koch's ``(**|AB)`` "integral distribution" -- the block of ERI columns
    sharing the ket shell pair ``AB`` that Koch computes together whenever a pivot lying in ``AB``
    is selected (Koch 2003, ALGORITHMS).

    Returns
    -------
    all_pairs : ndarray, shape (nshells**2, 2), int32
        Every ordered shell pair ``(A, B)`` in canonical (A outer, B inner) order.
    row_off : ndarray, shape (nshells**2 + 1,), int64
        Prefix sums of ``nA * nB`` over ``all_pairs``; block ``p`` occupies packed AO-pair rows
        ``[row_off[p], row_off[p + 1])``.
    perm : ndarray, shape (nbf**2,), int64
        Packed-to-global map: ``perm[packed] = global`` where ``global = m * nbf + n``.
    """
    first_size = basis.shell_first_and_size
    nsh = basis.nshells
    nbf = basis.size

    all_pairs = np.empty((nsh * nsh, 2), dtype=np.int32)
    row_off = np.zeros(nsh * nsh + 1, dtype=np.int64)
    perm = np.empty(nbf * nbf, dtype=np.int64)

    p = 0  # shell-pair ordinal
    packed = 0  # packed AO-pair index
    for A in range(nsh):
        fA, nA = first_size[A]
        for B in range(nsh):
            fB, nB = first_size[B]
            all_pairs[p] = (A, B)
            for iA in range(nA):
                base = (fA + iA) * nbf + fB
                perm[packed : packed + nB] = np.arange(base, base + nB)
                packed += nB
            row_off[p + 1] = row_off[p] + nA * nB
            p += 1

    return all_pairs, row_off, perm


def cholesky_otf(system, tol, basis=None):
    r"""
    On-the-fly pivoted Cholesky decomposition of the four-center ERIs.

    Parameters
    ----------
    system : System
        The molecular system. Also selects the integral backend for the primitives.
    tol : float
        Pivot threshold. This is Koch's decomposition accuracy ``D`` (Koch 2003, ALGORITHMS: "The
        decomposition to an accuracy D proceeds in the following manner"). The decomposition stops
        once the largest remaining Schur-complement diagonal drops to or below ``tol`` -- Koch's
        "The process now continues until all diagonal elements are smaller than D." If ``tol <= 0``,
        a machine-precision threshold ``nbf**2 * eps * max(diag)`` is used (matching LAPACK
        ``dpstrf``'s default).
    basis : Basis, optional
        The orbital basis. Defaults to ``system.basis``.

    Returns
    -------
    B : ndarray, shape (naux, nbf, nbf)
        The Cholesky factor, in global row-major AO order, with
        ``sum_J B[J] outer B[J] approx (mn|rs)``.
    naux : int
        The number of Cholesky vectors retained.

    Notes
    -----
    This is the production one-step driver. Relative to the frozen reference
    (:func:`forte2.integrals.cholesky_reference.cholesky_otf_reference`) it applies three of Koch's
    accelerations, each of which leaves the ``(mn|rs) ≈ Σ_J B[J] B[J]`` contract accurate to ``tol``
    but changes the *path* to it (so the two agree only entrywise to ``O(tol)``, not bit-for-bit):

    * **Significant-set restriction.** Columns whose diagonal ``M_pp`` is already below Koch's
      prescreen ``tol^2 / X_max`` can never contribute above ``tol`` to any matrix element (by
      Cauchy-Schwarz ``|M_pq| <= sqrt(M_pp X_max) < tol``), so they are dropped from all vectors and
      GEMMs up front. Because the Schur diagonal only decreases, this static set is a safe superset
      of every future significant set. On large sparse systems this shrinks the work from ``O(M N^2)``
      to ``O(M |D|)``; on small compact systems ``|D| = N^2`` and nothing is dropped.
    * **Schwarz-screened column builds.** Each ``(**|AB)`` block is computed via
      :func:`~forte2.integrals.integrals.coulomb_4c_pair_block_screened`, which skips bra shell pairs
      whose ``Q_AB Q_CD < tol`` before libint2 evaluates them.
    * **Proactive shell-pair draining.** Whenever a block ``AB`` is loaded for a pivot, *all* of its
      columns with Schur diagonal ``> X_max/1000`` are decomposed before returning to the global
      argmax (Koch, ALGORITHMS), so a shell pair is typically evaluated once rather than once per
      pivot. This replaces the reference's reactive single-block cache.
    """
    if basis is None:
        basis = system.basis
    nbf = basis.size
    n = nbf * nbf

    all_pairs, row_off, perm = _build_shell_pair_layout(basis)

    # Koch step 1: "Initially we calculate the diagonal elements M_pp = (ab|ab)." Here p = (m, n)
    # is a compound AO index, so diag[p] = (mn|mn). We hold it in packed shell-pair order (the
    # coulomb_4c_diagonal primitive returns global order, hence the [perm] gather) so it aligns
    # row-for-row with the pair-block columns pulled later. This diagonal is always exact (unscreened)
    # -- it drives both pivot selection and the significant-set screen.
    diag_full = np.asarray(integrals.coulomb_4c_diagonal(system, basis))[perm]

    # Dmax0 is Koch's X_max, the maximum diagonal element used both to set the screening scale and
    # (below) to bound the Schur complement via Cauchy-Schwarz. stop_tol is the accuracy D at which
    # "further improvements of the accuracy beyond D are not possible": we stop pivoting once the
    # largest remaining diagonal falls to or below it.
    Dmax0 = float(diag_full.max()) if n else 0.0
    eps = np.finfo(float).eps
    stop_tol = tol if (tol is not None and tol > 0.0) else n * eps * Dmax0

    # Significant set D (Koch's initial-diagonal prescreen "M_pp < D^2 / X_max"): a column whose
    # diagonal is below screen has |M_pq| <= sqrt(M_pp * X_max) < D for every q, so it contributes
    # nothing above the accuracy D and is excluded from all vectors, GEMMs, and pivot searches. Since
    # the Schur diagonal only decreases (Eq. (3), diagonal case), a column below screen initially
    # stays below it, so this static set safely bounds every future significant set. `sig` maps a
    # packed AO-pair index to its position within the significant set (-1 if screened out).
    screen = (stop_tol * stop_tol / Dmax0) if Dmax0 > 0.0 else 0.0
    keep = diag_full >= screen
    K = np.where(keep)[0]  # packed indices of the significant columns, ascending
    nK = K.size
    sig = np.full(n, -1, dtype=np.int64)
    sig[K] = np.arange(nK)
    diag = diag_full[K].astype(
        float, copy=True
    )  # Schur diagonal over the significant set

    # Shell-pair Schwarz factors Q_AB for Cauchy-Schwarz screening of the column builds. None on the
    # libcint (high angular momentum) backend, in which case the screened primitive falls back to the
    # exact block -- draining and the significant set still apply, only the per-block FLOP saving is
    # forgone.
    schwarz = integrals.coulomb_4c_schwarz_factors(system, basis)

    # L holds the Cholesky vectors L^J row-wise, restricted to the significant set (row Q == vector J
    # of Eq. (1)/(3), column i == significant AO pair K[i]). Koch's central result is that the vector
    # count M stays small -- "the number of elements needed to be stored scale as N^2 much less than
    # the potential N^4 number of raw two-electron integrals" -- so we size for O(M * |D|) and grow by
    # doubling rather than ever allocating the N^4 matrix. The initial guess 4*nbf reflects the
    # observed M ~ (a few)*N (Koch's Tables I/III report M/N ratios of roughly 9-13).
    cap = max(1, min(nK, 4 * nbf))
    L = np.zeros((cap, nK))
    pivot_pos = (
        []
    )  # significant-set positions of selected pivots (for the triangular zeroing)
    Q = 0

    # Outer loop: pick the global-argmax pivot, load its (**|AB) block once, and drain that block.
    while True:
        # Koch: "we find the largest diagonal element" -- the pivot J of Eq. (3), with Dmax = M_JJ
        # the current Schur-complement diagonal. Complete pivoting keeps the semidefinite
        # decomposition stable. Stop when the largest remaining diagonal is below the accuracy D:
        # by Cauchy-Schwarz, Eq. (7), no remaining matrix element can then exceed D.
        jK = int(np.argmax(diag)) if nK else 0
        Dmax = float(diag[jK]) if nK else 0.0
        if Dmax <= stop_tol:
            break

        # Locate the ket shell-pair holding the pivot. Because the bra list equals the ket list in the
        # same canonical order, the pivot's bra-row offset also equals its ket-column offset within kp.
        piv_packed = int(K[jK])
        kp = int(np.searchsorted(row_off, piv_packed, side="right") - 1)

        # Compute Koch's (**|AB) distribution once, Schwarz-screened: every bra AO pair against the
        # single ket shell pair AB = all_pairs[kp]. block[:, c] is the (screened) raw ERI column
        # M[:, q] for the AO pair q at column c of AB; restrict its rows to the significant set.
        block = np.asarray(
            integrals.coulomb_4c_pair_block_screened(
                system, all_pairs, all_pairs[kp : kp + 1], schwarz, stop_tol, basis
            )
        )
        blockK = block[K, :]

        # Significant columns of this ket shell pair and their positions in the significant set.
        kp_lo, kp_hi = int(row_off[kp]), int(row_off[kp + 1])
        posK_of_col = sig[
            kp_lo:kp_hi
        ]  # significant-set position per block column (-1 if screened)
        col_valid = posK_of_col >= 0
        valid_cols = np.nonzero(col_valid)[
            0
        ]  # block-local column indices that are significant
        valid_posK = posK_of_col[col_valid]  # their significant-set positions

        # Proactive draining (Koch, ALGORITHMS): decompose every column of the loaded block whose
        # Schur diagonal exceeds X_max/1000 before recomputing the global argmax, so AB is evaluated
        # once instead of once per pivot. drain_floor never drops below the stop accuracy D.
        drain_floor = max(stop_tol, Dmax / 1000.0)
        while valid_cols.size:
            dcols = diag[valid_posK]
            m = int(np.argmax(dcols))
            Dq = float(dcols[m])
            if Dq <= drain_floor:
                break
            c = int(valid_cols[m])
            jKq = int(valid_posK[m])

            # The pivot's raw column M_pJ (first factor of Eq. (3) before the Schur subtraction);
            # copied because it is updated in place while blockK must stay pristine for other pivots.
            col = blockK[:, c].astype(float, copy=True)
            # Current Schur complement column M~_pJ = M_pJ - sum_{K<J} L^K_p L^K_J (Eq. (3)):
            # L[:Q, jKq] are the L^K_J factors, contracted against the full earlier vectors L^K_p.
            if Q > 0:
                col -= L[:Q].T @ L[:Q, jKq]

            # New Cholesky vector L^J_p = M~_pJ / M_JJ^{1/2} (Eq. (3)); Dq is the pivot's Schur
            # diagonal M_JJ (= M~_JJ at this step).
            Lqq = np.sqrt(Dq)
            col /= Lqq
            # Enforce the exact triangular structure Eq. (3) implies but finite precision only
            # approximates: prior pivots have M~_qJ == 0, and this pivot's own entry is L^J_J = Lqq.
            if pivot_pos:
                col[pivot_pos] = 0.0
            col[jKq] = Lqq

            if Q == cap:
                new_cap = min(cap * 2, nK)
                new_L = np.zeros((new_cap, nK))
                new_L[:cap] = L
                L = new_L
                cap = new_cap
            L[Q] = col

            # Diagonal update, diagonal case of Eq. (3): M~_pp = M_pp - (L^J_p)^2, so the next argmax
            # picks the largest remaining diagonal (Koch's X~_max, Eq. (7)). The pivot's own diagonal
            # goes to exactly zero (M~_JJ = M_JJ - L^J_J^2 = 0), so it is never reselected and drops
            # out of the drain candidates on the next iteration.
            diag -= col * col
            diag[jKq] = 0.0

            pivot_pos.append(jKq)
            Q += 1

    if Q == 0:
        raise RuntimeError(
            "On-the-fly Cholesky produced no vectors; check the basis and cholesky_tol."
        )

    # Scatter each vector's significant columns back to global AO order and reshape to
    # (naux, nbf, nbf). The Q rows are Koch's M vectors L^J; after this the tensor satisfies Eq. (1),
    # (mn|rs) = sum_J B[J, m, n] B[J, r, s], to accuracy D. Columns outside the significant set stay
    # zero -- their true integrals are below D by construction. Reordering AO pairs is a symmetric
    # permutation of M, which leaves the Cholesky reconstruction valid.
    B = np.zeros((Q, n))
    B[:, perm[K]] = L[:Q]
    B = B.reshape(Q, nbf, nbf)

    memory_gb = 8 * (Q * nbf**2) / (1024**3)
    logger.log_info1(
        "Building B tensor using on-the-fly Cholesky decomposition of the ERIs"
    )
    logger.log_info1(f"Number of system basis functions: {nbf}")
    logger.log_info1(f"Number of Cholesky vectors: {Q}")
    logger.log_info1(f"B tensor memory: {memory_gb:.2f} GB")

    return B, Q


# ---------------------------------------------------------------------------
# Two-step decomposition (Folkestad, Kjønstad, Koch, JCP 150, 194112 (2019)).
#
# Everything below is independent of ``cholesky_otf`` above (the Koch one-step
# driver); it is routed through ``cholesky_tei="pivoted"``.
# ---------------------------------------------------------------------------


def _resolve_stop_tol(tol, diag):
    """Return the effective threshold ``τ`` and the maximum diagonal ``X_max``.

    Mirrors :func:`cholesky_otf`: a non-positive ``tol`` falls back to LAPACK ``dpstrf``'s default
    machine-precision threshold ``n * eps * max(diag)``. Folkestad calls this threshold ``τ``
    (Folkestad Sec. II: "These steps are repeated until all diagonal elements of M are below a given
    threshold τ > 0").
    """
    n = diag.size
    Dmax0 = float(diag.max()) if n else 0.0
    eps = np.finfo(float).eps
    stop_tol = tol if (tol is not None and tol > 0.0) else n * eps * Dmax0
    return stop_tol, Dmax0


# Default per-batch qualified-column budget, as a multiple of nbf (used when max_qual is None).
# A few * nbf keeps each qualified block (|D| x n_qual) modest without splitting so finely that
# batching overhead dominates; the final pivot set is threshold-exact regardless of the batch split,
# so this only trades batch size against the number of block evaluations.
_DEFAULT_MAX_QUAL_FACTOR = 10


def cholesky_pivots(
    system, tol, basis=None, *, span_factor=1e-2, max_qual=None, layout=None
):
    r"""
    Step I of the two-step CD: determine *only* the pivot indices ``B`` (the "Cholesky basis").

    This implements Folkestad's step-I procedure (Folkestad Sec. II, numbered list 1-7 and
    Eqs. 8-13, 16-17). No Cholesky vectors are returned: the transient vectors built here exist only
    to update the diagonal (Folkestad Eq. 12) and to Schur-subtract already-selected contributions
    (Folkestad Eqs. 9-10); they are discarded on return. Because the vectors are not the deliverable,
    the diagonal decreases monotonically and "a diagonal ``D_p = M_pp`` below ``τ`` will never be
    selected as a pivot", which is what licenses screening **both** rows and columns of ``M`` by the
    same threshold ``τ``.

    Parameters
    ----------
    system : System
        The molecular system (also selects the integral backend).
    tol : float
        The decomposition threshold ``τ`` (Folkestad Sec. II). ``tol <= 0`` selects a
        machine-precision fallback, as in :func:`cholesky_otf`.
    basis : Basis, optional
        The orbital basis. Defaults to ``system.basis``.
    span_factor : float, optional, default=1e-2
        Folkestad's span factor ``σ`` (Folkestad Eq. 8), "which ensures that qualified diagonals are
        not too small". The default ``σ = 10^-2`` is the value proposed in Folkestad (Sec. II, "we
        use σ = 10^-2 as proposed in Ref. 12").
    max_qual : int, optional
        Upper bound on the number of qualified AO-pair columns computed per batch (Folkestad Eq. 8:
        "such that the number of elements in Q does not exceed a user-specified maximum"). ``None``
        (the default) uses ``_DEFAULT_MAX_QUAL_FACTOR * nbf``, so a large system is decomposed in
        bounded-size batches rather than assembling one enormous qualified block. The pivot set is
        threshold-exact regardless of the batch split, so this only trades batch size against the
        number of block evaluations. Pass a positive integer to override, or ``0``/negative for no
        cap (one batch per qualification round).
    layout : tuple, optional
        A precomputed ``_build_shell_pair_layout(basis)`` triple, to avoid recomputation when this
        is called from :func:`cholesky_pivoted`.

    Returns
    -------
    pivots : ndarray of int64
        The pivot set ``B`` as *packed* AO-pair indices (see the module layout note), in selection
        order. ``len(pivots)`` is at least the numerical rank of ``M`` at threshold ``τ``; see the
        note on minimality below.

    Notes
    -----
    The two-step "Cholesky basis" is not guaranteed to be *minimal*. Within a qualified batch we
    decompose greedily over the qualified columns ``Q`` only (Folkestad step 6), so a column that is
    only near-dependent on the not-yet-selected part of the basis can still be taken as a pivot,
    whereas Koch's one-step algorithm -- which pivots on the single global maximum with all vectors
    in hand -- would have driven it below ``τ`` first. The overshoot is small (a few vectors) and
    shrinks with the span factor ``σ``; the reconstruction accuracy is unaffected. This matches
    Folkestad's observation that the two-step procedure trades a slightly larger basis for the much
    stronger row+column screening it enables.

    This production driver restricts the transient vectors and working diagonal to the significant
    set ``D = {p : D_p >= τ}`` (Folkestad Fig. 1). Because Step I *discards* the vectors -- a column
    ``p`` with ``D_p < τ`` is never a pivot (the inner loop stops at ``D_q <= τ``) and its diagonal
    is never read again -- the vectors need only span ``|D|`` columns, not ``n``. This is precisely
    the stronger row+column screening the two-step algorithm licenses (Folkestad: "a diagonal
    ``D_p = M_pp`` below ``τ`` will never be selected as a pivot"), and closes the ``O(rank * n)``
    scratch of the frozen reference to ``O(rank * |D|)``. The block builds are additionally
    Cauchy-Schwarz screened (Folkestad Eq. 6) via
    :func:`~forte2.integrals.integrals.coulomb_4c_pair_block_screened`.

    Qualification is done purely at shell-pair granularity (Folkestad Sec. II: "we modify the
    screening and qualification steps such that shell pairs are treated instead of AO pairs"),
    ordering shell pairs by their maximal diagonal ``D^{AB}_max`` (Folkestad Eq. 16).
    """
    if basis is None:
        basis = system.basis
    if layout is None:
        layout = _build_shell_pair_layout(basis)
    all_pairs, row_off, perm = layout

    nbf = basis.size
    n = nbf * nbf

    # Resolve the per-batch qualified-column budget (Folkestad Eq. 8's user maximum). None -> a few
    # times nbf so a large system is decomposed in bounded batches; <= 0 disables the cap.
    if max_qual is None:
        max_qual = _DEFAULT_MAX_QUAL_FACTOR * nbf
    batch_cap = max_qual if max_qual and max_qual > 0 else None

    # Initial diagonal D_p = (mn|mn), in packed order (Folkestad step 1's D0; identical to Koch's
    # M_pp). This is always exact (unscreened) -- it drives both qualification and the significant set.
    diag_full = np.asarray(integrals.coulomb_4c_diagonal(system, basis))[perm]
    stop_tol, _ = _resolve_stop_tol(tol, diag_full)

    # Significant set D = {p : D_p >= τ} (Folkestad step 3 / Eq. 17). Unlike Koch's one-step driver,
    # Step I *discards* the vectors, so a column with D_p < τ is never a pivot (the inner loop stops
    # at D_q <= τ) and its diagonal is never read again. The vectors therefore need span only these
    # |D| columns, and the diagonal itself only decreases, so this static set is a safe superset of
    # every future significant set. `sig` maps a packed AO-pair index to its position in D (-1 if
    # excluded). This closes the reference's O(rank * n) scratch to O(rank * |D|).
    keep = diag_full >= stop_tol
    K = np.where(keep)[0]  # packed indices of the significant columns, ascending
    nK = K.size
    if nK == 0:
        raise RuntimeError(
            "Two-step Cholesky (step I) produced no pivots; check the basis and cholesky_tol."
        )
    sig = np.full(n, -1, dtype=np.int64)
    sig[K] = np.arange(nK)
    diag = diag_full[K].astype(
        float, copy=True
    )  # working diagonal over the significant set

    # Shell-pair Schwarz factors for Cauchy-Schwarz screening of the qualified block builds (Folkestad
    # Eq. 6). None on the libcint backend, where the screened primitive falls back to the exact block.
    schwarz = integrals.coulomb_4c_schwarz_factors(system, basis)

    # Map each significant column to its ket shell pair and, from the ascending K order, the K-space
    # segment starts of the shell pairs that contain at least one significant column. These segments
    # let the per-shell-pair max reduction (Folkestad Eq. 16) run over the significant diagonal alone.
    Ksp = (
        np.searchsorted(row_off, K, side="right") - 1
    )  # shell-pair id per significant column
    seg_sp, seg_start = np.unique(
        Ksp, return_index=True
    )  # shell pairs present in D; K-space starts
    seg_start = seg_start.astype(np.intp)

    # Transient Cholesky vectors, restricted to the significant set (row Q == vector Q, column i ==
    # significant AO pair K[i]); kept only to update D and subtract prior contributions, discarded on
    # return. Grown by doubling exactly like cholesky_otf.
    cap = max(1, min(nK, 4 * nbf))
    L = np.zeros((cap, nK))
    pivots = []  # selected pivots as packed AO-pair indices (the deliverable)
    pivot_pos = []  # their significant-set positions (for the triangular zeroing)
    Q = 0

    def _grow(Q, cap, L):
        if Q == cap:
            new_cap = min(cap * 2, nK)
            new_L = np.zeros((new_cap, nK))
            new_L[:cap] = L
            return new_cap, new_L
        return cap, L

    # Outer loop == Folkestad steps 3-7: qualify a batch, decompose it, append its pivots to B,
    # repeat until no significant diagonal remains.
    while True:
        Dmax = float(diag.max()) if nK else 0.0
        # Folkestad step 3 / Eq. 17: once the largest remaining diagonal is below τ, by Cauchy-Schwarz
        # (Folkestad Eq. 6) every remaining matrix element is below τ and the basis B is complete.
        if Dmax <= stop_tol:
            break

        # Folkestad step 4 / Eq. 16: order shell pairs by their maximal diagonal D^{AB}_max (reduced
        # over the significant diagonal) and, per Eq. 8, qualify those reaching σ * Dmax. Qualifying
        # whole shell pairs (not individual AO pairs) matches Libint's shell-quartet evaluation.
        sp_max = np.maximum.reduceat(diag, seg_start)  # aligned with seg_sp
        qual_thresh = span_factor * Dmax
        qmask = sp_max >= qual_thresh
        qualified = seg_sp[qmask]
        # Take qualified shell pairs largest-D^{AB}_max first (Folkestad: "qualified from the AB with
        # the largest diagonal before the next shell pair ... is considered").
        qualified = qualified[np.argsort(sp_max[qmask])[::-1]]

        # Assemble the qualified ket shell pairs subject to the column budget batch_cap.
        ket_kps = []
        ncol = 0
        for kp in qualified:
            size_kp = int(row_off[kp + 1] - row_off[kp])
            if batch_cap is not None and ket_kps and ncol + size_kp > batch_cap:
                break
            ket_kps.append(int(kp))
            ncol += size_kp
        ket_kps = np.asarray(ket_kps, dtype=np.int64)

        # Packed indices of every qualified column, in block-column order (the block concatenates each
        # ket shell pair's contiguous packed range in ket_kps order). Only the significant ones can be
        # pivots, so restrict to those: valid_bc are their block columns, valid_sig their D positions.
        qcols = np.concatenate(
            [np.arange(row_off[kp], row_off[kp + 1]) for kp in ket_kps]
        ).astype(np.int64)
        qcols_sig = sig[qcols]
        valid_bc = np.nonzero(qcols_sig >= 0)[0]
        valid_sig = qcols_sig[valid_bc]
        valid_packed = qcols[valid_bc]

        # Folkestad step 5 / Eqs. 6, 9: compute the (Schwarz-screened) block M_pq for all rows p and
        # qualified columns q, keep only significant rows/columns, then subtract the pivots already in
        # B. colmat holds ~M_pq over D (only prior batches subtracted here; the within-batch
        # subtraction is Eq. 10 below).
        block = np.asarray(
            integrals.coulomb_4c_pair_block_screened(
                system, all_pairs, all_pairs[ket_kps], schwarz, stop_tol, basis
            )
        )
        colmat = block[np.ix_(K, valid_bc)].astype(float, copy=True)  # (nK, n_valid)
        if Q > 0:
            colmat -= L[:Q].T @ L[:Q][:, valid_sig]

        # Folkestad step 6 / Eqs. 10-12: greedily decompose the qualified block. batch_start marks
        # where this batch's pivots begin in L so Eq. 10's Σ_{J∈C} subtraction uses exactly them.
        batch_start = Q
        while True:
            # "select q ∈ Q such that D_q = max_{p∈Q} D_p" (Folkestad step 6). The global argmax pivot
            # always lives in a qualified shell pair, so progress is guaranteed each batch.
            dvals = diag[valid_sig]
            m = int(np.argmax(dvals))
            Dq = float(dvals[m])
            if Dq <= stop_tol:  # no significant diagonal left in Q -> back to step 3
                break
            jKq = int(valid_sig[m])  # significant-set position of the pivot

            # Folkestad Eq. 10: L^q_p = (~M_pq - Σ_{J∈C} L^J_p L^J_q) / sqrt(~M_qq).
            newvec = colmat[:, m].copy()
            if Q > batch_start:
                newvec -= L[batch_start:Q].T @ L[batch_start:Q, jKq]
            Lqq = np.sqrt(Dq)
            newvec /= Lqq
            # Exact triangular structure (finite precision only approximates it): earlier pivots have
            # a zero entry, and this pivot's own entry is L^q_q = sqrt(~M_qq).
            if pivot_pos:
                newvec[pivot_pos] = 0.0
            newvec[jKq] = Lqq

            cap, L = _grow(Q, cap, L)
            L[Q] = newvec

            # Folkestad Eq. 12: D_p ← D_p - (L^q_p)^2. Setting D_q = 0 removes q from Q (Eq. 11) and
            # from all future qualification, so it is never reselected.
            diag -= newvec * newvec
            diag[jKq] = 0.0

            pivots.append(int(valid_packed[m]))
            pivot_pos.append(jKq)
            Q += 1

        # Folkestad step 7 / Eq. 13: B ← B ∪ C, then return to step 3.

    if Q == 0:
        raise RuntimeError(
            "Two-step Cholesky (step I) produced no pivots; check the basis and cholesky_tol."
        )

    return np.asarray(pivots, dtype=np.int64)


def cholesky_vectors_ri(system, pivots, basis=None, *, tol=None, layout=None):
    r"""
    Step II of the two-step CD: build the Cholesky vectors from the pivot set ``B`` via RI.

    This implements Folkestad's step II (Folkestad Sec. II, Eqs. 2, 3, 14, 15). Each pivot
    ``J = γδ ∈ B`` defines a Cholesky-basis product function ``ρ_J(r) = χ_γ(r) χ_δ(r)`` (Folkestad:
    "each pivot J = γδ ∈ B defines a Cholesky basis function ρ_J(r) = χ_γ(r)χ_δ(r)"), and the
    vectors follow from the RI/inner-projection expression

    .. math::
        S_{JK} = (ρ_J | ρ_K) = (γδ | γ'δ') = (Q Q^T)_{JK},
        \qquad L^J_{αβ} = \sum_{K ∈ B} (αβ | K)\, Q^{-T}_{KJ}
        \tag{Folkestad Eqs. 3, 14}

    which reproduces the ERI matrix as the inner projection ``M ≈ A S^{-1} A^T`` with
    ``A_{αβ,K} = (αβ|K)`` (Folkestad Eq. 2). This is precisely a density fit onto the Cholesky basis,
    so the same linear algebra as
    :meth:`~forte2.jkbuilder.jkbuilder.FockBuilder._build_B_density_fitting` applies.

    Parameters
    ----------
    system : System
        The molecular system (also selects the integral backend).
    pivots : array_like of int
        The pivot set ``B`` as packed AO-pair indices, e.g. from :func:`cholesky_pivots`.
    basis : Basis, optional
        The orbital basis. Defaults to ``system.basis``.
    tol : float, optional
        The decomposition threshold ``τ``, used only for the Cauchy-Schwarz product-function
        screening of Folkestad Eq. 15. ``None`` disables that screening.
    layout : tuple, optional
        A precomputed ``_build_shell_pair_layout(basis)`` triple.

    Returns
    -------
    B : ndarray, shape (naux, nbf, nbf)
        The Cholesky factor in global row-major AO order, with
        ``sum_J B[J] outer B[J] approx (mn|rs)``.
    naux : int
        The number of Cholesky vectors, equal to ``len(pivots)``.
    """
    if basis is None:
        basis = system.basis
    if layout is None:
        layout = _build_shell_pair_layout(basis)
    all_pairs, row_off, perm = layout

    nbf = basis.size
    n = nbf * nbf
    pivots = np.asarray(pivots, dtype=np.int64)
    npiv = pivots.size

    # Each pivot is a packed AO pair sitting in a ket shell pair; gather the unique ket shell pairs
    # so a single pair-block call yields A_{αβ,K} = (αβ|K) for all rows αβ and all pivot columns K.
    ket_kps = np.searchsorted(row_off, pivots, side="right") - 1
    unique_kps, inv = np.unique(ket_kps, return_inverse=True)
    inv = np.asarray(inv).reshape(-1)  # numpy<2.1 may return a 2-D inverse

    # Column offset of each unique ket shell pair within the concatenated block, and hence the block
    # column of every pivot (its shell pair's offset + its local position row_off within the block).
    kp_sizes = (row_off[unique_kps + 1] - row_off[unique_kps]).astype(np.int64)
    col_off = np.zeros(unique_kps.size + 1, dtype=np.int64)
    np.cumsum(kp_sizes, out=col_off[1:])
    pivcol = col_off[inv] + (pivots - row_off[unique_kps[inv]])

    # Folkestad Eq. 15: Cauchy-Schwarz screening of the (αβ|K) build. |(αβ|K)| <= Q_αβ Q_K, so a bra
    # shell pair whose Q_αβ Q_K < screen for a pivot's ket shell pair contributes below screen to that
    # vector and can be skipped before libint2 evaluates it. Using the screened primitive makes this
    # skip *actual* rather than the cosmetic zero-after-compute of the reference. Threshold is
    # min(τ, 1e-8) (the accuracy of the RI fit); None -> no screening.
    #
    # The pivot rows are provably never screened, so the metric S below is exact: each pivot's ket
    # shell pair has Q_K >= sqrt((γδ|γδ)) > sqrt(τ) (its diagonal exceeded τ when selected), and
    # likewise Q_αβ > sqrt(τ) for a pivot bra row, so Q_αβ Q_K > τ >= screen -- above the threshold.
    screen = min(tol, 1e-8) if tol is not None else None
    schwarz = (
        integrals.coulomb_4c_schwarz_factors(system, basis)
        if screen is not None
        else None
    )
    G = np.asarray(
        integrals.coulomb_4c_pair_block_screened(
            system, all_pairs, all_pairs[unique_kps], schwarz, screen, basis
        )
    )
    A = np.ascontiguousarray(G[:, pivcol].astype(float, copy=False))  # (n, npiv)

    # S_{JK} = (ρ_J | ρ_K): the pivot rows of A (Folkestad Eq. 3 metric). Exact (never screened, see
    # above) and symmetric PSD by construction, so its Cholesky factor Q with S = Q Q^T exists.
    S = A[pivots, :]
    S = 0.5 * (S + S.T)  # symmetrize away round-off before factorizing

    # Folkestad Eq. 14: L^J_αβ = Σ_K (αβ|K) Q^{-T}_{KJ}. With S = Q Q^T (lower Q), this is
    # L = Q^{-1} A^T (shape (npiv, n)); the reconstruction L^T L = A (Q^T Q)^{-1}... = A S^{-1} A^T
    # recovers Folkestad Eq. 2. A near-singular S (should not occur -- the pivots are linearly
    # independent to τ) falls back to the eigen-based S^{-1/2}, which is equally valid up to an
    # orthogonal transform of the vectors.
    try:
        Qfac = sp.linalg.cholesky(S, lower=True)
        Qinv = sp.linalg.solve_triangular(Qfac, np.eye(npiv), lower=True)
        Lvec = Qinv @ A.T
    except sp.linalg.LinAlgError:
        X, _, _ = invsqrt_matrix(S, rtol=1e-12)
        Lvec = X @ A.T

    # Scatter packed -> global AO order and reshape to the (naux, nbf, nbf) B tensor, exactly as in
    # cholesky_otf.
    B = np.zeros((npiv, n))
    B[:, perm] = Lvec
    B = B.reshape(npiv, nbf, nbf)
    return B, npiv


def cholesky_pivoted(system, tol, basis=None, *, span_factor=1e-2, max_qual=None):
    r"""
    Two-step pivoted Cholesky decomposition of the four-center ERIs (Folkestad 2019).

    Orchestrates :func:`cholesky_pivots` (step I -- determine the pivot set ``B``) and
    :func:`cholesky_vectors_ri` (step II -- build the vectors from ``B`` by RI). Produces the same
    ``(naux, nbf, nbf)`` ``B_Pmn`` tensor as the other decomposition paths, such that
    ``(mn|rs) ≈ Σ_J B[J,m,n] B[J,r,s]`` to the threshold ``τ = tol``.

    Parameters
    ----------
    system : System
        The molecular system.
    tol : float
        The decomposition threshold ``τ``.
    basis : Basis, optional
        The orbital basis. Defaults to ``system.basis``.
    span_factor : float, optional, default=1e-2
        Folkestad's span factor ``σ`` (Folkestad Eq. 8), forwarded to :func:`cholesky_pivots`.
    max_qual : int, optional
        Per-batch qualified-column budget, forwarded to :func:`cholesky_pivots`.

    Returns
    -------
    B : ndarray, shape (naux, nbf, nbf)
    naux : int
    """
    if basis is None:
        basis = system.basis
    layout = _build_shell_pair_layout(basis)

    pivots = cholesky_pivots(
        system, tol, basis, span_factor=span_factor, max_qual=max_qual, layout=layout
    )
    B, naux = cholesky_vectors_ri(system, pivots, basis, tol=tol, layout=layout)

    nbf = basis.size
    memory_gb = 8 * (naux * nbf**2) / (1024**3)
    logger.log_info1(
        "Building B tensor using two-step (Folkestad) Cholesky decomposition of the ERIs"
    )
    logger.log_info1(f"Number of system basis functions: {nbf}")
    logger.log_info1(f"Number of Cholesky vectors: {naux}")
    logger.log_info1(f"B tensor memory: {memory_gb:.2f} GB")

    return B, naux
