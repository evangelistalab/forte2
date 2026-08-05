"""
Frozen reference (oracle) implementations of the on-the-fly Cholesky decompositions.

.. warning::
   This module is a **frozen numerical oracle**. It preserves, verbatim, the Phase 1/2 reference
   implementations of the Koch one-step (:func:`cholesky_otf_reference`) and Folkestad two-step
   (:func:`cholesky_pivots_reference`, :func:`cholesky_vectors_ri_reference`,
   :func:`cholesky_pivoted_reference`) drivers, each first validated against the dense LAPACK
   ``dpstrf`` path. The production drivers in :mod:`forte2.integrals.cholesky` are diffed against
   these functions by ``tests/integrals/test_cholesky_reference_oracle.py``. Do **not** optimize or
   otherwise change the behaviour of anything here -- edit the production module instead. It is kept
   self-contained (its own copy of the shared helpers) so production changes can never perturb it.

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
:func:`cholesky_pivoted_reference` implements the two-step algorithm of Folkestad, Kjønstad, and Koch
[J. Chem. Phys. 150, 194112 (2019)], routed through ``cholesky_tei="pivoted"``. Its central idea
(Folkestad Sec. II) is to split the decomposition into

* **Step I** (:func:`cholesky_pivots_reference`) -- determine *only* the pivot indices ``B = {J}``, the
  "Cholesky basis". Because the Cholesky vectors are not the deliverable of this step, the diagonal
  decreases monotonically (Folkestad Eq. 12) and "a diagonal ``D_p = M_pp`` below ``τ`` will never
  be selected as a pivot", so we may screen **both rows and columns** of ``M`` by the same
  threshold ``τ`` -- a stronger screening than Koch's one-step algorithm, which builds the vectors
  during pivoting and can therefore only screen rows via Cauchy-Schwarz.
* **Step II** (:func:`cholesky_vectors_ri_reference`) -- construct the Cholesky vectors from ``B`` using the
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


def cholesky_otf_reference(system, tol, basis=None):
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
    """
    if basis is None:
        basis = system.basis
    nbf = basis.size
    n = nbf * nbf

    all_pairs, row_off, perm = _build_shell_pair_layout(basis)

    # Koch step 1: "Initially we calculate the diagonal elements M_pp = (ab|ab)." Here p = (m, n)
    # is a compound AO index, so diag[p] = (mn|mn). We hold it in packed shell-pair order (the
    # coulomb_4c_diagonal primitive returns global order, hence the [perm] gather) so it aligns
    # row-for-row with the pair-block columns pulled later.
    diag = np.asarray(integrals.coulomb_4c_diagonal(system, basis))[perm].copy()

    # Dmax0 is Koch's X_max, the maximum diagonal element used both to set the screening scale and
    # (below) to bound the Schur complement via Cauchy-Schwarz. stop_tol is the accuracy D at which
    # "further improvements of the accuracy beyond D are not possible": we stop pivoting once the
    # largest remaining diagonal falls to or below it. (Koch additionally prescreens the initial
    # diagonal, zeroing M_pp < D^2 / X_max; here that is subsumed into the running argmax/stop test,
    # which never selects such a column because its diagonal is already below D.)
    Dmax0 = float(diag.max()) if n else 0.0
    eps = np.finfo(float).eps
    stop_tol = tol if (tol is not None and tol > 0.0) else n * eps * Dmax0

    # L holds the Cholesky vectors L^J row-wise in packed order (row Q == vector J of Eq. (1)/(3)).
    # Koch's central result is that the vector count M stays small -- "the number of elements needed
    # to be stored scale as N^2 much less than the potential N^4 number of raw two-electron
    # integrals" -- so we size for O(M * nbf^2) and grow by doubling rather than ever allocating the
    # N^4 matrix. The initial guess 4*nbf reflects the observed M ~ (a few)*N (Koch's Tables I/III
    # report M/N ratios of roughly 9-13).
    cap = max(1, min(n, 4 * nbf))
    L = np.zeros((cap, n))
    pivots = []
    Q = 0

    # Single-entry cache of the most recent (**|AB) integral distribution. Koch notes that pure full
    # pivoting would discard all but one column of each computed shell pair and force "a prohibitively
    # large number of integral recalculations"; his remedy is to proactively "decompose the remaining
    # integrals in the shell pair" (all diagonals > X_max/1000) while AB is loaded. We take a simpler
    # reactive variant: keep the standard global argmax pivot, but cache the block so consecutive
    # pivots landing in the same ket shell pair reuse it. A block is thus recomputed once per
    # *contiguous* run of pivots in AB -- cheaper than per-pivot, but (unlike Koch's proactive
    # draining) a shell pair can still be recomputed if pivots interleave across shell pairs.
    cached_kp = -1
    cached_block = None

    while Q < n:
        # Koch: "we find the largest diagonal element" -- this is the pivot J of Eq. (3), with
        # Dmax = M_JJ (the current Schur-complement diagonal at the pivot). This complete-pivoting
        # choice is what keeps the semidefinite decomposition stable.
        piv = int(np.argmax(diag))
        Dmax = float(diag[piv])
        # Stop when the largest remaining diagonal is below the accuracy D. By Cauchy-Schwarz,
        # Eq. (7), |M_pq| <= sqrt(M_pp * M_qq) <= sqrt(M_pp * X_max), so once the max diagonal is
        # below D no remaining matrix element can exceed the target accuracy.
        if Dmax <= stop_tol:
            break

        # Locate the ket shell-pair holding the pivot and its column within that block. Because the
        # bra list equals the ket list in the same canonical order, the pivot's bra-row offset also
        # equals its ket-column offset within shell-pair `kp`.
        kp = int(np.searchsorted(row_off, piv, side="right") - 1)
        local = piv - int(row_off[kp])
        if kp != cached_kp:
            # Compute Koch's (**|AB) distribution: every bra AO pair against the single ket shell
            # pair AB = all_pairs[kp]. cached_block[:, c] is the raw ERI column M[:, q] for the AO
            # pair q sitting at column c of AB.
            cached_block = np.asarray(
                integrals.coulomb_4c_pair_block(
                    system, all_pairs, all_pairs[kp : kp + 1], basis
                )
            )
            cached_kp = kp
        # The pivot's raw column M_pJ (the first factor of Eq. (3) before the Schur subtraction).
        # Copied because it is updated in place below while cached_block must stay pristine for the
        # other pivots in this shell pair.
        col = cached_block[:, local].astype(float, copy=True)  # raw M[:, piv]

        # Form the current Schur complement column M~_pJ = M_pJ - sum_{K<J} L^K_p L^K_J, i.e. the
        # subtracted term of Eq. (3) accumulated over all previously computed vectors. L[:Q, piv]
        # is the piv-th entry of each earlier vector (the L^K_J factors); L[:Q].T @ ... contracts
        # them against the full earlier vectors L^K_p.
        if Q > 0:
            col -= L[:Q].T @ L[:Q, piv]

        # Define the new Cholesky vector by dividing the updated column by sqrt(M_JJ): this is
        # exactly Eq. (3)'s L^J_p = M~_pJ / M_JJ^{1/2}, the implicitly defined vector. Dmax is the
        # pivot's Schur-complement diagonal M_JJ (equal to M~_JJ at this step).
        Lqq = np.sqrt(Dmax)
        col /= Lqq
        # Enforce exact triangular structure that Eq. (3) implies but finite-precision arithmetic
        # only approximates: previously selected pivots q have M~_qJ == 0 (their diagonals were
        # driven to zero), and the current pivot's own entry is L^J_J = sqrt(M_JJ) = Lqq.
        if pivots:
            col[pivots] = 0.0
        col[piv] = Lqq

        if Q == cap:
            new_cap = min(cap * 2, n)
            new_L = np.zeros((new_cap, n))
            new_L[:cap] = L
            L = new_L
            cap = new_cap
        L[Q] = col

        # Update the Schur-complement diagonal by the diagonal case p == q of Eq. (3):
        # M~_pp = M_pp - (L^J_p)^2. This is what lets the next iteration's argmax pick the largest
        # *remaining* diagonal (Koch's running X~_max in Eq. (7)) without touching off-diagonals.
        # The pivot's own diagonal is set to exactly zero (Eq. (3) gives M~_JJ = M_JJ - L^J_J^2 = 0)
        # so it is never reselected.
        diag -= col * col
        diag[piv] = 0.0

        pivots.append(piv)
        Q += 1

    if Q == 0:
        raise RuntimeError(
            "On-the-fly Cholesky produced no vectors; check the basis and cholesky_tol."
        )

    # Scatter each vector's packed columns back to global AO order and reshape to (naux, nbf, nbf).
    # The Q rows are Koch's M vectors L^J; after this the tensor satisfies Eq. (1),
    # (mn|rs) = sum_J B[J, m, n] B[J, r, s], to accuracy D. Reordering AO pairs is a symmetric
    # permutation of M, which leaves the Cholesky reconstruction valid.
    B = np.zeros((Q, n))
    B[:, perm] = L[:Q]
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
# Everything below is independent of ``cholesky_otf_reference`` above (the Koch one-step
# driver); it is routed through ``cholesky_tei="pivoted"``.
# ---------------------------------------------------------------------------


def _resolve_stop_tol(tol, diag):
    """Return the effective threshold ``τ`` and the maximum diagonal ``X_max``.

    Mirrors :func:`cholesky_otf_reference`: a non-positive ``tol`` falls back to LAPACK ``dpstrf``'s default
    machine-precision threshold ``n * eps * max(diag)``. Folkestad calls this threshold ``τ``
    (Folkestad Sec. II: "These steps are repeated until all diagonal elements of M are below a given
    threshold τ > 0").
    """
    n = diag.size
    Dmax0 = float(diag.max()) if n else 0.0
    eps = np.finfo(float).eps
    stop_tol = tol if (tol is not None and tol > 0.0) else n * eps * Dmax0
    return stop_tol, Dmax0


def cholesky_pivots_reference(
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
        machine-precision fallback, as in :func:`cholesky_otf_reference`.
    basis : Basis, optional
        The orbital basis. Defaults to ``system.basis``.
    span_factor : float, optional, default=1e-2
        Folkestad's span factor ``σ`` (Folkestad Eq. 8), "which ensures that qualified diagonals are
        not too small". The default ``σ = 10^-2`` is the value proposed in Folkestad (Sec. II, "we
        use σ = 10^-2 as proposed in Ref. 12").
    max_qual : int, optional
        Upper bound on the number of qualified AO-pair columns computed per batch (Folkestad Eq. 8:
        "such that the number of elements in Q does not exceed a user-specified maximum"). ``None``
        imposes no cap -- every qualified shell pair is taken in one batch, which is fine for the
        reference-scale systems this path targets.
    layout : tuple, optional
        A precomputed ``_build_shell_pair_layout(basis)`` triple, to avoid recomputation when this
        is called from :func:`cholesky_pivoted_reference`.

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

    Deviations from Folkestad, all benign for a small-system reference decomposition:

    * We keep the transient vectors as full-length ``(n,)`` rows rather than restricting them to the
      significant set ``D`` (Folkestad Fig. 1 and the "memory ... drops to zero" remark). This costs
      ``O(rank * n)`` scratch instead of the paper's shrinking footprint.
    * Qualification is done purely at shell-pair granularity (Folkestad Sec. II: "we modify the
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

    # Initial diagonal D_p = (mn|mn), in packed order (Folkestad step 1's D0; identical to Koch's
    # M_pp). We mutate a working copy in place as vectors are subtracted (Folkestad Eq. 12).
    diag0 = np.asarray(integrals.coulomb_4c_diagonal(system, basis))[perm].copy()
    stop_tol, _ = _resolve_stop_tol(tol, diag0)
    diag = diag0.copy()

    # Start of each shell pair's packed AO-pair range, for the per-shell-pair max reduction.
    sp_starts = row_off[:-1].astype(np.intp)

    # Transient Cholesky vectors, kept only to update D and subtract prior contributions; grown by
    # doubling exactly like cholesky_otf_reference. Discarded before returning (only ``pivots`` survives).
    cap = max(1, min(n, 4 * nbf))
    L = np.zeros((cap, n))
    pivots = []
    Q = 0

    def _grow(Q, cap, L):
        if Q == cap:
            new_cap = min(cap * 2, n)
            new_L = np.zeros((new_cap, n))
            new_L[:cap] = L
            return new_cap, new_L
        return cap, L

    # Outer loop == Folkestad steps 3-7: qualify a batch, decompose it, append its pivots to B,
    # repeat until no significant diagonal remains.
    while True:
        Dmax = float(diag.max()) if n else 0.0
        # Folkestad step 3 / Eq. 17: the standard significance criterion D = {p : D_p >= τ}. Once
        # the largest remaining diagonal is below τ, by Cauchy-Schwarz (Folkestad Eq. 6) every
        # remaining matrix element is below τ and the basis B is complete.
        if Dmax <= stop_tol:
            break

        # Folkestad step 4 / Eq. 16: order shell pairs by their maximal diagonal D^{AB}_max and, per
        # Eq. 8, qualify those reaching σ * Dmax. Qualifying whole shell pairs (not individual AO
        # pairs) matches Libint's shell-quartet evaluation -- the same reason Folkestad "treat[s]
        # shell pairs instead of AO pairs".
        sp_max = np.maximum.reduceat(diag, sp_starts)
        sp_max[row_off[1:] == row_off[:-1]] = (
            -np.inf
        )  # guard against any empty shell pair
        qual_thresh = span_factor * Dmax
        qualified = np.where(sp_max >= qual_thresh)[0]
        # Take qualified shell pairs largest-D^{AB}_max first (Folkestad: "qualified from the AB with
        # the largest diagonal before the next shell pair ... is considered").
        qualified = qualified[np.argsort(sp_max[qualified])[::-1]]

        # Assemble the qualified ket shell pairs subject to the optional column budget max_qual.
        ket_kps = []
        ncol = 0
        for kp in qualified:
            size_kp = int(row_off[kp + 1] - row_off[kp])
            if max_qual is not None and ket_kps and ncol + size_kp > max_qual:
                break
            ket_kps.append(int(kp))
            ncol += size_kp
        ket_kps = np.asarray(ket_kps, dtype=np.int64)

        # Packed indices of every qualified column, aligned 1:1 with the columns of the block below
        # (the block concatenates each ket shell pair's contiguous packed range in ket_kps order).
        qcols = np.concatenate(
            [np.arange(row_off[kp], row_off[kp + 1]) for kp in ket_kps]
        ).astype(np.int64)

        # Folkestad step 5 / Eq. 9: compute M_pq for all rows p and qualified columns q, then
        # subtract the contributions of the pivots already in B. colmat holds ~M_pq (only the
        # already-selected batches subtracted); the within-batch subtraction is Eq. 10 below.
        colmat = np.asarray(
            integrals.coulomb_4c_pair_block(
                system, all_pairs, all_pairs[ket_kps], basis
            )
        ).astype(float, copy=True)
        if Q > 0:
            colmat -= L[:Q].T @ L[:Q][:, qcols]

        # Folkestad step 6 / Eqs. 10-12: greedily decompose the qualified block. C is the set of
        # pivots built in this batch; batch_start marks where they begin in L so Eq. 10's Σ_{J∈C}
        # subtraction uses exactly the this-batch vectors.
        batch_start = Q
        while True:
            # "select q ∈ Q such that D_q = max_{p∈Q} D_p" (Folkestad step 6). The global argmax
            # pivot always lives in a qualified shell pair, so progress is guaranteed each batch.
            jpos = int(np.argmax(diag[qcols]))
            q = int(qcols[jpos])
            Dq = float(diag[q])
            if Dq <= stop_tol:  # no significant diagonal left in Q -> back to step 3
                break

            # Folkestad Eq. 10: L^q_p = (~M_pq - Σ_{J∈C} L^J_p L^J_q) / sqrt(~M_qq).
            newvec = colmat[:, jpos].copy()
            if Q > batch_start:
                newvec -= L[batch_start:Q].T @ L[batch_start:Q, q]
            Lqq = np.sqrt(Dq)
            newvec /= Lqq
            # Exact triangular structure (finite precision only approximates it): earlier pivots
            # have a zero entry, and this pivot's own entry is L^q_q = sqrt(~M_qq).
            if pivots:
                newvec[pivots] = 0.0
            newvec[q] = Lqq

            cap, L = _grow(Q, cap, L)
            L[Q] = newvec

            # Folkestad Eq. 12: D_p ← D_p - (L^q_p)^2. Setting D_q = 0 removes q from Q (Eq. 11) and
            # from all future qualification, so it is never reselected.
            diag -= newvec * newvec
            diag[q] = 0.0

            pivots.append(q)
            Q += 1

        # Folkestad step 7 / Eq. 13: B ← B ∪ C, then return to step 3.

    if Q == 0:
        raise RuntimeError(
            "Two-step Cholesky (step I) produced no pivots; check the basis and cholesky_tol."
        )

    return np.asarray(pivots, dtype=np.int64)


def cholesky_vectors_ri_reference(system, pivots, basis=None, *, tol=None, layout=None):
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
        The pivot set ``B`` as packed AO-pair indices, e.g. from :func:`cholesky_pivots_reference`.
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

    # A_{αβ,K} = (αβ|K): rows = all packed AO pairs αβ, columns = the pivot product functions K.
    G = np.asarray(
        integrals.coulomb_4c_pair_block(system, all_pairs, all_pairs[unique_kps], basis)
    )
    A = np.ascontiguousarray(G[:, pivcol].astype(float, copy=False))  # (n, npiv)

    # S_{JK} = (ρ_J | ρ_K): the pivot rows of A (Folkestad Eq. 3 metric). Symmetric PSD by
    # construction, so its Cholesky factor Q with S = Q Q^T exists.
    S = A[pivots, :]
    S = 0.5 * (S + S.T)  # symmetrize away round-off before factorizing

    # Folkestad Eq. 15: Cauchy-Schwarz screening (αβ|K)^2 <= (αβ|αβ) · max_γδ D_γδ <=
    # (min(τ, 1e-8))^2. Rows αβ failing this contribute nothing to any vector, so we zero them.
    if tol is not None:
        diag0 = np.asarray(integrals.coulomb_4c_diagonal(system, basis))[perm]
        Dmax0 = float(diag0.max()) if n else 0.0
        screen = min(tol, 1e-8)
        if Dmax0 > 0.0 and screen > 0.0:
            keep = diag0 * Dmax0 > screen * screen
            A[~keep, :] = 0.0

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
    # cholesky_otf_reference.
    B = np.zeros((npiv, n))
    B[:, perm] = Lvec
    B = B.reshape(npiv, nbf, nbf)
    return B, npiv


def cholesky_pivoted_reference(
    system, tol, basis=None, *, span_factor=1e-2, max_qual=None
):
    r"""
    Two-step pivoted Cholesky decomposition of the four-center ERIs (Folkestad 2019).

    Orchestrates :func:`cholesky_pivots_reference` (step I -- determine the pivot set ``B``) and
    :func:`cholesky_vectors_ri_reference` (step II -- build the vectors from ``B`` by RI). Produces the same
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
        Folkestad's span factor ``σ`` (Folkestad Eq. 8), forwarded to :func:`cholesky_pivots_reference`.
    max_qual : int, optional
        Per-batch qualified-column budget, forwarded to :func:`cholesky_pivots_reference`.

    Returns
    -------
    B : ndarray, shape (naux, nbf, nbf)
    naux : int
    """
    if basis is None:
        basis = system.basis
    layout = _build_shell_pair_layout(basis)

    pivots = cholesky_pivots_reference(
        system, tol, basis, span_factor=span_factor, max_qual=max_qual, layout=layout
    )
    B, naux = cholesky_vectors_ri_reference(
        system, pivots, basis, tol=tol, layout=layout
    )

    nbf = basis.size
    memory_gb = 8 * (naux * nbf**2) / (1024**3)
    logger.log_info1(
        "Building B tensor using two-step (Folkestad) Cholesky decomposition of the ERIs"
    )
    logger.log_info1(f"Number of system basis functions: {nbf}")
    logger.log_info1(f"Number of Cholesky vectors: {naux}")
    logger.log_info1(f"B tensor memory: {memory_gb:.2f} GB")

    return B, naux
