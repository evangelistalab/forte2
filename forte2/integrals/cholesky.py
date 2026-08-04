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
"""

import numpy as np

from forte2 import integrals
from forte2.helpers import logger


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
