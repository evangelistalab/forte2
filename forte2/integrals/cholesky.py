"""
On-the-fly Cholesky decomposition (CD) of the four-center two-electron integrals.

This implements the one-step, pivoted Cholesky algorithm of Koch, Sánchez de Merás, and
Pedersen [J. Chem. Phys. 118, 9481 (2003)], producing the same ``B_Pmn`` tensor of shape
``(naux, nbf, nbf)`` that :class:`~forte2.jkbuilder.jkbuilder.FockBuilder` consumes, such that

.. math::
    (\\mu\\nu|\\rho\\sigma) \\approx \\sum_J B^J_{\\mu\\nu} B^J_{\\rho\\sigma}.

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
        Pivot threshold. The decomposition stops once the largest remaining Schur-complement
        diagonal drops to or below ``tol``. If ``tol <= 0``, a machine-precision threshold
        ``nbf**2 * eps * max(diag)`` is used (matching LAPACK ``dpstrf``'s default).
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

    # Diagonal D_p = (mn|mn), permuted from global into packed order.
    diag = np.asarray(integrals.coulomb_4c_diagonal(system, basis))[perm].copy()

    Dmax0 = float(diag.max()) if n else 0.0
    eps = np.finfo(float).eps
    stop_tol = tol if (tol is not None and tol > 0.0) else n * eps * Dmax0

    # L holds the Cholesky vectors row-wise in packed order; grown by doubling so peak memory is
    # O(naux * nbf^2), never O(nbf^4).
    cap = max(1, min(n, 4 * nbf))
    L = np.zeros((cap, n))
    pivots = []
    Q = 0

    # Single-entry cache of the most recent raw column block. The raw integrals are invariant, so
    # consecutive pivots landing in the same ket shell-pair reuse the block instead of recomputing.
    cached_kp = -1
    cached_block = None

    while Q < n:
        piv = int(np.argmax(diag))
        Dmax = float(diag[piv])
        if Dmax <= stop_tol:
            break

        # Locate the ket shell-pair holding the pivot and its column within that block. Because the
        # bra list equals the ket list in the same canonical order, the pivot's bra-row offset also
        # equals its ket-column offset within shell-pair `kp`.
        kp = int(np.searchsorted(row_off, piv, side="right") - 1)
        local = piv - int(row_off[kp])
        if kp != cached_kp:
            cached_block = np.asarray(
                integrals.coulomb_4c_pair_block(
                    system, all_pairs, all_pairs[kp : kp + 1], basis
                )
            )
            cached_kp = kp
        col = cached_block[:, local].astype(float, copy=True)  # raw M_packed[:, piv]

        # Subtract the contributions of the previously computed vectors.
        if Q > 0:
            col -= L[:Q].T @ L[:Q, piv]

        Lqq = np.sqrt(Dmax)
        col /= Lqq
        # Enforce the triangular structure: zero the already-selected pivots, set this pivot.
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

        # Update the Schur-complement diagonal; the pivot's own entry becomes exactly zero.
        diag -= col * col
        diag[piv] = 0.0

        pivots.append(piv)
        Q += 1

    if Q == 0:
        raise RuntimeError(
            "On-the-fly Cholesky produced no vectors; check the basis and cholesky_tol."
        )

    # Permute packed -> global and reshape to the (naux, nbf, nbf) B tensor.
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
