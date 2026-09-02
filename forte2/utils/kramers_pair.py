import numpy as np


def split_spinor(Cp):
    """
    Split a two-component spinor into alpha and beta AO blocks.

    Parameters
    ----------
    Cp : ndarray, shape (2 * nao,)
        One spinor coefficient vector ordered as [alpha, beta].

    Returns
    -------
    Ca, Cb : ndarray, shape (nao,)
        Alpha and beta AO coefficient blocks.
    """
    Cp = np.asarray(Cp)

    if Cp.ndim != 1:
        raise ValueError("Cp must be a one-dimensional spinor vector.")

    if Cp.size % 2 != 0:
        raise ValueError("Spinor length must be even.")

    nao = Cp.size // 2
    return Cp[:nao], Cp[nao:]

def time_reverse_spinor(C):
    """
    Apply time reversal operator.

    C:
        (2*NAO, nspinor)

    Forte2 GHF ordering:
        alpha block
        beta block
    """

    nao2, nmo = C.shape
    nao = nao2 // 2

    Ca = C[:nao, :]
    Cb = C[nao:, :]

    C_theta = np.vstack([
        -Cb.conj(),
        Ca.conj()
    ])

    return C_theta

def spinor_inner(x, y, S):
    r"""
    Compute the two-component AO-metric inner product:

        <x|y> = xa^\dagger S ya + xb^\dagger S yb
    """
    xa, xb = split_spinor(x)
    ya, yb = split_spinor(y)

    S = np.asarray(S)

    if S.shape != (xa.size, xa.size):
        raise ValueError("S has an incompatible shape.")

    return (
        np.vdot(xa, S @ ya)
        + np.vdot(xb, S @ yb)
    )


def s_orthonormality_error(C, S):
    C = np.asarray(C)
    S_spin = S
    nmo = C.shape[1]

    gram = C.conj().T @ S_spin @ C

    error = (
        np.linalg.norm(
            gram - np.eye(nmo),
            ord="fro",
        )
        / np.sqrt(nmo)
    )

    return error

def kramers_matrix(C, S):
    """
    Compute Kramers score matrix.

    K[p,q] = | <p | Theta q> |

    C:
        (2*nao,nmo)

    S:
        (nao,nao)

    """
    S_spin = S

    C_theta = time_reverse_spinor(C)

    K = np.abs(
        C.conj().T @ S_spin @ C_theta
    )

    return K

## Kramer's amplitude matrix 
def kramers_amplitude_matrix(C, S):
    C = np.asarray(C)

    S_spin = S
    if C.shape[0] != S_spin.shape[0]:
        raise ValueError(
            "C and S have incompatible AO dimensions."
        )

    C_theta = time_reverse_spinor(C)

    A = C.conj().T @ S_spin @ C_theta

    return A


## Have two functions for finding kramers pairs, one that returns the scores and one that doesn't. 
# The one that returns the scores is used in the mutual correlation analysis, while the one that doesn't is used in the kramers pair analysis.
def find_kramers_pairs(C, S, threshold=0.9):

    K = kramers_matrix(C, S)

    nmo = C.shape[1]

    pairs = []

    for p in range(nmo):

        # avoid self
        scores = K[:, p].copy()
        scores[p] = 0.0

        q = int(np.argmax(scores))
        if p < q:
            pairs.append(
                (
                    p,
                    q,
                    scores[q],
                    scores[q] > threshold
                )
            )
            

    return pairs, K

def pure_find_kramers_pairs(C, S, threshold=0.9):

    K = kramers_matrix(C, S)

    nmo = C.shape[1]

    pairs = []

    for p in range(nmo):

        # avoid self
        scores = K[:, p].copy()
        scores[p] = 0.0

        q = int(np.argmax(scores))
        if p < q:
            pairs.append(
                (
                    p,
                    q,
                )
            )
            

    return pairs, K

def find_degenerate_blocks(
    orbital_energies,
    atol=1.0e-8,
    rtol=1.0e-7,
):
    eps = np.asarray(
        orbital_energies,
        dtype=float,
    ).reshape(-1)

    if not np.all(np.isfinite(eps)):
        raise ValueError(
            "orbital_energies contains a non-finite value."
        )

    order = np.argsort(eps, kind="stable")

    blocks = []
    current = [int(order[0])] if order.size else []

    for idx_ in order[1:]:
        idx = int(idx_)

        reference_energy = eps[current[0]]

        tolerance = (
            atol
            + rtol * max(
                1.0,
                abs(reference_energy),
                abs(eps[idx]),
            )
        )

        if abs(eps[idx] - reference_energy) <= tolerance:
            current.append(idx)

        else:
            if len(current) >= 2:
                blocks.append(
                    np.asarray(current, dtype=int)
                )

            current = [idx]

    if len(current) >= 2:
        blocks.append(
            np.asarray(current, dtype=int)
        )

    return blocks

def canonical_kramers_block(n):
    if n <= 0 or n % 2 != 0:
        raise ValueError(
            "A Kramers block must have positive even dimension."
        )

    J = np.zeros((n, n), dtype=complex)

    for k in range(0, n, 2):
        J[k, k + 1] = -1.0
        J[k + 1, k] = 1.0

    return J

def analyze_degenerate_blocks(
    A,
    orbital_energies,
    blocks,
    closure_tol=1.0e-6,
    pair_error_tol=1.0e-6,
):
    A = np.asarray(A)

    eps = np.asarray(
        orbital_energies,
        dtype=float,
    ).reshape(-1)

    reports = []

    for block in blocks:
        block = np.asarray(block, dtype=int)

        n = block.size

        A_B = A[np.ix_(block, block)]
        K_B = np.abs(A_B)

        even = (n % 2 == 0)

        unitarity_error = (
            np.linalg.norm(
                A_B.conj().T @ A_B - np.eye(n),
                ord="fro",
            )
            / np.sqrt(n)
        )

        skew_error = (
            np.linalg.norm(
                A_B.T + A_B,
                ord="fro",
            )
            / np.sqrt(n)
        )

        closure_weight = np.sum(
            K_B**2,
            axis=0,
        )

        if even:
            target = canonical_kramers_block(n)

            score_target_error = (
                np.linalg.norm(
                    K_B - np.abs(target),
                    ord="fro",
                )
                / np.sqrt(n)
            )

        else:
            score_target_error = np.inf

        closed = (
            even
            and unitarity_error <= closure_tol
            and skew_error <= closure_tol
        )

        already_adjacent = (
            closed
            and score_target_error <= pair_error_tol
        )

        reports.append({
            "indices": block,
            "size": n,
            "energy_min": float(np.min(eps[block])),
            "energy_max": float(np.max(eps[block])),
            "energy_spread": float(np.ptp(eps[block])),
            "even": even,
            "closed_under_time_reversal": closed,
            "already_adjacent": already_adjacent,
            "needs_rotation": (
                closed and not already_adjacent
            ),
            "unitarity_error": float(unitarity_error),
            "skew_error": float(skew_error),
            "score_target_error_before": float(
                score_target_error
            ),
            "closure_weight_per_orbital": closure_weight,
            "K_before": K_B,
        })

    return reports

def _orthogonal_seed(
    U_used,
    n,
    tol=1.0e-12,
):
    eye = np.eye(n, dtype=complex)

    best_v = None
    best_norm = -1.0

    for j in range(n):
        v = eye[:, j].copy()

        if U_used.shape[1]:
            v -= (
                U_used
                @ (U_used.conj().T @ v)
            )

        norm_v = np.linalg.norm(v)

        if norm_v > best_norm:
            best_v = v
            best_norm = norm_v

    if best_norm <= tol:
        raise np.linalg.LinAlgError(
            "Could not find an orthogonal seed vector."
        )

    return best_v / best_norm

def kramers_pair_rotation(
    A_B,
    structure_tol=1.0e-6,
    orthogonalization_tol=1.0e-12,
):
    A_B = np.asarray(
        A_B,
        dtype=complex,
    )

    if (
        A_B.ndim != 2
        or A_B.shape[0] != A_B.shape[1]
    ):
        raise ValueError("A_B must be square.")

    n = A_B.shape[0]

    if n == 0 or n % 2 != 0:
        raise ValueError(
            "A_B must have positive even dimension."
        )

    skew_error = (
        np.linalg.norm(
            A_B.T + A_B,
            ord="fro",
        )
        / np.sqrt(n)
    )

    unitary_error = (
        np.linalg.norm(
            A_B.conj().T @ A_B - np.eye(n),
            ord="fro",
        )
        / np.sqrt(n)
    )

    if (
        skew_error > structure_tol
        or unitary_error > structure_tol
    ):
        raise ValueError(
            "The block is not sufficiently antisymmetric "
            "and unitary: "
            f"skew_error={skew_error:.3e}, "
            f"unitary_error={unitary_error:.3e}."
        )

    # Remove tiny numerical violation of A_B^T = -A_B.
    A_work = 0.5 * (A_B - A_B.T)

    U_B = np.empty(
        (n, 0),
        dtype=complex,
    )

    for _ in range(n // 2):
        v = _orthogonal_seed(
            U_B,
            n,
            orthogonalization_tol,
        )

        # Coefficient vector of Theta(C_B v).
        w = A_work @ v.conj()

        # Numerical reorthogonalization.
        if U_B.shape[1]:
            w -= (
                U_B
                @ (U_B.conj().T @ w)
            )

        w -= v * np.vdot(v, w)

        norm_w = np.linalg.norm(w)

        if norm_w <= orthogonalization_tol:
            raise np.linalg.LinAlgError(
                "The Kramers partner became "
                "linearly dependent."
            )

        w /= norm_w

        U_B = np.column_stack([
            U_B,
            v,
            w,
        ])

    A_rot = (
        U_B.conj().T
        @ A_B
        @ U_B.conj()
    )

    target = canonical_kramers_block(n)

    target_error = (
        np.linalg.norm(
            A_rot - target,
            ord="fro",
        )
        / np.sqrt(n)
    )

    rotation_unitarity_error = (
        np.linalg.norm(
            U_B.conj().T @ U_B - np.eye(n),
            ord="fro",
        )
        / np.sqrt(n)
    )

    if target_error > 10.0 * structure_tol:
        raise np.linalg.LinAlgError(
            "Rotation did not reach the canonical "
            "Kramers form; "
            f"error={target_error:.3e}."
        )

    info = {
        "target_error": float(target_error),
        "rotation_unitarity_error": float(
            rotation_unitarity_error
        ),
    }

    return U_B, A_rot, info

def rotate_degenerate_kramers_blocks(
    C,
    S,
    orbital_energies,
    energy_atol=1.0e-8,
    energy_rtol=1.0e-7,
    closure_tol=1.0e-6,
    pair_error_tol=1.0e-6,
):
    C = np.asarray(C)

    eps = np.asarray(
        orbital_energies,
        dtype=float,
    ).reshape(-1)

    if C.ndim != 2 or C.shape[1] != eps.size:
        raise ValueError(
            "len(orbital_energies) "
            "must equal C.shape[1]."
        )

    A_before = kramers_amplitude_matrix(C, S)

    blocks = find_degenerate_blocks(
        eps,
        energy_atol,
        energy_rtol,
    )

    reports = analyze_degenerate_blocks(
        A_before,
        eps,
        blocks,
        closure_tol,
        pair_error_tol,
    )

    C_rot = np.array(
        C,
        dtype=np.result_type(C.dtype, complex),
        copy=True,
    )

    U_full = np.eye(
        C.shape[1],
        dtype=complex,
    )

    for report in reports:
        block = report["indices"]
        n = report["size"]

        if not report["even"]:
            report["status"] = (
                "skipped: odd-dimensional energy block"
            )
            continue

        if not report["closed_under_time_reversal"]:
            report["status"] = (
                "skipped: block is not "
                "time-reversal closed"
            )
            continue

        if not report["needs_rotation"]:
            report["status"] = (
                "unchanged: already adjacent "
                "in the score matrix"
            )

            report["score_target_error_after"] = (
                report["score_target_error_before"]
            )

            report["fock_offdiagonal_error_after"] = 0.0

            continue

        A_B = A_before[np.ix_(block, block)]

        U_B, _, rotation_info = (
            kramers_pair_rotation(
                A_B,
                structure_tol=closure_tol,
            )
        )

        # The important rectangular-times-square rotation.
        C_rot[:, block] = C[:, block] @ U_B

        U_full[np.ix_(block, block)] = U_B

        # Check preservation of the semi-canonical Fock block.
        F_B_before = np.diag(eps[block])

        F_B_after = (
            U_B.conj().T
            @ F_B_before
            @ U_B
        )

        F_B_offdiag = (
            F_B_after
            - np.diag(np.diag(F_B_after))
        )

        report["status"] = "rotated"
        report["U_B"] = U_B
        report.update(rotation_info)

        report["fock_offdiagonal_error_after"] = float(
            np.linalg.norm(
                F_B_offdiag,
                ord="fro",
            )
            / np.sqrt(n)
        )

    A_after = kramers_amplitude_matrix(
        C_rot,
        S,
    )

    for report in reports:
        block = report["indices"]

        if report.get("status") == "rotated":
            A_B_after = A_after[
                np.ix_(block, block)
            ]

            K_B_after = np.abs(A_B_after)

            target = canonical_kramers_block(
                report["size"]
            )

            report["A_after"] = A_B_after
            report["K_after"] = K_B_after

            report["score_target_error_after"] = float(
                np.linalg.norm(
                    K_B_after - np.abs(target),
                    ord="fro",
                )
                / np.sqrt(report["size"])
            )

    result = {
        "C_rot": C_rot,
        "U": U_full,
        "A_before": A_before,
        "K_before": np.abs(A_before),
        "A_after": A_after,
        "K_after": np.abs(A_after),
        "blocks": reports,
        "orthonormality_error_before":
            s_orthonormality_error(C, S),
        "orthonormality_error_after":
            s_orthonormality_error(C_rot, S),
    }

    return result


