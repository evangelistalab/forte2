import numpy as np
import scipy as sp

import forte2.integrals as integrals
from forte2.helpers.matrix_functions import block_diag_2x2


def mo_overlap(C_a, system_a, C_b, system_b=None):
    r"""
    Overlap between two sets of MO(-like) coefficients, :math:`C_a^\dagger S C_b`.

    If `system_a` is two-component (spinor-basis, e.g. GHF/X2C), the AO
    overlap is expanded to the corresponding spin-doubled block-diagonal
    form to match `C_a`'s/`C_b`'s spinor row dimension.

    Parameters
    ----------
    C_a : NDArray
        Coefficients in the AO basis of `system_a`, shape ``(nbf_a, n_a)``.
    system_a : System
        The system whose AO basis `C_a` is expressed in.
    C_b : NDArray
        Coefficients in the AO basis of `system_b`, shape ``(nbf_b, n_b)``.
    system_b : System, optional
        The system whose AO basis `C_b` is expressed in. If None, `C_b` is
        assumed to be in the same AO basis as `C_a` (`system_a`'s).

    Returns
    -------
    NDArray
        The overlap matrix, shape ``(n_a, n_b)``.
    """
    if system_b is None:
        S = integrals.overlap(system_a)
    else:
        S = integrals.overlap(system_a, system_a.basis, system_b.basis)
    if system_a.two_component:
        S = block_diag_2x2(S)
    return C_a.T.conj() @ S @ C_b


def project_orbitals(C_source, system_source, system_target, nocc):
    r"""
    Project `nocc` occupied orbitals from `system_source`'s basis into
    `system_target`'s basis, completing the result to a full orthonormal MO
    coefficient matrix.

    The occupied block is built from the cross-basis overlap,

    ``Q_occ = X_target^H S(target, source) C_occ_source``,

    where ``X_target`` is the canonical orthogonalizer for `system_target`'s AO
    basis. The projected occupied subspace is orthonormalized, then completed
    with an orthonormal virtual complement, so the result is a valid full MO
    guess.

    Parameters
    ----------
    C_source : NDArray
        Source MO coefficients, shape ``(nbf_source, n_source)``.
    system_source : System
        The system whose AO basis `C_source` is expressed in.
    system_target : System
        The system whose AO basis the projected orbitals should be expressed in.
    nocc : int
        Number of occupied orbitals to project from `C_source`.

    Returns
    -------
    NDArray | None
        Projected coefficients in `system_target`'s AO basis, shape
        ``(nbf_target, nmo_target)``, or None if the projection is numerically
        singular (occupied subspace not resolvable, or too small a virtual
        complement).
    """
    X_target = system_target.get_Xorth()
    if nocc == 0:
        return X_target.copy()
    if nocc > C_source.shape[1] or nocc > X_target.shape[1]:
        return None

    Q_occ_raw = mo_overlap(X_target, system_target, C_source[:, :nocc], system_source)
    svals = np.linalg.svd(Q_occ_raw, compute_uv=False)
    if len(svals) < nocc or svals[-1] < 1.0e-8:
        return None

    Q_occ, _ = np.linalg.qr(Q_occ_raw, mode="reduced")
    Q_occ = Q_occ[:, :nocc]

    nvirt = X_target.shape[1] - nocc
    if nvirt > 0:
        Q_virt = sp.linalg.null_space(Q_occ.T.conj())
        if Q_virt.shape[1] < nvirt:
            return None
        Q = np.hstack((Q_occ, Q_virt[:, :nvirt]))
    else:
        Q = Q_occ

    return X_target @ Q


def project_occupied_orbitals(source_method, method):
    """
    Project occupied orbitals from ``source_method`` into ``method.system``.

    The projection uses the cross-overlap between the new and old AO bases:

    ``Q_occ = X_new^T S(new, old) C_occ_old``,

    where ``X_new`` is the canonical orthogonalizer for the new AO basis. The
    projected occupied subspace is orthonormalized, then completed with an
    orthonormal virtual complement so the SCF object receives a full MO guess.

    Parameters
    ----------
    source_method : object
        A converged method whose orbitals are used as the source.
    method : object
        The method whose system defines the target AO basis.

    Returns
    -------
    list[NDArray] | None
        The projected MO coefficients, or None if the projection does not apply
        (two-component systems, mismatched basis sets, unsupported references).
    """
    if not _can_project_orbitals(source_method, method):
        return None

    source_C = source_method.mos.C
    occupied_counts = _occupied_counts(method)
    if occupied_counts is None or len(occupied_counts) != len(source_C):
        return None

    projected = []
    for C_old, nocc in zip(source_C, occupied_counts):
        C_new = project_orbitals(C_old, source_method.system, method.system, nocc)
        if C_new is None:
            return None
        projected.append(C_new)

    return projected


def _can_project_orbitals(source_method, method):
    if getattr(source_method, "mos", None) is None:
        return False
    if getattr(source_method.system, "two_component", False):
        return False
    if getattr(method.system, "two_component", False):
        return False
    if len(source_method.mos.C) not in [1, 2]:
        return False
    return True


def _occupied_counts(method):
    method_name = (
        method._scf_type() if hasattr(method, "_scf_type") else type(method).__name__
    )
    if method_name == "GHF":
        return None
    if not hasattr(method, "na") or not hasattr(method, "nb"):
        return None
    if method_name in ["UHF", "CUHF"]:
        return [method.na, method.nb]
    return [max(method.na, method.nb)]
