import numpy as np

import forte2.integrals as integrals
from forte2.helpers import block_diag_2x2, logger


def mo_overlap(C_a, system_a, C_b, system_b=None):
    r"""
    Overlap between two sets of MO(-like) coefficients, :math:`C_a^\dagger S C_b`.
    If system_a is two-component, then all other quantities are assumed
    to have two-component dimensions.

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
        S = system_a.ints_overlap()
    else:
        S = integrals.overlap(system_a, system_a.basis, system_b.basis)
        if system_a.two_component:
            S = block_diag_2x2(S)
    return C_a.T.conj() @ S @ C_b


def transfer_orbitals(C_source, system_source, system_target):
    r"""
    Express orbitals from `system_source` in the AO basis of `system_target`.

    If both systems carry the same shells on the same atoms, as with
    ``System.with_geometry``, the AOs move with their atoms and the coefficients
    carry over unchanged. Otherwise, the orbitals are projected onto the target
    basis through the cross-basis overlap. In both cases, the result is then
    orthonormalized with the symmetric (Löwdin) procedure, which changes the
    orbitals as little as possible and keeps their order.

    If the target has fewer orbitals than the source, the trailing source
    orbitals are dropped. If it has more, the set is completed with an
    orthonormal complement.

    Parameters
    ----------
    C_source : NDArray
        Source coefficients, shape ``(nbf_source, n_source)``.
    system_source : System
        The system whose AO basis `C_source` is expressed in.
    system_target : System
        The system whose AO basis the orbitals are transferred to.

    Returns
    -------
    NDArray | None
        Orthonormal coefficients in `system_target`'s AO basis, shape
        ``(nbf_target, nmo_target)``, or None if the target basis cannot
        represent the source orbitals.
    """
    X = system_target.get_Xorth()
    n = min(C_source.shape[1], X.shape[1])
    if _same_basis_layout(system_source, system_target):
        # transported: essentially C1 = C0 [C0^H S1 C0]^{-1/2}
        Q = mo_overlap(X, system_target, C_source[:, :n])
    else:
        # projected: essentially C1 = P [P^H S1 P]^{-1/2}, with P = S1^{-1} S10 C0
        Q = mo_overlap(X, system_target, C_source[:, :n], system_source)

    # Q = U s Vh = (U Vh) (V s Vh) = (U Vh) (Q^H Q)^{1/2}
    # therefore U Vh = Q (Q^H Q)^{-1/2},
    # where (Q^H Q)^{-1/2} is the symmetric/Lowdin orthogonalizer of Q
    # Therefore, X U Vh = X Q (Q^H Q)^{-1/2}

    # with transported orbitals, Q = X^H S1 C0, so
    #   X Q = X X^H S1 C0 = C0, and Q^H Q = C0^H S1 C0
    # with projected orbitals, Q = X^H S10 C0, so
    #   X Q = S1^{-1} S10 C0, and Q^H Q = C0^H S01 S1^{-1} S10 C0
    U, s, Vh = np.linalg.svd(Q)
    if s[-1] < 1.0e-8:
        logger.log_warning(
            "Cannot transfer orbitals: the target basis does not represent them "
            f"(smallest singular value {s[-1]:.2e})."
        )
        return None
    return X @ np.hstack((U[:, :n] @ Vh, U[:, n:]))


def _same_basis_layout(system_a, system_b):
    # Same shells on the same atoms, in the same order; only the centers may differ.
    if not np.array_equal(system_a.atomic_charges, system_b.atomic_charges):
        return False
    basis_a, basis_b = system_a.basis, system_b.basis
    if basis_a.nshells != basis_b.nshells:
        return False
    if basis_a.center_first_and_last_shell != basis_b.center_first_and_last_shell:
        return False
    for i in range(basis_a.nshells):
        a, b = basis_a[i], basis_b[i]
        if a.l != b.l or a.is_pure != b.is_pure:
            return False
        if a.exponents != b.exponents or a.coeff != b.coeff:
            return False
    return True
