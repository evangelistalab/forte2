import numpy as np


def determine_fno_n_keep(
    occ_desc: np.ndarray,
    p_o: float | None,
    n_kappa: float | None,
    degeneracy_tol: float,
) -> int:
    """
    Determine how many virtual natural orbitals to retain.

    Parameters
    ----------
    occ_desc : np.ndarray
        Natural occupation numbers, sorted in descending order.
    p_o : float, optional
        Retain the smallest set of leading NOs whose cumulative occupation is
        at least this fraction (0, 1] of the total. Mutually exclusive with n_kappa.
    n_kappa : float, optional
        Retain all NOs with occupation number >= n_kappa. Mutually exclusive with p_o.
    degeneracy_tol : float
        After applying the p_o/n_kappa criterion, the cutoff is pushed outward
        (more orbitals retained) while the occupation numbers straddling the
        boundary differ by less than this fraction of the larger one, so that
        near-degenerate NOs (e.g. Kramers partners) are never split between the
        retained and discarded sets.

    Returns
    -------
    int
        Number of virtual NOs to retain.
    """
    nvirt = occ_desc.shape[0]
    if p_o is not None:
        cumulative = np.cumsum(occ_desc)
        n_keep = int(np.searchsorted(cumulative, p_o * cumulative[-1]) + 1)
        n_keep = min(n_keep, nvirt)
    else:
        n_keep = int(np.sum(occ_desc >= n_kappa))

    while 0 < n_keep < nvirt and (
        occ_desc[n_keep - 1] - occ_desc[n_keep] < degeneracy_tol * occ_desc[n_keep - 1]
    ):
        n_keep += 1

    assert n_keep > 0, "FNO truncation criterion discards all virtual orbitals."
    return n_keep


def build_fno_virtual_space(pt2, gamma_vv, p_o, n_kappa, degeneracy_tol):
    """
    Build the frozen-natural-orbital virtual space for a full-space
    RelDSRG_MRPT2 calculation.

    Diagonalizes the (Hermitian) virtual-virtual unrelaxed 1-RDM, truncates it
    by cumulative occupation percentage (p_o) or a hard occupation threshold
    (n_kappa), and returns a new (mos, mo_space) pair with the virtual block
    rotated into the natural-orbital basis (composed with the semicanonical
    rotation already applied for pt2's own amplitude equations) and the
    discarded NOs marked frozen. Only the virtual columns of mos.C are
    touched; frozen_core/core/active columns are left exactly as inherited
    from the parent reference, since downstream DSRG cumulant rotations
    assume the active block stays in the reference's native basis.

    Parameters
    ----------
    pt2 : RelDSRG_MRPT2
        A RelDSRG_MRPT2 instance that has already run get_integrals() and
        solve_dsrg() in the full (untruncated) virtual space.
    gamma_vv : np.ndarray
        The virtual-virtual unrelaxed 1-RDM, in pt2's semicanonical virtual
        basis (as returned by pt2.compute_unrelaxed_gamma_vv()).
    p_o, n_kappa : float, optional
        Truncation criteria; see determine_fno_n_keep. Exactly one must be given.
    degeneracy_tol : float
        See determine_fno_n_keep.

    Returns
    -------
    tuple[MO, MOSpace]
        The truncated (mos, mo_space) pair.
    """
    occ, U_no = np.linalg.eigh(gamma_vv)
    order = np.argsort(occ)[::-1]
    occ, U_no = occ[order], U_no[:, order]

    n_keep = determine_fno_n_keep(occ, p_o, n_kappa, degeneracy_tol)
    n_discard = occ.shape[0] - n_keep

    virt = pt2.mo_space.virt
    U_semican_virt = pt2.semicanonicalizer.U[virt, virt]
    U_total_virt = U_semican_virt @ U_no

    C_contig_new = pt2._C.copy()
    C_contig_new[:, virt] = pt2._C[:, virt] @ U_total_virt
    C_orig_new = C_contig_new[:, pt2.mo_space.contig_to_orig]

    mos_trunc = pt2.mos.copy()
    mos_trunc.C[0] = C_orig_new

    # update_frozen_orbitals always recomputes core_orbitals from
    # (core_orbitals + frozen_core_orbitals) minus whatever it's asked to
    # newly freeze, so the existing frozen-core set must be re-passed
    # explicitly here or it silently gets merged back into "core" and
    # correlated.
    mo_space_trunc = pt2.mo_space.update_frozen_orbitals(
        frozen_core_orbitals=pt2.mo_space.frozen_core_orbitals,
        frozen_virtual_orbitals=n_discard,
    )

    return mos_trunc, mo_space_trunc
