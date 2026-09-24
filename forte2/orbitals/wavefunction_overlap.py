import numpy as np
from scipy.linalg import det

from .orbital_overlap import mo_overlap


def ci_overlap(ci_1, ci_2, root_1=0, root_2=0):
    r"""
    Computes the overlap between two CI wavefunctions :math:`\langle \Psi_1 | \Psi'_2 \rangle`

    The overlap accounts for the difference between the two orbital sets, not
    only the CI coefficients. The wavefunctions can differ in geometry, basis
    set, orbitals, and core and active spaces. Both must be nonrelativistic or
    both two-component, with the same numbers of alpha and beta electrons, or
    the same number of electrons if two-component.

    Parameters
    ----------
    ci_1, ci_2 : ActiveSpaceDriver
        Either CI or MCoptimizer holding the bra and ket wavefunctions.
    root_1, root_2 : int, optional, default=0
        The absolute root index of each wavefunction, counted across all states.

    Returns
    -------
    float or complex
        The overlap, complex if the wavefunctions are two-component. Its phase
        depends on the arbitrary phases of the orbitals and CI vectors.

    Raises
    ------
    TypeError
        If the solver of either driver isn't a `CISolver` or `RelCISolver`.
    ValueError
        If only one wavefunction is two-component, or if the electron counts
        differ.

    Notes
    -----
    The overlap is summed over all determinant pairs, using Eqs. 4 and 8 of
    Plasser et al., J. Chem. Theory Comput. 12, 1207 (2016), so the cost scales
    as the product of the two determinant counts. A two-component determinant
    holds all its electrons in one spinor string, so each pair contributes a
    single determinant.
    """
    from forte2.ci import CISolver, RelCISolver

    for ci in (ci_1, ci_2):
        if not isinstance(ci.ci_solver, (CISolver, RelCISolver)):
            raise TypeError(
                "ci_overlap requires a CISolver or RelCISolver, "
                f"got {type(ci.ci_solver).__name__}."
            )
    two_component = isinstance(ci_1.ci_solver, RelCISolver)
    if isinstance(ci_2.ci_solver, RelCISolver) != two_component:
        raise ValueError(
            "The wavefunctions must be both nonrelativistic or both two-component."
        )

    expansions = []
    for ci, root in ((ci_1, root_1), (ci_2, root_2)):
        if not ci.executed:
            ci.run()
        expansions.append(_determinant_expansion(ci, root, two_component))
    (dets_1, c_1, C_1, ndocc_1), (dets_2, c_2, C_2, ndocc_2) = expansions

    nel = []
    for dets, _, _, ndocc in expansions:
        na, nb = ndocc + dets[0].count_alpha(), ndocc + dets[0].count_beta()
        nel.append(na if two_component else (na, nb))
    if nel[0] != nel[1]:
        raise ValueError(
            "The wavefunctions must have the same numbers of electrons, "
            f"got {nel[0]} and {nel[1]}."
        )

    S = mo_overlap(C_1, ci_1.system, C_2, ci_2.system)
    return _determinant_overlap(
        S, dets_1, c_1, ndocc_1, dets_2, c_2, ndocc_2, two_component
    )


def _determinant_expansion(ci, root, two_component):
    """
    Gets the determinants, determinant-basis CI vector, core and active MO
    coefficients, and number of core orbitals of a root.
    """
    state, root_in_state = ci.ci_solver._get_state_root(root)
    solver = ci.ci_solver.sub_solvers[state]
    c = solver.evecs[:, root_in_state]
    if not two_component:
        c = solver.csf_C_to_det_C(c)
    space = ci.mo_space
    ndocc = space.nfrozen_core + space.ncore
    C = ci.mos.C[0][:, space.orig_to_contig][:, : ndocc + space.nactv]
    return solver.dets, c, C, ndocc


def _determinant_overlap(S, dets_1, c_1, ndocc_1, dets_2, c_2, ndocc_2, two_component):
    r"""
    Computes :math:`\langle\Psi_1|\Psi_2\rangle` as a Löwdin sum over all
    determinant pairs.

    Parameters
    ----------
    S : NDArray
        The MO overlap, shape ``(ndocc_1 + nactv_1, ndocc_2 + nactv_2)``. Each
        side is ordered as core orbitals, then active orbitals.
    dets_1, dets_2 : list[Determinant]
        The active-space determinants of each wavefunction.
    c_1, c_2 : NDArray
        The CI coefficients, in the order of ``dets_1`` and ``dets_2``.
    ndocc_1, ndocc_2 : int
        The number of core orbitals of each wavefunction.
    two_component : bool
        If True, the orbitals are spinors and each core spinor holds one
        electron. If False, each core orbital holds two.

    Returns
    -------
    float or complex
        The overlap, complex if ``two_component`` is True.
    """
    occ_1 = [
        _spin_occupations(d, ndocc_1, S.shape[0] - ndocc_1, two_component)
        for d in dets_1
    ]
    occ_2 = [
        _spin_occupations(d, ndocc_2, S.shape[1] - ndocc_2, two_component)
        for d in dets_2
    ]
    overlap = 0.0
    for c1, (alpha_1, beta_1) in zip(c_1, occ_1):
        for c2, (alpha_2, beta_2) in zip(c_2, occ_2):
            # all alpha creation operators precede all beta ones, so the
            # spin-orbital overlap matrix is block diagonal. scipy's det, unlike
            # numpy's, doesn't warn on exactly singular complex blocks
            det_alpha = det(S[np.ix_(alpha_1, alpha_2)])
            det_beta = det(S[np.ix_(beta_1, beta_2)])
            overlap += np.conj(c1) * c2 * det_alpha * det_beta
    return complex(overlap) if two_component else float(overlap)


def _spin_occupations(det, ndocc, nactv, two_component):
    """
    Return the occupied alpha and beta orbital indices of a determinant in ascending order.
    A two-component determinant holds all its electrons, core included, in the alpha string.
    """
    docc = list(range(ndocc))
    alpha = docc + [ndocc + p for p in range(nactv) if det.na(p)]
    if two_component:
        return alpha, []
    beta = docc + [ndocc + p for p in range(nactv) if det.nb(p)]
    return alpha, beta
