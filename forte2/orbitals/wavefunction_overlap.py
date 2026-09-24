from dataclasses import dataclass, replace
from math import comb

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import det

from .ci_rotation import transform_ci_vectors
from .orbital_overlap import mo_overlap


def ci_overlap(ci_1, ci_2, root_1=0, root_2=0, algorithm="biorthogonal"):
    r"""
    Computes the overlap between two CI wavefunctions, :math:`\langle \Psi | \Psi' \rangle`.

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
    algorithm : {"biorthogonal", "naive"}, optional, default="biorthogonal"
        ``"biorthogonal"`` requires the same core and active spaces in both
        wavefunctions and the complete CAS determinant space. ``"naive"``
        handles any pair of wavefunctions, at a cost that scales as the
        product of the two determinant counts.

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
        If ``algorithm`` is unknown, if only one wavefunction is
        two-component, if the electron counts differ, or if ``algorithm`` is
        ``"biorthogonal"`` and the wavefunctions don't meet its requirements.

    Notes
    -----
    ``"naive"`` sums over all determinant pairs, using Eqs. 4 and 8 of
    Plasser et al., J. Chem. Theory Comput. 12, 1207 (2016). A two-component
    determinant holds all its electrons in one spinor string, so each pair
    contributes a single determinant.

    ``"biorthogonal"`` transforms the two orbital sets into a biorthonormal
    pair, following the Appendix of Malmqvist, Int. J. Quantum Chem. 30, 479
    (1986), re-expresses each CI vector in its new orbitals, and takes the dot
    product of the two vectors.

    See docs/technical_notes/ci_overlap.tex for detailed derivations.
    """
    two_component = _check_pair(ci_1, ci_2, algorithm)
    bra = _Wavefunction.from_driver(ci_1, [root_1], two_component)
    ket = _Wavefunction.from_driver(ci_2, [root_2], two_component)
    counts_bra, counts_ket = bra.blocks[0].counts, ket.blocks[0].counts
    if counts_bra != counts_ket:
        raise ValueError(
            "The wavefunctions must have the same numbers of electrons, "
            f"got {counts_bra} and {counts_ket}."
        )
    overlap = _overlap_matrix(bra, ket, algorithm)[0, 0]
    return complex(overlap) if two_component else float(overlap)


def ci_overlap_matrix(ci_1, ci_2, roots_1=None, roots_2=None, algorithm="biorthogonal"):
    r"""
    Computes the overlaps :math:`\langle \Psi_i | \Psi'_j \rangle` between
    roots of two CI wavefunctions.

    The orbital transformation is built once and shared by every root pair,
    so the whole matrix costs about as much as a single `ci_overlap`. Roots
    with different numbers of alpha and beta electrons, which can only happen
    in a nonrelativistic calculation, have zero overlap.

    Parameters
    ----------
    ci_1, ci_2 : ActiveSpaceDriver
        Either CI or MCoptimizer holding the bra and ket wavefunctions.
    roots_1, roots_2 : list[int], optional
        The absolute root indices of each wavefunction, counted across all
        states. All roots if None.
    algorithm : {"biorthogonal", "naive"}, optional, default="biorthogonal"
        As in `ci_overlap`.

    Returns
    -------
    NDArray
        The overlaps, shape ``(len(roots_1), len(roots_2))``, complex if the
        wavefunctions are two-component.

    Raises
    ------
    TypeError
        If the solver of either driver isn't a `CISolver` or `RelCISolver`.
    ValueError
        If ``algorithm`` is unknown, if only one wavefunction is
        two-component, if the total electron counts differ, or if
        ``algorithm`` is ``"biorthogonal"`` and the wavefunctions don't meet
        its requirements.
    """
    two_component = _check_pair(ci_1, ci_2, algorithm)
    bra = _Wavefunction.from_driver(ci_1, roots_1, two_component)
    ket = _Wavefunction.from_driver(ci_2, roots_2, two_component)
    return _overlap_matrix(bra, ket, algorithm)


def _check_pair(ci_1, ci_2, algorithm):
    """
    Validates two CI drivers for an overlap and returns whether they're
    two-component.
    """
    from forte2.ci import CISolver, RelCISolver

    if algorithm not in ("biorthogonal", "naive"):
        raise ValueError(
            f"algorithm must be 'biorthogonal' or 'naive', got {algorithm!r}."
        )
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
    return two_component


@dataclass
class _RootBlock:
    """
    The requested roots of one CI sub-solver.

    Attributes
    ----------
    solver : _CISingleStateSolver or _RelCISingleStateSolver
        The sub-solver that holds the roots.
    positions : list[int]
        The positions of the roots among all requested roots.
    vectors : NDArray
        The CI vectors in the determinant basis of solver, shape
        (ndet, len(positions)).
    counts : tuple[int, ...]
        The numbers of alpha and beta electrons, or (number of electrons,) if
        two-component.
    """

    solver: object
    positions: list[int]
    vectors: NDArray
    counts: tuple[int, ...]


@dataclass
class _Wavefunction:
    """
    The requested roots of a CI wavefunction in the determinant basis, with
    the orbitals they're expanded in.

    Attributes
    ----------
    system : System
        The system that the orbitals belong to.
    C : NDArray
        The core, then active, MO coefficients, shape (nbasis, ndocc + nactv).
    ndocc, nactv : int
        The numbers of core orbitals, frozen ones included, and active orbitals.
    two_component : bool
        Whether the orbitals are spinors.
    blocks : list[_RootBlock]
        The requested roots, grouped by the sub-solver that holds them.
    nroots : int
        The number of requested roots.
    """

    system: object
    C: NDArray
    ndocc: int
    nactv: int
    two_component: bool
    blocks: list[_RootBlock]
    nroots: int

    @classmethod
    def from_driver(cls, ci, roots, two_component):
        """
        Gets roots of a CI driver, all of them if roots is None, running the
        driver first if needed.
        """
        if not ci.executed:
            ci.run()
        if roots is None:
            roots = range(ci.ci_solver.sa_info.nroots_sum)
        members = {}
        for position, root in enumerate(roots):
            state, root_in_state = ci.ci_solver._get_state_root(root)
            members.setdefault(state, []).append((position, root_in_state))

        space = ci.mo_space
        ndocc = space.nfrozen_core + space.ncore
        blocks = []
        for state, pairs in members.items():
            solver = ci.ci_solver.sub_solvers[state]
            vectors = solver.evecs[:, [root_in_state for _, root_in_state in pairs]]
            if not two_component:
                vectors = np.column_stack([solver.csf_C_to_det_C(c) for c in vectors.T])
            strings = solver.ci_strings
            if two_component:
                counts = (ndocc + strings.na,)
            else:
                counts = (ndocc + strings.na, ndocc + strings.nb)
            blocks.append(
                _RootBlock(solver, [position for position, _ in pairs], vectors, counts)
            )
        C = ci.mos.C[0][:, space.orig_to_contig][:, : ndocc + space.nactv]
        return cls(ci.system, C, ndocc, space.nactv, two_component, blocks, len(roots))

    def occupations(self, solver):
        """
        Gets the occupied alpha and beta orbital indices of each determinant of
        a sub-solver, in ascending order. A two-component determinant holds all
        its electrons, core included, in the alpha string.
        """
        docc = list(range(self.ndocc))
        actv = range(self.nactv)
        occupations = []
        for d in solver.dets:
            alpha = docc + [self.ndocc + p for p in actv if d.na(p)]
            if self.two_component:
                beta = []
            else:
                beta = docc + [self.ndocc + p for p in actv if d.nb(p)]
            occupations.append((alpha, beta))
        return occupations


def _overlap_matrix(bra, ket, algorithm):
    """
    Computes <Psi_i|Psi'_j> for every pair of requested roots of the bra and
    the ket.
    """
    for block_bra in bra.blocks:
        for block_ket in ket.blocks:
            if sum(block_bra.counts) != sum(block_ket.counts):
                raise ValueError(
                    "The wavefunctions must have the same numbers of electrons, "
                    f"got {sum(block_bra.counts)} and {sum(block_ket.counts)}."
                )
    S = mo_overlap(bra.C, bra.system, ket.C, ket.system)
    if algorithm == "naive":
        return _naive_overlaps(S, bra, ket)
    _check_biorthogonal_applies(bra, ket)
    return _biorthogonal_overlaps(S, bra, ket)


def _matching_blocks(blocks_bra, blocks_ket):
    """
    Yields the pairs of root blocks with the same numbers of alpha and beta
    electrons. Other pairs don't overlap.
    """
    for block_bra in blocks_bra:
        for block_ket in blocks_ket:
            if block_bra.counts == block_ket.counts:
                yield block_bra, block_ket


def _naive_overlaps(S, bra, ket):
    """
    Computes <Psi_i|Psi'_j> as Löwdin sums over all determinant pairs.

    Parameters
    ----------
    S : NDArray
        The MO overlap between the core, then active, orbitals of the bra and
        the ket, shape (bra.ndocc + bra.nactv, ket.ndocc + ket.nactv).
    bra, ket : _Wavefunction
        The two wavefunctions.

    Returns
    -------
    NDArray
        The overlaps, shape (bra.nroots, ket.nroots).
    """
    dtype = complex if bra.two_component else float
    overlaps = np.zeros((bra.nroots, ket.nroots), dtype=dtype)
    for block_bra, block_ket in _matching_blocks(bra.blocks, ket.blocks):
        D = _determinant_overlaps(
            S, bra.occupations(block_bra.solver), ket.occupations(block_ket.solver)
        )
        overlaps[np.ix_(block_bra.positions, block_ket.positions)] = (
            block_bra.vectors.conj().T @ D @ block_ket.vectors
        )
    return overlaps


def _determinant_overlaps(S, occupations_bra, occupations_ket):
    """
    Computes the Löwdin overlap <D_I|D'_J> of every pair of determinants, given
    their occupied alpha and beta orbital indices.
    """
    D = np.empty((len(occupations_bra), len(occupations_ket)), dtype=S.dtype)
    for I, (alpha_bra, beta_bra) in enumerate(occupations_bra):
        for J, (alpha_ket, beta_ket) in enumerate(occupations_ket):
            # all alpha creation operators precede all beta ones, so the
            # spin-orbital overlap matrix is block diagonal. scipy's det, unlike
            # numpy's, doesn't warn on exactly singular complex blocks
            D[I, J] = det(S[np.ix_(alpha_bra, alpha_ket)]) * det(
                S[np.ix_(beta_bra, beta_ket)]
            )
    return D


def _check_biorthogonal_applies(bra, ket):
    """
    Raises ValueError unless both wavefunctions have the same core and active
    spaces and their sub-solvers span the complete CAS determinant space.
    """
    if (bra.ndocc, bra.nactv) != (ket.ndocc, ket.nactv):
        raise ValueError(
            "The biorthogonal algorithm requires the same numbers of core and "
            f"active orbitals, got {(bra.ndocc, bra.nactv)} and "
            f"{(ket.ndocc, ket.nactv)}. Use algorithm='naive' instead."
        )
    for wfn in (bra, ket):
        for block in wfn.blocks:
            strings = block.solver.ci_strings
            ndet_cas = comb(wfn.nactv, strings.na) * comb(wfn.nactv, strings.nb)
            # a rotation of the active orbitals can move amplitude out of a
            # restricted determinant space
            if strings.ngas_spaces > 1 or strings.ndet != ndet_cas:
                raise ValueError(
                    "The biorthogonal algorithm requires the complete CAS "
                    "determinant space, without GAS or point-group restrictions. "
                    "Use algorithm='naive' instead."
                )


@dataclass
class BiorthonormalTransforms:
    r"""
    The transforms :math:`\mathbf{M}` of the bra orbitals and
    :math:`\mathbf{M}'` of the ket orbitals that make them biorthonormal,
    :math:`\mathbf{M}^\dagger \mathbf{S} \mathbf{M}' = \mathbf{1}`, with the
    singular value decompositions they're built from,
    :math:`\mathbf{S}_{CC'} = \mathbf{U}_C \mathbf{D}_C \mathbf{V}_C^\dagger`
    for the core block of the MO overlap and
    :math:`\bar{\mathbf{S}}_{AA'} = \mathbf{U}_A \mathbf{D}_A \mathbf{V}_A^\dagger`
    for its Schur complement.

    Attributes
    ----------
    M, M_prime : NDArray
        The bra and ket transforms, shape ``(ndocc + nactv, ndocc + nactv)``.
        Their core blocks are ``U_C`` and ``V_C @ diag(1 / d_C)``, and their
        active blocks are ``U_A`` and ``V_A @ diag(1 / d_A)``.
    U_C, V_C : NDArray
        The singular vectors of the core block, shape ``(ndocc, ndocc)``.
    d_C : NDArray
        The singular values of the core block, shape ``(ndocc,)``.
    U_A, V_A : NDArray
        The singular vectors of the Schur complement, shape ``(nactv, nactv)``.
        In the real case either can be improper.
    d_A : NDArray
        The singular values of the Schur complement, shape ``(nactv,)``.
    """

    M: NDArray
    M_prime: NDArray
    U_C: NDArray
    d_C: NDArray
    V_C: NDArray
    U_A: NDArray
    d_A: NDArray
    V_A: NDArray


def biorthogonalize_casscf_orbitals(S, ndocc, nactv):
    r"""
    Builds the transforms that make the bra and ket orbitals of two CASSCF
    wavefunctions biorthonormal,
    :math:`\mathbf{M}^\dagger \mathbf{S} \mathbf{M}' = \mathbf{1}`.

    Follows the pseudo-corresponding orbital construction in the Appendix of
    Malmqvist, Int. J. Quantum Chem. 30, 479 (1986), Eqs. A.1-A.7. Both
    transforms are block upper-triangular in the (core, active) ordering, so
    new core orbitals are combinations of old core orbitals only, which keeps
    a CAS expansion closed under the transform.

    Parameters
    ----------
    S : NDArray
        The MO overlap between the bra and ket orbitals, shape
        ``(ndocc + nactv, ndocc + nactv)``, with bra orbitals as rows and ket
        orbitals as columns, each ordered as core, then active.
    ndocc : int
        The number of core orbitals on each side.
    nactv : int
        The number of active orbitals on each side.

    Returns
    -------
    BiorthonormalTransforms
        The transforms and the singular value decompositions they're built from.

    Raises
    ------
    numpy.linalg.LinAlgError
        If the core block of ``S`` or its Schur complement is singular, in
        which case no biorthonormal pair exists.
    """
    n = ndocc + nactv
    core = slice(0, ndocc)
    actv = slice(ndocc, n)
    S_CC, S_CA = S[core, core], S[core, actv]
    S_AC, S_AA = S[actv, core], S[actv, actv]

    # core blocks (Malmqvist Eq. A.2)
    U_C, d_C, V_C_h = np.linalg.svd(S_CC)
    V_C = V_C_h.conj().T
    _check_nonsingular(d_C, "core")
    S_CC_inv = (V_C / d_C) @ U_C.conj().T

    # active blocks, from the Schur complement of the core block (Eqs. A.3-A.4)
    S_bar_AA = S_AA - S_AC @ S_CC_inv @ S_CA
    U_A, d_A, V_A_h = np.linalg.svd(S_bar_AA)
    V_A = V_A_h.conj().T
    _check_nonsingular(d_A, "active")

    # the core-active blocks mix each side's core orbitals into its active
    # orbitals, so that they're orthogonal to the other side's core orbitals
    # (Eqs. A.5-A.7)
    M = np.zeros((n, n), dtype=S.dtype)
    M[core, core] = U_C
    M[core, actv] = -S_CC_inv.conj().T @ S_AC.conj().T @ U_A
    M[actv, actv] = U_A

    M_prime = np.zeros((n, n), dtype=S.dtype)
    M_prime[core, core] = V_C / d_C
    M_prime[core, actv] = -S_CC_inv @ S_CA @ (V_A / d_A)
    M_prime[actv, actv] = V_A / d_A

    return BiorthonormalTransforms(M, M_prime, U_C, d_C, V_C, U_A, d_A, V_A)


def _check_nonsingular(singular_values, block):
    if np.any(singular_values < 1e-10):
        raise np.linalg.LinAlgError(
            f"The {block} block of the MO overlap is singular (smallest singular "
            f"value {singular_values.min():.3e}), so the orbital sets can't be "
            "biorthogonalized."
        )


def _biorthogonal_overlaps(S, bra, ket):
    """
    Computes <Psi_i|Psi'_j> as dot products of the CI vectors re-expressed in
    biorthonormal orbitals.
    """
    bio = biorthogonalize_casscf_orbitals(S, bra.ndocc, bra.nactv)
    # the core block of each transform multiplies every coefficient by its
    # determinant to the power -g, with g electrons per core orbital
    g = 1 if bra.two_component else 2
    scale_bra = det(bio.U_C) ** -g
    scale_ket = det(bio.V_C / bio.d_C) ** -g
    blocks_bra = [
        replace(
            block,
            vectors=scale_bra
            * transform_ci_vectors(
                block.solver, block.vectors, bio.U_A, bra.two_component
            ),
        )
        for block in bra.blocks
    ]
    blocks_ket = [
        replace(
            block,
            vectors=scale_ket
            * transform_ci_vectors(
                block.solver, block.vectors, bio.V_A, ket.two_component, d=bio.d_A
            ),
        )
        for block in ket.blocks
    ]
    dtype = complex if bra.two_component else float
    overlaps = np.zeros((bra.nroots, ket.nroots), dtype=dtype)
    for block_bra, block_ket in _matching_blocks(blocks_bra, blocks_ket):
        overlaps[np.ix_(block_bra.positions, block_ket.positions)] = (
            block_bra.vectors.conj().T @ block_ket.vectors
        )
    return overlaps
