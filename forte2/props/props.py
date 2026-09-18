import numpy as np

from forte2 import integrals
from forte2.data import DEBYE_TO_AU, ANGSTROM_TO_BOHR
from forte2.helpers.matrix_functions import block_diag_2x2
from .mutual_correlation import RMP2MPQOnTheFly, UMP2MPQOnTheFly


DEFAULT_MUTUAL_CORRELATION_SCORE_THRESHOLDS = (0.15, 0.05, 0.0)


def suggest_mutual_correlation_active_spaces(
    matrix,
    occupations,
    *,
    candidate_indices=None,
    relative_thresholds=DEFAULT_MUTUAL_CORRELATION_SCORE_THRESHOLDS,
    absolute_threshold=7.5e-4,
    mandatory_indices=(),
    degeneracy_rtol=1.0e-8,
    degeneracy_atol=1.0e-10,
):
    """Suggest active orbitals from significant mutual-correlation scores.

    The significant score of orbital ``p`` is the sum of ``M[p, q]`` over
    candidate-orbital edges greater than ``absolute_threshold``.  For each
    positive relative threshold ``eta``, orbitals with scores at least
    ``eta * max(score)`` are selected.  A zero threshold retains only
    endpoints of significant edges, rather than every candidate orbital.

    Any mandatory orbitals are added before completing numerical degeneracy
    groups defined by the supplied natural occupations.  Degeneracy completion
    is transitive, so a narrowly split multiplet is never cut into pieces.

    Returns
    -------
    dict
        The significant edges and scores, their maximum, and one selection
        record per requested relative threshold.
    """
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("The mutual-correlation matrix must be square.")
    if np.iscomplexobj(matrix):
        if not np.allclose(matrix.imag, 0.0, rtol=0.0, atol=1.0e-12):
            raise ValueError("The mutual-correlation matrix must be real.")
        matrix = matrix.real
    matrix = np.asarray(matrix, dtype=float)
    nmo = matrix.shape[0]

    if candidate_indices is None:
        candidates = tuple(range(nmo))
    else:
        candidates = tuple(int(index) for index in candidate_indices)
    if not candidates:
        raise ValueError("At least one candidate orbital is required.")
    if len(candidates) != len(set(candidates)):
        raise ValueError("Candidate orbital indices must be unique.")
    if min(candidates) < 0 or max(candidates) >= nmo:
        raise IndexError(f"Candidate orbital indices must lie in [0, {nmo}).")
    candidate_array = np.asarray(candidates, dtype=int)
    candidate_block = matrix[np.ix_(candidate_array, candidate_array)]
    if not np.all(np.isfinite(candidate_block)):
        raise ValueError("The mutual-correlation matrix contains non-finite values.")
    if not np.allclose(candidate_block, candidate_block.T, rtol=1.0e-8, atol=1.0e-12):
        raise ValueError("The mutual-correlation matrix must be symmetric.")
    if np.min(candidate_block) < -1.0e-12:
        raise ValueError("The mutual-correlation matrix must be nonnegative.")

    absolute_threshold = float(absolute_threshold)
    if absolute_threshold < 0.0:
        raise ValueError("absolute_threshold must be nonnegative.")

    occupations = np.asarray(occupations, dtype=float)
    if occupations.ndim != 1 or occupations.size < nmo:
        raise ValueError("Natural occupations must cover every matrix orbital.")
    if not np.all(np.isfinite(occupations[candidate_array])):
        raise ValueError("Candidate natural occupations must be finite.")
    if degeneracy_rtol < 0.0 or degeneracy_atol < 0.0:
        raise ValueError("Degeneracy tolerances must be nonnegative.")

    mandatory = {int(index) for index in mandatory_indices}
    if not mandatory.issubset(candidates):
        raise ValueError("Mandatory orbitals must belong to the candidate space.")

    thresholds = tuple(float(value) for value in relative_thresholds)
    if not thresholds:
        raise ValueError("At least one relative threshold is required.")
    if len(thresholds) != len(set(thresholds)):
        raise ValueError("Relative thresholds must be unique.")
    if any(value < 0.0 or value > 1.0 for value in thresholds):
        raise ValueError("Relative thresholds must lie in [0, 1].")

    significant_scores = np.zeros(nmo)
    significant_edges = []
    for position, p in enumerate(candidates):
        for q in candidates[position + 1 :]:
            value = max(0.5 * (matrix[p, q] + matrix[q, p]), 0.0)
            if value > absolute_threshold:
                significant_edges.append((p, q, float(value)))
                significant_scores[p] += value
                significant_scores[q] += value
    significant_edges.sort(key=lambda edge: (-edge[2], edge[0], edge[1]))
    maximum_score = float(np.max(significant_scores[candidate_array]))

    def complete_degenerate_groups(selected):
        completed = set(selected)
        additions = set()
        changed = True
        while changed:
            changed = False
            for p in tuple(completed):
                for q in candidates:
                    if q in completed:
                        continue
                    if np.isclose(
                        occupations[p],
                        occupations[q],
                        rtol=degeneracy_rtol,
                        atol=degeneracy_atol,
                    ):
                        completed.add(q)
                        additions.add(q)
                        changed = True
        return tuple(sorted(completed)), tuple(sorted(additions))

    suggestions = {}
    for threshold in thresholds:
        score_cutoff = threshold * maximum_score
        if maximum_score == 0.0:
            score_selected = set()
        elif threshold == 0.0:
            score_selected = {
                index for index in candidates if significant_scores[index] > 0.0
            }
        else:
            score_selected = {
                index
                for index in candidates
                if significant_scores[index] > score_cutoff
                or np.isclose(
                    significant_scores[index],
                    score_cutoff,
                    rtol=1.0e-8,
                    atol=1.0e-12,
                )
            }
        active_indices, degeneracy_additions = complete_degenerate_groups(
            score_selected | mandatory
        )
        suggestions[threshold] = {
            "relative_threshold": threshold,
            "score_cutoff": score_cutoff,
            "score_selected_indices": tuple(sorted(score_selected)),
            "mandatory_indices": tuple(sorted(mandatory)),
            "degeneracy_completed_indices": degeneracy_additions,
            "active_indices": active_indices,
        }

    return {
        "candidate_indices": candidates,
        "absolute_threshold": absolute_threshold,
        "significant_edges": tuple(significant_edges),
        "significant_scores": significant_scores,
        "maximum_significant_score": maximum_score,
        "degeneracy_rtol": float(degeneracy_rtol),
        "degeneracy_atol": float(degeneracy_atol),
        "suggestions": suggestions,
    }


def get_1e_property(system, g1, property_name, origin=None, unit="debye"):
    """
    Calculate a one-electron property using AO-basis quantities.

    Parameters
    ----------
    system : System
        The system for which the property is calculated.
    g1 : NDArray
        The 1-particle density matrix in the AO basis.
        Should be the spin-free density matrix (dm_aa + dm_bb) for the non-relativistic case.
    property_name : str
        The name of the property to calculate (e.g., "kinetic_energy", "nuclear_attraction_energy", "electric_dipole").
    origin: list[float], optional
        The origin point for properties that depend on it (e.g., electric dipole moment).
    unit: str, optional, default="debye"
        The unit for the property value, either "debye" or "au". Default is "debye".
        Only used for multipole moments. For quadrupole moments, "debye" stands for debye * angstrom, etc.

    Returns
    -------
    float or NDArray
        The calculated property value.
    """

    if system.two_component:
        assert (
            g1.shape[0] == 2 * system.nbf
        ), f"g1 shape {g1.shape[0]} does not match the number of basis functions, {2 * system.nbf} in the system."
    else:
        assert (
            g1.shape[0] == system.nbf
        ), f"g1 shape {g1.shape[0]} does not match the number of basis functions, {system.nbf} in the system."

    def _origin_check(origin):
        if origin is None:
            origin = [0.0, 0.0, 0.0]
        assert len(origin) == 3, "Origin must be a 3-element vector."
        return origin

    spin_independent_properties = [
        "kinetic_energy",
        "nuclear_attraction_energy",
        "electric_dipole",
        "dipole",
        "electric_quadrupole",
        "quadrupole",
    ]
    assert (
        property_name in spin_independent_properties
    ), f"Property '{property_name}' is not supported, must be one of {spin_independent_properties}."
    factor = 1.0

    match property_name:
        case "kinetic_energy":
            oei = integrals.kinetic(system)
        case "nuclear_attraction_energy":
            oei = integrals.nuclear(system)
        case "electric_dipole":
            origin = _origin_check(origin)
            _, *oei = integrals.emultipole1(system, origin=origin)
            factor = -1.0 / DEBYE_TO_AU if unit == "debye" else -1.0
        case "dipole":
            e_dip = get_1e_property(
                system, g1, "electric_dipole", origin=origin, unit=unit
            )
            nuc_dip = system.nuclear_dipole(origin=origin, unit=unit)
            return e_dip + nuc_dip
        case "electric_quadrupole":
            origin = _origin_check(origin)
            *_, xx, xy, xz, yy, yz, zz = integrals.emultipole2(system, origin=origin)
            oei = [xx, xy, xz, yy, yz, zz]
            factor = (
                -1.0 / (DEBYE_TO_AU * ANGSTROM_TO_BOHR) if unit == "debye" else -1.0
            )
        case "quadrupole":
            xx, xy, xz, yy, yz, zz = get_1e_property(
                system, g1, "electric_quadrupole", origin=origin, unit=unit
            )
            e_quad = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
            e_quad = 0.5 * (3 * e_quad - np.trace(e_quad) * np.eye(3))
            nuc_quad = system.nuclear_quadrupole(origin=origin, unit=unit)
            return e_quad + nuc_quad
        case _:
            raise ValueError(f"Property '{property_name}' is not supported.")

    if system.two_component:
        if isinstance(oei, list):
            oei = [(block_diag_2x2(_)) for _ in oei]
        else:
            oei = block_diag_2x2(oei)

    if not isinstance(oei, list):
        return np.einsum("pq,pq->", g1, oei) * factor
    return np.array([np.einsum("pq,pq->", g1, _) for _ in oei]) * factor


def mulliken_population(system, g1):
    """
    Perform Mulliken population analysis on the system using the given method.

    Parameters
    ----------
    system : System
        The system for which the Mulliken population is calculated.
    g1 : NDArray
        The 1-particle spin-free density matrix (dm_aa + dm_bb).

    Returns
    -------
    tuple(NDArray, NDArray)
        The Mulliken population for each basis function and the atomic charges.

    Notes
    -----
    See eq 3.196 in Szabo and Ostlund.
    """
    ovlp = integrals.overlap(system)
    psdiag = np.einsum("pq,qp->p", g1, ovlp)
    center_first_and_last = system.basis.center_first_and_last
    charges = system.atomic_charges
    pop = np.array([psdiag[_[0] : _[1]].sum() for _ in center_first_and_last])
    return (psdiag, charges - pop)


def iao_partial_charge(system, g1_iao):
    """
    Perform partial charge analysis using IAOs.

    Parameters
    ----------
    system : System
        The system for which the partial charge is calculated.
    g1_iao : NDArray
        The 1-particle spin-free density matrix in the IAO basis.
        Calulated using `forte2.orbitlas.iao.IAO.make_sf_1rdm`.

    Returns
    -------
    tuple(NDArray, NDArray)
        The diagonal elements of the 1-particle density matrix in the IAO basis and the
        partial charges for each atom.
    """
    g1diag = np.diag(g1_iao)
    center_first_and_last = system.minao_basis.center_first_and_last
    charges = system.atomic_charges
    pop = np.array([g1diag[_[0] : _[1]].sum() for _ in center_first_and_last])
    return (g1diag, charges - pop)


def _resolve_rdm_info_indices(
    mp2,
    C_no,
    occupations,
    *,
    indices=None,
    mo_range=None,
    avas=None,
    occupation_window=None,
):
    """Resolve one user-facing orbital selection into analysis-basis indices."""
    selectors = {
        "indices": indices,
        "mo_range": mo_range,
        "avas": avas,
        "occupation_window": occupation_window,
    }
    selected_names = [
        name for name, value in selectors.items() if value is not None
    ]
    if len(selected_names) > 1:
        raise ValueError(
            "Choose only one RDM-info orbital selector: indices, mo_range, "
            "avas, or occupation_window."
        )

    nmo = C_no.shape[1]
    details = {}
    if indices is not None:
        selected = tuple(dict.fromkeys(int(p) for p in indices))
        selection = "indices"
    elif mo_range is not None:
        if len(mo_range) != 2:
            raise ValueError("mo_range must contain exactly (start, stop).")
        start, stop = (int(value) for value in mo_range)
        if start < 0 or stop > nmo or start >= stop:
            raise ValueError(
                f"mo_range must satisfy 0 <= start < stop <= {nmo}; "
                f"got ({start}, {stop})."
            )
        selected = tuple(range(start, stop))
        selection = "mo_range"
        details["mo_range"] = (start, stop)
    elif occupation_window is not None:
        if len(occupation_window) != 2:
            raise ValueError(
                "occupation_window must contain exactly (minimum, maximum)."
            )
        minimum, maximum = (float(value) for value in occupation_window)
        if minimum > maximum:
            raise ValueError(
                "occupation_window minimum must not exceed its maximum."
            )
        selected = tuple(
            np.flatnonzero(
                (occupations >= minimum) & (occupations <= maximum)
            ).tolist()
        )
        selection = "natural_occupation"
        details["occupation_window"] = (minimum, maximum)
    elif avas is not None:
        if not getattr(avas, "executed", False):
            avas.run()
        if avas.system is not mp2.system:
            raise ValueError(
                "The AVAS and RMP2 calculations must use the same System object."
            )
        if getattr(avas.system, "two_component", False):
            raise TypeError(
                "AVAS-based RDM-info selection requires restricted spatial orbitals."
            )
        if not hasattr(avas, "mo_space") or not hasattr(avas, "mos"):
            raise TypeError("avas must be an executed AVAS calculation.")
        if len(avas.mos.C) != 1:
            raise TypeError(
                "AVAS-based RDM-info selection requires one restricted MO matrix."
            )

        active_indices = tuple(avas.mo_space.active_indices)
        if not active_indices:
            raise ValueError("The AVAS calculation did not select any active orbitals.")
        C_avas_active = np.asarray(avas.mos.C[0])[:, active_indices]
        if C_avas_active.shape[0] != C_no.shape[0]:
            raise ValueError(
                "The AVAS and RMP2 orbitals use incompatible AO dimensions."
            )

        overlap = mp2.system.ints_overlap()
        projection = C_avas_active.conj().T @ overlap @ C_no
        weights = np.sum(np.abs(projection) ** 2, axis=0).real
        # AVAS rotates and reorders its orbitals.  Select common natural
        # orbitals by subspace projection instead of reusing AVAS indices.
        best = np.argsort(-weights, kind="stable")[: len(active_indices)]
        selected = tuple(sorted(int(p) for p in best))
        selection = "avas"
        details["avas_projection_weights"] = weights
    else:
        selected = tuple(range(nmo))
        selection = "all"

    if any(p < 0 or p >= nmo for p in selected):
        raise IndexError(f"Every selected orbital index must be in [0, {nmo}).")
    if not selected:
        raise ValueError("The RDM-info orbital selection is empty.")
    return selected, selection, details


def _ump2_common_natural_orbitals(mp2, gamma1):
    """Build common spin-free UMP2 natural orbitals in the AO metric."""
    gamma1_a, gamma1_b = gamma1
    Ca, Cb = mp2.C
    overlap = mp2.system.ints_overlap()

    gamma1_ao = (
        Ca @ gamma1_a @ Ca.T.conj()
        + Cb @ gamma1_b @ Cb.T.conj()
    )
    gamma1_ao = 0.5 * (gamma1_ao + gamma1_ao.T.conj())

    overlap_evals, overlap_evecs = np.linalg.eigh(
        0.5 * (overlap + overlap.T.conj())
    )
    cutoff = mp2.system.overlap_ortho_rtol * np.max(overlap_evals)
    keep = overlap_evals > cutoff
    if np.count_nonzero(keep) < mp2.nmo:
        raise ValueError(
            "The overlap matrix rank is smaller than the number of UMP2 "
            "orbitals."
        )

    values = overlap_evals[keep]
    vectors = overlap_evecs[:, keep]
    overlap_sqrt = vectors * np.sqrt(values)
    overlap_inv_sqrt = vectors / np.sqrt(values)
    gamma1_orth = overlap_sqrt.T.conj() @ gamma1_ao @ overlap_sqrt
    gamma1_orth = 0.5 * (gamma1_orth + gamma1_orth.T.conj())

    occupations, C_orth = np.linalg.eigh(gamma1_orth)
    order = np.argsort(occupations)[::-1][: mp2.nmo]
    occupations = occupations[order].real
    C_no = overlap_inv_sqrt @ C_orth[:, order]
    Ua = Ca.T.conj() @ overlap @ C_no
    Ub = Cb.T.conj() @ overlap @ C_no
    return C_no, occupations, Ua, Ub


def _rotate_1rdm(gamma1, U):
    rotated = U.T.conj() @ gamma1 @ U
    return 0.5 * (rotated + rotated.T.conj())


def rmp2_mpq_onthefly_no(
    mp2,
    cache_pair_blocks=True,
    cache_fixed_slabs=False,
    compute=False,
    indices=None,
    mo_range=None,
    avas=None,
    occupation_window=None,
    include_quadratic=False,
):
    """Construct a low-cost RMP2 RDM-information analyzer.

    The RMP2 block natural orbitals provide a restricted orbital basis that is
    compatible with AVAS.  When ``avas`` is supplied, its active subspace is
    mapped into that basis by AO-overlap projection.  The other selectors have
    the same meanings as in :func:`ump2_mpq_onthefly_no`.  Constructing the
    returned analyzer exposes the block-NO coefficients, occupations, and
    canonical-MO transformation as ``C_no``, ``no_occs``, and ``U``.

    Parameters
    ----------
    mp2 : RMP2
        Executed restricted MP2 calculation.
    cache_pair_blocks, cache_fixed_slabs : bool, optional
        Control rotated-pair and canonical-slab amplitude caching.
    compute : bool, optional
        Compute M1 and M2 before returning the analyzer.
    indices : iterable[int], optional
        Explicit block-NO indices.
    mo_range : tuple[int, int], optional
        Half-open block-NO range ``(start, stop)``.
    avas : AVAS, optional
        Restricted AVAS calculation on the same system.
    occupation_window : tuple[float, float], optional
        Inclusive block-natural-occupation window.
    include_quadratic : bool, optional
        Include additional cumulant contractions quadratic in the first-order
        MP2 doubles amplitudes. Disabled by default; the mutual-correlation
        construction used in the paper retains only the first-order
        ``oovv``/``vvoo`` cumulant contribution.

    Returns
    -------
    RMP2MPQOnTheFly
        Analyzer configured with the selected RDM-info orbital space.
    """
    analyzer = RMP2MPQOnTheFly(
        mp2,
        cache_pair_blocks=cache_pair_blocks,
        cache_fixed_slabs=cache_fixed_slabs,
        include_quadratic=include_quadratic,
    )
    selected, selection, selection_details = _resolve_rdm_info_indices(
        mp2,
        analyzer.C_no,
        analyzer.occs,
        indices=indices,
        mo_range=mo_range,
        avas=avas,
        occupation_window=occupation_window,
    )
    analyzer.rdm_info_indices = selected
    analyzer.rdm_info_selection = selection
    analyzer.rdm_info_selection_details = selection_details

    if compute:
        analyzer.make_measures()

    return analyzer


def ump2_mpq_onthefly_no(
    mp2,
    cache_pair_blocks=True,
    cache_fixed_slabs=False,
    compute=False,
    indices=None,
    mo_range=None,
    avas=None,
    occupation_window=None,
    include_quadratic=False,
    common_no_mixing_tolerance=1.0e-10,
    common_no_transform="block_projected",
    max_exact_tensor_memory_mb=512.0,
    max_exact_orbitals=30,
):
    """Construct a low-cost UMP2 RDM-information analyzer.

    The analysis is performed in the common spin-free UMP2 natural-orbital
    basis.  Its cost can be restricted with exactly one of ``indices``,
    ``mo_range``, or ``occupation_window``.  ``mo_range`` follows Python's
    half-open convention, so ``(0, 50)`` selects orbitals 0 through 49.
    Passing ``occupation_window=(0.02, 1.98)`` selects partially occupied
    common natural orbitals.  AVAS is restricted-only and is therefore handled
    by :func:`rmp2_mpq_onthefly_no` instead.

    By default, the analyzer retains the first-order MP2 ``oovv``/``vvoo``
    cumulant contribution. Returned M1/M2 arrays keep their full-space shapes
    and contain zeros outside the selected RDM-info space.

    Parameters
    ----------
    mp2 : UMP2
        Executed UMP2 calculation.  Its density-fitting factors are used to
        generate amplitude blocks on demand.
    cache_pair_blocks : bool, optional
        Retain rotated occupied-pair amplitude blocks for reuse.
    cache_fixed_slabs : bool, optional
        Retain canonical fixed-occupied amplitude slabs.  This is faster but
        can grow to full-amplitude memory, so it is disabled by default.
    compute : bool, optional
        Compute M1 and M2 before returning the analyzer.
    indices : iterable[int], optional
        Explicit common-NO indices.  Mutually exclusive with the other orbital
        selectors.
    mo_range : tuple[int, int], optional
        Half-open common-NO range ``(start, stop)``.
    avas : AVAS, optional
        Unsupported for UMP2.  Passing it raises an error directing the caller
        to :func:`rmp2_mpq_onthefly_no`.
    occupation_window : tuple[float, float], optional
        Inclusive natural-occupation window.  Use ``(0.02, 1.98)`` for the
        conventional partially occupied space.
    include_quadratic : bool, optional
        Include additional cumulant contractions quadratic in the first-order
        MP2 doubles amplitudes. This diagnostic option is disabled by default
        and is not part of the paper's production definition.
    common_no_mixing_tolerance : float, optional
        Warning threshold for discarded occupied-virtual mixing in the
        low-cost block-projected common-NO transformation.
    common_no_transform : {"block_projected", "exact_selected"}, optional
        Use the legacy occupied/virtual block projection or apply the full
        alpha and beta MO-to-common-NO transformations to the retained
        cumulant blocks restricted to the selected orbital space.  When the
        optional selected-quadratic diagnostic is requested, its retained
        source blocks are transformed by the same exact selected-space route.
    max_exact_tensor_memory_mb : float, optional
        Explicit peak-memory budget for ``common_no_transform="exact_selected"``.
        Exceeding it raises instead of silently using block projection.
    max_exact_orbitals : int, optional
        Hard cap on the number of common natural orbitals transformed in exact
        selected-space mode.  The default is 30.

    Returns
    -------
    UMP2MPQOnTheFly
        Analyzer configured with the selected RDM-info orbital space.
    """
    if avas is not None:
        raise TypeError(
            "AVAS selection requires restricted orbitals; use "
            "rmp2_mpq_onthefly_no with an RMP2 calculation."
        )

    gamma1 = mp2.make_1rdm_sd()
    no_transform = _ump2_common_natural_orbitals(mp2, gamma1)
    C_no, occupations, Ua, Ub = no_transform
    selected, selection, selection_details = _resolve_rdm_info_indices(
        mp2,
        C_no,
        occupations,
        indices=indices,
        mo_range=mo_range,
        occupation_window=occupation_window,
    )

    analyzer = UMP2MPQOnTheFly(
        mp2,
        Ua=Ua,
        Ub=Ub,
        gamma1=gamma1,
        orbital_indices=selected,
        include_quadratic=include_quadratic,
        cache_pair_blocks=cache_pair_blocks,
        cache_fixed_slabs=cache_fixed_slabs,
        common_no_mixing_tolerance=common_no_mixing_tolerance,
        common_no_transform=common_no_transform,
        max_exact_tensor_memory_mb=max_exact_tensor_memory_mb,
        max_exact_orbitals=max_exact_orbitals,
    )

    gamma1_no_a = _rotate_1rdm(gamma1[0], Ua)
    gamma1_no_b = _rotate_1rdm(gamma1[1], Ub)
    gamma1_no = gamma1_no_a + gamma1_no_b

    analyzer.C_no = C_no
    analyzer.no_occs = occupations
    analyzer.no_transform = no_transform
    analyzer.gamma1_no_a = gamma1_no_a
    analyzer.gamma1_no_b = gamma1_no_b
    analyzer.gamma1_a = gamma1_no_a
    analyzer.gamma1_b = gamma1_no_b
    analyzer.γa = gamma1_no_a
    analyzer.γb = gamma1_no_b
    analyzer.Gamma1_no = gamma1_no
    analyzer.Gamma1 = gamma1_no
    analyzer.Γ1 = gamma1_no
    analyzer.rdm_info_selection = selection
    analyzer.rdm_info_selection_details = selection_details

    if compute:
        analyzer.make_measures()

    return analyzer
