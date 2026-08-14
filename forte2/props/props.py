import numpy as np

from forte2 import integrals
from forte2.data import DEBYE_TO_AU, ANGSTROM_TO_BOHR
from forte2.helpers.matrix_functions import block_diag_2x2
from .mutual_correlation import RMP2MPQOnTheFly, UMP2MPQOnTheFly


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
                "The AVAS and UMP2 calculations must use the same System object."
            )
        if getattr(avas.system, "two_component", False):
            raise TypeError(
                "AVAS-based RDM-info selection requires restricted spatial orbitals."
            )
        if not hasattr(avas, "mo_space") or not hasattr(avas, "C"):
            raise TypeError("avas must be an executed AVAS calculation.")
        if len(avas.C) != 1:
            raise TypeError(
                "AVAS-based RDM-info selection requires one restricted MO matrix."
            )

        active_indices = tuple(avas.mo_space.active_indices)
        if not active_indices:
            raise ValueError("The AVAS calculation did not select any active orbitals.")
        C_avas_active = np.asarray(avas.C[0])[:, active_indices]
        if C_avas_active.shape[0] != C_no.shape[0]:
            raise ValueError(
                "The AVAS and UMP2 orbitals use incompatible AO dimensions."
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


def rmp2_mpq_onthefly_no(
    mp2,
    cache_pair_blocks=True,
    cache_fixed_slabs=False,
    compute=False,
    indices=None,
    mo_range=None,
    avas=None,
    occupation_window=None,
    include_quadratic=True,
):
    """Construct a low-cost RMP2 RDM-information analyzer.

    The RMP2 block natural orbitals provide a restricted orbital basis that is
    compatible with AVAS.  When ``avas`` is supplied, its active subspace is
    mapped into that basis by AO-overlap projection.  The other selectors have
    the same meanings as in :func:`ump2_mpq_onthefly_no`.

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
        Include retained cumulant terms quadratic in the MP2 amplitudes.

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
    include_quadratic=True,
    common_no_mixing_tolerance=1.0e-10,
):
    """Construct a low-cost UMP2 RDM-information analyzer.

    The analysis is performed in the common spin-free UMP2 natural-orbital
    basis.  Its cost can be restricted with exactly one of ``indices``,
    ``mo_range``, or ``occupation_window``.  ``mo_range`` follows Python's
    half-open convention, so ``(0, 50)`` selects orbitals 0 through 49.
    Passing ``occupation_window=(0.02, 1.98)`` selects partially occupied
    common natural orbitals.  AVAS is restricted-only and is therefore handled
    by :func:`rmp2_mpq_onthefly_no` instead.

    Set ``include_quadratic=False`` to retain only cumulant terms linear in the
    MP2 doubles amplitudes.  Returned M1/M2 arrays keep their full-space shapes
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
        Include the retained cumulant terms quadratic in the MP2 amplitudes.
    common_no_mixing_tolerance : float, optional
        Warning threshold for discarded occupied-virtual mixing in the
        low-cost block-projected common-NO transformation.

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
    no_transform = mp2.make_natural_orbital_transform(gamma1)
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
    )

    gamma1_no_a, gamma1_no_b = mp2.make_1rdm_no_sd(gamma1, no_transform)
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
