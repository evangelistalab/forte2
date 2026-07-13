import forte2
import numpy as np
from itertools import combinations, permutations
from subspaces import subspaces

_CUMULANT_CACHE = {}


class MutualCorrelationEnergyAnalysis:
    """
    Container for fragment correlation energy analysis.

    Examples
    --------
    mca = MutualCorrelationEnergyAnalysis(ci)

    mca.M1[(0,)]
    mca.M2[(0,1)]
    mca.total_correlation
    """

    def __init__(
        self,
        ci,
        fragments=None,
        root=0,
        max_body_order=4,
        core_orbitals=None,
        nat_orbs=False,
    ):

        self.ci = ci
        self.root = root
        self.max_body_order = max_body_order
        self.core_orbitals = core_orbitals
        self.nat_orbs = nat_orbs

        if fragments is None:
            self.fragments = [[orb] for orb in ci.mo_space.active_indices]
        else:
            self.fragments = [list(f) for f in fragments]

        self.nfragments = len(self.fragments)

        basis = _analysis_basis(
                                ci,
                                root=root,
                                nat_orbs=nat_orbs,
                                )

        self.C = basis["C"]
        self.gamma1 = basis["gamma1"]
        self.no_occupations = basis["occupations"]
        self.orbital_rotation = basis["orbital_rotation"]
        # self.Gamma1 = self.gamma1
        # setattr(self, "\u03931", self.gamma1)

        self.M1 = {}
        self.M2 = {}
        self.M3 = {}
        self.M4 = {}

        self._build_terms()

    def _build_terms(self):

        n = self.nfragments

        max_order = min(self.max_body_order, n, 4)
        if max_order >= 1:
            for i in range(n):
                self.M1[(i,)] = onefrag_correlation_energy_enumerated(
                    self.ci,
                    self.fragments[i],
                    core_orbitals=self.core_orbitals,
                    root=self.root,
                    C=self.C,
                    orbital_rotation=self.orbital_rotation,
                )

        if max_order >= 2:
            for i, j in combinations(range(n), 2):
                self.M2[(i, j)] = twofrag_correlation_energy_enumerated(
                    self.ci,
                    self.fragments[i],
                    self.fragments[j],
                    core_orbitals=self.core_orbitals,
                    root=self.root,
                    C=self.C,
                    orbital_rotation=self.orbital_rotation,
                )

        if max_order >= 3:
            for i, j, k in combinations(range(n), 3):
                self.M3[(i, j, k)] = threefrag_correlation_energy_enumerated(
                    self.ci,
                    self.fragments[i],
                    self.fragments[j],
                    self.fragments[k],
                    core_orbitals=self.core_orbitals,
                    root=self.root,
                    C=self.C,
                    orbital_rotation=self.orbital_rotation,
                )

        if max_order >= 4:
            for i, j, k, l in combinations(range(n), 4):
                self.M4[(i, j, k, l)] = fourfrag_correlation_energy_enumerated(
                    self.ci,
                    self.fragments[i],
                    self.fragments[j],
                    self.fragments[k],
                    self.fragments[l],
                    core_orbitals=self.core_orbitals,
                    root=self.root,
                    C=self.C,
                    orbital_rotation=self.orbital_rotation,
                )

        self.total_correlation = (
            sum(self.M1.values())
            + sum(self.M2.values())
            + sum(self.M3.values())
            + sum(self.M4.values())
        )

    @property
    def fragment_labels(self):
        return [
            frag[0] if len(frag) == 1 else tuple(frag)
            for frag in self.fragments
        ]

    def get_M2_matrix(self):
        """
        Return a symmetric NxN M2 matrix.
        """
        n = self.nfragments
        M2 = np.zeros((n, n))

        for (i, j), value in self.M2.items():
            M2[i, j] = value
            M2[j, i] = value

        return M2

    def get_M3_tensor(self):
        """
        Return full symmetric M3 tensor.
        """
        n = self.nfragments
        M3 = np.zeros((n, n, n))

        from itertools import permutations

        for inds, value in self.M3.items():
            for perm in set(permutations(inds)):
                M3[perm] = value

        return M3

    def get_M4_tensor(self):
        """
        Return full symmetric M4 tensor.
        """
        n = self.nfragments
        M4 = np.zeros((n, n, n, n))

        from itertools import permutations

        for inds, value in self.M4.items():
            for perm in set(permutations(inds)):
                M4[perm] = value

        return M4

    def mutual_correlation_matrix_summary(
        self,
        print_threshold=7.5e-4,
    ):

        lines = [
            f"Total Correlation Energy (Sum of the 2-Cumulant Terms): {self.total_correlation:16.10f}",
            "",
            f"M2 Terms (|value| > {print_threshold:.1e})",
            "------------------------------------------",
        ]

        entries = [
            (abs(v), v, i, j)
            for (i, j), v in self.M2.items()
        ]
        entries.sort(reverse=True)
        labels = self.fragment_labels

        for _, value, i, j in entries:
            if abs(value) < print_threshold:
                break

            lines.append(
                f"{str(labels[i]):>8} "
                f"{str(labels[j]):>8} "
                f"{value:16.10f}"
            )

        return "\n".join(lines)

    def __repr__(self):
        return (
            f"MutualCorrelationEnergyAnalysis("
            f"nfragments={self.nfragments}, "
            f"total_correlation={self.total_correlation:.8f})"
        )

def natural_orbitals_from_gamma1(C_act, gamma1_act):
    """
    Construct active-space natural orbitals from the spin-summed active-space
    one-particle RDM.

    Only the active orbitals are rotated. Core and external virtual orbitals
    remain unchanged.

    Parameters
    ----------
    C_act : ndarray
        Active-space molecular orbital coefficient matrix.
    gamma1_act : ndarray
        Spin-summed active-space 1-RDM.

    Returns
    -------
    C_no : ndarray
        Active-space orbital coefficients in the natural orbital basis.
    occupations : ndarray
        Natural orbital occupation numbers.
    orbital_rotation : ndarray
        Active-space orbital rotation matrix.
    """

    gamma1_act = 0.5 * (gamma1_act + gamma1_act.T)

    occupations, orbital_rotation = np.linalg.eigh(gamma1_act)

    order = np.argsort(occupations)[::-1]

    occupations = occupations[order]
    orbital_rotation = orbital_rotation[:, order]

    C_no = C_act @ orbital_rotation

    return C_no, occupations, orbital_rotation


def _analysis_basis(ci, root=0, nat_orbs=False):
    """
    Build the orbital basis used throughout the correlation analysis.
    When ``nat_orbs`` is False, the molecular orbital basis is returned
    unchanged.
    When ``nat_orbs`` is True, the active-space orbitals are rotated into
    the natural orbital basis obtained by diagonalizing the active-space
    spin-summed 1-RDM. Inactive orbitals remain in the input orbital basis.
    
    Returns
    -------
    dict: Dictionary containing
        C: Orbital coefficient matrix (assumed to be semi-canonical, user-supplied).
        gamma1: Spin-summed one-particle RDM expressed in the analysis basis.
        occupations: Natural orbital occupation numbers in the transformed active basis, 
            or None if molecular orbitals are used.
        orbital_rotation: Active-space orbital rotation matrix, or None.
            (previously denoted as 'U')
    """
    
    gamma1 = _spin_summed_1rdm(ci, root=root)
    if not nat_orbs:
        return {
            "C": ci.C[0],
            "gamma1": gamma1,
            "occupations": None,
            "orbital_rotation": None,
        }

    C = np.array(ci.C[0], copy=True)

    active_indices = list(ci.mo_space.active_indices)

    C_no, occupations, orbital_rotation = natural_orbitals_from_gamma1(
        C[:, active_indices],
        gamma1,
    )

    C[:, active_indices] = C_no
    gamma1 = _transform_1rdm(gamma1, orbital_rotation)

    return {"C": C, 
            "gamma1": gamma1, 
            "occupations": occupations, 
            "orbital_rotation": orbital_rotation}

def _transform_1rdm(gamma1, orbital_rotation):
    """
    Rotate a one-particle RDM into a new orbital basis.
    """
    return orbital_rotation.T @ gamma1 @ orbital_rotation if orbital_rotation is not None else gamma1

def _transform_2tensor(tensor, orbital_rotation):
    """
    Rotate a four-index tensor into a new orbital basis.
    """
    if orbital_rotation is None:
        return tensor

    return np.einsum(
        "pqrs,pi,qj,rk,sl->ijkl",
        tensor,
        orbital_rotation,
        orbital_rotation,
        orbital_rotation,
        orbital_rotation,
        optimize=True,
    )

def _spin_summed_1rdm(ci, root=0):
    ci_solver = ci.sub_solvers[0]
    a1, b1 = ci_solver.make_sd_1rdm(root)
    return a1 + b1

def _hashable_value(value):
    if isinstance(value, np.ndarray):
        return tuple(value.reshape(-1).tolist())
    if isinstance(value, list):
        return tuple(value)
    return value

#generate index maps for the 1, 2RDM and 2-electron integral V, 
# which use different indexing conventions
def _fragment_index_maps(ci, orbital_fragments):

    frag_orbs, global_subs, local_subs = subspaces(
        *orbital_fragments,
        rdm_orbitals=ci.mo_space.active_indices,
    )

    nlocal = len(frag_orbs)
    local_to_global = np.empty(nlocal, dtype=int)
    local_to_fragment = np.empty(nlocal, dtype=int)

    for frag_id, (global_sub, local_sub) in enumerate(zip(global_subs, local_subs)):
        for g, l in zip(global_sub, local_sub):
            local_to_global[l] = g
            local_to_fragment[l] = frag_id

    return frag_orbs, local_to_global, local_to_fragment

def clear_cache():
    """Clear cached spin-dependent cumulants."""
    _CUMULANT_CACHE.clear()

def _cumulant_cache_key(ci, root):
    return (
        id(ci),
        root,
        _hashable_value(getattr(ci, "E", None)),
        tuple(ci.mo_space.active_indices),
    )

#Build spin-dependent 2-cumulants with forte2.
#the cached method avoids rebuilding the spin-summed cumulants inside each iteration of orbitals pq..

def _spin_dependent_cumulants(ci, root=0, orbital_rotation=None):
    """
    Returns
    -------
    aa2_cumulant
    ab2_cumulant
    bb2_cumulant
    spin_summed_cumulant:
        aa2_cumulant + ab2_cumulant + transpose(ab2_cumulant) + bb2_cumulant
    """
    if orbital_rotation is not None:
        aa2, ab2, bb2, _ = _spin_dependent_cumulants(
            ci, root=root, orbital_rotation=None
        )
        aa2 = _transform_2tensor(aa2, orbital_rotation)
        ab2 = _transform_2tensor(ab2, orbital_rotation)
        bb2 = _transform_2tensor(bb2, orbital_rotation)
        spin_summed_cumulant = (
            aa2
            + ab2
            + ab2.transpose(1, 0, 3, 2)
            + bb2
        )
        return aa2, ab2, bb2, spin_summed_cumulant

    cache_key = _cumulant_cache_key(ci, root)
    if cache_key in _CUMULANT_CACHE:
        return _CUMULANT_CACHE[cache_key]

    ci_solver = ci.sub_solvers[0]

    a1, b1 = ci_solver.make_sd_1rdm(root)
    aa_pair, ab2, bb_pair = ci_solver.make_sd_2rdm(root)

    aa2 = forte2.cpp_helpers.packed_tensor4_to_tensor4(aa_pair)
    bb2 = forte2.cpp_helpers.packed_tensor4_to_tensor4(bb_pair)

    aa2_cumulant = (
        aa2
        - np.einsum("pr,qs->pqrs", a1, a1)
        + np.einsum("ps,qr->pqrs", a1, a1)
    )
    ab2_cumulant = ab2 - np.einsum("pr,qs->pqrs", a1, b1)
    bb2_cumulant = (
        bb2
        - np.einsum("pr,qs->pqrs", b1, b1)
        + np.einsum("ps,qr->pqrs", b1, b1)
    )
    # still setting 1-rdms to zero, after 2-cumulant is built
    spin_summed_cumulant = (
        aa2_cumulant
        + ab2_cumulant
        + ab2_cumulant.transpose(1, 0, 3, 2)
        + bb2_cumulant
    )
    cumulants = (aa2_cumulant, ab2_cumulant, bb2_cumulant, spin_summed_cumulant)
    _CUMULANT_CACHE[cache_key] = cumulants
    return cumulants

#Build spin-summed 2-cumulant with forte2
def _spin_summed_2cumulant(ci, root=0, orbital_rotation=None):
    return _spin_dependent_cumulants(
        ci, root=root, orbital_rotation=orbital_rotation
    )[3]

#an accelerated way to count fragments and determine body order
def _unique_fragment_count_mask(local_to_fragment, body_order):
    frag_ids = np.asarray(local_to_fragment)
    f0 = frag_ids[:, None, None, None]
    f1 = frag_ids[None, :, None, None]
    f2 = frag_ids[None, None, :, None]
    f3 = frag_ids[None, None, None, :]
    unique_count = (
        1
        + (f1 != f0)
        + ((f2 != f0) & (f2 != f1))
        + ((f3 != f0) & (f3 != f1) & (f3 != f2))
    )
    return unique_count == body_order

def _fragment_combo_mask(local_to_fragment, combo):
    frag_ids = np.asarray(local_to_fragment)
    combo = np.asarray(combo)
    allowed = np.isin(frag_ids, combo)
    return (
        _unique_fragment_count_mask(frag_ids, len(combo))
        & allowed[:, None, None, None]
        & allowed[None, :, None, None]
        & allowed[None, None, :, None]
        & allowed[None, None, None, :]
    )

def _local_spin_summed_cumulant(
    ci, local_to_global, root=0, orbital_rotation=None
):
    local_to_global = np.asarray(local_to_global)
    return _spin_summed_2cumulant(
        ci, root=root, orbital_rotation=orbital_rotation
    )[
        np.ix_(local_to_global, local_to_global, local_to_global, local_to_global)
    ]

#Combine 1, 2, 3, and 4-body mutual correlation energy into one function
def fragment_correlation_energy_enumerated(
    ci,
    *orbital_fragments,
    body_order=None,
    core_orbitals=None,
    root=0,
    C=None,
    orbital_rotation=None,
):
    
    """
    Compute an n-fragment correlation contribution by explicit
    enumeration of orbital indices.

    One-electron contributions are included only for one- and two-body
    terms, while two-electron cumulant contributions are included for
    body orders one through four.

    If ``orbital_rotation`` is supplied, the cumulants are rotated into
    the corresponding orbital basis before the energy is evaluated.
    """

    if body_order is None:
        body_order = len(orbital_fragments) 
    
    if core_orbitals is None:
        core_orbitals = tuple(ci.core_indices)
    else:
        core_orbitals = tuple(core_orbitals)
        #check that user-supplied core orbs match ci calc
        if core_orbitals != tuple(ci.core_indices):
            raise ValueError(
                "The supplied core_orbitals do not match ci.core_indices.\n"
                f"core_orbitals = {core_orbitals}\n"
                f"ci.core_indices = {tuple(ci.core_indices)}"
            )
        
    if C is None:
        C = ci.C[0]    
    
    #One-electron terms contribute only to body_order 1 or 2. Two-electron terms
    #can contribute to body_order 1, 2, 3, or 4.
    if body_order < 1 or body_order > 4:
        raise ValueError(f"body_order must be between 1 and 4; got {body_order}")
    if body_order > len(orbital_fragments):
        raise ValueError(
            f"body_order={body_order} cannot exceed the number of fragments "
            f"({len(orbital_fragments)})"
        )

    #Check to see if an orbital is repeated in different frags
    fragments = [list(fragment) for fragment in orbital_fragments]
    seen = set()
    overlap = set()
    for fragment in fragments:
        fragment_set = set(fragment)
        overlap |= seen & fragment_set
        seen |= fragment_set
    if overlap:
        raise ValueError(
            f"Orbital fragments must be disjoint; shared orbitals found: {sorted(overlap)}"
        )
    #build index maps, rdms using helper functions
    frag_orbs, local_to_global, local_to_fragment = _fragment_index_maps(ci, fragments)
    ints = forte2.jkbuilder.RestrictedMOIntegrals(
        system=ci.system,
        C=C,
        orbitals=frag_orbs,
        core_orbitals=list(core_orbitals),
    )
    V = ints.V

    #The mod variant intentionally suppresses one-electron contributions.
    e1 = 0.0

    #two-electron terms
    mask = _unique_fragment_count_mask(local_to_fragment, body_order)
    spin_block_sum = _local_spin_summed_cumulant(
        ci,
        local_to_global,
        root=root,
        orbital_rotation=orbital_rotation,
    )
    e2 = 0.5 * np.sum(V[mask] * spin_block_sum[mask])

    return e1 + e2

def onefrag_correlation_energy_enumerated(
    ci, A_orbs, core_orbitals=None, root=0, **kwargs
):
    return fragment_correlation_energy_enumerated(
        ci,
        A_orbs,
        body_order=1,
        core_orbitals=core_orbitals,
        root=root,
        **kwargs,
    )

def twofrag_correlation_energy_enumerated(
    ci, A_orbs, B_orbs, core_orbitals=None, root=0, **kwargs
):
    return fragment_correlation_energy_enumerated(
        ci,
        A_orbs,
        B_orbs,
        body_order=2,
        core_orbitals=core_orbitals,
        root=root,
        **kwargs,
    )

def threefrag_correlation_energy_enumerated(
    ci, A_orbs, B_orbs, C_orbs, core_orbitals=None, root=0, **kwargs
):
    return fragment_correlation_energy_enumerated(
        ci,
        A_orbs,
        B_orbs,
        C_orbs,
        body_order=3,
        core_orbitals=core_orbitals,
        root=root,
        **kwargs,
    )

def fourfrag_correlation_energy_enumerated(
    ci, A_orbs, B_orbs, C_orbs, D_orbs, core_orbitals=None, root=0, **kwargs
):
    return fragment_correlation_energy_enumerated(
        ci,
        A_orbs,
        B_orbs,
        C_orbs,
        D_orbs,
        body_order=4,
        core_orbitals=core_orbitals,
        root=root,
        **kwargs,
    )


def fragment_decomposition_energy_enumerated(
    ci,
    fragments,
    max_body_order=4,
    core_orbitals=None,
    root=0,
    nat_orbs=False,
):
      
    #Return Ecore plus all enumerated fragment terms through max_body_order.
    fragments = [list(fragment) for fragment in fragments]
    all_fragment_orbs, local_to_global, local_to_fragment = _fragment_index_maps(
        ci, fragments
    )
    basis = _analysis_basis(ci, root=root, nat_orbs=nat_orbs)

    if core_orbitals is None:
        core_orbitals = tuple(ci.core_indices)
    else:
        core_orbitals = tuple(core_orbitals)
        #check that user-supplied core orbs match ci calc
        if core_orbitals != tuple(ci.core_indices):
            raise ValueError(
                "The supplied core_orbitals do not match ci.core_indices.\n"
                f"core_orbitals = {core_orbitals}\n"
                f"ci.core_indices = {tuple(ci.core_indices)}"
            )

    ints = forte2.jkbuilder.RestrictedMOIntegrals(
        system=ci.system,
        C=basis["C"],
        orbitals=all_fragment_orbs,
        core_orbitals=core_orbitals,
    )
    spin_block_sum = _local_spin_summed_cumulant(
        ci,
        local_to_global,
        root=root,
        orbital_rotation=basis["orbital_rotation"],
    )

    terms = {}
    total = ints.E
    for order in range(1, min(max_body_order, len(fragments), 4) + 1):
        order_total = 0.0
        for combo in combinations(range(len(fragments)), order):
            mask = _fragment_combo_mask(local_to_fragment, combo)
            value = 0.5 * np.sum(ints.V[mask] * spin_block_sum[mask])
            terms[combo] = value
            order_total += value
        total += order_total

    return MutualCorrelationEnergyAnalysis(
        ci,
        fragments,
        root=root,
        max_body_order=max_body_order,
        core_orbitals=core_orbitals,
        nat_orbs=nat_orbs,
    )
