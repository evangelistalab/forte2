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

    def __init__(self, ci, fragments=None, root=0):

        self.ci = ci
        self.root = root

        if fragments is None:
            self.fragments = [[orb] for orb in ci.active_indices]
        else:
            self.fragments = [list(f) for f in fragments]

        self.nfragments = len(self.fragments)

        self.gamma1 = _spin_summed_1rdm(ci, root)

        self.M1 = {}
        self.M2 = {}
        self.M3 = {}
        self.M4 = {}

        self._build_terms()

    def _build_terms(self):

        n = self.nfragments

        for i in range(n):
            self.M1[(i,)] = onefrag_correlation_energy_enumerated(
                self.ci,
                self.fragments[i],
                root=self.root,
            )

        for i, j in combinations(range(n), 2):
            self.M2[(i, j)] = twofrag_correlation_energy_enumerated(
                self.ci,
                self.fragments[i],
                self.fragments[j],
                root=self.root,
            )

        for i, j, k in combinations(range(n), 3):
            self.M3[(i, j, k)] = threefrag_correlation_energy_enumerated(
                self.ci,
                self.fragments[i],
                self.fragments[j],
                self.fragments[k],
                root=self.root,
            )

        for i, j, k, l in combinations(range(n), 4):
            self.M4[(i, j, k, l)] = fourfrag_correlation_energy_enumerated(
                self.ci,
                self.fragments[i],
                self.fragments[j],
                self.fragments[k],
                self.fragments[l],
                root=self.root,
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

def _spin_summed_1rdm(ci, root=0):
    ci_solver = ci.sub_solvers[0]
    a1, b1 = ci_solver.make_sd_1rdm(root)
    return a1 + b1


def _fragment_1rdm(ci, fragments, root=0):
    gamma1 = _spin_summed_1rdm(ci, root=root)
    _, global_subs, _ = subspaces(
        *fragments,
        rdm_orbitals=ci.mo_space.active_indices,
    )
    fragment_gamma1 = np.zeros((len(fragments), len(fragments)))
    for i, p_subspace in enumerate(global_subs):
        for j, q_subspace in enumerate(global_subs):
            fragment_gamma1[i, j] = np.sum(gamma1[np.ix_(p_subspace, q_subspace)])
    return fragment_gamma1


def _fragment_labels(fragments):
    return [fragment[0] if len(fragment) == 1 else tuple(fragment) for fragment in fragments]

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

def _spin_dependent_2cumulants(ci, root=0):
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
def _spin_summed_2cumulant(ci, root=0):
    return _spin_dependent_2cumulants(ci, root=root)[3]

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

def _local_spin_summed_cumulant(ci, local_to_global, root=0):
    local_to_global = np.asarray(local_to_global)
    return _spin_summed_2cumulant(ci, root=root)[
        np.ix_(local_to_global, local_to_global, local_to_global, local_to_global)
    ]

#Build 1-RDMs and 2-cumulants with forte2
#This function is defunct but preserved in case needed later on
def _spin_dependent_rdms(ci, root=0):
    n = len(ci.mo_space.active_indices)
    zeros = np.zeros((n, n))
    aa2_cumulant, ab2_cumulant, bb2_cumulant, _ = _spin_dependent_2cumulants(
        ci, root=root
    )

    return zeros, zeros, aa2_cumulant, ab2_cumulant, bb2_cumulant

#Combine 1, 2, 3, and 4-body mutual correlation energy into one function
def fragment_correlation_energy_enumerated(
    ci,
    *orbital_fragments,
    body_order=None,
    core_orbitals=(),
    root=0,
):
    #Compute an n-fragment energy contribution by explicit index enumeration.

    #One-electron terms contribute only to body_order 1 or 2. Two-electron terms
    #can contribute to body_order 1, 2, 3, or 4.
    
    if body_order is None:
        body_order = len(orbital_fragments) 

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
        C=ci.C[0],
        orbitals=frag_orbs,
        core_orbitals=list(core_orbitals),
    )
    V = ints.V

    #The mod variant intentionally suppresses one-electron contributions.
    e1 = 0.0

    #two-electron terms
    mask = _unique_fragment_count_mask(local_to_fragment, body_order)
    spin_block_sum = _local_spin_summed_cumulant(ci, local_to_global, root=root)
    e2 = 0.5 * np.sum(V[mask] * spin_block_sum[mask])

    return e1 + e2

def onefrag_correlation_energy_enumerated(ci, A_orbs, core_orbitals=(), root=0):
    return fragment_correlation_energy_enumerated(
        ci,
        A_orbs,
        body_order=1,
        core_orbitals=core_orbitals,
        root=root,
    )

def twofrag_correlation_energy_enumerated(ci, A_orbs, B_orbs, core_orbitals=(), root=0):
    return fragment_correlation_energy_enumerated(
        ci,
        A_orbs,
        B_orbs,
        body_order=2,
        core_orbitals=core_orbitals,
        root=root,
    )

def threefrag_correlation_energy_enumerated(ci, A_orbs, B_orbs, C_orbs, core_orbitals=(), root=0):
    return fragment_correlation_energy_enumerated(
        ci,
        A_orbs,
        B_orbs,
        C_orbs,
        body_order=3,
        core_orbitals=core_orbitals,
        root=root,
    )

def fourfrag_correlation_energy_enumerated(ci, A_orbs, B_orbs, C_orbs, D_orbs, core_orbitals=(), root=0):
    return fragment_correlation_energy_enumerated(
        ci,
        A_orbs,
        B_orbs,
        C_orbs,
        D_orbs,
        body_order=4,
        core_orbitals=core_orbitals,
        root=root,
    )


def fragment_decomposition_energy_enumerated(
    ci,
    fragments,
    max_body_order=4,
    core_orbitals=(),
    root=0,
):
      
    #Return Ecore plus all enumerated fragment terms through max_body_order.
    fragments = [list(fragment) for fragment in fragments]
    all_fragment_orbs, local_to_global, local_to_fragment = _fragment_index_maps(
        ci, fragments
    )

    ints = forte2.jkbuilder.RestrictedMOIntegrals(
        system=ci.system,
        C=ci.C[0],
        orbitals=all_fragment_orbs,
        core_orbitals=list(core_orbitals),
    )
    spin_block_sum = _local_spin_summed_cumulant(ci, local_to_global, root=root)

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
        Ecore=ints.E,
        terms=terms,
        decomposition_energy=total,
        ci_energy=ci.E,
        residual=ci.E - total,
        gamma1=_fragment_1rdm(ci, fragments, root=root),
        active_mo_indices=_fragment_labels(fragments),
        fragments=fragments,
        max_body_order=max_body_order,
    )
