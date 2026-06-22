import forte2
import numpy as np
from itertools import combinations, product
from subspaces import subspaces

#generate index maps for the 1, 2RDM and 2-electron integral V, which use different indexing conventions
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

#Build 1 and 2-RDMS with forte2
def _spin_dependent_rdms(ci, root=0):
    
    ci_solver = ci.sub_solvers[0]

    a1, b1 = ci_solver.make_sd_1rdm(root)
    aa_pair, ab2, bb_pair = ci_solver.make_sd_2rdm(root)

    aa2 = forte2.cpp_helpers.packed_tensor4_to_tensor4(aa_pair)
    bb2 = forte2.cpp_helpers.packed_tensor4_to_tensor4(bb_pair)

    return a1, b1, aa2, ab2, bb2

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
    a1, b1, aa2, ab2, bb2 = _spin_dependent_rdms(ci, root=root)

    ints = forte2.jkbuilder.RestrictedMOIntegrals(
        system=ci.system,
        C=ci.C[0],
        orbitals=frag_orbs,
        core_orbitals=list(core_orbitals),
    )
    H = ints.H
    V = ints.V

    nlocal = len(frag_orbs)
    local_range = range(nlocal)

    #one-electron contribution to 1- and 2-frag correlation energy
    e1 = 0.0
    if body_order <= 2:
        for p, q in product(local_range, repeat=2):
            if len({local_to_fragment[p], local_to_fragment[q]}) == body_order:
                gp, gq = local_to_global[[p, q]]
                e1 += H[p, q] * (a1[gp, gq] + b1[gp, gq])
    #two-electron terms
    e2 = 0.0
    for p, q, r, s in product(local_range, repeat=4): #iterates over all indices instead of exploiting symmetry
        if len({local_to_fragment[p],local_to_fragment[q],local_to_fragment[r],local_to_fragment[s]}) == body_order:
            gp, gq, gr, gs = local_to_global[[p, q, r, s]]
            spin_block_sum = (aa2[gp, gq, gr, gs] + ab2[gp, gq, gr, gs] + ab2[gq, gp, gs, gr] + bb2[gp, gq, gr, gs]) #make rdm with all three (four) spins instead of three einsum blocks
            e2 += 0.5 * V[p, q, r, s] * spin_block_sum #V is a spin-free quantity

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
    all_fragment_orbs = [orb for fragment in fragments for orb in fragment]

    ints = forte2.jkbuilder.RestrictedMOIntegrals(
        system=ci.system,
        C=ci.C[0],
        orbitals=all_fragment_orbs,
        core_orbitals=list(core_orbitals),
    )

    terms = {}
    total = ints.E
    for order in range(1, min(max_body_order, len(fragments), 4) + 1):
        order_total = 0.0
        for combo in combinations(range(len(fragments)), order):
            value = fragment_correlation_energy_enumerated(
                ci,
                *(fragments[i] for i in combo),
                body_order=order,
                core_orbitals=core_orbitals,
                root=root,
            )
            terms[combo] = value
            order_total += value
        total += order_total

    return {
        "Ecore": ints.E,
        "terms": terms,
        "decomposition_energy": total,
        "ci_energy": ci.E,
        "residual": ci.E - total,
    }
