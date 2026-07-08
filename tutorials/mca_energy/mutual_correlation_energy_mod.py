import forte2
import numpy as np
from itertools import combinations, product
from subspaces import subspaces

class MutualCorrelationEnergyAnalysis:
    """
    Container for fragment correlation energy analysis using 2-cumulant quantity.

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
        core_orbitals=(),
        nat_orbs=False,
        nocc=None,
    ):

        self.ci = ci
        self.root = root
        self.max_body_order = max_body_order
        self.core_orbitals = core_orbitals
        self.nat_orbs = nat_orbs
        self.nocc = nocc

        if fragments is None:
            self.fragments = [[orb] for orb in ci.mo_space.active_indices]
        else:
            self.fragments = [list(f) for f in fragments]

        self.nfragments = len(self.fragments)

        basis = _analysis_basis(
                                ci,
                                root=root,
                                nat_orbs=nat_orbs,
                                nocc=nocc,
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


def natural_orbitals_from_gamma1(C_mo, gamma1_mo, nocc):
    """
    Construct block natural orbitals from the spin-summed one-particle RDM.

    The occupied-occupied and virtual-virtual blocks of the 1-RDM are
    diagonalized independently so that occupied and virtual orbital spaces
    remain separate. The resulting orbital rotation therefore differs from
    the global natural orbital transformation obtained by diagonalizing the
    full 1-RDM.

    Parameters
    ----------
    C_mo : ndarray
        Molecular orbital coefficient matrix for the active space.
    gamma1_mo : ndarray
        Spin-summed active-space one-particle RDM in the molecular orbital
        basis.
    nocc : int
        Number of occupied active orbitals.

    Returns
    -------
    C_no : ndarray
        Active-space molecular orbital coefficients rotated into the block
        natural orbital basis.
    occupations : ndarray
        Occupation numbers obtained from the occupied and virtual blocks.
    orbital_rotation : ndarray
        Orbital rotation matrix acting within the active space.
    """

    if nocc is None:
        raise ValueError("nocc must be provided when nat_orbs=True.")

    gamma1 = 0.5 * (gamma1_mo + gamma1_mo.T)
    nocc = int(nocc)

    if not (0 <= nocc <= gamma1.shape[0]):
        raise ValueError(
            f"nocc must be between 0 and {gamma1.shape[0]}; got {nocc}."
        )

    gamma_occ = gamma1[:nocc, :nocc]
    gamma_vir = gamma1[nocc:, nocc:]

    occ_occ, U_occ = np.linalg.eigh(gamma_occ)
    occ_vir, U_vir = np.linalg.eigh(gamma_vir)

    occ_order = np.argsort(occ_occ)[::-1]
    vir_order = np.argsort(occ_vir)[::-1]

    occ_occ = occ_occ[occ_order]
    occ_vir = occ_vir[vir_order]

    U_occ = U_occ[:, occ_order]
    U_vir = U_vir[:, vir_order]

    orbital_rotation = np.zeros_like(gamma1)
    orbital_rotation[:nocc, :nocc] = U_occ
    orbital_rotation[nocc:, nocc:] = U_vir

    C_no = C_mo @ orbital_rotation
    occupations = np.concatenate((occ_occ, occ_vir))

    return C_no, occupations, orbital_rotation


def _analysis_basis(ci, root=0, nat_orbs=False, nocc=None):
    """
    Build the orbital basis used throughout the correlation analysis.
    When ``nat_orbs`` is False, the molecular orbital basis is returned
    unchanged.
    When ``nat_orbs`` is True, the active-space orbitals are rotated into
    the block natural orbital basis while inactive orbitals remain fixed.

    Returns
    -------
    dict: Dictionary containing
        C: Orbital coefficient matrix.
        gamma1: Spin-summed one-particle RDM expressed in the analysis basis.
        occupations: Block natural orbital occupations, or None if molecular
            orbitals are used.
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
        nocc,
    )
    C[:, active_indices] = C_no

    gamma1 = _transform_1rdm(gamma1, orbital_rotation)

    return {
        "C": C,
        "gamma1": gamma1,
        "occupations": occupations,
        "orbital_rotation": orbital_rotation, 
    }


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


def _fragment_index_maps(ci, orbital_fragments):
    """
    generate index maps for the 1, 2RDM and 2-electron integral V, 
    which use different indexing conventions.
    """
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

#Build 1-RDMs and 2-cumulants with forte2
def _spin_dependent_rdms(ci, root=0, orbital_rotation=None):
    """
    Returns
    -------
    a1: alpha 1rdm
    b1: beta 1rdm
    aa2_cumulant: alpha-alpha 2 cumulant built from 1- and 2-rdms
    ab2_cumulant
    bb2_cumulant
    """
    
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
    if orbital_rotation is not None:
        a1 = _transform_1rdm(a1, orbital_rotation)
        b1 = _transform_1rdm(b1, orbital_rotation)
        aa2_cumulant = _transform_2tensor(aa2_cumulant, orbital_rotation)
        ab2_cumulant = _transform_2tensor(ab2_cumulant, orbital_rotation)
        bb2_cumulant = _transform_2tensor(bb2_cumulant, orbital_rotation)

    # still setting 1-rdms to zero, after 2-cumulant is built
    n = a1.shape[0]
    a1 = np.zeros((n, n))
    b1 = np.zeros((n, n))

    return a1, b1, aa2_cumulant, ab2_cumulant, bb2_cumulant

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

    #Compute an n-fragment energy contribution by explicit index enumeration.

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
    a1, b1, aa2, ab2, bb2 = _spin_dependent_rdms(
        ci, root=root, orbital_rotation=orbital_rotation
    )

    ints = forte2.jkbuilder.RestrictedMOIntegrals(
        system=ci.system,
        C=C,
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
            #aa2, ab2, bb2 are the 2-cumulants, NOT the 2RDMs 
            spin_block_sum = (aa2[gp, gq, gr, gs] + ab2[gp, gq, gr, gs] + ab2[gq, gp, gs, gr]+ bb2[gp, gq, gr, gs]) 
            e2 += 0.5 * V[p, q, r, s] * spin_block_sum #V is a spin-free quantity

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
    nocc=None,
):
      
    #Return Ecore plus all enumerated fragment terms through max_body_order.
    fragments = [list(fragment) for fragment in fragments]
    all_fragment_orbs = [orb for fragment in fragments for orb in fragment]
    basis = _analysis_basis(ci, root=root, nat_orbs=nat_orbs, nocc=nocc)
    
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
                C=basis["C"],
                orbital_rotation=basis["orbital_rotation"],
            )
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
        nocc=nocc,
    )
