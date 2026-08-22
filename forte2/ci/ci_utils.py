from collections import OrderedDict
from itertools import permutations

import numpy as np

from forte2.lib.ci_helpers import CIStrings
from forte2.state import MOSpace, StateAverageInfo
from forte2.helpers import logger
from forte2.data import EH_TO_EV


def pretty_print_gas_info(ci_strings: CIStrings):
    num_spaces = ci_strings.ngas_spaces
    gas_sizes = ci_strings.gas_size
    alpha_occupation = ci_strings.gas_alpha_occupations
    beta_occupation = ci_strings.gas_beta_occupations
    occupation_pairs = ci_strings.gas_occupations

    logger.log_info1("\nGAS information:")
    for i in range(num_spaces):
        logger.log_info1(f"GAS{i + 1}: size = {gas_sizes[i]}, ")

    table = []  # table[space_index] = [(aocc, bocc) for each config]

    for i in range(num_spaces):
        row = []
        for aocc_idx, bocc_idx in occupation_pairs:
            aocc = alpha_occupation[aocc_idx]
            bocc = beta_occupation[bocc_idx]
            row.append((aocc[i], bocc[i]))
        table.append(row)

    # Build header
    header = "Config.    "
    for conf_num in range(len(occupation_pairs)):
        header += f"{conf_num + 1:>3}"
    table_width = len(header)
    dash = "\n" + "-" * table_width
    eq_dash = "=" * table_width

    # Print rows: one per space
    rows = [header]
    for space_idx, row in enumerate(table, start=1):
        s_row = f"\nGAS{space_idx:1d} α Occ."
        for a_val, b_val in row:
            s_row += f" {a_val:2d}"
        s_row += f"\nGAS{space_idx:1d} β Occ."
        for a_val, b_val in row:
            s_row += f" {b_val:2d}"
        rows.append(s_row)

    s = f"\nGAS Occupation Configurations:\n{eq_dash}\n"
    s += dash.join(rows)
    s += f"\n{eq_dash}"

    logger.log_info1(s)


def pretty_print_ci_summary(
    sa_info: StateAverageInfo,
    eigvals_per_solver: list[list[float]],
    header="\nCI energy summary",
):
    """
    Pretty print the CI energy summary for the given CI states and eigenvalues.

    Parameters
    ----------
    sa_info : StateAverageInfo
        An instance of `StateAverageInfo` that holds information about the states and their properties.
    eigvals_per_solver : list[list[float]]
        A list of lists containing the eigenvalues (energies) for each CI solver.
    header : str, optional, default="CI energy summary"
        A header string to display at the top of the summary.
    """
    ncis = sa_info.ncis
    mult = [state.multiplicity for state in sa_info.states]
    ms = [state.ms for state in sa_info.states]
    irrep = [state.symmetry for state in sa_info.states]
    weights = sa_info.weights
    nroots = sa_info.nroots

    logger.log_info1(f"{header}:")
    width = 64
    logger.log_info1("=" * width)
    logger.log_info1(
        f"{'Root':>6} {'Mult.':>6} {'Ms':>6} {'Irrep':>6} {'Energy':>20} {'Weight':>15}"
    )
    logger.log_info1("-" * width)
    E_avg = 0.0
    iroot = 0
    for i in range(ncis):
        for j in range(nroots[i]):
            logger.log_info1(
                f"{iroot:>6d} {mult[i]:>6d} {ms[i]:>6.1f} {irrep[i]:>6d} {eigvals_per_solver[i][j].real:>20.10f} {weights[i][j]:>15.5f}"
            )
            iroot += 1
            E_avg += eigvals_per_solver[i][j] * weights[i][j]
        logger.log_info1("-" * width)
    logger.log_info1(f"{'Ensemble average energy':<27} {E_avg.real:>20.10f}")
    logger.log_info1("=" * width)


def pretty_print_ci_nat_occ_numbers(
    sa_info: StateAverageInfo,
    mo_space: MOSpace,
    nat_occs: np.ndarray,
    nat_occs_avg: np.ndarray | None = None,
) -> None:
    """
    Pretty print the natural occupation numbers for the CI states.
    Roots are rows, orbitals are columns.

    Parameters
    ----------
    sa_info : StateAverageInfo
        An instance of `StateAverageInfo` that holds information about the states and their properties.
    mo_space : MOSpace
        An instance of `MOSpace` that holds information about the partitioning of the molecular orbitals.
    nat_occs : np.ndarray
        A 2D numpy array containing the natural occupation numbers for each root and orbital.
        This should be calculated from CISolver.compute_natural_occupation_numbers.
    nat_occs_avg : np.ndarray, optional
        A 1D numpy array containing the state-averaged natural occupation numbers,
        computed from the state-averaged 1-RDM. Only meaningful (and printed) when
        more than one root is present.
    """
    nroots = sa_info.nroots_sum
    norb = mo_space.nactv
    width = 5 + 11 * norb
    logger.log_info1("\nNatural occupation numbers*:")
    logger.log_info1("=" * width)

    # Header with orbital indices
    header = "Orb     " + "".join(
        [f"{mo_space.active_indices[i]:<11d}" for i in range(norb)]
    )
    logger.log_info1(header)
    logger.log_info1("-" * width)

    # Data rows (one per root)
    for j in range(nroots):
        line = f"Root {j:<3d}"
        line += "".join([f"{nat_occs[i, j]:<11.6f}" for i in range(norb)])
        logger.log_info1(line)
    # if state-averaging, also print the average natural occupation numbers from the average 1-RDM
    if nroots > 1 and nat_occs_avg is not None:
        logger.log_info1("-" * width)
        avg_line = "Avg     " + "".join(
            [f"{nat_occs_avg[i]:<11.6f}" for i in range(norb)]
        )
        logger.log_info1(avg_line)

    logger.log_info1("=" * width)
    logger.log_info1(
        "* The occupation numbers are sorted in descending order\n"
        "  and do not correspond one-to-one to the active MOs."
    )


def pretty_print_ci_dets(
    sa_info: StateAverageInfo, mo_space: MOSpace, top_dets: list[list[list[tuple]]]
):
    """
    Pretty print the top determinants for each root of the CI states.

    Parameters
    ----------
    sa_info : StateAverageInfo
        An instance of `StateAverageInfo` that holds information about the states and their properties.
    mo_space : MOSpace
        An instance of `MOSpace` that holds information about the partitioning of the molecular orbitals.
    top_dets : list[list[list[tuple]]]
        A list of lists containing the top determinants and their coefficients for each root.
        This should be obtained from CISolver.get_top_determinants.
    """
    width_per_det = 1 + max(12, mo_space.nactv + 2)  # '|2222000>'
    ndets_per_root = len(top_dets[0])
    width = 10 + width_per_det * ndets_per_root
    nroots = sa_info.nroots_sum
    norb = mo_space.nactv
    is_complex = isinstance(top_dets[0][0][1], complex)

    logger.log_info1("\nTop determinants:")
    logger.log_info1("=" * width)
    logger.log_info1(
        f"{'Contrib.':<10}"
        + "".join([f"{'#'+str(i+1):<{width_per_det}}" for i in range(ndets_per_root)])
    )
    logger.log_info1("-" * width)
    for i in range(nroots):
        dets = [det for det, _ in top_dets[i]]
        coeffs = [coeff for _, coeff in top_dets[i]]
        logstr = f"Root {i:<5}" + "".join(
            [f"{d.str(norb):<{width_per_det}}" for d in dets]
        )
        logstr += (
            "\n"
            + " " * 10
            + "".join([f"{c.real:<+{width_per_det}.6f}" for c in coeffs])
        )
        if is_complex:
            logstr += (
                "\n"
                + " " * 10
                + "".join([f"{f'{c.imag:<+.6f}'+'i':<{width_per_det}}" for c in coeffs])
            )
        logger.log_info1(logstr)
        if i < nroots - 1:
            logger.log_info1("-" * width)
    logger.log_info1("=" * width)


def pretty_print_ci_transition_props(
    sa_info: StateAverageInfo,
    transition_dipoles: OrderedDict,
    oscillator_strengths: OrderedDict,
    eigvals_per_solver: list[list[float]],
    thres=1e-4,
):
    """
    Pretty print the dipole moments of CI states, as well as the bright transitions between them,
    including the oscillator strengths and vertical transition energies (VTE).

    Parameters
    ----------
    sa_info : StateAverageInfo
        An instance of `StateAverageInfo` that holds information about the states and their properties.
    transition_dipoles : OrderedDict
        A dictionary with keys as tuples (i, j) representing the initial and final states,
        and values as the transition dipole moments for each component (x, y, z).
    oscillator_strengths : OrderedDict
        A dictionary with keys as tuples (i, j) representing the initial and final states,
        and values as the oscillator strengths for each transition.
    eigvals_per_solver : list[list[float]]
        A list of lists containing the eigenvalues (energies) for each CI solver.
    """

    logger.log_info1("\nDipole moments (a.u.) of CI states (nuclear + electronic):")
    width = 43
    logger.log_info1("=" * width)
    logger.log_info1(f"{'State':<12} {'Dipole moment':<30}")
    logger.log_info1("-" * width)
    for iroot in range(sa_info.nroots_sum):
        dip = transition_dipoles[(iroot, iroot)]
        dip_str = "[" + ", ".join(f"{d:>7.4f}" for d in dip) + "]"
        logger.log_info1(f"{f'{iroot}':<12} {dip_str:<30}")
    logger.log_info1("=" * width)

    logger.log_info1(f"\nBright transitions (oscillator strength > {thres:5.2e}):")
    iroot = 0
    width = 64
    logger.log_info1("=" * width)
    logger.log_info1(
        f"{'Transition':<12} {'fosc':<10} {'VTE (eV)':<10} {'Electronic trans. dip. (a.u.)':<30}"
    )
    logger.log_info1("-" * width)
    nbright = 0
    for k, v in transition_dipoles.items():
        i, j = k
        dip = v
        isolver, iroot_in_solver = sa_info.absolute_root_map[i]
        jsolver, jroot_in_solver = sa_info.absolute_root_map[j]
        vte = (
            eigvals_per_solver[jsolver][jroot_in_solver]
            - eigvals_per_solver[isolver][iroot_in_solver]
        ) * EH_TO_EV
        osc = oscillator_strengths[k]
        if osc > thres:
            nbright += 1
            info = f"{f'{i}->{j}':<12} "
            info += f"{osc:<10.6f} {vte:<10.6f} "
            dip = "[" + ", ".join(f"{d:>7.4f}" for d in dip) + "]"
            info += f"{dip:<30}"
            logger.log_info1(info)
    if nbright == 0:
        logger.log_info1("No bright transitions found.")
    logger.log_info1("=" * width)


def spin_free_1rdm(gamma1_a, gamma1_b):
    """
    Assemble the spin-free one-particle RDM from the alpha and beta 1-RDMs.

    Parameters
    ----------
    gamma1_a : np.ndarray
        The alpha one-particle reduced density matrix, shape (norb, norb).
    gamma1_b : np.ndarray
        The beta one-particle reduced density matrix, shape (norb, norb).

    Returns
    -------
    np.ndarray
        The spin-free one-particle reduced density matrix (sf-1-RDM).
    """
    return gamma1_a + gamma1_b


def pair_indices_gt(norb):
    """Canonical (p, q) pairs with p > q, in the same order as ``pair_index_gt``."""
    return np.tril_indices(norb, -1)


def triplet_indices_gt(norb):
    """Canonical (p, q, r) triplets with p > q > r, in the same order as ``triplet_index_gt``."""
    if norb < 3:
        empty = np.array([], dtype=int)
        return empty, empty, empty
    p_list, q_list, r_list = [], [], []
    for p in range(2, norb):
        q_idx, r_idx = np.tril_indices(p, -1)
        p_list.append(np.full(q_idx.shape, p))
        q_list.append(q_idx)
        r_list.append(r_idx)
    return np.concatenate(p_list), np.concatenate(q_list), np.concatenate(r_list)


# Sign of each of the 6 permutations of 3 elements, used to unpack a packed antisymmetric
# same-spin 3-RDM block into the full 6-index tensor.
def _permutation_sign(perm):
    n = len(perm)
    seen = [False] * n
    sign = 1
    for i in range(n):
        if seen[i]:
            continue
        j, cycle_len = i, 0
        while not seen[j]:
            seen[j] = True
            j = perm[j]
            cycle_len += 1
        if cycle_len % 2 == 0:
            sign = -sign
    return sign


_PERM3 = [(perm, _permutation_sign(perm)) for perm in permutations(range(3))]


def unpack_ss_2rdm(gamma2_ss, norb):
    """
    Unpack a packed same-spin 2-RDM block into the full antisymmetric 4-index tensor.

    Parameters
    ----------
    gamma2_ss : np.ndarray
        The same-spin (aa or bb) two-particle RDM in packed form, shape (npairs, npairs),
        stored as gamma2_ss[p>q][r>s] = <L| a^+_p a^+_q a_s a_r |R>.
    norb : int
        The number of orbitals.

    Returns
    -------
    np.ndarray
        The full antisymmetric tensor, shape (norb, norb, norb, norb).
    """
    full = np.zeros((norb,) * 4, dtype=gamma2_ss.dtype)
    if norb < 2:
        return full

    p_idx, q_idx = pair_indices_gt(norb)
    P, Q = p_idx[:, None], q_idx[:, None]
    R, S = p_idx[None, :], q_idx[None, :]
    full[P, Q, R, S] += gamma2_ss
    full[Q, P, R, S] -= gamma2_ss
    full[P, Q, S, R] -= gamma2_ss
    full[Q, P, S, R] += gamma2_ss
    return full


def spin_free_2rdm(gamma2_ab, gamma2_aa, gamma2_bb):
    """
    Assemble the full spin-free two-particle RDM from the spin-resolved 2-RDM building blocks.

    Parameters
    ----------
    gamma2_ab : np.ndarray
        The alpha-beta two-particle RDM, full tensor of shape (norb, norb, norb, norb), stored as
        gamma2_ab[p][q][r][s] = <L| a^+_{p,a} a^+_{q,b} a_{s,b} a_{r,a} |R>.
    gamma2_aa : np.ndarray
        The alpha-alpha two-particle RDM in packed form, shape (npairs, npairs), stored as
        gamma2_aa[p>q][r>s] = <L| a^+_{p,a} a^+_{q,a} a_{s,a} a_{r,a} |R>.
    gamma2_bb : np.ndarray
        The beta-beta two-particle RDM in packed form, analogous to ``gamma2_aa``.

    Returns
    -------
    np.ndarray
        The full spin-free two-particle reduced density matrix (sf-2-RDM), shape
        (norb, norb, norb, norb).
    """
    norb = gamma2_ab.shape[0]
    sf = gamma2_ab + np.transpose(gamma2_ab, (1, 0, 3, 2))
    sf += unpack_ss_2rdm(gamma2_aa, norb)
    sf += unpack_ss_2rdm(gamma2_bb, norb)
    return sf


def unpack_sss_3rdm(packed, norb):
    """Unpack a packed same-spin 3-RDM block (ntriplets, ntriplets) into a full 6-index tensor."""
    full = np.zeros((norb,) * 6, dtype=packed.dtype)
    if norb < 3:
        return full

    triplet_idx = triplet_indices_gt(norb)
    for bra_perm, bra_sign in _PERM3:
        P = triplet_idx[bra_perm[0]][:, None]
        Q = triplet_idx[bra_perm[1]][:, None]
        R = triplet_idx[bra_perm[2]][:, None]
        for ket_perm, ket_sign in _PERM3:
            S = triplet_idx[ket_perm[0]][None, :]
            T = triplet_idx[ket_perm[1]][None, :]
            U = triplet_idx[ket_perm[2]][None, :]
            full[P, Q, R, S, T, U] += (bra_sign * ket_sign) * packed
    return full


def unpack_aab_3rdm(gamma3_aab, norb):
    """Unpack the AAB 3-RDM block, shape (npair, norb, npair, norb), into a full 6-index tensor."""
    full = np.zeros((norb,) * 6, dtype=gamma3_aab.dtype)
    if norb < 2:
        return full

    p_idx, q_idx = pair_indices_gt(norb)
    r_idx = np.arange(norb)
    P = p_idx[:, None, None, None]
    Q = q_idx[:, None, None, None]
    Rr = r_idx[None, :, None, None]
    S = p_idx[None, None, :, None]
    T = q_idx[None, None, :, None]
    U = r_idx[None, None, None, :]

    terms = (
        ((P, Q, Rr, S, T, U), +1),
        ((P, Q, Rr, T, S, U), -1),
        ((Q, P, Rr, S, T, U), -1),
        ((Q, P, Rr, T, S, U), +1),
        ((P, Rr, Q, S, U, T), +1),
        ((P, Rr, Q, T, U, S), -1),
        ((Q, Rr, P, S, U, T), -1),
        ((Q, Rr, P, T, U, S), +1),
        ((Rr, P, Q, U, S, T), +1),
        ((Rr, P, Q, U, T, S), -1),
        ((Rr, Q, P, U, S, T), -1),
        ((Rr, Q, P, U, T, S), +1),
    )
    for idx, sign in terms:
        full[idx] += sign * gamma3_aab
    return full


def unpack_abb_3rdm(gamma3_abb, norb):
    """Unpack the ABB 3-RDM block, shape (norb, npair, norb, npair), into a full 6-index tensor."""
    full = np.zeros((norb,) * 6, dtype=gamma3_abb.dtype)
    if norb < 2:
        return full

    q_idx, r_idx = pair_indices_gt(norb)
    p_idx = np.arange(norb)
    P = p_idx[:, None, None, None]
    Q = q_idx[None, :, None, None]
    Rr = r_idx[None, :, None, None]
    S = p_idx[None, None, :, None]
    T = q_idx[None, None, None, :]
    U = r_idx[None, None, None, :]

    terms = (
        ((P, Q, Rr, S, T, U), +1),
        ((P, Rr, Q, S, T, U), -1),
        ((Q, P, Rr, T, S, U), +1),
        ((Rr, P, Q, T, S, U), -1),
        ((P, Q, Rr, S, U, T), -1),
        ((P, Rr, Q, S, U, T), +1),
        ((Q, Rr, P, T, U, S), +1),
        ((Rr, Q, P, T, U, S), -1),
        ((Q, P, Rr, U, S, T), -1),
        ((Rr, P, Q, U, S, T), +1),
        ((Q, Rr, P, U, T, S), -1),
        ((Rr, Q, P, U, T, S), +1),
    )
    for idx, sign in terms:
        full[idx] += sign * gamma3_abb
    return full


def spin_free_3rdm(gamma3_aab, gamma3_abb, gamma3_aaa, gamma3_bbb):
    """
    Assemble the full spin-free three-particle RDM from the spin-resolved 3-RDM building blocks.

    Parameters
    ----------
    gamma3_aab : np.ndarray
        The alpha-alpha-beta three-particle RDM, packed shape (npair, norb, npair, norb).
    gamma3_abb : np.ndarray
        The alpha-beta-beta three-particle RDM, packed shape (norb, npair, norb, npair).
    gamma3_aaa : np.ndarray
        The alpha-alpha-alpha three-particle RDM, packed shape (ntriplets, ntriplets).
    gamma3_bbb : np.ndarray
        The beta-beta-beta three-particle RDM, packed shape (ntriplets, ntriplets).

    Returns
    -------
    np.ndarray
        The full spin-free three-particle reduced density matrix (sf-3-RDM), shape
        (norb, norb, norb, norb, norb, norb).
    """
    norb = gamma3_abb.shape[0]
    sf = unpack_aab_3rdm(gamma3_aab, norb)
    sf += unpack_abb_3rdm(gamma3_abb, norb)
    sf += unpack_sss_3rdm(gamma3_aaa, norb)
    sf += unpack_sss_3rdm(gamma3_bbb, norb)
    return sf


def make_2cumulant_so(gamma1, gamma2):
    """
    Compute the 2-cumulant from the spinorbital 1- and 2-RDMs.

    This can be useful for computing averaged cumulants, since one cannot simply average
    the 2-cumulants directly, as the relation between RDMs and cumulants is nonlinear.
    See text around eq. 7 om J. Chem. Phys. 148, 124106 (2018) for more details.

    Parameters
    ----------
    gamma1 : np.ndarray
        The one-particle reduced density matrix (1-RDM).
    gamma2 : np.ndarray
        The two-particle reduced density matrix (2-RDM).

    Returns
    -------
    np.ndarray
        The two-particle cumulant (2-cumulant).
    """
    l2 = (
        gamma2
        - np.einsum("pr,qs->pqrs", gamma1, gamma1, optimize=True)
        + np.einsum("ps,qr->pqrs", gamma1, gamma1, optimize=True)
    )
    return l2


def make_3cumulant_so(gamma1, gamma2, gamma3):
    """
    Compute the 3-cumulant from the spinorbital 1-, 2-, and 3-RDMs.

    This can be useful for computing averaged cumulants, since one cannot simply average
    the 3-cumulants directly, as the relation between RDMs and cumulants is nonlinear.
    See text around eq. 7 om J. Chem. Phys. 148, 124106 (2018) for more details.

    Parameters
    ----------
    gamma1 : np.ndarray
        The one-particle reduced density matrix (1-RDM).
    gamma2 : np.ndarray
        The two-particle reduced density matrix (2-RDM).
    gamma3 : np.ndarray
        The three-particle reduced density matrix (3-RDM).

    Returns
    -------
    np.ndarray
        The three-particle cumulant (3-cumulant).
    """
    l3 = (
        gamma3
        - np.einsum("ps,qrtu->pqrstu", gamma1, gamma2)
        + np.einsum("pt,qrsu->pqrstu", gamma1, gamma2)
        + np.einsum("pu,qrts->pqrstu", gamma1, gamma2)
        - np.einsum("qt,prsu->pqrstu", gamma1, gamma2)
        + np.einsum("qs,prtu->pqrstu", gamma1, gamma2)
        + np.einsum("qu,prst->pqrstu", gamma1, gamma2)
        - np.einsum("ru,pqst->pqrstu", gamma1, gamma2)
        + np.einsum("rs,pqut->pqrstu", gamma1, gamma2)
        + np.einsum("rt,pqsu->pqrstu", gamma1, gamma2)
        + 2
        * (
            np.einsum("ps,qt,ru->pqrstu", gamma1, gamma1, gamma1)
            + np.einsum("pt,qu,rs->pqrstu", gamma1, gamma1, gamma1)
            + np.einsum("pu,qs,rt->pqrstu", gamma1, gamma1, gamma1)
        )
        - 2
        * (
            np.einsum("ps,qu,rt->pqrstu", gamma1, gamma1, gamma1)
            + np.einsum("pu,qt,rs->pqrstu", gamma1, gamma1, gamma1)
            + np.einsum("pt,qs,ru->pqrstu", gamma1, gamma1, gamma1)
        )
    )

    return l3


def make_2cumulant_sf(gamma1, gamma2):
    """
    Compute the spin-free 2-cumulant from the 1- and 2- spin-free RDMs.

    This can be useful for computing averaged cumulants, since one cannot simply average
    the 2-cumulants directly, as the relation between RDMs and cumulants is nonlinear.

    Parameters
    ----------
    gamma1 : np.ndarray
        The one-particle spin-free reduced density matrix (sf-1-RDM).
    gamma2 : np.ndarray
        The two-particle spin-free reduced density matrix (sf-2-RDM).

    Returns
    -------
    np.ndarray
        The two-particle spin-free cumulant (sf-2-cumulant).
    """
    l2 = (
        gamma2
        - np.einsum("pr,qs->pqrs", gamma1, gamma1, optimize=True)
        + 0.5 * np.einsum("ps,qr->pqrs", gamma1, gamma1, optimize=True)
    )
    return l2


def make_3cumulant_sf(gamma1, gamma2, gamma3):
    """
    Compute the spin-free 3-cumulant from the 1-, 2-, and 3- spin-free RDMs.

    This can be useful for computing averaged cumulants, since one cannot simply average
    the 3-cumulants directly, as the relation between RDMs and cumulants is nonlinear.
    See text around eq. 7 om J. Chem. Phys. 148, 124106 (2018) for more details.

    Parameters
    ----------
    gamma1 : np.ndarray
        The one-particle spin-free reduced density matrix (sf-1-RDM).
    gamma2 : np.ndarray
        The two-particle spin-free reduced density matrix (sf-2-RDM).
    gamma3 : np.ndarray
        The three-particle spin-free reduced density matrix (sf-3-RDM).

    Returns
    -------
    np.ndarray
        The three-particle spin-free cumulant (sf-3-cumulant).
    """
    l3 = gamma3 - (
        +np.einsum("ps,qrtu->pqrstu", gamma1, gamma2, optimize=True)
        + np.einsum("qt,prsu->pqrstu", gamma1, gamma2, optimize=True)
        + np.einsum("ru,pqst->pqrstu", gamma1, gamma2, optimize=True)
    )
    l3 += 0.5 * (
        +np.einsum("pt,qrsu->pqrstu", gamma1, gamma2, optimize=True)
        + np.einsum("pu,qrts->pqrstu", gamma1, gamma2, optimize=True)
        + np.einsum("qs,prtu->pqrstu", gamma1, gamma2, optimize=True)
        + np.einsum("qu,prst->pqrstu", gamma1, gamma2, optimize=True)
        + np.einsum("rs,pqut->pqrstu", gamma1, gamma2, optimize=True)
        + np.einsum("rt,pqsu->pqrstu", gamma1, gamma2, optimize=True)
    )
    l3 += 2.0 * np.einsum("ps,qt,ru->pqrstu", gamma1, gamma1, gamma1, optimize=True)
    l3 -= (
        np.einsum("ps,qu,rt->pqrstu", gamma1, gamma1, gamma1, optimize=True)
        + np.einsum("pu,qt,rs->pqrstu", gamma1, gamma1, gamma1, optimize=True)
        + np.einsum("pt,qs,ru->pqrstu", gamma1, gamma1, gamma1, optimize=True)
    )
    l3 += 0.5 * (
        np.einsum("pt,qu,rs->pqrstu", gamma1, gamma1, gamma1, optimize=True)
        + np.einsum("pu,qs,rt->pqrstu", gamma1, gamma1, gamma1, optimize=True)
    )
    return l3
