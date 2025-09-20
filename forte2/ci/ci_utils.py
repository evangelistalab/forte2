import numpy as np

from forte2 import CIStrings
from forte2.state import MOSpace, StateAverageInfo
from forte2.helpers import logger
from forte2.system.atom_data import EH_TO_EV


def pretty_print_gas_info(ci_strings: CIStrings):
    num_spaces = ci_strings.ngas_spaces
    gas_sizes = ci_strings.gas_size
    alfa_occupation = ci_strings.gas_alfa_occupations
    beta_occupation = ci_strings.gas_beta_occupations
    occupation_pairs = ci_strings.gas_occupations

    logger.log_info1("\nGAS information:")
    for i in range(num_spaces):
        logger.log_info1(f"GAS{i + 1}: size = {gas_sizes[i]}, ")

    table = []  # table[space_index] = [(aocc, bocc) for each config]

    for i in range(num_spaces):
        row = []
        for aocc_idx, bocc_idx in occupation_pairs:
            aocc = alfa_occupation[aocc_idx]
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
    sa_info: StateAverageInfo, eigvals_per_solver: list[list[float]]
):
    """
    Pretty print the CI energy summary for the given CI states and eigenvalues.

    Parameters
    ----------
    sa_info : StateAverageInfo
        An instance of `StateAverageInfo` that holds information about the states and their properties.
    eigvals_per_solver : list[list[float]]
        A list of lists containing the eigenvalues (energies) for each CI solver.
    """
    ncis = sa_info.ncis
    mult = [state.multiplicity for state in sa_info.states]
    ms = [state.ms for state in sa_info.states]
    irrep = [state.symmetry for state in sa_info.states]
    weights = sa_info.weights
    nroots = sa_info.nroots

    logger.log_info1("CI energy summary:")
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
    sa_info: StateAverageInfo, mo_space: MOSpace, nat_occs: np.ndarray
):
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
    tdm_per_solver,
    fosc_per_solver,
    eigvals_per_solver,
    thres=1e-4,
):
    """
    Pretty print the dipole moments of CI states, as well as the bright transitions between them,
    including the oscillator strengths and vertical transition energies (VTE).

    Parameters
    ----------
    sa_info : StateAverageInfo
        An instance of `StateAverageInfo` that holds information about the states and their properties.
    tdm_per_solver : OrderedDict
        A dictionary with keys as tuples (i, j) representing the initial and final states,
        and values as the transition dipole moments for each component (x, y, z).
    eigvals_per_solver : list[list[float]]
        A list of lists containing the eigenvalues (energies) for each CI solver.
    """

    logger.log_info1("\nDipole moments (a.u.) of CI states (nuclear + electronic):")
    width = 43
    logger.log_info1("=" * width)
    logger.log_info1(f"{'State':<12} {'Dipole moment':<30}")
    logger.log_info1("-" * width)
    for ici in range(sa_info.ncis):
        for iroot in range(sa_info.nroots[ici]):
            dip = tdm_per_solver[ici][(iroot, iroot)]
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
    for ici in range(sa_info.ncis):
        for k, v in tdm_per_solver[ici].items():
            i, j = k
            dip = v
            vte = (eigvals_per_solver[ici][j] - eigvals_per_solver[ici][i]) * EH_TO_EV
            osc = fosc_per_solver[ici][k]
            if osc > thres:
                nbright += 1
                info = f"{f'{iroot+i}->{iroot+j}':<12} "
                info += f"{osc:<10.6f} {vte:<10.6f} "
                dip = "[" + ", ".join(f"{d:>7.4f}" for d in dip) + "]"
                info += f"{dip:<30}"
                logger.log_info1(info)
        iroot += sa_info.nroots[ici]
    if nbright == 0:
        logger.log_info1("No bright transitions found.")
    logger.log_info1("=" * width)
