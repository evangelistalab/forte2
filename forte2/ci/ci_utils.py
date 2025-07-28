from dataclasses import dataclass

import numpy as np

from forte2 import CIStrings
from forte2.state import State, MOSpace
from forte2.orbitals import AVAS
from forte2.helpers import logger
from forte2.system.atom_data import EH_TO_EV


@dataclass
class StateAverageInfo:
    """
    A class to hold information about state averaging in multireference calculations.

    Parameters
    ----------
    states : list[State] | State
        A list of `State` objects or a single `State` object representing the electronic states.
        This also includes the gas_min and gas_max attributes.
    nroots : list[int] | int, optional, default=1
        A list of integers specifying the number of roots for each state.
        If only one state is provided, this can be a single integer.
    weights : list[list[float]], optional
        A list of lists of floats specifying the weights for each root in each state.
        These do not have to be normalized, but must be non-negative.
        If not provided, equal weights are assigned to each root.

    Attributes
    ----------
    ncis : int
        The number of CI states, which is the length of the `states` list.
    nroots_sum : int
        The total number of roots across all states.
    weights_flat : NDArray
        A flattened array of weights for all roots across all states.
    """

    states: list[State] | State
    nroots: list[int] | int = 1
    weights: list[list[float]] = None

    def __post_init__(self):
        # 1. Validate states
        if isinstance(self.states, State):
            self.states = [self.states]
        assert isinstance(self.states, list), "states_and_mo_spaces must be a list"
        assert all(
            isinstance(state, State) for state in self.states
        ), "All elements in states_and_mo_spaces must be State instances"
        assert len(self.states) > 0, "states_and_mo_spaces cannot be empty"
        self.ncis = len(self.states)

        # 2. Validate nroots
        if isinstance(self.nroots, int):
            assert (
                self.ncis == 1
            ), "If nroots is an integer, there must be exactly one state."
            self.nroots = [self.nroots]
        assert isinstance(self.nroots, list), "nroots must be a list"
        assert all(
            isinstance(n, int) and n > 0 for n in self.nroots
        ), "nroots must be a list of positive integers"
        self.nroots_sum = sum(self.nroots)

        # 3. Validate weights
        if self.weights is None:
            self.weights = [[1.0 / self.nroots_sum] * n for n in self.nroots]
            self.weights_flat = np.concatenate(self.weights)
        else:
            assert (
                sum(len(w) for w in self.weights) == self.nroots_sum
            ), "Weights must match the total number of roots across all states"
            self.weights_flat = np.array(
                [w for sublist in self.weights for w in sublist], dtype=float
            )
            n = self.weights_flat.sum()
            self.weights = [[w / n for w in sublist] for sublist in self.weights]
            self.weights_flat /= n
            assert np.all(self.weights_flat >= 0), "Weights must be non-negative"

    def pretty_print_sa_info(self):
        """
        Pretty print the state averaging information.
        """
        width = 33
        logger.log_info1("\nRequested states:")
        logger.log_info1("=" * width)
        logger.log_info1(
            f"{'Root':>4} {'Nel':>5} {'Mult.':>6} {'Ms':>4} {'Weight':>10}"
        )
        logger.log_info1("-" * width)
        iroot = 0
        for i, state in enumerate(self.states):
            for j in range(self.nroots[i]):
                logger.log_info1(
                    f"{iroot:>4} {state.nel:>5} {state.multiplicity:>6d} {state.ms:>4.1f} {self.weights[i][j]:>10.6f}"
                )
                iroot += 1
            if i < len(self.states) - 1:
                logger.log_info1("-" * width)
        logger.log_info1("=" * width + "\n")


def pretty_print_gas_info(ci_strings: CIStrings):
    num_spaces = ci_strings.ngas_spaces
    gas_sizes = ci_strings.gas_size
    alfa_occupation = ci_strings.gas_alfa_occupations
    beta_occupation = ci_strings.gas_beta_occupations
    occupation_pairs = ci_strings.gas_occupations

    logger.log_info1("\nGAS information:")
    for i in range(num_spaces):
        logger.log_info1(f"GAS space {i + 1} : size = {gas_sizes[i]}, ")

    s = "\n    Config."
    for i in range(num_spaces):
        s += f"   Space {i + 1}"
    s += "\n            "
    for i in range(num_spaces):
        s += "   α    β "
    ndash = 7 + 10 * num_spaces
    dash = "-" * ndash
    s += f"\n    {dash}"
    num_conf = 0
    for aocc_idx, bocc_idx in occupation_pairs:
        num_conf += 1
        aocc = alfa_occupation[aocc_idx]
        bocc = beta_occupation[bocc_idx]
        s += f"\n    {num_conf:6d} "
        for i in range(num_spaces):
            s += f" {aocc[i]:4d} {bocc[i]:4d}"

    logger.log_info1(s)


def pretty_print_ci_summary(sa_info: StateAverageInfo, eigvals_per_solver: list[list[float]]):
    """
    Pretty print the CI energy summary for the given CI states and eigenvalues.

    Parameters
    ----------
    cistates : CIStates
        An instance of `CIStates` that holds information about the states and their properties.
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
                f"{iroot:>6d} {mult[i]:>6d} {ms[i]:>6.1f} {irrep[i]:>6d} {eigvals_per_solver[i][j]:>20.10f} {weights[i][j]:>15.5f}"
            )
            iroot += 1
            E_avg += eigvals_per_solver[i][j] * weights[i][j]
        logger.log_info1("-" * width)
    logger.log_info1(f"{'Ensemble average energy':<27} {E_avg:>20.10f}")
    logger.log_info1("=" * width)


def pretty_print_ci_nat_occ_numbers(sa_info: StateAverageInfo, mo_space: MOSpace, nat_occs: np.ndarray):
    """
    Pretty print the natural occupation numbers for the CI states.
    Roots are rows, orbitals are columns.
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


def pretty_print_ci_dets(sa_info: StateAverageInfo, mo_space: MOSpace, top_dets: list[list[list[tuple]]]):
    """
    Pretty print the top determinants for each root of the CI states.

    Parameters
    ----------
    cistates : CIStates
        An instance of `CIStates` that holds information about the states and their properties.
    top_dets : list[list[list[tuple]]]
        A list of lists containing the top determinants and their coefficients for each root.
    """
    width_per_det = 1 + max(12, mo_space.nactv + 2)  # '|2222000>'
    ndets_per_root = len(top_dets[0])
    width = 10 + width_per_det * ndets_per_root
    nroots = sa_info.nroots_sum
    norb = mo_space.nactv

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
        logstr += "\n"
        logstr += " " * 10 + "".join([f"{c:<+{width_per_det}.6f}" for c in coeffs])
        logger.log_info1(logstr)
        if i < nroots - 1:
            logger.log_info1("-" * width)
    logger.log_info1("=" * width)


def pretty_print_ci_transition_props(
    sa_info: StateAverageInfo, tdm_per_solver, fosc_per_solver, eigvals_per_solver, thres=1e-4
):
    """
    Pretty print the dipole moments of CI states, as well as the bright transitions between them,
    including the oscillator strengths and vertical transition energies (VTE).

    Parameters
    ----------
    cistates : CIStates
        An instance of `CIStates` that holds information about the states and their properties.
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
