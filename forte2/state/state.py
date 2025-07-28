import numpy as np
from dataclasses import dataclass, field

from forte2.helpers.multiplicity_labels import multiplicity_labels
from forte2.helpers import logger


@dataclass(order=True)
class State:
    """
    Class to represent a state of a quantum system.

    Parameters
    ----------
    nel : int
        Total number of electrons.
    multiplicity : int
        Multiplicity of the state (2S+1).
    ms : float
        Spin projection (Ms) of the state.
    irrep : int, optional
        Irreducible representation of the state in Cotton ordering.
    gas_min : list[int], optional, default=[]
        Minimum number of electrons in each GAS.
        If not provided, no occupation restrictions will be applied.
        If the length of `gas_min` is smaller than the number of GASes, the restrictions
        will be sequentially applied to GAS1, GAS2, until the end of `gas_min`,
        beyond which no restrictions will be applied.
    gas_max : list[int], optional, default=[]
        Maximum number of electrons in each GAS. Processed similarly to `gas_min`.
    symmetry: int, optional, default=0
        Symmetry of the state.
    symmetry_label: str, optional, default=None
        Label for the symmetry of the state.
    """

    nel: int
    multiplicity: int
    ms: float
    gas_min: list[int] = field(default_factory=list)
    gas_max: list[int] = field(default_factory=list)
    symmetry: int = field(default=0)
    symmetry_label: str = field(default=None)

    # Values derived from the above
    na: int = field(init=False)
    nb: int = field(init=False)
    twice_ms: int = field(init=False)

    def __post_init__(self):
        self.twice_ms = int(round(self.ms * 2))

        ### Sanity checks
        # 1. Basic checks
        assert np.isclose(
            int(round(self.nel)), self.nel
        ), "Number of electrons must be an integer!"
        self.nel = int(round(self.nel))
        assert (
            self.nel >= 0
        ), f"Number of electrons must be non-negative, got {self.nel}."
        assert np.isclose(
            int(round(self.multiplicity)), self.multiplicity
        ), "Multiplicity must be an integer!"
        self.multiplicity = int(round(self.multiplicity))
        assert (
            self.multiplicity >= 1
        ), f"Multiplicity must be at least 1! Got {self.multiplicity}."
        assert np.isclose(
            int(round(self.ms * 2)), self.ms * 2
        ), "ms must be a multiple of 0.5."

        # 2. Is the multiplicity compatible with the number of electrons?
        assert (
            self.multiplicity <= self.nel + 1
        ), f"Multiplicity {self.multiplicity} is incompatible with {self.nel} electrons."

        # 3. Is the Ms compatible with the number of electrons?
        if self.nel % 2 != self.twice_ms % 2:
            raise ValueError(f"{self.nel} electrons is incompatible with ms={self.ms}!")

        # 4. Is the Ms compatible with the multiplicity?
        # multiplicity = 2 S + 1
        # Ms \in [-S, -S+1, ..., S-1, S]
        twice_S = self.multiplicity - 1
        allowed_twice_ms_values = [i for i in range(-twice_S, twice_S + 1, 2)]
        if self.twice_ms not in allowed_twice_ms_values:
            raise ValueError(
                f"Requested Ms ({self.ms}) incompatible with multiplicity ({self.multiplicity}). Change the value of Ms."
            )
        ###

        self.na = int(round(self.nel + self.twice_ms) / 2)
        self.nb = int(round(self.nel - self.twice_ms) / 2)
        assert (
            self.nel == self.na + self.nb
        ), f"Number of electrons {self.nel} does not match na + nb = {self.na} + {self.nb}."
        assert (
            self.na >= 0 and self.nb >= 0
        ), f"Non-negative number of alpha and beta electrons is required."

    def multiplicity_label(self) -> str:
        """Return the label for the multiplicity of the state."""
        return multiplicity_labels[self.multiplicity - 1]

    def str_minimum(self) -> str:
        """Return a minimal string representation of the state."""
        symmetry_label1 = (
            self.symmetry_label if self.symmetry_label else f"Irrep {self.symmetry}"
        )
        return f"Nα = {self.na} Nβ = {self.nb} {self.multiplicity_label()} (Ms = {self.get_ms_string(self.twice_ms)}) {symmetry_label1}"

    def __str__(self) -> str:
        gas_restrictions = ""
        if self.gas_min:
            gas_restrictions += (
                " GAS min: " + " ".join(str(i) for i in self.gas_min) + ";"
            )
        if self.gas_max:
            gas_restrictions += (
                " GAS max: " + " ".join(str(i) for i in self.gas_max) + ";"
            )
        return self.str_minimum() + gas_restrictions

    def str_short(self) -> str:
        """Return a short string representation of the state."""
        multi = f"m{self.multiplicity}.z{self.twice_ms}"
        sym = f".h{self.irrep}"
        gmin = ".g" + "".join(f"_{i}" for i in self.gas_min) if self.gas_min else ""
        gmax = ".g" + "".join(f"_{i}" for i in self.gas_max) if self.gas_max else ""
        return multi + sym + gmin + gmax

    def __hash__(self) -> int:
        repr_str = (
            f"{self.na}_{self.nb}_{self.multiplicity}_{self.twice_ms}_{self.irrep}"
        )
        repr_str += "".join(f"_{i}" for i in self.gas_min)
        repr_str += "".join(f"_{i}" for i in self.gas_max)
        return hash(repr_str)

    @staticmethod
    def get_ms_string(twice_ms: int) -> str:
        """Return a string representation of Ms based on its value."""
        return str(twice_ms // 2) if twice_ms % 2 == 0 else f"{twice_ms}/2"


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
