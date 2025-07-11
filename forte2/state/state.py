import numpy as np
from dataclasses import dataclass, field
from forte2.helpers.multiplicity_labels import multiplicity_labels


@dataclass(order=True)
class State:
    """Class to represent a state of a quantum system.

    Attributes:
        nel (int): Total number of electrons.
        multiplicity (int): Multiplicity of the state (2S+1).
        ms (float): Spin projection (Ms) of the state.
        irrep (int, optional): Irreducible representation of the state in Cotton ordering.
        gas_min (list[int], optional): Minimum GAS restrictions.
        gas_max (list[int], optional): Maximum GAS restrictions.
    """

    nel: int
    multiplicity: int
    ms: float
    gas_min: list[int] = field(default_factory=list)
    gas_max: list[int] = field(default_factory=list)

    # Values derived from the above
    symmetry: int = field(default=0)
    symmetry_label: str = field(default=None)
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
        return multiplicity_labels[self.multiplicity - 1]

    def str_minimum(self) -> str:
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
        return str(twice_ms // 2) if twice_ms % 2 == 0 else f"{twice_ms}/2"
