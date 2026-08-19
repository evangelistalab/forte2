"""Determinants and their operations"""

from collections.abc import Sequence
from typing import Annotated, Final, overload

import numpy
from numpy.typing import NDArray


class Determinant:
    @overload
    def __init__(self, arg: Determinant) -> None: ...

    @overload
    def __init__(self, str: str) -> None:
        """Build a determinant from a string representation"""

    @staticmethod
    def zero() -> Determinant:
        """Create a zero determinant with no electrons"""

    maxnorb: Final[int] = ...
    """The maximum number of orbitals supported by the Determinant class"""

    def __eq__(self, arg: Determinant, /) -> bool:
        """Check if two determinants are equal"""

    def __lt__(self, arg: Determinant, /) -> bool:
        """Check if a determinant is less than another determinant"""

    def __hash__(self) -> int:
        """Get the hash of the determinant"""

    def __repr__(self) -> str:
        """String representation of the determinant"""

    def set_na(self, n: int, value: bool) -> None:
        """Set the occupation of an alpha orbital"""

    def set_nb(self, n: int, value: bool) -> None:
        """Set the occupation of a beta orbital"""

    def na(self, n: int) -> bool:
        """Is orbital n occupied by an alpha electron?"""

    def nb(self, n: int) -> bool:
        """Is orbital n occupied by a beta electron?"""

    def count_alpha(self) -> int:
        """Count the number of alpha electrons"""

    def count_beta(self) -> int:
        """Count the number of beta electrons"""

    def count(self) -> int:
        """Count the total number of electrons"""

    def create_alpha(self, n: int) -> float:
        """
        Apply an alpha creation operator to the determinant at the specified orbital index and return the sign
        """

    def create_beta(self, n: int) -> float:
        """
        Apply a beta creation operator to the determinant at the specified orbital index and return the sign
        """

    def destroy_alpha(self, n: int) -> float:
        """
        Apply an alpha destruction operator to the determinant at the specified orbital index and return the sign
        """

    def destroy_beta(self, n: int) -> float:
        """
        Apply a beta destruction operator to the determinant at the specified orbital index and return the sign
        """

    def excitation_connection(self, arg: Determinant, /) -> tuple[list[int], list[int], list[int], list[int]]:
        """
        Describe the excitation connection of a determinant d, relative to this one.The excitation connection is defined as the creation and annihilation operators that need to be applied to this determinant to obtain d. The excitation connection is a vector of 4 vectors:[[alfa annihilation], [alfa creation],[beta annihilation], [beta creation]]
        """

    def spin_flip(self) -> Determinant:
        """Spin flip the determinant, i.e., swap alpha and beta orbitals"""

    def str(self, n: int = 64) -> str:
        """Get the string representation of the Slater determinant"""

class Configuration:
    @overload
    def __init__(self) -> None:
        """Build an empty configuration"""

    @overload
    def __init__(self, arg: Determinant, /) -> None:
        """Build a configuration from a determinant"""

    def str(self, n: int = 64) -> str:
        """Get the string representation of the Slater determinant"""

    def is_empty(self, n: int) -> bool:
        """Is orbital n empty?"""

    def is_docc(self, n: int) -> bool:
        """Is orbital n doubly occupied?"""

    def is_socc(self, n: int) -> bool:
        """Is orbital n singly occupied?"""

    def set_occ(self, n: int, value: int) -> None:
        """Set the occupation value of an orbital"""

    def count_docc(self) -> int:
        """Count the number of doubly occupied orbitals"""

    def count_socc(self) -> int:
        """Count the number of singly occupied orbitals"""

    def get_docc_vec(self) -> list[int]:
        """Get a list of the doubly occupied orbitals"""

    def get_socc_vec(self) -> list[int]:
        """Get a list of the singly occupied orbitals"""

    def __repr__(self) -> str:
        """Get the string representation of the configuration"""

    def __str__(self) -> str:
        """Get the string representation of the configuration"""

    def __eq__(self, arg: Configuration, /) -> bool:
        """Check if two configurations are equal"""

    def __lt__(self, arg: Configuration, /) -> bool:
        """Check if a configuration is less than another configuration"""

    def __hash__(self) -> int:
        """Get the hash of the configuration"""

@overload
def hilbert_space(nmo: int, na: int, nb: int, nirrep: int = 1, mo_symmetry: Sequence[int] = [], symmetry: int = 0) -> list[Determinant]:
    """
    Generate the Hilbert space for a given number of electrons and orbitals.If information about the symmetry of the MOs is not provided, it assumes that all MOs have symmetry 0.
    """

@overload
def hilbert_space(nmo: int, na: int, nb: int, ref: Determinant, truncation: int, nirrep: int = 1, mo_symmetry: Sequence[int] = [], symmetry: int = 0) -> list[Determinant]:
    """
    Generate the Hilbert space for a given number of electrons, orbitals, and the truncation level.If information about the symmetry of the MOs is not provided, it assumes that all MOs have symmetry 0.A reference determinant must be provided to establish the excitation rank.
    """

def spin2(arg0: Determinant, arg1: Determinant, /) -> float:
    """Compute the S^2 value between two determinants"""

class SlaterRules:
    def __init__(self, norb: int, scalar_energy: float, one_electron_integrals: Annotated[NDArray[numpy.float64], dict(shape=(None, None))], two_electron_integrals: Annotated[NDArray[numpy.float64], dict(shape=(None, None, None, None))]) -> None:
        """
        Initialize a SlaterRules object with the number of orbitals, scalar energy, one-electron integrals, and two-electron integrals in physicist's notation.
        """

    def energy(self, arg: Determinant, /) -> float: ...

    def energies(self, dets: Sequence[Determinant]) -> Annotated[NDArray[numpy.float64], dict(shape=(None,))]:
        """Compute the energies of a vector of determinants"""

    def slater_rules(self, lhs: Determinant, rhs: Determinant) -> float: ...

class RelSlaterRules:
    def __init__(self, nspinor: int, scalar_energy: float, one_electron_integrals: Annotated[NDArray[numpy.complex128], dict(shape=(None, None))], two_electron_integrals: Annotated[NDArray[numpy.complex128], dict(shape=(None, None, None, None))]) -> None:
        """
        Initialize a RelSlaterRules object with the number of spinor(orbitals), scalar energy, one-electron integrals, and two-electron integrals in physicist's notation.
        """

    def energy(self, arg: Determinant, /) -> float: ...

    def energies(self, dets: Sequence[Determinant]) -> Annotated[NDArray[numpy.float64], dict(shape=(None,))]:
        """Compute the energies of a vector of determinants"""

    def slater_rules(self, lhs: Determinant, rhs: Determinant) -> complex: ...
