from collections.abc import Sequence
from typing import Annotated, overload

import numpy
from numpy.typing import NDArray


class Shell:
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, l: int, exponents: Sequence[float], coeffs: Sequence[float], center: Sequence[float], is_pure: bool = True, embed_normalization_into_coefficients: bool = True) -> None:
        """
        Construct a shell from the angular momentum (l) and a list of exponents and coefficients.
        """

    def __repr__(self) -> str: ...

    @property
    def size(self) -> int:
        """
        The number of basis functions in the shell (e.g., for l = 2, size = 5).
        """

    @property
    def ncontr(self) -> int:
        """The number of contractions in the shell."""

    @property
    def nprim(self) -> int:
        """The number of primitive Gaussians in the shell."""

    @property
    def l(self) -> int:
        """The angular momentum of the shell."""

    @property
    def coeff(self) -> list[float]:
        """The coefficients of the primitives in the shell."""

    @property
    def exponents(self) -> list[float]:
        """The exponents of the primitives in the shell."""

    @property
    def is_pure(self) -> bool:
        """Is the shell pure? (i.e., we have 5d and 7f functions)"""

    @property
    def center(self) -> list[float]:
        """The center of the shell (x, y, z) in bohr."""

class Basis:
    def __init__(self) -> None: ...

    def add(self, shell: Shell) -> None: ...

    def set_name(self, name: str) -> None: ...

    def __getitem__(self, i: int) -> Shell: ...

    def __len__(self) -> int: ...

    @property
    def shell_first_and_size(self) -> list[tuple[int, int]]: ...

    @property
    def center_first_and_last(self) -> list[tuple[int, int]]: ...

    @property
    def size(self) -> int: ...

    @property
    def max_l(self) -> int: ...

    @property
    def name(self) -> str: ...

    @property
    def max_nprim(self) -> int: ...

    @property
    def nprim(self) -> int: ...

    @property
    def nshells(self) -> int: ...

    def __repr__(self) -> str: ...

def shell_label(l: int, idx: int) -> str:
    """Returns a label for a given angular momentum (l) and index (idx)."""

def evaluate_shell(shell: Shell, point: Sequence[float]) -> list[float]:
    """
    Evaluate the shell at a given point. Returns a list of values for each basis function.
    """

def nuclear_repulsion(charges: Sequence[tuple[float, Sequence[float]]]) -> float: ...

@overload
def overlap(basis1: Basis, basis2: Basis) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None))]:
    """
    Compute the overlap integral matrix.

    Parameters
    ----------
    b1 : forte2.Basis
        First basis set.
    b2 : forte2.Basis
        Second basis set.

    Returns
    -------
    ndarray, shape = (nb1, nb2)
        Overlap integrals matrix.
    """

@overload
def overlap(basis: Basis) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None))]: ...

@overload
def kinetic(basis1: Basis, basis2: Basis) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None))]: ...

@overload
def kinetic(basis: Basis) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None))]: ...

@overload
def nuclear(basis1: Basis, basis2: Basis, charges: Sequence[tuple[float, Sequence[float]]]) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None))]: ...

@overload
def nuclear(basis: Basis, charges: Sequence[tuple[float, Sequence[float]]]) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None))]: ...

@overload
def emultipole1(basis1: Basis, basis2: Basis, origin: Sequence[float] = [0.0, 0.0, 0.0]) -> list[Annotated[NDArray[numpy.float64], dict(shape=(None, None))]]: ...

@overload
def emultipole1(basis: Basis, origin: Sequence[float] = [0.0, 0.0, 0.0]) -> list[Annotated[NDArray[numpy.float64], dict(shape=(None, None))]]: ...

@overload
def emultipole2(basis1: Basis, basis2: Basis, origin: Sequence[float] = [0.0, 0.0, 0.0]) -> list[Annotated[NDArray[numpy.float64], dict(shape=(None, None))]]: ...

@overload
def emultipole2(basis: Basis, origin: Sequence[float] = [0.0, 0.0, 0.0]) -> list[Annotated[NDArray[numpy.float64], dict(shape=(None, None))]]: ...

@overload
def emultipole3(basis1: Basis, basis2: Basis, origin: Sequence[float] = [0.0, 0.0, 0.0]) -> list[Annotated[NDArray[numpy.float64], dict(shape=(None, None))]]: ...

@overload
def emultipole3(basis: Basis, origin: Sequence[float] = [0.0, 0.0, 0.0]) -> list[Annotated[NDArray[numpy.float64], dict(shape=(None, None))]]: ...

@overload
def opVop(basis1: Basis, basis2: Basis, charges: Sequence[tuple[float, Sequence[float]]]) -> list[Annotated[NDArray[numpy.float64], dict(shape=(None, None))]]: ...

@overload
def opVop(basis: Basis, charges: Sequence[tuple[float, Sequence[float]]]) -> list[Annotated[NDArray[numpy.float64], dict(shape=(None, None))]]: ...

def erf_nuclear(basis1: Basis, basis2: Basis, omega_charges: "std::__1::tuple<double, std::__1::vector<std::__1::pair<double, std::__1::array<double, 3ul>>, std::__1::allocator<std::__1::pair<double, std::__1::array<double, 3ul>>>>>") -> Annotated[NDArray[numpy.float64], dict(shape=(None, None))]: ...

def erfc_nuclear(basis1: Basis, basis2: Basis, omega_charges: "std::__1::tuple<double, std::__1::vector<std::__1::pair<double, std::__1::array<double, 3ul>>, std::__1::allocator<std::__1::pair<double, std::__1::array<double, 3ul>>>>>") -> Annotated[NDArray[numpy.float64], dict(shape=(None, None))]: ...

@overload
def coulomb_4c(basis1: Basis, basis2: Basis, basis3: Basis, basis4: Basis) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None, None, None))]: ...

@overload
def coulomb_4c(basis: Basis) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None, None, None))]: ...

def coulomb_3c(basis1: Basis, basis2: Basis, basis3: Basis) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None, None))]: ...

@overload
def coulomb_2c(basis1: Basis, basis2: Basis) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None))]: ...

@overload
def coulomb_2c(basis: Basis) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None))]: ...

def erf_coulomb_3c(basis1: Basis, basis2: Basis, basis3: Basis, omega: float) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None, None))]: ...

def erf_coulomb_2c(basis1: Basis, basis2: Basis, omega: float) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None))]: ...

def erfc_coulomb_3c(basis1: Basis, basis2: Basis, basis3: Basis, omega: float) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None, None))]: ...

def erfc_coulomb_2c(basis1: Basis, basis2: Basis, omega: float) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None))]: ...

def basis_at_points(basis: Basis, points: Sequence[Sequence[float]]) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None))]: ...

def orbitals_at_points(basis: Basis, points: Sequence[Sequence[float]], C: Annotated[NDArray[numpy.float64], dict(shape=(None, None))]) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None))]:
    """
    Evaluate the orbitals on a set of points. Returns a 2D array of shape (npoints, norb).
    """

def orbitals_on_grid(basis: Basis, C: Annotated[NDArray[numpy.float64], dict(shape=(None, None))], min: Sequence[float], npoints: Sequence[int], axis: Sequence[Sequence[float]]) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None))]: ...
