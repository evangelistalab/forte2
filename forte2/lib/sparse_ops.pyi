"""Sparse operators, states, and their operations"""

from collections.abc import Iterator, Mapping, Sequence
import enum
from typing import Annotated, overload

import numpy
from numpy.typing import NDArray

import forte2.lib.det


class SQOperatorString:
    """A class to represent a string of creation/annihilation operators"""

    def __init__(self, arg0: forte2.lib.det.Determinant, arg1: forte2.lib.det.Determinant, /) -> None: ...

    def cre(self) -> forte2.lib.det.Determinant:
        """Get the creation operator string"""

    def ann(self) -> forte2.lib.det.Determinant:
        """Get the annihilation operator string"""

    def sign_mask(self) -> forte2.lib.det.Determinant:
        """Get the precomputed sign mask"""

    def str(self) -> str:
        """Get the string representation of the operator string"""

    def count(self) -> int:
        """Get the number of operators"""

    def adjoint(self) -> SQOperatorString:
        """Get the adjoint operator string"""

    def spin_flip(self) -> SQOperatorString:
        """Get the spin-flipped operator string"""

    def number_component(self) -> SQOperatorString:
        """Get the number component of the operator string"""

    def non_number_component(self) -> SQOperatorString:
        """Get the non-number component of the operator string"""

    def __str__(self) -> str:
        """Get the string representation of the operator string"""

    def __repr__(self) -> str:
        """Get the string representation of the operator string"""

    def latex(self) -> str:
        """Get the LaTeX representation of the operator string"""

    def latex_compact(self) -> str:
        """Get the compact LaTeX representation of the operator string"""

    def is_identity(self) -> bool:
        """Check if the operator string is the identity operator"""

    def is_nilpotent(self) -> bool:
        """Check if the operator string is nilpotent"""

    def op_tuple(self) -> list["std::__1::tuple<bool, bool, int>"]:
        """Get the operator tuple"""

    def __eq__(self, arg: SQOperatorString, /) -> bool:
        """Check if two operator strings are equal"""

    def __lt__(self, arg: SQOperatorString, /) -> bool:
        """Check if an operator string is less than another"""

    def __mul__(self, arg: complex, /) -> SparseOperator:
        """Multiply an operator string by a scalar"""

    def __rmul__(self, arg: complex, /) -> SparseOperator:
        """Multiply an operator string by a scalar"""

class CommutatorType(enum.Enum):
    commute = 0

    anticommute = 1

    may_not_commute = 2

def sqop(s: str, allow_reordering: bool = False) -> tuple[SQOperatorString, float]:
    """
    Create an operator string from a string representation (default: no not allow reordering)
    """

def compute_sign_mask(cre: forte2.lib.det.Determinant, ann: forte2.lib.det.Determinant) -> forte2.lib.det.Determinant:
    """
    Compute the sign mask associated with a set of creation and annihilation operators
    """

def commutator_type(lhs: SQOperatorString, rhs: SQOperatorString) -> CommutatorType:
    """Get the commutator type of two operator strings"""

class SparseState:
    """A class to represent a vector of determinants"""

    @overload
    def __init__(self) -> None:
        """Default constructor"""

    @overload
    def __init__(self, arg: SparseState) -> None:
        """Copy constructor"""

    @overload
    def __init__(self, arg: Mapping[forte2.lib.det.Determinant, complex], /) -> None:
        """Create a SparseState from a container of Determinants"""

    @overload
    def __init__(self, det: forte2.lib.det.Determinant, val: complex = 1) -> None:
        """Create a SparseState with a single determinant"""

    def items(self) -> Iterator[tuple[forte2.lib.det.Determinant, complex]]: ...

    def str(self, arg: int, /) -> str: ...

    def size(self) -> int: ...

    def norm(self, p: int = 2) -> float:
        """
        Calculate the p-norm of the SparseState (default p = 2, p = -1 for infinity norm)
        """

    def add(self, arg0: forte2.lib.det.Determinant, arg1: complex, /) -> None: ...

    def __add__(self, arg: SparseState, /) -> SparseState:
        """Add two SparseStates"""

    def __sub__(self, arg: SparseState, /) -> SparseState:
        """Subtract two SparseStates"""

    def __mul__(self, arg: complex, /) -> SparseState:
        """Multiply this SparseState by a scalar"""

    def __rmul__(self, arg: complex, /) -> SparseState:
        """Multiply a scalar by this SparseState"""

    def __iadd__(self, arg: SparseState, /) -> SparseState:
        """Add a SparseState to this SparseState"""

    def __isub__(self, arg: SparseState, /) -> SparseState:
        """Subtract a SparseState from this SparseState"""

    def __imul__(self, arg: complex, /) -> SparseState:
        """Multiply this SparseState by a scalar"""

    def __len__(self) -> int: ...

    def __eq__(self, arg: SparseState, /) -> bool: ...

    def __repr__(self) -> str: ...

    def __str__(self) -> str: ...

    def map(self) -> "ankerl::unordered_dense::v4_8_1::detail::table<forte2::DeterminantImpl<128ul>, std::__1::complex<double>, std::__1::hash<forte2::DeterminantImpl<128ul>>, std::__1::equal_to<forte2::DeterminantImpl<128ul>>, std::__1::allocator<std::__1::pair<forte2::DeterminantImpl<128ul>, std::__1::complex<double>>>, ankerl::unordered_dense::v4_8_1::bucket_type::standard, ankerl::unordered_dense::v4_8_1::detail::default_container_t, false>": ...

    def elements(self) -> "ankerl::unordered_dense::v4_8_1::detail::table<forte2::DeterminantImpl<128ul>, std::__1::complex<double>, std::__1::hash<forte2::DeterminantImpl<128ul>>, std::__1::equal_to<forte2::DeterminantImpl<128ul>>, std::__1::allocator<std::__1::pair<forte2::DeterminantImpl<128ul>, std::__1::complex<double>>>, ankerl::unordered_dense::v4_8_1::bucket_type::standard, ankerl::unordered_dense::v4_8_1::detail::default_container_t, false>": ...

    def __getitem__(self, arg: forte2.lib.det.Determinant, /) -> complex: ...

    def __setitem__(self, arg0: forte2.lib.det.Determinant, arg1: complex, /) -> None: ...

    def __contains__(self, arg: forte2.lib.det.Determinant, /) -> int: ...

    def apply(self, arg: SparseOperator, /) -> SparseState:
        """Apply an operator to this SparseState and return a new SparseState"""

    def apply_antiherm(self, arg: SparseOperator, /) -> SparseState:
        """
        Apply the antihermitian combination of the operator (op - op^dagger) to this SparseState and return a new SparseState
        """

    def number_project(self, arg0: int, arg1: int, /) -> SparseState: ...

    def spin2(self) -> complex:
        """Calculate the expectation value of S^2 for this SparseState"""

    def overlap(self, arg: SparseState, /) -> complex:
        """Calculate the overlap between this SparseState and another SparseState"""

def apply_op(sop: SparseOperator, state0: SparseState, screen_thresh: float = 1e-12) -> SparseState: ...

def apply_antiherm(sop: SparseOperator, state0: SparseState, screen_thresh: float = 1e-12) -> SparseState: ...

def apply_number_projector(arg0: int, arg1: int, arg2: SparseState, /) -> SparseState: ...

def get_projection(arg0: SparseOperatorList, arg1: SparseState, arg2: SparseState, /) -> list[complex]: ...

def spin2(arg0: SparseState, arg1: SparseState, /) -> complex:
    """Calculate the <left_state|S^2|right_state> expectation value"""

def overlap(arg0: SparseState, arg1: SparseState, /) -> complex: ...

def normalize(arg: SparseState, /) -> SparseState:
    """Returns a normalized version of the input SparseState"""

class SparseOperator:
    """A class to represent a sparse operator"""

    @overload
    def __init__(self) -> None:
        """Default constructor"""

    @overload
    def __init__(self, arg: SparseOperator) -> None:
        """Copy constructor"""

    @overload
    def __init__(self, arg: Mapping[SQOperatorString, complex], /) -> None:
        """Create a SparseOperator from a container of terms"""

    @overload
    def __init__(self, sqop: SQOperatorString, coefficient: complex = ...) -> None:
        """Create a SparseOperator with a single term"""

    @overload
    def add(self, sqop: SQOperatorString, coefficient: complex = ...) -> None:
        """Add a term to the operator"""

    @overload
    def add(self, str: str, coefficient: complex = ..., allow_reordering: bool = False) -> None:
        """Add a term to the operator from a string representation"""

    @overload
    def add(self, acre: Sequence[int], bcre: Sequence[int], aann: Sequence[int], bann: Sequence[int], coeff: complex = ...) -> None:
        """
        Add a term to the operator by passing lists of creation and annihilation indices. This version is faster than the string version and does not check for reordering
        """

    def remove(self, arg: str, /) -> None:
        """Remove a term"""

    def __iter__(self) -> Iterator[tuple[SQOperatorString, complex]]: ...

    def __getitem__(self, arg: str, /) -> complex:
        """Get the coefficient of a term"""

    def __len__(self) -> int:
        """Get the number of terms in the operator"""

    def coefficient(self, arg: str, /) -> complex:
        """Get the coefficient of a term"""

    def set_coefficient(self, arg0: str, arg1: complex, /) -> None:
        """Set the coefficient of a term"""

    def __add__(self, arg: SparseOperator, /) -> SparseOperator:
        """Add two SparseOperators"""

    def __sub__(self, arg: SparseOperator, /) -> SparseOperator:
        """Subtract two SparseOperators"""

    def __iadd__(self, arg: SparseOperator, /) -> SparseOperator:
        """Add a SparseOperator to this SparseOperator"""

    def __isub__(self, arg: SparseOperator, /) -> SparseOperator:
        """Subtract a SparseOperator from this SparseOperator"""

    def __imul__(self, arg: complex, /) -> SparseOperator:
        """Multiply this SparseOperator by a scalar"""

    @overload
    def __matmul__(self, arg: SparseOperator, /) -> SparseOperator:
        """Multiply two SparseOperator objects"""

    @overload
    def __matmul__(self, arg: SparseState, /) -> SparseState:
        """Multiply a SparseOperator and a SparseState"""

    def commutator(self, arg: SparseOperator, /) -> SparseOperator:
        """Compute the commutator of two SparseOperator objects"""

    def __itruediv__(self, arg: complex, /) -> SparseOperator:
        """Divide this SparseOperator by a scalar"""

    def __truediv__(self, arg: complex, /) -> SparseOperator:
        """Divide this SparseOperator by a scalar"""

    def __mul__(self, arg: complex, /) -> SparseOperator:
        """Multiply a SparseOperator by a scalar"""

    def __rmul__(self, arg: complex, /) -> SparseOperator:
        """Multiply a scalar by a SparseOperator"""

    def __rdiv__(self, arg: complex, /) -> SparseOperator:
        """Divide a scalar by a SparseOperator"""

    def __neg__(self) -> SparseOperator:
        """Negate the operator"""

    def copy(self, arg: SparseOperator, /) -> None:
        """Create a copy of this SparseOperator"""

    def norm(self) -> float:
        """Compute the norm of the operator"""

    def str(self) -> list[str]:
        """Get a string representation of the operator"""

    def latex(self) -> str:
        """Get a LaTeX representation of the operator"""

    def adjoint(self) -> SparseOperator:
        """Get the adjoint"""

    def __eq__(self, arg: SparseOperator, /) -> bool:
        """Check if two SparseOperators are equal"""

    def __repr__(self) -> str:
        """Get a string representation of the operator"""

    def __str__(self) -> str:
        """Get a string representation of the operator"""

    def apply_to_state(self, state: SparseState, screen_thresh: float = 1e-12) -> SparseState:
        """Apply the operator to a state"""

    def matrix(self, dets: Sequence[forte2.lib.det.Determinant], screen_thresh: float = 1e-12) -> Annotated[NDArray[numpy.complex128], dict(shape=(None, None))]:
        """
        Compute the matrix elements of the operator between a list of determinants
        """

@overload
def sparse_operator(s: str, coefficient: complex = ..., allow_reordering: bool = False) -> SparseOperator:
    """Create a SparseOperator object from a string and a complex"""

@overload
def sparse_operator(list: Sequence[tuple[str, complex]], allow_reordering: bool = False) -> SparseOperator:
    """Create a SparseOperator object from a list of Tuple[str, complex]"""

@overload
def sparse_operator(s: SQOperatorString, coefficient: complex = ...) -> SparseOperator:
    """Create a SparseOperator object from a SQOperatorString and a complex"""

@overload
def sparse_operator(list: Sequence[tuple[SQOperatorString, complex]]) -> SparseOperator:
    """
    Create a SparseOperator object from a list of Tuple[SQOperatorString, complex]
    """

def new_product(arg0: SparseOperator, arg1: SparseOperator, /) -> SparseOperator: ...

@overload
def sparse_operator_hamiltonian(scalar_energy: float, one_electron_integrals: Annotated[NDArray[numpy.float64], dict(shape=(None, None))], two_electron_integrals: Annotated[NDArray[numpy.float64], dict(shape=(None, None, None, None))], screen_thresh: float = 1e-12) -> SparseOperator:
    """
    Create a SparseOperator object representing the second quantized Hamiltonian.
    """

@overload
def sparse_operator_hamiltonian(scalar_energy: float, one_electron_integrals: Annotated[NDArray[numpy.complex128], dict(shape=(None, None))], two_electron_integrals: Annotated[NDArray[numpy.complex128], dict(shape=(None, None, None, None))], screen_thresh: float = 1e-12) -> SparseOperator: ...

class SparseOperatorList:
    """A class to represent a list of sparse operators"""

    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, arg: SparseOperatorList) -> None: ...

    @overload
    def add(self, arg0: SQOperatorString, arg1: complex, /) -> None: ...

    @overload
    def add(self, str: str, coefficient: complex = ..., allow_reordering: bool = False) -> None: ...

    @overload
    def add(self, acre: Sequence[int], bcre: Sequence[int], aann: Sequence[int], bann: Sequence[int], coeff: complex = ...) -> None:
        """
        Add a term to the operator by passing lists of creation and annihilation indices. This version is faster than the string version and does not check for reordering
        """

    def add_term(self, op_list: Sequence["std::__1::tuple<bool, bool, int>"], value: float = 0.0, allow_reordering: bool = False) -> None: ...

    def to_operator(self) -> SparseOperator: ...

    def remove(self, arg: str, /) -> None:
        """Remove a specific element from the vector space"""

    def __len__(self) -> int: ...

    def __iter__(self) -> Iterator[tuple[SQOperatorString, complex]]: ...

    def __repr__(self) -> str: ...

    def __str__(self) -> str: ...

    @overload
    def __getitem__(self, arg: int, /) -> complex:
        """Get the coefficient of a term"""

    @overload
    def __getitem__(self, arg: str, /) -> complex: ...

    def __setitem__(self, arg0: int, arg1: complex, /) -> None:
        """Set the coefficient of a term"""

    def coefficients(self) -> list[complex]: ...

    def set_coefficients(self, arg: Sequence[complex], /) -> None: ...

    def reverse(self) -> SparseOperatorList:
        """Reverse the order of the operators"""

    def pop_left(self) -> SparseOperatorList:
        """Remove the leftmost operator"""

    def pop_right(self) -> SparseOperatorList:
        """Remove the rightmost operator"""

    def slice(self, start: int, end: int) -> SparseOperatorList:
        """Return a slice of the operator"""

    def __call__(self, arg: int, /) -> tuple[SQOperatorString, complex]:
        """Get the nth operator"""

    def __matmul__(self, arg: SparseState, /) -> SparseState:
        """Multiply a SparseOperator and a SparseState"""

    def __add__(self, arg: SparseOperatorList, /) -> SparseOperatorList:
        """Add (concatenate) two SparseOperatorList objects"""

    def __iadd__(self, arg: SparseOperatorList, /) -> SparseOperatorList:
        """
        Add (concatenate) a SparseOperatorList object to this SparseOperatorList object
        """

    def apply_to_state(self, state: SparseState, screen_thresh: float = 1e-12) -> SparseState:
        """Apply the operator to a state"""

@overload
def operator_list(s: str, coefficient: complex = ..., allow_reordering: bool = False) -> SparseOperatorList:
    """Create a SparseOperatorList object from a string and a complex"""

@overload
def operator_list(list: Sequence[tuple[str, complex]], allow_reordering: bool = False) -> SparseOperatorList:
    """Create a SparseOperatorList object from a list of Tuple[str, complex]"""

@overload
def operator_list(s: SQOperatorString, coefficient: complex = ...) -> SparseOperatorList:
    """
    Create a SparseOperatorList object from a SQOperatorString and a complex
    """

@overload
def operator_list(list: Sequence[tuple[SQOperatorString, complex]]) -> SparseOperatorList:
    """
    Create a SparseOperatorList object from a list of Tuple[SQOperatorString, complex]
    """

class SparseExp:
    """A class to compute the exponential of a sparse operator"""

    def __init__(self, maxk: int = 19, screen_thresh: float = 1e-12) -> None: ...

    @overload
    def apply_op(self, sop: SparseOperator, state: SparseState, scaling_factor: float = 1.0) -> SparseState:
        """
        Apply the exponential of a SparseOperator to a state: exp(scaling_factor * sop) |state>
        """

    @overload
    def apply_op(self, sop: SparseOperatorList, state: SparseState, scaling_factor: float = 1.0) -> SparseState:
        """
        Apply the exponential of a SparseOperatorList to a state: exp(scaling_factor * sop) |state>
        """

    @overload
    def apply_antiherm(self, sop: SparseOperator, state: SparseState, scaling_factor: float = 1.0) -> SparseState:
        """
        Apply the antihermitian exponential of a SparseOperator to a state: exp(scaling_factor * (sop - sop^dagger)) |state>
        """

    @overload
    def apply_antiherm(self, sop: SparseOperatorList, state: SparseState, scaling_factor: float = 1.0) -> SparseState:
        """
        Apply the antihermitian exponential of a SparseOperatorList to a state: exp(scaling_factor * (sop - sop^dagger)) |state
        """

class SparseFactExp:
    """
    A class to compute the product exponential of a sparse operator using factorization
    """

    def __init__(self, screen_thresh: float = 1e-12) -> None: ...

    def apply_op(self, sop: SparseOperatorList, state: SparseState, inverse: bool = False, reverse: bool = False) -> SparseState:
        """
        Apply the factorized exponential of a SparseOperator to a state: ... exp(op2) exp(op1) |state>. inverse=True computes the inverse, and reverse=Trueapplies the operators in reverse order
        """

    def apply_antiherm(self, sop: SparseOperatorList, state: SparseState, inverse: bool = False, reverse: bool = False) -> SparseState:
        """
        Apply the factorized antihermitian exponential of a SparseOperator to a state: ... exp(op2 - op2^dagger) exp(op1 - op1^dagger) |state>. inverse=True computes the inverse, and reverse=True applies the operators in reverse order
        """

    def apply_antiherm_deriv(self, sqop: SQOperatorString, t: complex, state: SparseState) -> tuple[SparseState, SparseState]: ...
