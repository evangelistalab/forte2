from collections.abc import Iterator, Mapping, Sequence
import enum
from typing import Annotated, overload

from numpy.typing import ArrayLike

from . import cpp_helpers as cpp_helpers, ints as ints


class CIStrings:
    def __init__(self, na: int, nb: int, symmetry: int, orbital_symmetry: Sequence[Sequence[int]], gas_min: Sequence[int], gas_max: Sequence[int]) -> None:
        """
        Initialize the CIStrings with number of alpha and beta electrons, symmetry, orbital symmetry, minimum and maximum number of electrons in each GAS space, and logging level
        """

    @property
    def alfa_address(self) -> "std::__1::shared_ptr<forte2::StringAddress>": ...

    @property
    def na(self) -> int: ...

    @property
    def nb(self) -> int: ...

    @property
    def symmetry(self) -> int: ...

    @property
    def nas(self) -> int: ...

    @property
    def nbs(self) -> int: ...

    @property
    def ndet(self) -> int: ...

    @property
    def ngas_spaces(self) -> int: ...

    @property
    def gas_size(self) -> list[int]: ...

    @property
    def gas_alfa_occupations(self) -> list[list[int]]: ...

    @property
    def gas_beta_occupations(self) -> list[list[int]]: ...

    @property
    def gas_occupations(self) -> list[tuple[int, int]]: ...

    def determinant(self, address: int) -> Determinant: ...

    def determinant_index(self, d: Determinant) -> int: ...

    def make_determinants(self) -> list[Determinant]: ...

class CISigmaBuilder:
    def __init__(self, lists: CIStrings, E: float, H: Annotated[ArrayLike, dict(dtype='float64', shape=(None, None))], V: Annotated[ArrayLike, dict(dtype='float64', shape=(None, None, None, None))], log_level: int = 3) -> None:
        """
        Initialize the CISigmaBuilder with CIStrings, energy, Hamiltonian, and integrals
        """

    def set_algorithm(self, algorithm: str) -> None:
        """Set the sigma build algorithm (options = kh, hz)"""

    def get_algorithm(self) -> str:
        """Get the current sigma build algorithm"""

    def set_memory(self, memory: int) -> None:
        """Set the memory limit for the builder (in MB)"""

    def form_Hdiag_csf(self, dets: Sequence[Determinant], spin_adapter: CISpinAdapter, spin_adapt_full_preconditioner: bool = False) -> Annotated[ArrayLike, dict(dtype='float64', shape=(None))]: ...

    def energy_csf(self, dets: Sequence[Determinant], spin_adapter: CISpinAdapter, I: int) -> float:
        """Compute the energy of a CSF"""

    def form_H_csf(self, dets: Sequence[Determinant], spin_adapter: CISpinAdapter) -> Annotated[ArrayLike, dict(dtype='float64', shape=(None, None))]:
        """Form the full Hamiltonian matrix in the CSF basis"""

    def slater_rules_csf(self, dets: Sequence[Determinant], spin_adapter: CISpinAdapter, I: int, J: int) -> float: ...

    def Hamiltonian(self, basis: Annotated[ArrayLike, dict(dtype='float64', shape=(None))], sigma: Annotated[ArrayLike, dict(dtype='float64', shape=(None))]) -> None: ...

    def sf_1rdm(self, C_left: Annotated[ArrayLike, dict(dtype='float64', shape=(None))], C_right: Annotated[ArrayLike, dict(dtype='float64', shape=(None))]) -> Annotated[ArrayLike, dict(dtype='float64', shape=(None, None))]:
        """Compute the spin-free one-electron reduced density matrix"""

    def sf_2rdm(self, C_left: Annotated[ArrayLike, dict(dtype='float64', shape=(None))], C_right: Annotated[ArrayLike, dict(dtype='float64', shape=(None))]) -> Annotated[ArrayLike, dict(dtype='float64', shape=(None, None, None, None))]:
        """Compute the spin-free two-electron reduced density matrix"""

    def sf_3rdm(self, C_left: Annotated[ArrayLike, dict(dtype='float64', shape=(None))], C_right: Annotated[ArrayLike, dict(dtype='float64', shape=(None))]) -> Annotated[ArrayLike, dict(dtype='float64', shape=(None, None, None, None, None, None))]:
        """Compute the spin-free three-electron reduced density matrix"""

    def sf_2cumulant(self, C_left: Annotated[ArrayLike, dict(dtype='float64', shape=(None))], C_right: Annotated[ArrayLike, dict(dtype='float64', shape=(None))]) -> Annotated[ArrayLike, dict(dtype='float64', shape=(None, None, None, None))]:
        """Compute the spin-free two-electron cumulant"""

    def sf_3cumulant(self, C_left: Annotated[ArrayLike, dict(dtype='float64', shape=(None))], C_right: Annotated[ArrayLike, dict(dtype='float64', shape=(None))]) -> Annotated[ArrayLike, dict(dtype='float64', shape=(None, None, None, None, None, None))]:
        """Compute the spin-free three-electron cumulant"""

    def a_1rdm(self, C_left: Annotated[ArrayLike, dict(dtype='float64', shape=(None))], C_right: Annotated[ArrayLike, dict(dtype='float64', shape=(None))]) -> Annotated[ArrayLike, dict(dtype='float64', shape=(None, None))]:
        """Compute the alpha one-electron reduced density matrix"""

    def b_1rdm(self, C_left: Annotated[ArrayLike, dict(dtype='float64', shape=(None))], C_right: Annotated[ArrayLike, dict(dtype='float64', shape=(None))]) -> Annotated[ArrayLike, dict(dtype='float64', shape=(None, None))]:
        """Compute the beta one-electron reduced density matrix"""

    def aa_2rdm(self, C_left: Annotated[ArrayLike, dict(dtype='float64', shape=(None))], C_right: Annotated[ArrayLike, dict(dtype='float64', shape=(None))]) -> Annotated[ArrayLike, dict(dtype='float64', shape=(None, None))]:
        """Compute the alpha-alpha two-electron reduced density matrix"""

    def bb_2rdm(self, C_left: Annotated[ArrayLike, dict(dtype='float64', shape=(None))], C_right: Annotated[ArrayLike, dict(dtype='float64', shape=(None))]) -> Annotated[ArrayLike, dict(dtype='float64', shape=(None, None))]:
        """Compute the beta-beta two-electron reduced density matrix"""

    def ab_2rdm(self, C_left: Annotated[ArrayLike, dict(dtype='float64', shape=(None))], C_right: Annotated[ArrayLike, dict(dtype='float64', shape=(None))]) -> Annotated[ArrayLike, dict(dtype='float64', shape=(None, None, None, None))]:
        """Compute the alpha-beta two-electron reduced density matrix"""

    def aaa_3rdm(self, C_left: Annotated[ArrayLike, dict(dtype='float64', shape=(None))], C_right: Annotated[ArrayLike, dict(dtype='float64', shape=(None))]) -> Annotated[ArrayLike, dict(dtype='float64', shape=(None, None))]:
        """Compute the alpha-alpha-alpha three-electron reduced density matrix"""

    def aab_3rdm(self, C_left: Annotated[ArrayLike, dict(dtype='float64', shape=(None))], C_right: Annotated[ArrayLike, dict(dtype='float64', shape=(None))]) -> Annotated[ArrayLike, dict(dtype='float64', shape=(None, None, None, None))]:
        """Compute the alpha-alpha-beta three-electron reduced density matrix"""

    def abb_3rdm(self, C_left: Annotated[ArrayLike, dict(dtype='float64', shape=(None))], C_right: Annotated[ArrayLike, dict(dtype='float64', shape=(None))]) -> Annotated[ArrayLike, dict(dtype='float64', shape=(None, None, None, None))]:
        """Compute the alpha-beta-beta three-electron reduced density matrix"""

    def bbb_3rdm(self, C_left: Annotated[ArrayLike, dict(dtype='float64', shape=(None))], C_right: Annotated[ArrayLike, dict(dtype='float64', shape=(None))]) -> Annotated[ArrayLike, dict(dtype='float64', shape=(None, None))]:
        """Compute the beta-beta-beta three-electron reduced density matrix"""

    def avg_build_time(self) -> list[float]: ...

    def set_log_level(self, level: int) -> None:
        """Set the logging level for the class"""

    def a_1rdm_debug(self, C_left: Annotated[ArrayLike, dict(dtype='float64', shape=(None))], C_right: Annotated[ArrayLike, dict(dtype='float64', shape=(None))], alfa: bool) -> Annotated[ArrayLike, dict(dtype='float64', shape=(None, None))]: ...

    def aa_2rdm_debug(self, C_left: Annotated[ArrayLike, dict(dtype='float64', shape=(None))], C_right: Annotated[ArrayLike, dict(dtype='float64', shape=(None))], alfa: bool) -> Annotated[ArrayLike, dict(dtype='float64', shape=(None, None))]:
        """
        Compute the two-electron same-spin reduced density matrix for debugging purposes
        """

    def ab_2rdm_debug(self, C_left: Annotated[ArrayLike, dict(dtype='float64', shape=(None))], C_right: Annotated[ArrayLike, dict(dtype='float64', shape=(None))]) -> Annotated[ArrayLike, dict(dtype='float64', shape=(None, None, None, None))]:
        """
        Compute the two-electron mixed-spin reduced density matrix for debugging purposes
        """

    def aaa_3rdm_debug(self, C_left: Annotated[ArrayLike, dict(dtype='float64', shape=(None))], C_right: Annotated[ArrayLike, dict(dtype='float64', shape=(None))], alfa: bool) -> Annotated[ArrayLike, dict(dtype='float64', shape=(None, None))]:
        """
        Compute the three-electron same-spin reduced density matrix for debugging purposes
        """

    def aab_3rdm_debug(self, C_left: Annotated[ArrayLike, dict(dtype='float64', shape=(None))], C_right: Annotated[ArrayLike, dict(dtype='float64', shape=(None))]) -> Annotated[ArrayLike, dict(dtype='float64', shape=(None, None, None, None))]:
        """
        Compute the aab mixed-spin three-electron reduced density matrix for debugging purposes
        """

    def abb_3rdm_debug(self, C_left: Annotated[ArrayLike, dict(dtype='float64', shape=(None))], C_right: Annotated[ArrayLike, dict(dtype='float64', shape=(None))]) -> Annotated[ArrayLike, dict(dtype='float64', shape=(None, None, None, None))]:
        """
        Compute the abb mixed-spin three-electron reduced density matrix for debugging purposes
        """

    def aaaa_4rdm_debug(self, C_left: Annotated[ArrayLike, dict(dtype='float64', shape=(None))], C_right: Annotated[ArrayLike, dict(dtype='float64', shape=(None))], alfa: bool) -> Annotated[ArrayLike, dict(dtype='float64', shape=(None, None))]:
        """
        Compute the four-electron same-spin reduced density matrix for debugging purposes
        """

    def aaab_4rdm_debug(self, C_left: Annotated[ArrayLike, dict(dtype='float64', shape=(None))], C_right: Annotated[ArrayLike, dict(dtype='float64', shape=(None))]) -> Annotated[ArrayLike, dict(dtype='float64', shape=(None, None, None, None))]:
        """
        Compute the aaab mixed-spin four-electron reduced density matrix for debugging purposes
        """

    def aabb_4rdm_debug(self, C_left: Annotated[ArrayLike, dict(dtype='float64', shape=(None))], C_right: Annotated[ArrayLike, dict(dtype='float64', shape=(None))]) -> Annotated[ArrayLike, dict(dtype='float64', shape=(None, None, None, None))]:
        """
        Compute the aabb mixed-spin four-electron reduced density matrix for debugging purposes
        """

    def abbb_4rdm_debug(self, C_left: Annotated[ArrayLike, dict(dtype='float64', shape=(None))], C_right: Annotated[ArrayLike, dict(dtype='float64', shape=(None))]) -> Annotated[ArrayLike, dict(dtype='float64', shape=(None, None, None, None))]:
        """
        Compute the abbb mixed-spin four-electron reduced density matrix for debugging purposes
        """

    def sf_1rdm_debug(self, C_left: Annotated[ArrayLike, dict(dtype='float64', shape=(None))], C_right: Annotated[ArrayLike, dict(dtype='float64', shape=(None))]) -> Annotated[ArrayLike, dict(dtype='float64', shape=(None, None))]:
        """
        Compute the spin-free one-electron reduced density matrix for debugging purposes
        """

    def sf_2rdm_debug(self, C_left: Annotated[ArrayLike, dict(dtype='float64', shape=(None))], C_right: Annotated[ArrayLike, dict(dtype='float64', shape=(None))]) -> Annotated[ArrayLike, dict(dtype='float64', shape=(None, None, None, None))]:
        """
        Compute the spin-free two-electron reduced density matrix for debugging purposes
        """

    def sf_3rdm_debug(self, C_left: Annotated[ArrayLike, dict(dtype='float64', shape=(None))], C_right: Annotated[ArrayLike, dict(dtype='float64', shape=(None))]) -> Annotated[ArrayLike, dict(dtype='float64', shape=(None, None, None, None, None, None))]:
        """
        Compute the spin-free three-electron reduced density matrix for debugging purposes
        """

    def sf_2cumulant_debug(self, C_left: Annotated[ArrayLike, dict(dtype='float64', shape=(None))], C_right: Annotated[ArrayLike, dict(dtype='float64', shape=(None))]) -> Annotated[ArrayLike, dict(dtype='float64', shape=(None, None, None, None))]:
        """Compute the spin-free two-electron cumulant for debugging purposes"""

    def sf_3cumulant_debug(self, C_left: Annotated[ArrayLike, dict(dtype='float64', shape=(None))], C_right: Annotated[ArrayLike, dict(dtype='float64', shape=(None))]) -> Annotated[ArrayLike, dict(dtype='float64', shape=(None, None, None, None, None, None))]:
        """Compute the spin-free three-electron cumulant for debugging purposes"""

class CISpinAdapter:
    def __init__(self, twoS: int, twoMs: int, norb: int) -> None: ...

    def prepare_couplings(self, dets: Sequence[Determinant]) -> None: ...

    def csf_C_to_det_C(self, csf_C: Annotated[ArrayLike, dict(dtype='float64', shape=(None))], det_C: Annotated[ArrayLike, dict(dtype='float64', shape=(None))]) -> None: ...

    def det_C_to_csf_C(self, det_C: Annotated[ArrayLike, dict(dtype='float64', shape=(None))], csf_C: Annotated[ArrayLike, dict(dtype='float64', shape=(None))]) -> None: ...

    def ncsf(self) -> int: ...

    def set_log_level(self, level: int) -> None:
        """Set the logging level for the class"""

class Determinant:
    @overload
    def __init__(self, arg: Determinant) -> None: ...

    @overload
    def __init__(self, arg: str, /) -> None: ...

    @staticmethod
    def zero() -> Determinant: ...

    def __eq__(self, arg: Determinant, /) -> bool: ...

    def __lt__(self, arg: Determinant, /) -> bool: ...

    def __hash__(self) -> int: ...

    def __repr__(self) -> str:
        """String representation of the determinant"""

    def set_na(self, arg0: int, arg1: bool, /) -> None: ...

    def set_nb(self, arg0: int, arg1: bool, /) -> None: ...

    def na(self, arg: int, /) -> bool: ...

    def nb(self, arg: int, /) -> bool: ...

    def count_a(self) -> int: ...

    def count_b(self) -> int: ...

    def count(self) -> int: ...

    def create_a(self, n: int) -> float:
        """
        Apply an alpha creation operator to the determinant at the specified orbital index and return the sign
        """

    def create_b(self, n: int) -> float:
        """
        Apply a beta creation operator to the determinant at the specified orbital index and return the sign
        """

    def destroy_a(self, n: int) -> float:
        """
        Apply an alpha destruction operator to the determinant at the specified orbital index and return the sign
        """

    def destroy_b(self, n: int) -> float:
        """
        Apply a beta destruction operator to the determinant at the specified orbital index and return the sign
        """

    def spin_flip(self) -> Determinant:
        """Spin flip the determinant, i.e., swap alpha and beta orbitals"""

    def slater_sign(self, arg: int, /) -> float:
        """Get the sign of the Slater determinant"""

    def slater_sign_reverse(self, arg: int, /) -> float:
        """Get the sign of the Slater determinant"""

    def gen_excitation(self, arg0: Sequence[int], arg1: Sequence[int], arg2: Sequence[int], arg3: Sequence[int], /) -> float:
        """Apply a generic excitation"""

    def excitation_connection(self, arg: Determinant, /) -> list[list[int]]:
        """Get the excitation connection between this and another determinant"""

    def str(self, n: int = 64) -> str:
        """Get the string representation of the Slater determinant"""

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

class Configuration:
    @overload
    def __init__(self) -> None:
        """Build an empty configuration"""

    @overload
    def __init__(self, arg: Determinant, /) -> None:
        """Build a configuration from a determinant"""

    def str(self, n: int = 64) -> str:
        """Get the string representation of the Slater determinant"""

    def is_empt(self, n: int) -> bool:
        """Is orbital n empty?"""

    def is_docc(self, n: int) -> bool:
        """Is orbital n doubly occupied?"""

    def is_socc(self, n: int) -> bool:
        """Is orbital n singly occupied?"""

    def set_occ(self, n: int, value: int) -> None:
        """Set the value of an alpha bit"""

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

def set_log_level(arg: int, /) -> None:
    """
    Set the logging verbosity level (0=NONE, 1=ERROR, 2=WARNING, 3=INFO, 4=DEBUG)
    """

def get_log_level() -> int:
    """Get the current logging verbosity level"""

class SlaterRules:
    def __init__(self, norb: int, scalar_energy: float, one_electron_integrals: Annotated[ArrayLike, dict(dtype='float64', shape=(None, None))], two_electron_integrals: Annotated[ArrayLike, dict(dtype='float64', shape=(None, None, None, None))]) -> None: ...

    def energy(self, arg: Determinant, /) -> float: ...

    def slater_rules(self, lhs: Determinant, rhs: Determinant) -> float: ...

class SparseState:
    """A class to represent a vector of determinants"""

    @overload
    def __init__(self) -> None:
        """Default constructor"""

    @overload
    def __init__(self, arg: SparseState) -> None:
        """Copy constructor"""

    @overload
    def __init__(self, arg: Mapping[Determinant, complex], /) -> None:
        """Create a SparseState from a container of Determinants"""

    @overload
    def __init__(self, det: Determinant, val: complex = 1) -> None:
        """Create a SparseState with a single determinant"""

    def items(self) -> Iterator[tuple[Determinant, complex]]: ...

    def str(self, arg: int, /) -> str: ...

    def size(self) -> int: ...

    def norm(self, p: int = 2) -> float:
        """
        Calculate the p-norm of the SparseState (default p = 2, p = -1 for infinity norm)
        """

    def add(self, arg0: Determinant, arg1: complex, /) -> None: ...

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

    def map(self) -> "ankerl::unordered_dense::v4_5_0::detail::table<forte2::DeterminantImpl<128ul>, std::__1::complex<double>, std::__1::hash<forte2::DeterminantImpl<128ul>>, std::__1::equal_to<forte2::DeterminantImpl<128ul>>, std::__1::allocator<std::__1::pair<forte2::DeterminantImpl<128ul>, std::__1::complex<double>>>, ankerl::unordered_dense::v4_5_0::bucket_type::standard, ankerl::unordered_dense::v4_5_0::detail::default_container_t, false>": ...

    def elements(self) -> "ankerl::unordered_dense::v4_5_0::detail::table<forte2::DeterminantImpl<128ul>, std::__1::complex<double>, std::__1::hash<forte2::DeterminantImpl<128ul>>, std::__1::equal_to<forte2::DeterminantImpl<128ul>>, std::__1::allocator<std::__1::pair<forte2::DeterminantImpl<128ul>, std::__1::complex<double>>>, ankerl::unordered_dense::v4_5_0::bucket_type::standard, ankerl::unordered_dense::v4_5_0::detail::default_container_t, false>": ...

    def __getitem__(self, arg: Determinant, /) -> complex: ...

    def __setitem__(self, arg0: Determinant, arg1: complex, /) -> None: ...

    def __contains__(self, arg: Determinant, /) -> int: ...

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

    def matrix(self, dets: Sequence[Determinant], screen_thresh: float = 1e-12) -> Annotated[ArrayLike, dict(dtype='complex128', shape=(None, None))]:
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

def sparse_operator_hamiltonian(scalar_energy: float, one_electron_integrals: Annotated[ArrayLike, dict(dtype='float64', shape=(None, None))], two_electron_integrals: Annotated[ArrayLike, dict(dtype='float64', shape=(None, None, None, None))], screen_thresh: float = 1e-12) -> SparseOperator:
    """
    Create a SparseOperator object representing the Hamiltonian from integrals
    """

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

class SQOperatorString:
    """A class to represent a string of creation/annihilation operators"""

    def __init__(self, arg0: Determinant, arg1: Determinant, /) -> None: ...

    def cre(self) -> Determinant:
        """Get the creation operator string"""

    def ann(self) -> Determinant:
        """Get the annihilation operator string"""

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

    def op_tuple(self) -> "std::__1::vector<std::__1::tuple<bool, bool, int>, std::__1::allocator<std::__1::tuple<bool, bool, int>>>":
        """Get the operator tuple"""

    def __eq__(self, arg: SQOperatorString, /) -> bool:
        """Check if two operator strings are equal"""

    def __lt__(self, arg: SQOperatorString, /) -> bool:
        """Check if an operator string is less than another"""

    def __mul__(self, arg: complex, /) -> SparseOperator:
        """Multiply an operator string by a scalar"""

    def __rmul__(self, arg: complex, /) -> SparseOperator:
        """Multiply an operator string by a scalar"""

def sqop(s: str, allow_reordering: bool = False) -> tuple[SQOperatorString, float]:
    """
    Create an operator string from a string representation (default: no not allow reordering)
    """

class CommutatorType(enum.Enum):
    commute = 0

    anticommute = 1

    may_not_commute = 2

def commutator_type(lhs: SQOperatorString, rhs: SQOperatorString) -> CommutatorType:
    """Get the commutator type of two operator strings"""

__author__: str = 'Forte2 Developers'
