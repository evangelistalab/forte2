"""CI and selected CI helper classes"""

from collections.abc import Iterable, Iterator, Sequence
from typing import Annotated, overload

import numpy
from numpy.typing import NDArray

import forte2.lib.det
import forte2.lib.sparse_ops


class DeterminantVector:
    @overload
    def __init__(self) -> None:
        """Default constructor"""

    @overload
    def __init__(self, arg: DeterminantVector) -> None:
        """Copy constructor"""

    @overload
    def __init__(self, arg: Iterable[forte2.lib.det.Determinant], /) -> None:
        """Construct from an iterable object"""

    def __len__(self) -> int: ...

    def __bool__(self) -> bool:
        """Check whether the vector is nonempty"""

    def __repr__(self) -> str: ...

    def __iter__(self) -> Iterator[forte2.lib.det.Determinant]: ...

    @overload
    def __getitem__(self, arg: int, /) -> forte2.lib.det.Determinant: ...

    @overload
    def __getitem__(self, arg: slice, /) -> DeterminantVector: ...

    def clear(self) -> None:
        """Remove all items from list."""

    def append(self, arg: forte2.lib.det.Determinant, /) -> None:
        """Append `arg` to the end of the list."""

    def insert(self, arg0: int, arg1: forte2.lib.det.Determinant, /) -> None:
        """Insert object `arg1` before index `arg0`."""

    def pop(self, index: int = -1) -> forte2.lib.det.Determinant:
        """Remove and return item at `index` (default last)."""

    def extend(self, arg: DeterminantVector, /) -> None:
        """Extend `self` by appending elements from `arg`."""

    @overload
    def __setitem__(self, arg0: int, arg1: forte2.lib.det.Determinant, /) -> None: ...

    @overload
    def __setitem__(self, arg0: slice, arg1: DeterminantVector, /) -> None: ...

    @overload
    def __delitem__(self, arg: int, /) -> None: ...

    @overload
    def __delitem__(self, arg: slice, /) -> None: ...

    def __eq__(self, arg: object, /) -> bool: ...

    def __ne__(self, arg: object, /) -> bool: ...

    @overload
    def __contains__(self, arg: forte2.lib.det.Determinant, /) -> bool: ...

    @overload
    def __contains__(self, arg: object, /) -> bool: ...

    def count(self, arg: forte2.lib.det.Determinant, /) -> int:
        """Return number of occurrences of `arg`."""

    def remove(self, arg: forte2.lib.det.Determinant, /) -> None:
        """Remove first occurrence of `arg`."""

class CIStrings:
    def __init__(self, na: int, nb: int, symmetry: int, orbital_symmetry: Sequence[Sequence[int]], gas_min: Sequence[int], gas_max: Sequence[int]) -> None:
        """
        Initialize the CIStrings with number of alpha and beta electrons, symmetry, orbital symmetry, minimum and maximum number of electrons in each GAS space
        """

    @property
    def alpha_address(self) -> "std::__1::shared_ptr<forte2::StringAddress>": ...

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
    def gas_alpha_occupations(self) -> list[list[int]]: ...

    @property
    def gas_beta_occupations(self) -> list[list[int]]: ...

    @property
    def gas_occupations(self) -> list[tuple[int, int]]: ...

    def determinant(self, address: int) -> forte2.lib.det.Determinant: ...

    def determinant_index(self, d: forte2.lib.det.Determinant) -> int: ...

    def make_determinants(self) -> DeterminantVector: ...

class CISigmaBuilder:
    def __init__(self, lists: CIStrings, E: float, H: Annotated[NDArray[numpy.float64], dict(shape=(None, None))], V: Annotated[NDArray[numpy.float64], dict(shape=(None, None, None, None))], log_level: int = 3, algorithm: str = 'kh') -> None:
        """
        Initialize the CISigmaBuilder with CIStrings, energy, Hamiltonian, and integrals
        """

    def get_algorithm(self) -> str:
        """Get the current sigma build algorithm"""

    def set_memory(self, memory: int) -> None:
        """Set the memory limit for the builder (in MB)"""

    def form_Hdiag_csf(self, dets: DeterminantVector, spin_adapter: CISpinAdapter, spin_adapt_full_preconditioner: bool = False) -> Annotated[NDArray[numpy.float64], dict(shape=(None,))]: ...

    def energy_csf(self, dets: DeterminantVector, spin_adapter: CISpinAdapter, I: int) -> float:
        """Compute the energy of a CSF"""

    def form_H_csf(self, dets: DeterminantVector, spin_adapter: CISpinAdapter) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None))]:
        """Form the full Hamiltonian matrix in the CSF basis"""

    def slater_rules_csf(self, dets: DeterminantVector, spin_adapter: CISpinAdapter, I: int, J: int) -> float: ...

    def Hamiltonian(self, basis: Annotated[NDArray[numpy.float64], dict(shape=(None,))], sigma: Annotated[NDArray[numpy.float64], dict(shape=(None,))]) -> None: ...

    def sigma_one_electron(self, basis: Annotated[NDArray[numpy.float64], dict(shape=(None,))], sigma: Annotated[NDArray[numpy.float64], dict(shape=(None,))]) -> None:
        """
        Apply the scalar and one-electron part of the Hamiltonian to the wave function
        """

    def sigma_two_electron(self, basis: Annotated[NDArray[numpy.float64], dict(shape=(None,))], sigma: Annotated[NDArray[numpy.float64], dict(shape=(None,))]) -> None:
        """Apply the two-electron part of the Hamiltonian to the wave function"""

    def set_Hamiltonian(self, E: float | None = None, H: Annotated[NDArray[numpy.float64], dict(shape=(None, None))] | None = None, V: Annotated[NDArray[numpy.float64], dict(shape=(None, None, None, None))] | None = None) -> None:
        """
        Swap in a new Hamiltonian with the same number of orbitals, without reallocating scratch buffers. Any argument left as None keeps its current value.
        """

    def make_sparse_state(self, C: Annotated[NDArray[numpy.float64], dict(shape=(None,))], threshold: float = 1e-12) -> forte2.lib.sparse_ops.SparseState:
        """Convert a CI vector to a sparse state"""

    def sf_1rdm(self, C_left: Annotated[NDArray[numpy.float64], dict(shape=(None,))], C_right: Annotated[NDArray[numpy.float64], dict(shape=(None,))]) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None))]:
        """Compute the spin-free one-electron reduced density matrix"""

    def sf_2rdm(self, C_left: Annotated[NDArray[numpy.float64], dict(shape=(None,))], C_right: Annotated[NDArray[numpy.float64], dict(shape=(None,))]) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None, None, None))]:
        """Compute the spin-free two-electron reduced density matrix"""

    def sf_3rdm(self, C_left: Annotated[NDArray[numpy.float64], dict(shape=(None,))], C_right: Annotated[NDArray[numpy.float64], dict(shape=(None,))]) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None, None, None, None, None))]:
        """Compute the spin-free three-electron reduced density matrix"""

    def sf_2cumulant(self, C_left: Annotated[NDArray[numpy.float64], dict(shape=(None,))], C_right: Annotated[NDArray[numpy.float64], dict(shape=(None,))]) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None, None, None))]:
        """Compute the spin-free two-electron cumulant"""

    def sf_3cumulant(self, C_left: Annotated[NDArray[numpy.float64], dict(shape=(None,))], C_right: Annotated[NDArray[numpy.float64], dict(shape=(None,))]) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None, None, None, None, None))]:
        """Compute the spin-free three-electron cumulant"""

    def a_1rdm(self, C_left: Annotated[NDArray[numpy.float64], dict(shape=(None,))], C_right: Annotated[NDArray[numpy.float64], dict(shape=(None,))]) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None))]:
        """Compute the alpha one-electron reduced density matrix"""

    def b_1rdm(self, C_left: Annotated[NDArray[numpy.float64], dict(shape=(None,))], C_right: Annotated[NDArray[numpy.float64], dict(shape=(None,))]) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None))]:
        """Compute the beta one-electron reduced density matrix"""

    def aa_2rdm(self, C_left: Annotated[NDArray[numpy.float64], dict(shape=(None,))], C_right: Annotated[NDArray[numpy.float64], dict(shape=(None,))]) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None))]:
        """Compute the alpha-alpha two-electron reduced density matrix"""

    def bb_2rdm(self, C_left: Annotated[NDArray[numpy.float64], dict(shape=(None,))], C_right: Annotated[NDArray[numpy.float64], dict(shape=(None,))]) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None))]:
        """Compute the beta-beta two-electron reduced density matrix"""

    def ab_2rdm(self, C_left: Annotated[NDArray[numpy.float64], dict(shape=(None,))], C_right: Annotated[NDArray[numpy.float64], dict(shape=(None,))]) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None, None, None))]:
        """Compute the alpha-beta two-electron reduced density matrix"""

    def aaa_3rdm(self, C_left: Annotated[NDArray[numpy.float64], dict(shape=(None,))], C_right: Annotated[NDArray[numpy.float64], dict(shape=(None,))]) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None))]:
        """Compute the alpha-alpha-alpha three-electron reduced density matrix"""

    def aab_3rdm(self, C_left: Annotated[NDArray[numpy.float64], dict(shape=(None,))], C_right: Annotated[NDArray[numpy.float64], dict(shape=(None,))]) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None, None, None))]:
        """Compute the alpha-alpha-beta three-electron reduced density matrix"""

    def abb_3rdm(self, C_left: Annotated[NDArray[numpy.float64], dict(shape=(None,))], C_right: Annotated[NDArray[numpy.float64], dict(shape=(None,))]) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None, None, None))]:
        """Compute the alpha-beta-beta three-electron reduced density matrix"""

    def bbb_3rdm(self, C_left: Annotated[NDArray[numpy.float64], dict(shape=(None,))], C_right: Annotated[NDArray[numpy.float64], dict(shape=(None,))]) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None))]:
        """Compute the beta-beta-beta three-electron reduced density matrix"""

    def a_1trdm(self, sigmabuilder_right: CISigmaBuilder, C_left: Annotated[NDArray[numpy.float64], dict(shape=(None,))], C_right: Annotated[NDArray[numpy.float64], dict(shape=(None,))]) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None))]:
        """Compute the alpha one-electron transition reduced density matrix"""

    def b_1trdm(self, sigmabuilder_right: CISigmaBuilder, C_left: Annotated[NDArray[numpy.float64], dict(shape=(None,))], C_right: Annotated[NDArray[numpy.float64], dict(shape=(None,))]) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None))]:
        """Compute the beta one-electron transition reduced density matrix"""

    def sf_1trdm(self, sigmabuilder_right: CISigmaBuilder, C_left: Annotated[NDArray[numpy.float64], dict(shape=(None,))], C_right: Annotated[NDArray[numpy.float64], dict(shape=(None,))]) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None))]:
        """Compute the spin-free one-electron transition reduced density matrix"""

    def avg_build_time(self) -> list[float]: ...

    def set_log_level(self, level: int) -> None:
        """Set the logging level for the class"""

    def a_1rdm_debug(self, C_left: Annotated[NDArray[numpy.float64], dict(shape=(None,))], C_right: Annotated[NDArray[numpy.float64], dict(shape=(None,))], alpha: bool) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None))]: ...

    def aa_2rdm_debug(self, C_left: Annotated[NDArray[numpy.float64], dict(shape=(None,))], C_right: Annotated[NDArray[numpy.float64], dict(shape=(None,))], alpha: bool) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None))]:
        """
        Compute the two-electron same-spin reduced density matrix for debugging purposes
        """

    def ab_2rdm_debug(self, C_left: Annotated[NDArray[numpy.float64], dict(shape=(None,))], C_right: Annotated[NDArray[numpy.float64], dict(shape=(None,))]) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None, None, None))]:
        """
        Compute the two-electron mixed-spin reduced density matrix for debugging purposes
        """

    def aaa_3rdm_debug(self, C_left: Annotated[NDArray[numpy.float64], dict(shape=(None,))], C_right: Annotated[NDArray[numpy.float64], dict(shape=(None,))], alpha: bool) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None))]:
        """
        Compute the three-electron same-spin reduced density matrix for debugging purposes
        """

    def aab_3rdm_debug(self, C_left: Annotated[NDArray[numpy.float64], dict(shape=(None,))], C_right: Annotated[NDArray[numpy.float64], dict(shape=(None,))]) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None, None, None))]:
        """
        Compute the aab mixed-spin three-electron reduced density matrix for debugging purposes
        """

    def abb_3rdm_debug(self, C_left: Annotated[NDArray[numpy.float64], dict(shape=(None,))], C_right: Annotated[NDArray[numpy.float64], dict(shape=(None,))]) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None, None, None))]:
        """
        Compute the abb mixed-spin three-electron reduced density matrix for debugging purposes
        """

    def aaaa_4rdm_debug(self, C_left: Annotated[NDArray[numpy.float64], dict(shape=(None,))], C_right: Annotated[NDArray[numpy.float64], dict(shape=(None,))], alpha: bool) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None))]:
        """
        Compute the four-electron same-spin reduced density matrix for debugging purposes
        """

    def aaab_4rdm_debug(self, C_left: Annotated[NDArray[numpy.float64], dict(shape=(None,))], C_right: Annotated[NDArray[numpy.float64], dict(shape=(None,))]) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None, None, None))]:
        """
        Compute the aaab mixed-spin four-electron reduced density matrix for debugging purposes
        """

    def aabb_4rdm_debug(self, C_left: Annotated[NDArray[numpy.float64], dict(shape=(None,))], C_right: Annotated[NDArray[numpy.float64], dict(shape=(None,))]) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None, None, None))]:
        """
        Compute the aabb mixed-spin four-electron reduced density matrix for debugging purposes
        """

    def abbb_4rdm_debug(self, C_left: Annotated[NDArray[numpy.float64], dict(shape=(None,))], C_right: Annotated[NDArray[numpy.float64], dict(shape=(None,))]) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None, None, None))]:
        """
        Compute the abbb mixed-spin four-electron reduced density matrix for debugging purposes
        """

    def sf_1rdm_debug(self, C_left: Annotated[NDArray[numpy.float64], dict(shape=(None,))], C_right: Annotated[NDArray[numpy.float64], dict(shape=(None,))]) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None))]:
        """
        Compute the spin-free one-electron reduced density matrix for debugging purposes
        """

    def sf_2rdm_debug(self, C_left: Annotated[NDArray[numpy.float64], dict(shape=(None,))], C_right: Annotated[NDArray[numpy.float64], dict(shape=(None,))]) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None, None, None))]:
        """
        Compute the spin-free two-electron reduced density matrix for debugging purposes
        """

    def sf_3rdm_debug(self, C_left: Annotated[NDArray[numpy.float64], dict(shape=(None,))], C_right: Annotated[NDArray[numpy.float64], dict(shape=(None,))]) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None, None, None, None, None))]:
        """
        Compute the spin-free three-electron reduced density matrix for debugging purposes
        """

    def sf_2cumulant_debug(self, C_left: Annotated[NDArray[numpy.float64], dict(shape=(None,))], C_right: Annotated[NDArray[numpy.float64], dict(shape=(None,))]) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None, None, None))]:
        """Compute the spin-free two-electron cumulant for debugging purposes"""

    def sf_3cumulant_debug(self, C_left: Annotated[NDArray[numpy.float64], dict(shape=(None,))], C_right: Annotated[NDArray[numpy.float64], dict(shape=(None,))]) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None, None, None, None, None))]:
        """Compute the spin-free three-electron cumulant for debugging purposes"""

class CISpinAdapter:
    def __init__(self, twoS: int, twoMs: int, norb: int) -> None: ...

    def prepare_couplings(self, dets: DeterminantVector) -> None: ...

    def csf_C_to_det_C(self, csf_C: Annotated[NDArray[numpy.float64], dict(shape=(None,))], det_C: Annotated[NDArray[numpy.float64], dict(shape=(None,))]) -> None: ...

    def det_C_to_csf_C(self, det_C: Annotated[NDArray[numpy.float64], dict(shape=(None,))], csf_C: Annotated[NDArray[numpy.float64], dict(shape=(None,))]) -> None: ...

    @property
    def nconf(self) -> int: ...

    @property
    def ncsf(self) -> int: ...

    def set_log_level(self, level: int) -> None:
        """Set the logging level for the class"""

class RelCISigmaBuilder:
    def __init__(self, lists: CIStrings, E: float, H: Annotated[NDArray[numpy.complex128], dict(shape=(None, None))], V: Annotated[NDArray[numpy.complex128], dict(shape=(None, None, None, None))], log_level: int = 3, algorithm: str = 'hz') -> None:
        """
        Initialize the CISigmaBuilder with CIStrings, energy, Hamiltonian, and integrals
        """

    def get_algorithm(self) -> str:
        """Get the current sigma build algorithm"""

    def set_memory(self, memory: int) -> None:
        """Set the memory limit for the builder (in MB)"""

    def form_Hdiag(self, dets: DeterminantVector) -> Annotated[NDArray[numpy.complex128], dict(shape=(None,))]: ...

    def slater_rules(self, dets: DeterminantVector, I: int, J: int) -> complex: ...

    def Hamiltonian(self, basis: Annotated[NDArray[numpy.complex128], dict(shape=(None,))], sigma: Annotated[NDArray[numpy.complex128], dict(shape=(None,))]) -> None: ...

    def sigma_one_electron(self, basis: Annotated[NDArray[numpy.complex128], dict(shape=(None,))], sigma: Annotated[NDArray[numpy.complex128], dict(shape=(None,))]) -> None:
        """
        Apply the scalar and one-electron part of the Hamiltonian to the wave function
        """

    def sigma_two_electron(self, basis: Annotated[NDArray[numpy.complex128], dict(shape=(None,))], sigma: Annotated[NDArray[numpy.complex128], dict(shape=(None,))]) -> None:
        """Apply the two-electron part of the Hamiltonian to the wave function"""

    def set_Hamiltonian(self, E: float | None = None, H: Annotated[NDArray[numpy.complex128], dict(shape=(None, None))] | None = None, V: Annotated[NDArray[numpy.complex128], dict(shape=(None, None, None, None))] | None = None) -> None:
        """
        Swap in a new Hamiltonian with the same number of orbitals, without reallocating scratch buffers. Any argument left as None keeps its current value.
        """

    def so_1rdm(self, C_left: Annotated[NDArray[numpy.complex128], dict(shape=(None,))], C_right: Annotated[NDArray[numpy.complex128], dict(shape=(None,))]) -> Annotated[NDArray[numpy.complex128], dict(shape=(None, None))]:
        """Compute the spin-orbital one-electron reduced density matrix"""

    def so_2rdm(self, C_left: Annotated[NDArray[numpy.complex128], dict(shape=(None,))], C_right: Annotated[NDArray[numpy.complex128], dict(shape=(None,))]) -> Annotated[NDArray[numpy.complex128], dict(shape=(None, None, None, None))]:
        """Compute the spin-orbital two-electron reduced density matrix"""

    def so_2cumulant(self, C_left: Annotated[NDArray[numpy.complex128], dict(shape=(None,))], C_right: Annotated[NDArray[numpy.complex128], dict(shape=(None,))]) -> Annotated[NDArray[numpy.complex128], dict(shape=(None, None, None, None))]:
        """Compute the spin-orbital two-electron cumulant"""

    def so_3rdm(self, C_left: Annotated[NDArray[numpy.complex128], dict(shape=(None,))], C_right: Annotated[NDArray[numpy.complex128], dict(shape=(None,))]) -> Annotated[NDArray[numpy.complex128], dict(shape=(None, None, None, None, None, None))]:
        """Compute the spin-orbital three-electron reduced density matrix"""

    def so_3cumulant(self, C_left: Annotated[NDArray[numpy.complex128], dict(shape=(None,))], C_right: Annotated[NDArray[numpy.complex128], dict(shape=(None,))]) -> Annotated[NDArray[numpy.complex128], dict(shape=(None, None, None, None, None, None))]:
        """Compute the spin-orbital three-electron cumulant"""

    def so_1rdm_debug(self, C_left: Annotated[NDArray[numpy.complex128], dict(shape=(None,))], C_right: Annotated[NDArray[numpy.complex128], dict(shape=(None,))]) -> Annotated[NDArray[numpy.complex128], dict(shape=(None, None))]: ...

    def so_2rdm_debug(self, C_left: Annotated[NDArray[numpy.complex128], dict(shape=(None,))], C_right: Annotated[NDArray[numpy.complex128], dict(shape=(None,))]) -> Annotated[NDArray[numpy.complex128], dict(shape=(None, None, None, None))]: ...

    def so_3rdm_debug(self, C_left: Annotated[NDArray[numpy.complex128], dict(shape=(None,))], C_right: Annotated[NDArray[numpy.complex128], dict(shape=(None,))]) -> Annotated[NDArray[numpy.complex128], dict(shape=(None, None, None, None, None, None))]: ...

class SelectedCIHelper:
    def __init__(self, norb: int, dets: DeterminantVector, c: Annotated[NDArray[numpy.float64], dict(shape=(None, None))], E: float, H: Annotated[NDArray[numpy.float64], dict(shape=(None, None))], V: Annotated[NDArray[numpy.float64], dict(shape=(None, None, None, None))], log_level: int = 3, screening_criterion: str = 'hbci', frozen_creation: Sequence[int] = [], frozen_annihilation: Sequence[int] = []) -> None:
        """
        Initialize the SelectedCIHelper with the number of orbitals, initial determinants, energy, Hamiltonian, and integrals
        """

    def set_Hamiltonian(self, E: float | None = None, H: Annotated[NDArray[numpy.float64], dict(shape=(None, None))] | None = None, V: Annotated[NDArray[numpy.float64], dict(shape=(None, None, None, None))] | None = None) -> None:
        """
        Set the Hamiltonian integrals. Any argument left as None keeps its current value.
        """

    def Hamiltonian(self, basis: Annotated[NDArray[numpy.float64], dict(shape=(None,))], sigma: Annotated[NDArray[numpy.float64], dict(shape=(None,))]) -> None:
        """Apply the Hamiltonian to the basis and store the result in sigma"""

    def Hdiag(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None,))]:
        """Return the diagonal of the Hamiltonian matrix"""

    def set_c(self, c: Annotated[NDArray[numpy.float64], dict(shape=(None, None))]) -> None:
        """Set the CI coefficients"""

    def set_num_batches_per_thread(self, n: int) -> None:
        """
        Set the number of batches each thread will process in parallel sections
        """

    def set_energies(self, e: Annotated[NDArray[numpy.float64], dict(shape=(None,))]) -> None:
        """Set the energies of the roots"""

    def set_frozen_creation(self, frozen_creation: Sequence[int]) -> None:
        """Set orbitals excluded from creation in selection"""

    def set_frozen_annihilation(self, frozen_annihilation: Sequence[int]) -> None:
        """Set orbitals excluded from annihilation in selection"""

    def set_screening_criterion(self, criterion: str) -> None:
        """Set the screening criterion for selection ('hbci' or 'ehbci')"""

    def set_energy_correction(self, correction: str) -> None:
        """
        Set the energy correction method for selection ('variational' or 'pt2')
        """

    def set_pt2_regularizer(self, regularizer: str, strength: float = 0.5) -> None:
        """
        Set the PT2 regularization method ('none', 'shift', 'dsrg') and its strength
        """

    def select_hbci_ref(self, var_threshold: float, pt2_threshold: float) -> None:
        """Perform HBCI selection with the given threshold"""

    def select_hbci(self, var_threshold: float, pt2_threshold: float) -> None:
        """Perform HBCI selection with the given thresholds"""

    def compute_pt2_determ(self, eps2: float, num_batches: int) -> None:
        """
        Compute the second-order correction deterministically without modifying the variational space
        """

    def ept2(self) -> list[float]:
        """
        Return the second-order correction of the roots from the last compute_pt2 call
        """

    def ept2_stddev(self) -> list[float]:
        """
        Return the standard deviation of the second-order correction of the roots
        """

    def compute_pt2_semistoch(self, eps2: float, eps2_pseudostoch: float, eps2_determ: float, num_batches: int, min_batches_pseudostoch: int, target_error: float, num_batches_stoch: int, batches_per_sample: int, num_samples: int, sample_size: int, seed: int) -> None:
        """
        Compute the second-order correction with the three-step semistochastic algorithm
        """

    def ept2_determ(self) -> list[float]:
        """Return the deterministic term of the last semistochastic correction"""

    def ept2_pseudostoch(self) -> list[float]:
        """
        Return the pseudo-stochastic term of the last semistochastic correction
        """

    def ept2_stoch(self) -> list[float]:
        """Return the stochastic term of the last semistochastic correction"""

    def ept2_pseudostoch_stddev(self) -> list[float]:
        """Return the standard deviation of the pseudo-stochastic term"""

    def ept2_stoch_stddev(self) -> list[float]:
        """Return the standard deviation of the stochastic term"""

    def num_pseudostoch_batches(self) -> int:
        """Return the number of batches the pseudo-stochastic step evaluated"""

    def num_pt2_dets(self) -> int:
        """
        Return the number of external determinants included in the last compute_pt2 call
        """

    def pt2_time(self) -> float:
        """Return the wall time of the last compute_pt2 call"""

    def compute_spin2(self) -> list[float]:
        """
        Compute the expectation value of S^2 for each root and return as a list
        """

    def a_1rdm(self, left_root: int, right_root: int) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None))]:
        """Compute the alpha-spin 1-RDM between two roots"""

    def b_1rdm(self, left_root: int, right_root: int) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None))]:
        """Compute the beta-spin 1-RDM between two roots"""

    def sf_1rdm(self, left_root: int, right_root: int) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None))]:
        """Compute the spin-free 1-RDM between two roots"""

    def aa_2rdm(self, left_root: int, right_root: int) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None))]:
        """Compute the alpha-alpha 2-RDM between two roots"""

    def bb_2rdm(self, left_root: int, right_root: int) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None))]:
        """Compute the beta-beta 2-RDM between two roots"""

    def ab_2rdm(self, left_root: int, right_root: int) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None, None, None))]:
        """Compute the alpha-beta 2-RDM between two roots"""

    def sf_2rdm(self, left_root: int, right_root: int) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None, None, None))]:
        """Compute the spin-free 2-RDM between two roots"""

    def a_1trdm(self, right_helper: SelectedCIHelper, left_root: int, right_root: int) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None))]:
        """
        Compute the alpha-spin 1-transition RDM between two roots in different helpers
        """

    def b_1trdm(self, right_helper: SelectedCIHelper, left_root: int, right_root: int) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None))]:
        """
        Compute the beta-spin 1-transition RDM between two roots in different helpers
        """

    def sf_1trdm(self, right_helper: SelectedCIHelper, left_root: int, right_root: int) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None))]:
        """
        Compute the spin-free 1-transition RDM between two roots in different helpers
        """

    def dets(self) -> DeterminantVector:
        """Return the determinants in the variational space"""

    def ndets(self) -> int:
        """Return the number of determinants in the variational space"""

    def slater_rules(self, dets: DeterminantVector, I: int, J: int) -> float:
        """Compute the Hamiltonian matrix element <I|H|J>"""

    def energies(self) -> list[float]:
        """Return the energies of the roots"""

    def ept2_var(self) -> list[float]:
        """
        Return the variational part of the Epstein-Nesbet second-order energy correction
        """

    def ept2_pt(self) -> list[float]:
        """
        Return the perturbative part of the Epstein-Nesbet second-order energy correction
        """

    def num_new_dets_var(self) -> int:
        """
        Return the number of new variational determinants added in the last selection
        """

    def num_new_dets_pt2(self) -> int:
        """
        Return the number of new perturbative determinants added in the last selection
        """

    def selection_time(self) -> float:
        """Return the total selection time"""

class RelSelectedCIHelper:
    def __init__(self, norb: int, dets: DeterminantVector, c: Annotated[NDArray[numpy.complex128], dict(shape=(None, None))], E: float, H: Annotated[NDArray[numpy.complex128], dict(shape=(None, None))], V: Annotated[NDArray[numpy.complex128], dict(shape=(None, None, None, None))], log_level: int = 3, screening_criterion: str = 'hbci', frozen_creation: Sequence[int] = [], frozen_annihilation: Sequence[int] = []) -> None:
        """
        Initialize the RelSelectedCIHelper with the number of spinors, initial determinants, energy, complex Hamiltonian, and complex integrals
        """

    def set_Hamiltonian(self, E: float | None = None, H: Annotated[NDArray[numpy.complex128], dict(shape=(None, None))] | None = None, V: Annotated[NDArray[numpy.complex128], dict(shape=(None, None, None, None))] | None = None) -> None:
        """
        Set the (complex) Hamiltonian integrals. Any argument left as None keeps its current value.
        """

    def Hamiltonian(self, basis: Annotated[NDArray[numpy.complex128], dict(shape=(None,))], sigma: Annotated[NDArray[numpy.complex128], dict(shape=(None,))]) -> None:
        """
        Apply the Hamiltonian to the (complex) basis and store the result in sigma
        """

    def Hdiag(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None,))]:
        """Return the (real) diagonal of the Hamiltonian matrix"""

    def set_c(self, c: Annotated[NDArray[numpy.complex128], dict(shape=(None, None))]) -> None:
        """Set the (complex) CI coefficients"""

    def set_num_batches_per_thread(self, n: int) -> None:
        """
        Set the number of batches each thread will process in parallel sections
        """

    def set_energies(self, e: Annotated[NDArray[numpy.float64], dict(shape=(None,))]) -> None:
        """Set the energies of the roots"""

    def set_frozen_creation(self, frozen_creation: Sequence[int]) -> None:
        """Set orbitals excluded from creation in selection"""

    def set_frozen_annihilation(self, frozen_annihilation: Sequence[int]) -> None:
        """Set orbitals excluded from annihilation in selection"""

    def set_screening_criterion(self, criterion: str) -> None:
        """Set the screening criterion for selection (only 'hbci' is supported)"""

    def set_energy_correction(self, correction: str) -> None:
        """
        Set the energy correction method for selection ('variational' or 'pt2')
        """

    def set_pt2_regularizer(self, regularizer: str, strength: float = 0.5) -> None:
        """
        Set the PT2 regularization method ('none', 'shift', 'dsrg') and its strength
        """

    def select_hbci_ref(self, var_threshold: float, pt2_threshold: float) -> None:
        """Perform HBCI selection with the reference implementation"""

    def select_hbci(self, var_threshold: float, pt2_threshold: float) -> None:
        """Perform HBCI selection with the batched implementation"""

    def a_1rdm(self, left_root: int, right_root: int) -> Annotated[NDArray[numpy.complex128], dict(shape=(None, None))]:
        """
        Compute the complex alpha 1-RDM (or transition 1-RDM) between two roots
        """

    def aa_2rdm(self, left_root: int, right_root: int) -> Annotated[NDArray[numpy.complex128], dict(shape=(None, None, None, None))]:
        """
        Compute the complex alpha-alpha 2-RDM (or transition 2-RDM) between two roots
        """

    def dets(self) -> DeterminantVector:
        """Return the determinants in the variational space"""

    def ndets(self) -> int:
        """Return the number of determinants in the variational space"""

    def slater_rules(self, dets: DeterminantVector, I: int, J: int) -> complex:
        """Compute the Hamiltonian matrix element <I|H|J>"""

    def energies(self) -> list[float]:
        """Return the energies of the roots"""

    def ept2_var(self) -> list[float]:
        """
        Return the variational part of the Epstein-Nesbet second-order energy correction
        """

    def ept2_pt(self) -> list[float]:
        """
        Return the perturbative part of the Epstein-Nesbet second-order energy correction
        """

    def num_new_dets_var(self) -> int:
        """
        Return the number of new variational determinants added in the last selection
        """

    def num_new_dets_pt2(self) -> int:
        """
        Return the number of new perturbative determinants added in the last selection
        """

    def selection_time(self) -> float:
        """Return the total selection time"""
