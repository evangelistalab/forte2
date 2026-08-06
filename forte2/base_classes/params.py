from dataclasses import dataclass, field
from abc import ABC

from forte2.lib.det import Determinant


@dataclass
class ParamsBase(ABC):
    def copy(self, **kwargs):
        """Create a copy of this Params object, optionally overriding some fields."""
        # copy all fields from self
        fields = {
            f.name: getattr(self, f.name) for f in self.__dataclass_fields__.values()
        }
        # override with any provided kwargs
        for key, value in kwargs.items():
            if key not in fields:
                raise ValueError(
                    f"{self.__class__.__name__} has no field named '{key}'"
                )
            fields[key] = value
        # initialize a new instance of the same class with the updated fields
        # this ordering makes sure that the __post_init__ method is called with the updated fields,
        # so any validation logic there will be applied to the new values
        new_instance = type(self)(**fields)
        return new_instance

    def _validate_list(self, lst, pred):
        """
        validate that pred(i) for i in list is all true
        """
        for i in lst:
            if not pred(i):
                return False
        return True

    @staticmethod
    def _is_nonneg_integer(i):
        return isinstance(i, int) and i >= 0

    @staticmethod
    def _is_nonneg_float(i):
        return isinstance(i, int | float) and i >= 0


@dataclass
class DavidsonLiuParams(ParamsBase):
    """
    Parameters for the Davidson-Liu eigenvalue solver.

    Parameters
    ----------
    guess_per_root : int, optional, default=2
        The number of guess vectors for each root.
    ndets_per_guess : int, optional, default=10
        The number of determinants per guess vector.
    collapse_per_root : int, optional, default=2
        The number of determinants to collapse per root.
    basis_per_root : int, optional, default=4
        The maximum number of basis vectors per root.
    maxiter : int, optional, default=100
        The maximum number of iterations for the Davidson-Liu solver.
    e_tol : float, optional, default=1e-10
        The energy convergence threshold for the solver.
    r_tol : float, optional, default=1e-5
        The residual convergence threshold for the solver.
    """

    guess_per_root: int = 2
    ndets_per_guess: int = 10
    collapse_per_root: int = 2
    basis_per_root: int = 4
    maxiter: int = 100
    e_tol: float = 1e-12
    r_tol: float = 1e-6

    def __post_init__(self):
        if self.collapse_per_root < 1:
            raise ValueError(
                f"Davidson-Liu solver: collapse_per_root ({self.collapse_per_root}) must be greater than or equal to 1."
            )
        if self.basis_per_root < self.collapse_per_root + 1:
            raise ValueError(
                f"Davidson-Liu solver: basis_per_root ({self.basis_per_root}) must be greater than or equal to collapse_per_root + 1 ({self.collapse_per_root + 1})."
            )


@dataclass
class CIParams(ParamsBase):
    """
    Parameters for the CI solver.

    Parameters
    ----------
    ci_algorithm: str, optional, default="hz"
        The algorithm used for the CI sigma builder.
        Non-relativistic options are:
            - "hz" / "Harrison-Zarrabian"
            - "kh" / "Knowles-Handy"
            - "exact": Exact diagonalization
        Two-component (relativistic) options are:
            - "hz" / "Harrison-Zarrabian"
            - "exact": Exact diagonalization
            - "sparse": Sigma builder using sparse representation of the Hamiltonian and states.
                Recommended for debug use only.

    ci_builder_memory: int, optional, default=1024
        The maximum memory (in MB) to use for the CI sigma builder. This is used only if ci_algorithm is "hz" or "kh".
    energy_shift: float, optional, default=None
        An energy shift, used to find roots around a specific energy. If None, no shift is applied.
    """

    ci_algorithm: str = "hz"
    ci_builder_memory: int = 1024
    energy_shift: float = None

    def __post_init__(self):
        assert self.ci_algorithm.lower() in [
            "hz",
            "harrison-zarrabian",
            "kh",
            "knowles-handy",
            "exact",
            "sparse",
        ], "ci_algorithm must be one of 'hz', 'kh', 'exact', or 'sparse'."


@dataclass
class DMRGParams(ParamsBase):
    """
    Parameters for the DMRG (block2) active-space solver.

    Parameters
    ----------
    bond_dims : list[int], optional
        The MPS bond-dimension schedule over the DMRG sweeps. If shorter than
        ``n_sweeps``, the last value is repeated for the remaining sweeps.
        Defaults to a ramp from 250 to 500.
    noises : list[float], optional
        The perturbative-noise schedule over the DMRG sweeps. Defaults to a
        decreasing schedule ending at 0.
    thrds : list[float], optional
        The Davidson residual-convergence threshold schedule over the sweeps.
    n_sweeps : int, optional, default=20
        The maximum number of DMRG sweeps.
    tol : float, optional, default=1e-8
        The energy-convergence threshold used to terminate the sweeps early.
    symm_type : str, optional, default="SU2"
        The block2 symmetry type. Currently only ``"SU2"`` (spin-adapted) is
        supported.
    scratch : str, optional, default=None
        The scratch directory used by block2. If ``None``, a temporary
        directory is created.
    n_threads : int, optional, default=4
        The number of threads used by block2.
    iprint : int, optional, default=0
        The block2 verbosity level.
    reorder_orbitals : bool, optional, default=False
        Whether to let block2 reorder the active orbitals (via its Fiedler/gaopt
        heuristics) before building the MPO.
    """

    bond_dims: list[int] = None
    noises: list[float] = None
    thrds: list[float] = None
    n_sweeps: int = 20
    tol: float = 1e-8
    symm_type: str = "SU2"
    scratch: str = None
    n_threads: int = 4
    iprint: int = 0
    reorder_orbitals: bool = False

    def __post_init__(self):
        if not self.bond_dims:
            self.bond_dims = [250] * 4 + [500] * 4
        self._validate_list(self.bond_dims, self._is_nonneg_integer)
        if not self.noises:
            self.noises = [1e-4] * 4 + [1e-5] * 4 + [0.0]
        self._validate_list(self.noises, self._is_nonneg_float)
        if not self.thrds:
            self.thrds = [1e-10] * 8
        self._validate_list(self.thrds, self._is_nonneg_float)
        if self.symm_type.upper() != "SU2":
            raise ValueError(
                f"Only 'SU2' symm_type is currently supported, got '{self.symm_type}'."
            )
        if self.n_sweeps < 1:
            raise ValueError(f"n_sweeps needs to be >= 1, got {self.n_sweeps}.")
        if self.tol < 0:
            raise ValueError(f"tol needs to be > 0, got {self.tol}.")
        if self.n_threads < 1:
            raise ValueError(f"n_threads needs to be >= 1, got {self.n_threads}.")


@dataclass
class SelectedCIParams(ParamsBase):
    """
    Parameters for the Selected CI solver.

    Parameters
    ----------
    maxcycle: int, optional, default=15
        The maximum number of selection cycles.
    e_tol: float, optional, default=1e-8
        The energy convergence threshold for selected CI macroiterations.
    var_threshold: float, optional, default=5e-4
        The threshold for including determinants in the variational space based on their contribution to the wavefunction.
    pt2_threshold: float, optional, default=1e-8
        The threshold for including determinants in the perturbative correction based on their second-order energy contribution.
    selection_algorithm: str, optional, default="hbci"
        The algorithm used for selecting determinants. Options are "hbci" and "hbci_ref".
    guess_occ_window: int, optional, default=2
        The number of occupied orbitals to consider when generating guess determinants.
    guess_vir_window: int, optional, default=2
        The number of virtual orbitals to consider when generating guess determinants.
    num_threads: int, optional, default=4
        The number of threads to use for parallel selection and diagonalization.
    ci_algorithm: str, optional, default="sparse"
        The algorithm used for the CI diagonalization. Options are "exact" and "sparse".
    num_batches_per_thread: int, optional, default=4
        The number of batches of determinants to process per thread during selection and diagonalization.
    do_spin_penalty: bool, optional, default=True
        Whether to apply a spin penalty to the Hamiltonian to enforce correct spin symmetry.
    guess_dets: list[Determinant], optional
        A list of determinants to use as the initial guess for the CI wavefunction.
        Note that this set will be further filtered by `DavidsonLiuParams.ndets_per_guess` using the determinantal energies,
        before finally being enlarged to a spin-complete set.
        Therefore, it is not recommended to provide energetically disjoint guess determinants, as the higher energy ones will likely be filtered out.
        Use `pinned_guess_dets` to ensure certain determinants are included in the guess without relying on their energies.
        If not provided, the guess determinants will be generated based on the guess_occ_window and guess_vir_window parameters.
    pinned_guess_dets: list[Determinant], optional
        A list of determinants that are pinned to the initial guess, ensuring they are included in the variational space.
    frozen_creation: list[int], optional
        A list of orbital indices for which creation operators are frozen (i.e., not allowed to be occupied in the selected determinants).
        This is used to enforce certain symmetries or to exclude certain orbitals from the selection process.
    frozen_annihilation: list[int], optional
        A list of orbital indices for which annihilation operators are frozen (i.e., not allowed to be unoccupied in the selected determinants).
        This is used to enforce certain symmetries or to exclude certain orbitals from the selection process.
    screening_criterion: str, optional, default="hbci"
        The criterion used to screen determinants during selection. Options are "hbci" and "ehbci".
    energy_correction: str, optional, default="pt2"
        The method used to compute the energy correction from the determinants that are not included in the variational space.
        Options are "pt2" and "variational".
    energy_shift: float, optional, default=None
        An energy shift applied during selection to target specific roots. If None, no shift is applied.
    pt2_regularizer: str, optional, default="none"
        The method used to regularize the PT2 energy correction.
        Options are:
            - "none": No regularization.
            - "shift": Apply a small shift to the denominators in the PT2 expression to avoid divergences:
                1 / denom -> 1 / (denom + pt2_regularizer_strength)
                Ept2 -> 0 as pt2_regularizer_strength -> inf
            - "dsrg": Use a DSRG-inspired regularization of the PT2 correction:
                1 / denom -> (1 / denom) * (1 - exp(-denom^2 * pt2_regularizer_strength))
                Ept2 -> 0 as pt2_regularizer_strength -> 0, Ept2 -> unregularized PT2 as pt2_regularizer_strength -> inf
    pt2_regularizer_strength: float, optional, default=0.0
        The strength of the PT2 regularization.
        Note that the interpretation of this parameter depends on the choice of pt2_regularizer (see above).
    """

    maxcycle: int = 15
    e_tol: float = 1e-8
    var_threshold: float = 5e-4
    pt2_threshold: float = 1e-8
    selection_algorithm: str = "hbci"
    guess_occ_window: int = 2
    guess_vir_window: int = 2
    num_threads: int = 4
    ci_algorithm: str = "sparse"
    num_batches_per_thread: int = 4
    do_spin_penalty: bool = True
    guess_dets: list[Determinant] = field(default_factory=list)
    pinned_guess_dets: list[Determinant] = field(default_factory=list)
    frozen_creation: list[int] = field(default_factory=list)
    frozen_annihilation: list[int] = field(default_factory=list)
    screening_criterion: str = "hbci"
    energy_correction: str = "pt2"
    energy_shift: float = None
    pt2_regularizer: str = "none"
    pt2_regularizer_strength: float = 0.0

    def __post_init__(self):
        if self.ci_algorithm.lower() not in ["exact", "sparse"]:
            raise ValueError("ci_algorithm must be 'exact' or 'sparse'")
        if self.selection_algorithm.lower() not in ["hbci", "hbci_ref"]:
            raise ValueError("selection_algorithm must be 'hbci' or 'hbci_ref'")
        if self.screening_criterion.lower() not in ["hbci", "ehbci"]:
            raise ValueError("screening_criterion must be 'hbci' or 'ehbci'")
        if self.energy_correction.lower() not in ["variational", "pt2"]:
            raise ValueError("energy_correction must be 'variational' or 'pt2'")
        if self.pt2_regularizer.lower() not in ["none", "shift", "dsrg"]:
            raise ValueError("pt2_regularizer must be 'none', 'shift', or 'dsrg'")
