from dataclasses import dataclass, field, fields
from abc import ABC
from typing import Literal, get_args, get_type_hints, get_origin

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

    def __post_init__(self):
        self._validate_literals()

    def _validate_literals(self):
        """Check every Literal-annotated field against the values its annotation allows."""
        hints = get_type_hints(type(self))
        for f in fields(type(self)):
            t = hints[f.name]
            if get_origin(t) is Literal:
                allowed_args = get_args(t)
                value = getattr(self, f.name)
                if value not in allowed_args:
                    raise ValueError(
                        f"{type(self).__name__}.{f.name} must be one of {allowed_args}, "
                        f"but got {value!r}."
                    )

    @classmethod
    def is_valid_input(cls, *args, **kwargs):
        try:
            cls(*args, **kwargs)
        except Exception:
            return False
        return True


@dataclass
class X2CParams(ParamsBase):
    """
    Parameters for the exact two-component (X2C) relativistic Hamiltonian.

    Parameters
    ----------
    x2c_type : str | None, optional, default=None
        The spin structure of the X2C transformation. Options are:
            - None
            - "sf": Spin-free (scalar) X2C.
            - "so": Spin-orbit (two-component) X2C.
    x2c_model : str, optional, default="1e"
        The decoupling model used to build the X2C Hamiltonian. Options are:
            - None
            - "1e": One-electron X2C (bare-nucleus decoupling).
            - "sap": Superposition of atomic potentials X2C.
    snso_type : str | None, optional, default=None
        The screened-nuclear-spin-orbit (SNSO) scaling applied to the spin-orbit
        coupling. Only valid when ``x2c_type == "so"`` and ``x2c_model == "1e"``.
        Options are:
            - None
            - "boettger": Boettger scaling.
            - "dc": Dirac-Coulomb scaling.
            - "dcb": Dirac-Coulomb-Breit scaling.
            - "row-dependent": Row-dependent scaling.
    """

    x2c_type: Literal[None, "sf", "so"] = None
    x2c_model: Literal[None, "1e", "sap"] = "1e"
    snso_type: Literal[None, "boettger", "dc", "dcb", "row-dependent"] = None

    def __post_init__(self):
        super().__post_init__()

        if self.x2c_type == None and (self.x2c_model or self.snso_type):
            raise ValueError("x2c_model and snso_type must be None if x2c_type is None")

        if self.x2c_type is not None and self.x2c_model == None:
            raise ValueError("x2c_model must be set if x2c_type isn't None")

        # SNSO scaling only makes sense for so, 1e.
        if self.snso_type is not None and not (
            self.x2c_type == "so" and self.x2c_model == "1e"
        ):
            raise ValueError(
                "snso_type is only valid when x2c_type == 'so' and x2c_model == '1e', "
                f"but got x2c_type={self.x2c_type!r}, x2c_model={self.x2c_model!r}."
            )


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
    e_tol : float, optional, default=1e-12
        The energy convergence threshold for the solver.
    r_tol : float, optional, default=1e-6
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
        super().__post_init__()
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

    ci_algorithm: Literal[
        "hz", "harrison-zarrabian", "kh", "knowles-handy", "exact", "sparse"
    ] = "hz"
    ci_builder_memory: int = 1024
    energy_shift: float = None


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
    ci_algorithm: str, optional, default="iterative"
        The algorithm used for the CI diagonalization. Options are "exact" and "iterative".
        "iterative" runs a Davidson-Liu solve whose sigma build is the C++
        `SelectedCIHelper`/`RelSelectedCIHelper`; "exact" builds and diagonalizes the
        dense Hamiltonian via `SlaterRules`.
    num_batches_per_thread: int, optional, default=4
        The number of batches of determinants to process per thread during selection and diagonalization.
        The number of threads is determined automatically from the environment (affinity mask,
        `OMP_NUM_THREADS`, `OMP_THREAD_LIMIT`, `SLURM_CPUS_PER_TASK`); set `FORTE_NUM_THREADS_OVERRIDE`
        to override it.
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
    pt2_algorithm: str, optional, default="deterministic"
        How the final perturbative correction is computed. Options are:
            - "deterministic": sum every connection above `pt2_threshold` exactly. The cost and the
              memory both grow as `pt2_threshold` is lowered.
            - "semistochastic": write the correction as a telescoping sum of three terms, of which
              only the first is exact. This reaches a `pt2_threshold` far below what is affordable
              deterministically, at the cost of a statistical error that is reported alongside the
              energy.
        The semistochastic algorithm replaces the final selection cycle, so it neither grows the
        variational space nor reports a `var'` energy.
    pt2_threshold_determ: float, optional, default=2e-6
        The threshold reached by the deterministic term of the semistochastic correction.
        Must be at least `pt2_threshold_pseudostoch`.
    pt2_threshold_pseudostoch: float, optional, default=1e-7
        The threshold reached by the pseudo-stochastic term of the semistochastic correction.
        Must lie between `pt2_threshold` and `pt2_threshold_determ`.
    pt2_num_batches_per_thread: int, optional, default=20
        The number of batches per thread for the deterministic and pseudo-stochastic terms. This
        bounds the memory held at any one time and does not affect the deterministic term's value.
        The pseudo-stochastic term extrapolates from a subset of these batches, so a larger number
        also makes its stopping rule finer grained.
    pt2_min_batches_pseudostoch: int, optional, default=8
        The number of batches the pseudo-stochastic term evaluates before its stopping rule may
        fire. The spread over a handful of batches is itself too uncertain to stop on.
    pt2_num_batches_stoch: int, optional, default=1
        The number of batches the stochastic term splits its connected space into. The default of 1
        evaluates each sample exactly; raise it to bound the memory a single sample needs.
    pt2_batches_per_sample: int, optional, default=1
        The number of batches evaluated per sample, scaled up by the fraction of batches sampled.
        This trades the memory and time of a sample against its variance. It must not exceed
        `pt2_num_batches_stoch`, and setting the two equal makes each sample exact.
    pt2_num_samples: int, optional, default=10
        The number of stochastic samples, or 0 to skip the stochastic term. At least two are needed
        to estimate an error.
    pt2_sample_size: int, optional, default=200
        The number of variational determinants drawn with replacement per sample.
    pt2_target_error: float, optional, default=1e-5
        The error the semistochastic correction aims for. The pseudo-stochastic term stops once its
        own error falls well inside this, leaving the rest of the budget to the stochastic term.
    pt2_seed: int, optional, default=0
        The seed for the sampling. Runs are reproducible for a fixed seed, thread count and
        variational space; vary it to obtain independent estimates.
    """

    maxcycle: int = 15
    e_tol: float = 1e-8
    var_threshold: float = 5e-4
    pt2_threshold: float = 1e-8
    selection_algorithm: Literal["hbci", "hbci_ref"] = "hbci"
    guess_occ_window: int = 2
    guess_vir_window: int = 2
    ci_algorithm: Literal["iterative", "exact"] = "iterative"
    num_batches_per_thread: int = 4
    do_spin_penalty: bool = True
    guess_dets: list[Determinant] = field(default_factory=list)
    pinned_guess_dets: list[Determinant] = field(default_factory=list)
    frozen_creation: list[int] = field(default_factory=list)
    frozen_annihilation: list[int] = field(default_factory=list)
    screening_criterion: Literal["hbci", "ehbci"] = "hbci"
    energy_correction: Literal["variational", "pt2"] = "pt2"
    energy_shift: float = None
    pt2_regularizer: Literal["none", "shift", "dsrg"] = "none"
    pt2_regularizer_strength: float = 0.0
    pt2_algorithm: Literal["deterministic", "semistochastic"] = "deterministic"
    pt2_threshold_determ: float = 2e-6
    pt2_threshold_pseudostoch: float = 1e-7
    pt2_num_batches_per_thread: int = 20
    pt2_min_batches_pseudostoch: int = 8
    pt2_num_batches_stoch: int = 1
    pt2_batches_per_sample: int = 1
    pt2_num_samples: int = 10
    pt2_sample_size: int = 200
    pt2_target_error: float = 1e-5
    pt2_seed: int = 0

    def __post_init__(self):
        super().__post_init__()
        if self.var_threshold < 0:
            raise RuntimeError(
                f"var_threshold cannot be negative, got {self.var_threshold}"
            )
        if self.pt2_threshold < 0:
            raise RuntimeError(
                f"var_threshold cannot be negative, got {self.pt2_threshold}"
            )
        if self.pt2_algorithm != "semistochastic":
            return
        if not (
            self.pt2_threshold
            <= self.pt2_threshold_pseudostoch
            <= self.pt2_threshold_determ
        ):
            raise ValueError(
                "The semistochastic PT2 algorithm requires "
                "pt2_threshold <= pt2_threshold_pseudostoch <= pt2_threshold_determ, got "
                f"{self.pt2_threshold}, {self.pt2_threshold_pseudostoch}, {self.pt2_threshold_determ}. "
                "Each term of the correction covers the gap between two thresholds, and an "
                "inverted pair silently contributes with the wrong sign."
            )
        if self.pt2_num_samples == 1:
            raise ValueError(
                "pt2_num_samples must be 0 to skip the stochastic term, or at least 2 to "
                "estimate an error from it."
            )
        if self.pt2_num_samples > 0 and self.pt2_sample_size < 2:
            raise ValueError("pt2_sample_size must be at least 2.")
        if self.pt2_min_batches_pseudostoch < 2:
            raise ValueError(
                "pt2_min_batches_pseudostoch must be at least 2: the pseudo-stochastic error is the "
                "spread over the batches evaluated so far, and a single batch has none to "
                "measure, so stopping on it would report a one-batch extrapolation as exact."
            )
        if not (1 <= self.pt2_batches_per_sample <= self.pt2_num_batches_stoch):
            raise ValueError(
                "pt2_batches_per_sample must lie between 1 and pt2_num_batches_stoch, got "
                f"{self.pt2_batches_per_sample} and {self.pt2_num_batches_stoch}."
            )
        if self.pt2_num_samples > 0 and self.energy_correction == "variational":
            raise ValueError(
                "The variational energy correction cannot be combined with the stochastic term: "
                "it is not linear in the square of the coupling, which is what the sampling "
                "estimates without bias. Set pt2_num_samples=0 or use energy_correction='pt2'."
            )
