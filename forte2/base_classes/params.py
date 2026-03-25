from dataclasses import dataclass, field

from forte2 import Determinant

@dataclass
class DavidsonLiuParams:
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
class CIParams:
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
class SelectedCIParams:
    ### Selected CI parameters
    maxcycle: int = 10
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
    frozen_creation: list[int] = field(default_factory=list)
    screening_criterion: str = "hbci"
    energy_correction: str = "pt2"
    energy_shift: float = None
