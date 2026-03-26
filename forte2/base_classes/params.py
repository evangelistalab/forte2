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
    """
    Parameters for the Selected CI solver.

    Parameters
    ----------
    maxcycle: int, optional, default=10
        The maximum number of selection cycles.
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
        If not provided, the guess determinants will be generated based on the guess_occ_window and guess_vir_window parameters.
    frozen_creation: list[int], optional
        A list of orbital indices for which creation operators are frozen (i.e., not allowed to be occupied in the selected determinants).
        This is used to enforce certain symmetries or to exclude certain orbitals from the selection process.
    screening_criterion: str, optional, default="hbci"
        The criterion used to screen determinants during selection. Options are "hbci" and "ehbci".
    energy_correction: str, optional, default="pt2"
        The method used to compute the energy correction from the determinants that are not included in the variational space.
        Options are "pt2" and "none".
    energy_shift: float, optional, default=None
        An energy shift applied during selection to target specific roots. If None, no shift is applied.
    pt2_renormalizer: str, optional, default="none"
        The method used to renormalize the PT2 energy correction.
        Options are:
            - "none": No renormalization.
            - "shift": Apply a small shift to the denominators in the PT2 expression to avoid divergences:
                1 / denom -> 1 / (denom + pt2_renormalizer_strength)
                Ept2 -> 0 as pt2_renormalizer_strength -> inf
            - "dsrg": Use a DSRG-inspired renormalization of the PT2 correction:
                1 / denom -> (1 / denom) * (1 - exp(-denom^2 * pt2_renormalizer_strength))
                Ept2 -> 0 as pt2_renormalizer_strength -> 0, Ept2 -> unrenormalized PT2 as pt2_renormalizer_strength -> inf
    pt2_renormalizer_strength: float, optional, default=0.0
        The strength of the PT2 renormalization.
        Note that the interpretation of this parameter depends on the choice of pt2_renormalizer (see above).
    """

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
    pt2_renormalizer: str = "none"
    pt2_renormalizer_strength: float = 0.0

    def __post_init__(self):
        if self.ci_algorithm.lower() not in ["exact", "sparse"]:
            raise ValueError("ci_algorithm must be 'exact' or 'sparse'")
        if self.selection_algorithm.lower() not in ["hbci", "hbci_ref"]:
            raise ValueError("selection_algorithm must be 'hbci' or 'hbci_ref'")
        if self.screening_criterion.lower() not in ["hbci", "ehbci"]:
            raise ValueError("screening_criterion must be 'hbci' or 'ehbci'")
        if self.energy_correction.lower() not in ["variational", "pt2"]:
            raise ValueError("energy_correction must be 'variational' or 'pt2'")
        if self.pt2_renormalizer.lower() not in ["none", "shift", "dsrg"]:
            raise ValueError("pt2_renormalizer must be 'none', 'shift', or 'dsrg'")
