from dataclasses import dataclass, field, fields
from abc import ABC
from typing import Literal, get_args, get_type_hints, get_origin

import numpy as np

from forte2.data.atom_data import LIGHT_SPEED
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
    ``X2CParams()`` (all defaults) means no X2C transformation.

    Parameters
    ----------
    x2c_type : str | None, optional, default=None
        The spin structure of the X2C transformation. Options are:
            - None
            - "sf": Spin-free (scalar) X2C.
            - "so": Spin-orbit (two-component) X2C.
    x2c_model : str | None, optional, default=None
        The decoupling model used to build the X2C Hamiltonian; "1e" if
        ``x2c_type`` is set and this is left as None. Options are:
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
    skip_picture_change : bool, optional, default=False
        If True, evaluate one-electron property operators without the picture change
        correction. Individual property calls can override this.
    light_speed : float, optional, default=LIGHT_SPEED
        The speed of light in atomic units used by the X2C transformation and its
        gradient. Scalar-relativistic and spin-orbit effects scale as
        ``1 / light_speed**2``, so changing it tunes their strength; the
        nonrelativistic limit is ``light_speed -> inf``. The SNSO factors are not
        rescaled with it.
    """

    x2c_type: Literal[None, "sf", "so"] = None
    x2c_model: Literal[None, "1e", "sap"] = None
    snso_type: Literal[None, "boettger", "dc", "dcb", "row-dependent"] = None
    skip_picture_change: bool = False
    light_speed: float = LIGHT_SPEED

    def __post_init__(self):
        super().__post_init__()

        if self.x2c_type is None and (self.x2c_model or self.snso_type):
            raise ValueError("x2c_model and snso_type must be None if x2c_type is None")

        if self.x2c_type is not None and self.x2c_model is None:
            self.x2c_model = "1e"

        if not (np.isfinite(self.light_speed) and self.light_speed > 0):
            raise ValueError(
                f"light_speed must be positive and finite, but got {self.light_speed}."
            )
        if self.x2c_type is None and self.light_speed != LIGHT_SPEED:
            raise ValueError("light_speed only affects X2C; set x2c_type as well.")

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
class DMRGParams(ParamsBase):
    """
    Parameters for the DMRG solver.

    The per-sweep schedules ``bond_dims``, ``noises``, and ``davidson_tols`` are padded to
    ``n_sweeps`` by repeating their last entry.

    Parameters
    ----------
    bond_dims : list[int], optional, default=[250] * 4 + [500] * 4
        The MPS bond dimension of each sweep.
    noises : list[float], optional, default=[1e-4] * 4 + [1e-5] * 4 + [0.0]
        The noise of each sweep.
    noise_type : str, optional, default="ReducedPerturbativeCollected"
        The block2 noise method:

        - "ReducedPerturbativeCollected", "ReducedPerturbative", "Perturbative": noise
          from the Hamiltonian acting on the wavefunction. It vanishes at an exact
          eigenstate, so in an active space small enough for DMRG to solve exactly, a
          solve can stay on an excited state.
        - "Wavefunction": random noise on the wavefunction, within its current blocks.
        - "DensityMatrix": random noise on the reduced density matrix, so no state of a
          bond is dropped. Use it for active spaces that DMRG solves exactly.
    davidson_tols : list[float], optional, default=[1e-10] * 8
        The Davidson convergence threshold of each sweep, on the squared residual norm.
    n_sweeps : int, optional, default=20
        The maximum number of sweeps.
    twosite_to_onesite : int, optional, default=None
        The number of two-site sweeps before the remaining sweeps switch to the one-site
        algorithm. The two-site sweeps don't stop early. If None, every sweep is a
        two-site sweep.
    tol : float, optional, default=1e-8
        The energy convergence threshold that ends the sweeps early.
    orbital_ordering : str, optional, default="fiedler"
        The order of the active orbitals along the MPS chain. "fiedler" orders them by the
        Fiedler vector of the exchange integrals, which places strongly coupled orbitals
        close together. "gaopt" refines the Fiedler order with a genetic algorithm, which
        takes longer. "original" keeps the order of the active space.
    random_seed : int, optional, default=1234
        The seed of block2's random number generator, set before each initial MPS is
        built. It must be positive.
    stack_memory : int, optional, default=1024
        The working memory (in MB) that block2 reserves for each solver.
    scratch : str, optional, default=None
        The directory in which each solver creates its scratch directory. If None, the
        system temporary directory is used.
    iprint : int, optional, default=0
        The block2 verbosity level.
    """

    bond_dims: list[int] = field(default_factory=lambda: [250] * 4 + [500] * 4)
    noises: list[float] = field(default_factory=lambda: [1e-4] * 4 + [1e-5] * 4 + [0.0])
    noise_type: Literal[
        "ReducedPerturbativeCollected",
        "ReducedPerturbative",
        "Perturbative",
        "Wavefunction",
        "DensityMatrix",
    ] = "ReducedPerturbativeCollected"
    davidson_tols: list[float] = field(default_factory=lambda: [1e-10] * 8)
    n_sweeps: int = 20
    twosite_to_onesite: int = None
    tol: float = 1e-8
    orbital_ordering: Literal["fiedler", "gaopt", "original"] = "fiedler"
    random_seed: int = 1234
    stack_memory: int = 1024
    scratch: str = None
    iprint: int = 0

    def __post_init__(self):
        super().__post_init__()
        if not self.bond_dims or not all(
            isinstance(m, int) and m > 0 for m in self.bond_dims
        ):
            raise ValueError(
                f"bond_dims must be a non-empty list of positive integers, got {self.bond_dims}."
            )
        if not self.noises or not all(n >= 0 for n in self.noises):
            raise ValueError(
                f"noises must be a non-empty list of non-negative numbers, got {self.noises}."
            )
        if not self.davidson_tols or not all(t > 0 for t in self.davidson_tols):
            raise ValueError(
                f"davidson_tols must be a non-empty list of positive numbers, got {self.davidson_tols}."
            )
        if self.n_sweeps < 1:
            raise ValueError(f"n_sweeps must be >= 1, got {self.n_sweeps}.")
        if self.twosite_to_onesite is not None and not (
            0 < self.twosite_to_onesite < self.n_sweeps
        ):
            raise ValueError(
                f"twosite_to_onesite must be between 1 and n_sweeps - 1, got "
                f"{self.twosite_to_onesite}."
            )
        if self.tol <= 0:
            raise ValueError(f"tol must be > 0, got {self.tol}.")
        # block2 treats a seed of 0 as a request for a time-based seed
        if self.random_seed < 1:
            raise ValueError(f"random_seed must be >= 1, got {self.random_seed}.")
        if self.stack_memory < 1:
            raise ValueError(f"stack_memory must be >= 1, got {self.stack_memory}.")
        # if n_sweeps > length of any of these, pad them with their last entries
        for name in ("bond_dims", "noises", "davidson_tols"):
            schedule = list(getattr(self, name))
            schedule += [schedule[-1]] * (self.n_sweeps - len(schedule))
            setattr(self, name, schedule)

    @classmethod
    def auto_schedule(
        cls, max_bond_dim, start_bond_dim=250, n_sweeps=None, tol=1e-5, **kwargs
    ):
        """
        Automatic schedule, matches block2's "default" scheduler.

        The automatic bond dimensions are 50, 100, 250, 500, range(1000, 10001, 1000).
        The ``start_bond_dim`` and ``max_bond_dim`` do not have to take one of the values
        above, e.g., start_bond_dim=200, max_bond_dim=900 gives [200, 250, 500, 900].
        Each bond dimension runs 8 sweeps up to 1000 and 4 after 1000,
        with noise and Davidson tolerances that decrease along the ladder, and the last
        bond dimension runs 8 sweeps.
        If more sweeps are left over, Noise-free sweeps at ``max_bond_dim``, with a Davidson
        tolerance of ``tol / 10``, are added.
        After the first 2 noise-free sweeps, the remaining sweeps switch to the one-site
        algorithm, unless ``twosite_to_onesite`` is passed.

        Parameters
        ----------
        max_bond_dim : int
            The final bond dimension.
        start_bond_dim : int, optional, default=250
            The first bond dimension.
        n_sweeps : int, optional
            The maximum number of sweeps.
            It must exceed the number of sweeps required by the schedule.
            If None, 4 noise-free sweeps are added at the end of the schedule.
        tol : float, optional, default=1e-5
            The energy convergence threshold.
        **kwargs
            Any other ``DMRGParams`` field.

        Returns
        -------
        DMRGParams
            The parameters with an automatic schedule filled in.
        """
        if not 0 < start_bond_dim <= max_bond_dim:
            raise ValueError(
                f"Need 0 < start_bond_dim <= max_bond_dim, got {start_bond_dim} and "
                f"{max_bond_dim}."
            )
        # Each rung is (bond dimension, sweeps, Davidson tolerance, noise)
        ladder = [50, 100, 250, 500] + [1000 * x for x in range(1, 11)]
        rungs = list(
            zip(
                ladder,
                [8] * 5 + [4] * 9,
                [1e-4] * 3 + [1e-5] * 2 + [5e-6] * 9,
                [1e-3] * 3 + [1e-4] * 2 + [5e-5] * 9,
            )
        )
        if start_bond_dim < ladder[0]:
            rungs.insert(0, (start_bond_dim, *rungs[0][1:]))
        elif start_bond_dim > ladder[-1]:
            while start_bond_dim > rungs[-1][0]:
                rungs.append((rungs[-1][0] + 1000, *rungs[-1][1:]))
        else:
            i = next(
                (i for i in range(1, len(ladder)) if start_bond_dim < ladder[i]), None
            )
            if i is not None:
                rungs[i - 1] = (start_bond_dim, *rungs[i - 1][1:])
        steps = []
        for m, n, dav_tol, noise in rungs:
            if m >= max_bond_dim:
                steps.append((max_bond_dim, n, dav_tol, noise))
                break
            if m >= start_bond_dim:
                steps.append((m, n, dav_tol, noise))
        # The last step always runs 8 sweeps
        steps[-1] = (steps[-1][0], 8, *steps[-1][2:])

        bond_dims, davidson_tols, noises = [], [], []
        for m, n, dav_tol, noise in steps:
            bond_dims += [m] * n
            davidson_tols += [dav_tol] * n
            noises += [noise] * n
        n_ladder = len(bond_dims)
        if n_sweeps is None:
            n_sweeps = n_ladder + 4
        elif n_sweeps <= n_ladder:
            raise ValueError(
                f"n_sweeps must exceed the {n_ladder} ladder sweeps, got {n_sweeps}."
            )
        if n_ladder + 2 < n_sweeps:
            kwargs.setdefault("twosite_to_onesite", n_ladder + 2)
        return cls(
            bond_dims=bond_dims + [max_bond_dim],
            noises=noises + [0.0],
            davidson_tols=davidson_tols + [tol / 10],
            n_sweeps=n_sweeps,
            tol=tol,
            **kwargs,
        )


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

    def __post_init__(self):
        super().__post_init__()
        if self.var_threshold < 0:
            raise RuntimeError(
                f"var_threshold cannot be negative, got {self.var_threshold}"
            )
        if self.pt2_threshold < 0:
            raise RuntimeError(
                f"pt2_threshold cannot be negative, got {self.pt2_threshold}"
            )
