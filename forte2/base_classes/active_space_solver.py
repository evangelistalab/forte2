from dataclasses import dataclass

from .method import Method
from forte2.state import StateAverageInfo, State, MOSpace
from forte2.helpers import logger


@dataclass
class ActiveSpaceSolver(Method):
    states: State | list[State]
    nroots: int | list[int] = 1
    weights: list[float] | list[list[float]] = None
    mo_space: MOSpace = None
    frozen_core_orbitals: list[int] = None
    core_orbitals: list[int] = None
    active_orbitals: list[int] | list[list[int]] = None
    frozen_virtual_orbitals: list[int] = None
    die_if_not_converged: bool = False

    def __post_init__(self):
        self.dtype = float
        self.sa_info = StateAverageInfo(
            states=self.states,
            nroots=self.nroots,
            weights=self.weights,
        )
        self.ncis = self.sa_info.ncis
        self.weights = self.sa_info.weights
        self.weights_flat = self.sa_info.weights_flat
        self.requires = {"system", "mo_coeff"}
        self.requires_flags["two_component"] = False
        self.provides = {"system", "mo_coeff", "mo_space"}

    def _startup(self, two_component=False):
        if not self.parent_method.executed:
            self.parent_method.run()

        self.system = self.parent_method.system
        # UHF will only provide alpha MOs, others are unchanged by the only_alpha kwarg
        self.mo_coeff = self.parent_method.mo_coeff.copy()
        self._make_mo_space()

    def _make_mo_space(self):
        two_component = self.two_component
        # Ways of providing the MO space:
        # 1. Via the parent method (if it has mo_space).
        # 2. Via the mo_space parameter.
        # 3. Via the *_orbitals parameters (core_orbitals, active_orbitals, frozen_core_orbitals, frozen_virtual_orbitals).
        # If any one of 2-3 is provided, then 1 is ignored.
        # If more than one of 2-3 is provided, then an error is raised.
        provided_via_parent = "mo_space" in self.parent_method.provides
        provided_via_mo_space = self.mo_space is not None
        provided_via_orbitals = any(
            [
                self.frozen_core_orbitals is not None,
                self.core_orbitals is not None,
                self.active_orbitals is not None,
                self.frozen_virtual_orbitals is not None,
            ]
        )

        provided_via_args = provided_via_mo_space + provided_via_orbitals

        if (not provided_via_parent) and (provided_via_args == 0):
            raise ValueError(
                "Parent_method cannot provide MOSpace. "
                "Either mo_space, *_orbitals, or nel_active and norb_active must be provided."
            )

        if provided_via_args > 1:
            raise ValueError(
                "Only one of mo_space, *_orbitals, or nel_active and norb_active can be provided."
            )

        # override parent_method if any arguments are provided
        if provided_via_args == 1:
            if provided_via_mo_space:
                # mo_space is provided directly
                logger.log_info1("ActiveSpaceSolver: Using provided mo_space.")
                return

            if provided_via_orbitals:
                # construct mo_space from *_orbitals arguments
                nmo = self.system.nmo * 2 if two_component else self.system.nmo
                self.mo_space = MOSpace(
                    nmo=nmo,
                    active_orbitals=(
                        self.active_orbitals if self.active_orbitals is not None else []
                    ),
                    core_orbitals=(
                        self.core_orbitals if self.core_orbitals is not None else []
                    ),
                    frozen_core_orbitals=(
                        self.frozen_core_orbitals
                        if self.frozen_core_orbitals is not None
                        else []
                    ),
                    frozen_virtual_orbitals=(
                        self.frozen_virtual_orbitals
                        if self.frozen_virtual_orbitals is not None
                        else []
                    ),
                )
                logger.log_info1(
                    "ActiveSpaceSolver: mo_space constructed from provided orbital lists."
                )
                return
        elif provided_via_parent:
            self.mo_space = self.parent_method.mo_space
            logger.log_info1("ActiveSpaceSolver: mo_space copied from parent method.")
            return


@dataclass
class RelActiveSpaceSolver(ActiveSpaceSolver):
    nel: int = None
    states: State | list[State] = None

    def __post_init__(self):
        if self.nel is None and self.states is None:
            raise ValueError("Either nel or states must be provided.")
        if self.nel is not None and self.states is not None:
            raise ValueError("Only one of nel or states can be provided.")
        if self.nel is not None:
            mult = 1 if self.nel % 2 == 0 else 2
            ms = 0.0 if mult == 1 else 0.5
            self.states = State(nel=self.nel, multiplicity=mult, ms=ms)
        super().__post_init__()
        self.requires_flags["two_component"] = True
        self.dtype = complex
