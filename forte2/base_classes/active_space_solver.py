from dataclasses import dataclass
from abc import ABC

from .mixins import MOsMixin, SystemMixin, MOSpaceMixin
from forte2.state import StateAverageInfo, State, MOSpace
from forte2.helpers import logger


@dataclass
class ActiveSpaceSolver(ABC, MOsMixin, SystemMixin, MOSpaceMixin):
    states: State | list[State]
    nroots: int | list[int] = 1
    weights: list[float] | list[list[float]] = None
    mo_space: MOSpace = None
    frozen_core_orbitals: list[int] = None
    core_orbitals: list[int] = None
    active_orbitals: list[int] | list[list[int]] = None
    frozen_virtual_orbitals: list[int] = None
    final_orbital: str = "semicanonical"

    def __post_init__(self):
        self.sa_info = StateAverageInfo(
            states=self.states,
            nroots=self.nroots,
            weights=self.weights,
        )
        self.ncis = self.sa_info.ncis
        self.weights = self.sa_info.weights
        self.weights_flat = self.sa_info.weights_flat
        assert self.final_orbital in [
            "semicanonical",
            "original",
        ], "final_orbital must be either 'semicanonical' or 'original'."

    def _startup(self):
        if not self.parent_method.executed:
            self.parent_method.run()

        SystemMixin.copy_from_upstream(self, self.parent_method)
        MOsMixin.copy_from_upstream(self, self.parent_method)
        self._make_mo_space()

    def _make_mo_space(self):
        # Ways of providing the MO space:
        # 1. Via the parent method (if it has MOSpaceMixin).
        # 2. Via the mo_space parameter.
        # 3. Via the *_orbitals parameters (core_orbitals, active_orbitals, frozen_core_orbitals, frozen_virtual_orbitals).
        # If any one of 2-3 is provided, then 1 is ignored.
        # If more than one of 2-3 is provided, then an error is raised.
        provided_via_parent = isinstance(self.parent_method, MOSpaceMixin)
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
                return

            if provided_via_orbitals:
                # construct mo_space from *_orbitals arguments
                self.mo_space = MOSpace(
                    nmo=self.system.nmo,
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
                return
        elif provided_via_parent:
            MOSpaceMixin.copy_from_upstream(self, self.parent_method)
            return
