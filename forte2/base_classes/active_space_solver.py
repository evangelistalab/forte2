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

    def __post_init__(self):
        self.sa_info = StateAverageInfo(
            states=self.states,
            nroots=self.nroots,
            weights=self.weights,
        )
        self.ncis = self.sa_info.ncis
        self.weights = self.sa_info.weights
        self.weights_flat = self.sa_info.weights_flat

    def _startup(self):
        if not self.parent_method.executed:
            self.parent_method.run()

        SystemMixin.copy_from_upstream(self, self.parent_method)
        MOsMixin.copy_from_upstream(self, self.parent_method)
        self._parse_mo_space()

    def _parse_mo_space(self):
        # If either mo_space is provided or speficied via the kwargs,
        # then use it, otherwise copy from the parent method.
        provided_via_kw = any(
            [
                self.frozen_core_orbitals is not None,
                self.core_orbitals is not None,
                self.active_orbitals is not None,
                self.frozen_virtual_orbitals is not None,
            ]
        )

        if provided_via_kw and self.mo_space is not None:
            raise ValueError("Both mo_space and *_orbitals parameters are provided. ")

        if provided_via_kw:
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

        if isinstance(self.parent_method, MOSpaceMixin):
            if self.mo_space is not None:
                logger.log_warning(
                    "Using the provided mo_space instead of the one from the parent method."
                )
            else:
                MOSpaceMixin.copy_from_upstream(self, self.parent_method)
        else:
            assert self.mo_space is not None, (
                "If the parent method does not have MOSpaceMixin, "
                "then mo_space must be provided."
            )
