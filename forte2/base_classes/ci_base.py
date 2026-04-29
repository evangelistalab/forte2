from abc import abstractmethod
from dataclasses import dataclass, field

from .active_space_solver import ActiveSpaceSolver, RelActiveSpaceSolver


@dataclass
class CIBase(ActiveSpaceSolver):
    ### Non-init attributes
    first_run: bool = field(default=True, init=False)
    executed: bool = field(default=False, init=False)

    def __call__(self, parent_method):
        self.parent_method = parent_method
        return self

    def _collect_child_kwargs(self, target_cls):
        """Collect keyword arguments for child solvers."""
        # Defer import to avoid polluting top-level namespace
        from dataclasses import fields as _dc_fields

        # Take all init fields of the target dataclass and copy values from `self` if present
        names = {f.name for f in _dc_fields(target_cls) if f.init}
        return {n: getattr(self, n) for n in names if hasattr(self, n)}

    def _startup(self):
        super()._startup()

    def _get_state_root(self, absolute_root) -> tuple[int, int]:
        if absolute_root < 0 or absolute_root >= self.sa_info.nroots_sum:
            raise ValueError(
                f"absolute_root must be between 0 and {self.sa_info.nroots_sum - 1}, but got {absolute_root}."
            )
        return self.sa_info.absolute_root_map[absolute_root]

    def _validate_rdm_inputs(self, left_root, right_root):
        left_state, left_root_in_state = self._get_state_root(left_root)
        if right_root is not None:
            right_state, right_root_in_state = self._get_state_root(right_root)
        else:
            right_state = left_state
            right_root_in_state = left_root_in_state

        if left_state != right_state:
            if (
                self.sa_info.states[left_state].na != self.sa_info.states[right_state].na
                or self.sa_info.states[left_state].nb
                != self.sa_info.states[right_state].nb
            ):
                raise ValueError(
                    "Cross-state RDMs are only supported for states with the same number of alpha and beta electrons."
                )

        return left_state, right_state, left_root_in_state, right_root_in_state


    @abstractmethod
    def run(self): ...

    @abstractmethod
    def reset_eigensolver(self): ...

    @abstractmethod
    def get_convergence_status(self): ...


@dataclass
class RelCIBase(RelActiveSpaceSolver):
    ### Non-init attributes
    first_run: bool = field(default=True, init=False)
    executed: bool = field(default=False, init=False)

    __call__ = CIBase.__call__
    _collect_child_kwargs = CIBase._collect_child_kwargs

    def _startup(self):
        super()._startup(two_component=True)

    @abstractmethod
    def run(self, use_asym_ints=False): ...

    @abstractmethod
    def reset_eigensolver(self): ...

    @abstractmethod
    def get_convergence_status(self): ...