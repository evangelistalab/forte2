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