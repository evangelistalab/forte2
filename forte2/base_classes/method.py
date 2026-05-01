from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Method(ABC):
    # set of attributes that is required from the parent method
    requires: set[str] = field(default_factory=set, init=False)
    # set of flags that this method requires the parent method to have set
    requires_flags: dict[str, bool] = field(default_factory=dict, init=False)
    # set of attributes that this method provides to downstream methods
    provides: set[str] = field(default_factory=set, init=False)
    two_component: bool | None = field(default=None, init=False)
    executed: bool = field(default=False, init=False)

    @abstractmethod
    def __call__(self, upstream): ...

    @abstractmethod
    def run(self): ...

    def _register_parent_method(self, parent_method):
        if not isinstance(parent_method, Method):
            raise ValueError(
                f"Parent method must be an instance of Method, but got {type(parent_method)}."
            )
        for req in self.requires:
            if req not in parent_method.provides:
                raise RuntimeError(
                    f"Parent method {parent_method.__class__.__name__} does not provide required data '{req}' for {self.__class__.__name__}."
                )
            
        for flag, value in self.requires_flags.items():
            if not hasattr(parent_method, flag):
                raise RuntimeError(
                    f"Parent method {parent_method.__class__.__name__} does not have required flag '{flag}' for {self.__class__.__name__}."
                )
            if getattr(parent_method, flag) != value:
                raise RuntimeError(
                    f"Parent method {parent_method.__class__.__name__} has flag '{flag}'={getattr(parent_method, flag)}, but {self.__class__.__name__} requires it to be {value}."
                )
        self.parent_method = parent_method

        if parent_method.two_component is None:
            raise RuntimeError(
                f"Parent method {parent_method.__class__.__name__} must have two_component set to True or False, but got None."
            )
        self.two_component = parent_method.two_component
