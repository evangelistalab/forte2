from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Method(ABC):
    # set of attributes that is required from the parent method
    # these should not be class attributes and are instantiated at runtime
    # these are therefore only checked against "provides"
    requires: set[str] = field(default_factory=set, init=False)
    # attributes this method requires the parent method to have, keyed by name.
    # A value of None checks existence only, 
    # any other value additionally checks for equality.
    requires_attrs: dict[str, object | None] = field(
        default_factory=dict, init=False
    )
    # set of attributes that this method provides to downstream methods
    provides: set[str] = field(default_factory=set, init=False)
    # Flags that all methods need to have
    two_component: bool | None = field(default=None, init=False)
    # Whether run() has been called and returned successfully
    executed: bool = field(default=False, init=False)
    # Whether this method has been bound to its upstream
    called: bool = field(default=False, init=False)

    @abstractmethod
    def __call__(self, upstream): ...

    @abstractmethod
    def run(self): ...

    def _register_parent_method(self, parent_method):
        """
        These checks help perform pre-run sanity checks so that incompatible methods
        raise errors at init time, instead at run time.
        """
        if not isinstance(parent_method, Method):
            raise ValueError(
                f"Parent method must be an instance of Method, but got {type(parent_method)}."
            )

        if not parent_method.called:
            raise RuntimeError(
                f"Parent method {parent_method.__class__.__name__} has not been bound "
                f"to a System or an upstream method, so {self.__class__.__name__} "
                "cannot be attached to it."
            )

        for req in self.requires:
            if req not in parent_method.provides:
                raise RuntimeError(
                    f"Parent method {parent_method.__class__.__name__} does not provide required data '{req}' for {self.__class__.__name__}."
                )

        for iattr, vattr in self.requires_attrs.items():
            if not hasattr(parent_method, iattr):
                raise RuntimeError(
                    f"Parent method {parent_method.__class__.__name__} does not have required attr '{iattr}' for {self.__class__.__name__}."
                )
            if vattr is not None and getattr(parent_method, iattr) != vattr:
                raise RuntimeError(
                    f"Parent method {parent_method.__class__.__name__} has attr '{iattr}'={getattr(parent_method, iattr)}, but {self.__class__.__name__} requires it to be {vattr}."
                )

        if parent_method.two_component is None:
            raise RuntimeError(
                f"Parent method {parent_method.__class__.__name__} must have two_component set to True or False, but got None."
            )
        self.parent_method = parent_method
        self.two_component = parent_method.two_component
        self.called = True
