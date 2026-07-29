from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Method(ABC):
    # set of attributes that is required from the parent method
    # these should not be class attributes and are instantiated at runtime
    # these are therefore only checked against "provides"
    requires: set[str] = field(default_factory=set, init=False)
    # set of attributes that this method requires the parent method to have
    # they can either be a str (in which case only existence is checked)
    # or a tuple of (attr_name, attr_value), in which both existence and value are checked
    requires_attrs: set[str | tuple] = field(default_factory=set, init=False)
    # set of attributes that this method provides to downstream methods
    provides: set[str] = field(default_factory=set, init=False)
    # Flags that all methods need to have
    two_component: bool | None = field(default=None, init=False)
    executed: bool = field(default=False, init=False)

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
        for req in self.requires:
            if req not in parent_method.provides:
                raise RuntimeError(
                    f"Parent method {parent_method.__class__.__name__} does not provide required data '{req}' for {self.__class__.__name__}."
                )

        for attr in self.requires_attrs:
            if isinstance(attr, str):
                iattr, vattr = attr, None
            elif isinstance(attr, tuple):
                iattr, vattr = attr
            else:
                raise RuntimeError(
                    f"Got unexpected requires_attrs entry {attr} of {type(attr)}, needs to be either str or tuple!"
                )

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
