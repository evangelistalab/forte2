from dataclasses import dataclass, field
import numpy as np

from forte2.base_classes import Method
from forte2.base_classes.params import X2CParams


@dataclass
class SpinorUpcaster(Method):
    """
    A converter class to convert a spatial-orbital-based method to a spinor-based method by converting the MO coefficients to spinor basis and updating the system object.

    Parameters
    ----------
    x2c_override : X2CParams | None, optional
        The X2C Hamiltonian to use, as an :class:`X2CParams` instance. If provided,
        :attr:`System.x2c` is replaced by this value and the X2C helper is rebuilt.
        If None, the System option is retained.
    apply_random_phase : bool, optional, default=False
        Whether to apply a random phase to the MO coefficients after conversion. This can be useful for testing the robustness of downstream methods to the choice of MO phases.
    rng : np.random.Generator or int, optional, default=np.random.default_rng()
        The random number generator to use for generating the random phase. Can be an instance of `np.random.Generator` or an integer seed.
    """

    x2c_override: X2CParams | None = None
    apply_random_phase: bool = False
    rng: np.random.Generator | int = field(default_factory=np.random.default_rng)

    def __post_init__(self):
        if self.x2c_override is not None and not isinstance(
            self.x2c_override, X2CParams
        ):
            raise ValueError(
                f"x2c_override must be an X2CParams instance or None, but got {type(self.x2c_override)}."
            )

        if self.apply_random_phase:
            if not isinstance(self.rng, np.random.Generator | int):
                raise ValueError(
                    f"rng must be an instance of np.random.Generator or an integer seed, but got {type(self.rng)}."
                )
        if isinstance(self.rng, int):
            self.rng = np.random.default_rng(self.rng)

        self.requires = {"system", "mos"}
        self.provides = {"system", "mos"}

    def __call__(self, parent_method):
        self._register_parent_method(parent_method)
        if "mo_space" in self.parent_method.provides:
            self.provides.add("mo_space")
        self.two_component = True
        return self

    def run(self):
        if not self.parent_method.executed:
            self.parent_method.run()

        self.system = self.parent_method.system
        self.mos = self.parent_method.mos.copy()

        if "mo_space" in self.parent_method.provides:
            self.mo_space = self.parent_method.mo_space

        if not self.system.two_component:
            self.mos = self.mos.to_spinorbital_basis()
            if "mo_space" in self.parent_method.provides:
                self.mo_space = self.mo_space.to_spinorbital_basis()
            self.system.two_component = True
        if self.apply_random_phase:
            random_phase = np.diag(
                np.exp(1j * self.rng.uniform(-np.pi, np.pi, size=self.mos.nmo))
            )
            self.mos.C[0] = self.mos.C[0] @ random_phase
        if self.x2c_override is not None:
            self.system.x2c = self.x2c_override
            self.system._init_x2c()

        self.executed = True
        return self
