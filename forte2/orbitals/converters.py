from dataclasses import dataclass, field
import numpy as np

from forte2.base_classes import Method


@dataclass
class SpatialToSpinorConverter(Method):
    """
    A converter class to convert a spatial-orbital-based method to a spinor-based method by converting the MO coefficients to spinor basis and updating the system object.

    Parameters
    ----------
    x2c_type_override : str | None, optional
        The type of X2C Hamiltonian to use. Must be either 'so' (spin-orbit) or 'sf' (spin-free).
        If provided, `System.x2c_type` will be overwritten by this value.
        If None, the X2C type will be determined by the `System` object.
    snso_type_override : str | None, optional
        The type of SNSO correction to use. Must be one of 'boettger', 'dc', 'dcb', or 'row-dependent'.
        If provided, `System.snso_type` will be overwritten by this value.
        If None, the SNSO type will be determined by the `System` object.
    apply_random_phase : bool, optional, default=False
        Whether to apply a random phase to the MO coefficients after conversion. This can be useful for testing the robustness of downstream methods to the choice of MO phases.
    rng : np.random.Generator or int, optional, default=np.random.default_rng()
        The random number generator to use for generating the random phase. Can be an instance of `np.random.Generator` or an integer seed.
    """

    x2c_type_override: str | None = None
    snso_type_override: str | None = None
    apply_random_phase: bool = False
    rng: np.random.Generator | int = field(default_factory=np.random.default_rng)

    def __post_init__(self):
        if not isinstance(self.x2c_type_override, (str, type(None))):
            raise ValueError(
                f"x2c_type_override must be a string or None, but got {type(self.x2c_type_override)}."
            )
        if not isinstance(self.snso_type_override, (str, type(None))):
            raise ValueError(
                f"snso_type_override must be a string or None, but got {type(self.snso_type_override)}."
            )
        if self.x2c_type_override is not None:
            self.x2c_type_override = self.x2c_type_override.lower()
            if self.x2c_type_override not in ["so", "sf"]:
                raise ValueError(
                    f"Invalid x2c_type_override: {self.x2c_type_override}. Must be 'so' or 'sf'."
                )
        if self.snso_type_override is not None:
            self.snso_type_override = self.snso_type_override.lower()
            if self.snso_type_override not in [
                "boettger",
                "dc",
                "dcb",
                "row-dependent",
            ]:
                raise ValueError(
                    f"Invalid snso_type_override: {self.snso_type_override}. Must be 'boettger', 'dc', 'dcb', or 'row-dependent'."
                )

        if self.apply_random_phase:
            if not isinstance(self.rng, np.random.Generator | int):
                raise ValueError(
                    f"rng must be an instance of np.random.Generator or an integer seed, but got {type(self.rng)}."
                )
        if isinstance(self.rng, int):
            self.rng = np.random.default_rng(self.rng)

        self.requires = {"system", "mo_coeff"}
        self.provides = {"system", "mo_coeff"}

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
        self.mo_coeff = self.parent_method.mo_coeff.copy()

        if "mo_space" in self.parent_method.provides:
            self.mo_space = self.parent_method.mo_space

        if not self.system.two_component:
            self.mo_coeff = self.mo_coeff.to_spinorbital_basis()
            if "mo_space" in self.parent_method.provides:
                self.mo_space = self.mo_space.to_spinorbital_basis()
            self.system.two_component = True
        if self.apply_random_phase:
            random_phase = np.diag(
                np.exp(1j * self.rng.uniform(-np.pi, np.pi, size=self.mo_coeff.nmo))
            )
            self.mo_coeff.C[0] = self.mo_coeff.C[0] @ random_phase
        if self.x2c_type_override is not None:
            self.system.x2c_type = self.x2c_type_override
        if self.snso_type_override is not None:
            self.system.snso_type = self.snso_type_override
        # if system.x2c_type was None at system init, the x2c_helper object
        # was never built. This if clause will catch that and build the x2c_helper object with the correct x2c_type and snso_type.
        if self.system.x2c_helper is None:
            self.system._init_x2c()

        self.executed = True
        return self
