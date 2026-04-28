from dataclasses import dataclass, field

from forte2.scf import rhf
import numpy as np

from forte2.base_classes.mixins import MOsMixin, SystemMixin


def convert_coeff_spatial_to_spinor(C, complex=True):
    """
    Convert spatial orbital MO coefficients to spinor(bital) MO coefficients

    Parameters
    ----------
    C : list of NDArray
        The MO coefficients in spatial orbital basis.
    complex : bool, optional, default=True
        Whether to cast to complex dtype.

    Returns
    -------
    list of NDArray
        The MO coefficients in spinor(bital) basis.
    """
    assert isinstance(C, list)
    dtype = np.complex128 if complex else np.float64
    nbf = C[0].shape[0]
    C_2c = np.zeros((nbf * 2,) * 2, dtype=dtype)
    if len(C) == 2:
        # UHF
        assert C[0].shape[0] == nbf
        assert C[1].shape[0] == nbf
        # |a^0_{alfa AO} b^0_{alfa AO} ... |
        # |a^0_{beta AO} b^0_{beta AO} ... |
        C_2c[:nbf, ::2] = C[0]
        C_2c[nbf:, 1::2] = C[1]
    elif len(C) == 1:
        # RHF/ROHF
        C_2c[:nbf, ::2] = C[0]
        C_2c[nbf:, 1::2] = C[0]
    else:
        raise RuntimeError(f"Coefficient of length {len(C)} not recognized!")
    return [C_2c]


@dataclass
class NonRelToRelConverter(MOsMixin, SystemMixin):
    """
    A converter class to convert a non-relativistic method to a relativistic method by converting the MO coefficients to spinor basis and updating the system object.

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

    executed: bool = field(init=False, default=False)

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

    def __call__(self, parent_method):
        self.parent_method = parent_method
        return self

    def run(self):
        if not self.parent_method.executed:
            self.parent_method.run()
        MOsMixin.copy_from_upstream(self, self.parent_method)
        SystemMixin.copy_from_upstream(self, self.parent_method)
        if not self.system.two_component:
            self.C = convert_coeff_spatial_to_spinor(self.C)
            self.irrep_labels = [l for sub in self.irrep_labels for l in (sub, sub)]
            self.irrep_indices = [i for sub in self.irrep_indices for i in (sub, sub)]
            self.system.two_component = True
        if self.apply_random_phase:
            nmo = self.C[0].shape[1]
            random_phase = np.diag(
                np.exp(1j * self.rng.uniform(-np.pi, np.pi, size=nmo))
            )
            self.C[0] = self.C[0] @ random_phase
        x2c_type_save = self.system.x2c_type
        if self.x2c_type_override is not None:
            self.system.x2c_type = self.x2c_type_override
        if self.snso_type_override is not None:
            self.system.snso_type = self.snso_type_override
        # if system.x2c_type was None at system init, the x2c_helper object
        # was never built. This if clause will catch that and build the x2c_helper object with the correct x2c_type and snso_type.
        if x2c_type_save is None and self.system.x2c_type is not None:
            self.system._init_x2c()

        self.executed = True
        return self
