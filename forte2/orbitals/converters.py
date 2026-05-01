from dataclasses import dataclass, field
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
    nmo = C[0].shape[1]
    C_2c = np.zeros((nbf * 2, nmo * 2), dtype=dtype)
    if len(C) == 2:
        # UHF
        if C[1].shape[0] != nbf or C[1].shape[1] != nmo:
            raise ValueError(
                f"C[1] has shape {C[1].shape}, but expected ({nbf}, {nmo})."
            )
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


def list_spatial_to_spinor(lst):
    """
    Convert a list of spatial orbital objects to a list of spinor orbital objects by either duplicating each element (RHF/ROHF case) or interleaving two lists (UHF case).

    Parameters
    ----------
    lst : list of list of objects
        A list of lists of spatial orbital objects. The outer list is over spin (length 1 for RHF/ROHF, length 2 for UHF), and the inner lists are over nmos.

    Returns
    -------
    list[list]
        A list of spinor orbital objects.
    """
    assert isinstance(lst, list)
    if len(lst) == 2:
        # UHF case: interleave the two lists
        if len(lst[0]) != len(lst[1]):
            raise ValueError(
                f"lst[0] has length {len(lst[0])} but lst[1] has length {len(lst[1])}."
            )
        lst_2c = [obj for pair in zip(lst[0], lst[1]) for obj in pair]
    elif len(lst) == 1:
        # RHF/ROHF case: duplicate each element
        lst_2c = [obj for obj in lst[0] for _ in (0, 1)]
    else:
        raise RuntimeError(f"List of length {len(lst)} not recognized!")
    return [lst_2c]


@dataclass
class SpatialToSpinorConverter(MOsMixin, SystemMixin):
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
        MOsMixin.copy_from_upstream(self, self.parent_method, only_alpha=True)
        SystemMixin.copy_from_upstream(self, self.parent_method)
        if hasattr(self.parent_method, "mo_space"):
            self.mo_space = self.parent_method.mo_space

        if not self.system.two_component:
            self.C = convert_coeff_spatial_to_spinor(self.C)
            self.irrep_labels = list_spatial_to_spinor(self.irrep_labels)
            self.irrep_indices = list_spatial_to_spinor(self.irrep_indices)
            if hasattr(self.parent_method, "mo_space"):
                self.mo_space = self.mo_space.to_spinorbital_basis()
            self.system.two_component = True
        if self.apply_random_phase:
            nmo = self.C[0].shape[1]
            random_phase = np.diag(
                np.exp(1j * self.rng.uniform(-np.pi, np.pi, size=nmo))
            )
            self.C[0] = self.C[0] @ random_phase
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
