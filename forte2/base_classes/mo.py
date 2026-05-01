from dataclasses import dataclass, field

import numpy as np


@dataclass
class MO:
    """
    Class to hold molecular orbital coefficients and their associated irrep labels and indices.
    """

    C: list
    spinorbital: bool
    irrep_labels: list
    irrep_indices: list

    nmo: int = field(init=False)
    nbf: int = field(init=False)
    restricted: bool = field(init=False)

    def __post_init__(self):
        if not isinstance(self.C, list):
            raise ValueError(
                f"C must be a list of numpy arrays, but got {type(self.C)}."
            )
        if self.spinorbital and len(self.C) != 1:
            raise ValueError(
                f"For spinorbital coefficients, C must be a list of length 1, but got length {len(self.C)}."
            )
        for i, arr in enumerate(self.C):
            if not isinstance(arr, np.ndarray):
                raise ValueError(f"C[{i}] must be a numpy array, but got {type(arr)}.")
        self.nbf, self.nmo = self.C[0].shape
        for i, arr in enumerate(self.C):
            if arr.shape != (self.nbf, self.nmo):
                raise ValueError(
                    f"C[{i}] has shape {arr.shape} but expected {(self.nbf, self.nmo)}."
                )
        self.restricted = False
        if len(self.C) == 1 and not self.spinorbital:
            self.restricted = True

        if len(self.irrep_labels) != len(self.C):
            raise ValueError(
                f"Length of irrep_labels ({len(self.irrep_labels)}) must match length of C ({len(self.C)})."
            )
        if len(self.irrep_indices) != len(self.C):
            raise ValueError(
                f"Length of irrep_indices ({len(self.irrep_indices)}) must match length of C ({len(self.C)})."
            )
        for i, (labels, indices) in enumerate(
            zip(self.irrep_labels, self.irrep_indices)
        ):
            if len(labels) != self.nmo or len(indices) != self.nmo:
                raise ValueError(
                    f"Length of irrep_labels[{i}] and irrep_indices[{i}] must match number of MOs ({self.nmo}), but got {len(labels)} and {len(indices)}."
                )
            
    @property
    def Ca(self):
        if self.spinorbital:
            raise ValueError("Ca is not defined for generalized MO objects.")
        return self.C[0]
    
    @property
    def Cb(self):
        if self.spinorbital:
            raise ValueError("Cb is not defined for generalized MO objects.")
        if self.restricted:
            return self.C[0]
        else:
            return self.C[1]
        
    @property
    def Cso(self):
        if not self.spinorbital:
            raise ValueError("Cso is only defined for generalized MO objects.")
        return self.C[0]


    def copy(self):
        return self.__class__(
            C=[arr.copy() for arr in self.C],
            spinorbital=self.spinorbital,
            irrep_labels=[label.copy() for label in self.irrep_labels],
            irrep_indices=[ind.copy() for ind in self.irrep_indices],
        )

    def to_spinorbital_basis(self, cmplx=True):
        def list_spatial_to_spinor(lst):
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

        if self.spinorbital:
            return self.copy()
        dtype = np.complex128 if cmplx else np.float64
        C_2c = np.zeros((self.nbf * 2, self.nmo * 2), dtype=dtype)
        if not self.restricted:
            # |a^0_{alfa AO} b^0_{alfa AO} ... |
            # |a^0_{beta AO} b^0_{beta AO} ... |
            C_2c[: self.nbf, ::2] = self.C[0]
            C_2c[self.nbf :, 1::2] = self.C[1]
        else:
            # RHF/ROHF
            C_2c[: self.nbf, ::2] = self.C[0]
            C_2c[self.nbf :, 1::2] = self.C[0]

        return self.__class__(
            C=[C_2c],
            spinorbital=True,
            irrep_labels=list_spatial_to_spinor(self.irrep_labels),
            irrep_indices=list_spatial_to_spinor(self.irrep_indices),
        )
