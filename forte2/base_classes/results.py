from dataclasses import dataclass, fields, asdict
from abc import ABC

from .mo import MO
import numpy as np
from numpy.typing import NDArray
from typing import Any


@dataclass(frozen=True)
class ResultBase(ABC):
    def copy(self):
        res = dict()
        for f in fields(self):
            res[f.name] = getattr(self, f.name)
        return type(self)(**res)

    def to_npz(self, filepath: str) -> None:
        """Write dataclass to NPZ file."""
        d = asdict(self)
        d["cls_name"] = self.__class__.__name__
        np.savez(filepath, **d)


@dataclass(frozen=True)
class SCFResult(ResultBase):
    """
    Class that holds the results of an SCF calculation.

    Attributes
    ----------
    energy : float
        Total energy of the SCF calculation.
    orbital_energies : list[NDArray]
        List of arrays containing the orbital energies.
        If restricted or generalized SCF, this will be a list of length 1.
        If unrestricted SCF, this will be a list of length 2, with the first element containing the alpha orbital energies and the second element containing the beta orbital energies.
    mo_coeff : MO
        MO object holding MO coefficients and associated irrep labels and indices.
    ao_fock_matrix : list[NDArray]
        List of Fock matrices in the AO basis. Length convention is the same as for `orbital_energies`.
    """

    # Total energy
    energy: float
    # Orbital energies
    orbital_energies: list[NDArray]
    # Orbital coefficients
    mo_coeff: MO
    # Fock matrix in the AO basis
    ao_fock_matrix: list[NDArray]


@dataclass(frozen=True)
class ActiveSpaceResult(ResultBase):
    """
    Class that holds the results of an ActiveSpaceDriver (CI or MCOptimizer) calculation.

    Attributes
    ----------
    energy : float
        Total energy of the SCF calculation.
    orbital_energies : list[NDArray]
        List of arrays containing the orbital energies.
        If restricted or generalized SCF, this will be a list of length 1.
        If unrestricted SCF, this will be a list of length 2, with the first element containing the alpha orbital energies and the second element containing the beta orbital energies.
    mo_coeff : MO
        MO object holding MO coefficients and associated irrep labels and indices.
    ao_fock_matrix : list[NDArray]
        List of Fock matrices in the AO basis. Length convention is the same as for `orbital_energies`.
    """

    # Total energy
    energy: float
    # Orbital energies
    orbital_energies: list[NDArray]
    # Orbital coefficients
    mo_coeff: MO
    # Fock matrix in the AO basis
    ao_fock_matrix: list[NDArray]
