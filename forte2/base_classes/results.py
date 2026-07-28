from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
import json

from .mo import MO
import numpy as np
from numpy.typing import NDArray
from typing import Any


class DataclassEncoder(json.JSONEncoder):
    """Custom encoder that handles numpy arrays and dataclasses."""
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if hasattr(obj, '__dataclass_fields__'):
            return asdict(obj)
        return super().default(obj)


@dataclass(frozen=True)
class ResultBase(ABC):
    @abstractmethod
    def copy(self):...

    def to_json(self, filepath: str) -> None:
        """Write dataclass to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(asdict(self), f, cls=DataclassEncoder, indent=4)

    def to_npz(self, filepath: str) -> None:
        """Write dataclass to NPZ file."""
        d = asdict(self)
        d["cls_name"] = self.__class__.__name__
        np.savez(filepath, **d)

    @classmethod
    def from_npz(cls, filepath: str)-> "ResultBase":
        """Load dataclass from NPZ file."""
        data = np.load(filepath, allow_pickle=True)
        if "cls_name" not in data:
            raise ValueError("NPZ file does not contain 'cls_name' key.")
        if data["cls_name"] != cls.__name__:
            raise ValueError(f"NPZ file contains data for class {data['cls_name']}, but expected {cls.__name__}.")
        # Remove cls_name from data before passing to constructor
        data_dict = {key: data[key] for key in data if key != "cls_name"}
        return cls(**data_dict)

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

    def copy(self):
        ...