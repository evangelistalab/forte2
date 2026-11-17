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

    All quantities are expressed in the basis of `mo_coeff`. `nroots` is the total
    number of roots over all states, and `nact` is the number of active orbitals
    (spinors in two-component calculations).

    Attributes
    ----------
    ci_energies : NDArray
        Energies of all roots, shape (nroots,). For selected CI, these are the
        total energies, including the PT2 correction.
    mo_coeff : MO
        Final MO coefficients.
    mo_coeff_unrotated : MO | None
        MO coefficients before the rotation to semicanonical or natural orbitals.
        None if the orbitals were not rotated.
    active_rotation : NDArray | None
        Rotation of the active orbitals from `mo_coeff_unrotated` to `mo_coeff`,
        shape (nact, nact), in the index order of `rdm1`: the active columns of
        `mo_coeff` equal those of `mo_coeff_unrotated` times `active_rotation`.
        None if the orbitals were not rotated.
    rdm1 : NDArray
        Active-space 1-RDMs of all roots, with shape (nroots, 2, nact, nact) for the
        alpha and beta blocks, or (nroots, nact, nact) in two-component calculations.
    dipoles : NDArray | None
        Dipole moments (nuclear + electronic) of all roots in atomic units, shape
        (nroots, 3). None for model systems.
    trdm1 : dict[tuple[int, int], NDArray] | None
        Active-space 1-TRDMs between distinct roots, ``trdm1[(i, j)]`` =
        <i|a^+_p a_q|j>, with the trailing dimensions of `rdm1`.
    transition_dipoles : dict[tuple[int, int], NDArray] | None
        ``transition_dipoles[(i, j)]`` = <i|mu|j> in atomic units. The
        ``(i, i)`` entries hold `dipoles`.
    oscillator_strengths : dict[tuple[int, int], float] | None
        Oscillator strengths of the transitions, zero for the ``(i, i)`` entries.

    Notes
    -----
    The transition dictionaries are only computed if transition properties are
    requested. They share the keys of
    :meth:`CIBase.compute_transition_properties`: each pair appears once, ordered
    from the lower- to the higher-energy root, and pairs that could not be
    computed are absent.
    """

    ci_energies: NDArray
    mo_coeff: MO
    mo_coeff_unrotated: MO | None
    active_rotation: NDArray | None
    rdm1: NDArray
    dipoles: NDArray | None
    trdm1: dict[tuple[int, int], NDArray] | None = None
    transition_dipoles: dict[tuple[int, int], NDArray] | None = None
    oscillator_strengths: dict[tuple[int, int], float] | None = None
