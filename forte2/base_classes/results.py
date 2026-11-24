from dataclasses import dataclass, fields, asdict
from abc import ABC
from datetime import datetime
from pathlib import Path
import glob
import itertools
import os

from forte2.helpers import logger
from .mo import MO
import numpy as np
from numpy.typing import NDArray

# File-name prefix (including the directory) of saved results; None when saving is off
_save_prefix = None
# Numbers saved results in the order they are produced, over the whole process
_save_counter = itertools.count(1)


def save_results(name=None, directory=None):
    """
    Save every result produced to an NPZ file. Saving is off by default.

    Files are written as ``{prefix}_{NNN}_{Method}.npz``. ``NNN`` numbers the
    results in the order they are produced, and ``Method`` is the class that
    produced the result. A file that cannot be written is skipped with a warning,
    and the calculation continues.

    Parameters
    ----------
    name : str | bool | None, optional, default=None
        The file-name prefix. If None, the prefix is the script's name followed by
        the current date and time, with ``-2``, ``-3``, ... appended if files with
        that prefix already exist. False turns saving off.
    directory : str | os.PathLike | None, optional, default=None
        The directory to save to, created if it does not exist. If None, the
        directory of the invoking script is used, or the working directory when
        there is no plain script (IPython, Jupyter, pytest).

    Raises
    ------
    ValueError
        If the directory cannot be created or is not writable.
    """
    from forte2.helpers.echo_script import _plain_script_path

    global _save_prefix
    if name is False:
        _save_prefix = None
        return

    script = _plain_script_path()
    if directory is None:
        directory = Path(script).resolve().parent if script else Path.cwd()
    else:
        directory = Path(directory).expanduser()
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise ValueError(
                f"Cannot create the results directory {directory}: {e}"
            ) from e
    if not os.access(directory, os.W_OK):
        raise ValueError(f"The results directory {directory} is not writable.")

    if name is not None:
        _save_prefix = directory / name
        return

    stem = Path(script).stem if script else "forte2"
    base = f"{stem}_{datetime.now():%Y%m%d-%H%M%S}"
    prefix, serial = base, 1
    while any(directory.glob(f"{glob.escape(prefix)}_*.npz")):
        serial += 1
        prefix = f"{base}-{serial}"
    _save_prefix = directory / prefix


def _save_result(result, method_name):
    """Write `result` to the next numbered NPZ file if saving is on."""
    if _save_prefix is None:
        return
    path = f"{_save_prefix}_{next(_save_counter):03d}_{method_name}.npz"
    try:
        result.to_npz(path)
    except OSError as e:
        logger.log_warning(f"Could not save the {method_name} result to {path}: {e}")


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
