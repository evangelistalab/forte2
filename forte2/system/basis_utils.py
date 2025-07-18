from dataclasses import dataclass

import numpy as np

import forte2
from forte2.system import System
from forte2.helpers import logger
from forte2.system.atom_data import Z_TO_ATOM_SYMBOL

SPH_LABELS = [
    ["s"],
    ["py", "pz", "px"],
    ["dxy", "dyz", "dz2", "dxz", "dx2-y2"],
    ["fy(3x2-y2)", "fxyz", "fyz2", "fz3", "fxz2", "fz(x2-y2)", "fx(x2-3y2)"],
]
"""
The labels for spherical harmonics up to f orbitals.
We follow the `Libint2 convention <https://github.com/evaleev/libint/wiki/using-modern-CPlusPlus-API#solid-harmonic-gaussians-ordering-and-normalization>`_.
"""

AM_LABELS = ["s", "p", "d", "f", "g", "h", "i", "j", "k", "l", "m", "n"]


def get_shell_label(l, idx):
    """
    Get the label for a shell based on its angular momentum quantum number (l) and index (idx).

    Parameters
    ----------
    l : int
        Angular momentum quantum number.
    idx : int
        Index of the shell within the angular momentum type.

    Returns
    -------
    str
        The label for the shell.
    """

    if l < 0 or idx < 0:
        raise ValueError(f"Invalid angular momentum index: {l} or index: {idx}")

    if idx > 2 * l:
        raise ValueError(
            f"Index {idx} exceeds maximum allowed for angular momentum {l}: {2*l}"
        )

    if l > len(AM_LABELS):
        raise ValueError(f"Angular momentum {l} exceeds defined labels.")

    if l < len(SPH_LABELS) and idx < len(SPH_LABELS[l]):
        return SPH_LABELS[l][idx]
    else:
        return f"{AM_LABELS[l]}({idx})"


def shell_label_to_lm(shell_label):
    """
    Convert a shell label to its angular momentum quantum number (l) and index (m).

    Parameters
    ----------
    shell_label : str
        The label of the shell, e.g., "s", "p", "dx2-y2", "h(9)"

    Returns
    -------
    list[tuple]
        A list of a single tuple containing the angular momentum quantum number (l) and the index (m), if both l and m can be determined.
        If only l can be determined, returns a list of tuples of all possible (l, m) pairs.
    """
    am = shell_label[0].lower()
    if am not in AM_LABELS:
        raise ValueError(f"Invalid angular momentum label: {am}")
    l = AM_LABELS.index(am)

    if len(shell_label) == 1:
        return [(l, m) for m in range(2 * l + 1)]
    elif l < len(SPH_LABELS):
        m = SPH_LABELS[l].index(shell_label.lower())
    else:
        m = int(shell_label.split("(")[-1].split(")")[0])
        assert (
            m <= 2 * l
        ), f"Index {m} exceeds maximum allowed for angular momentum {l}: {2*l}"
        assert m >= 0, f"Index {m} must be non-negative."
    return [(l, m)]


@dataclass
class BasisInfo:
    """
    A class to hold information about the basis set of a system.

    Parameters
    ----------
    system : forte2.System
        The system for which the basis set information is to be generated.
    basis : forte2.ints.Basis
        The basis set.

    Attributes
    ----------
    basis_labels : list[_AOLabel]
        A data structure with the following attributes:
        - iatom: int, the index of the atom in the system.
        - Z: int, the atomic number of the atom.
        - Zidx: int, the index of the atom in the system (1-based).
        - n: int, the principal quantum number for the shell.
        - l: int, the angular momentum quantum number.
        - m: int, the index of the basis function within the shell.

    atom_to_aos : dict[int : dict[int : list[int]]]
        A dict of dict where e.g., ``atom_to_aos[6][2]`` gives a list of absolute indices of all AOs on C2.
    """

    system: System
    basis: forte2.ints.Basis

    @dataclass
    class _AOLabel:
        abs_idx: int
        iatom: int
        Z: int
        Zidx: int
        n: int
        l: int
        m: int

        def __str__(self):
            return f"{self.abs_idx:<5} {self.iatom:<5} {Z_TO_ATOM_SYMBOL[self.Z]+str(self.Zidx):<5} {str(self.n)+get_shell_label(self.l,self.m):<10}"

    def __post_init__(self):
        self.basis_labels = self._label_basis_functions()
        self.atom_to_aos = self._make_atom_to_ao_map()

    def _label_basis_functions(self):
        basis_labels = []

        shell_first_and_size = self.basis.shell_first_and_size
        center_first = np.array([_[0] for _ in self.basis.center_first_and_last])
        center_given_shell = (
            lambda ishell: np.searchsorted(center_first, ishell, side="right") - 1
        )
        charges = self.system.atomic_charges
        Z_counts = {}
        center_to_atom_label = []
        for i in range(self.system.natoms):
            Z = charges[i]
            if Z not in Z_counts:
                Z_counts[Z] = 0
            Z_counts[Z] += 1
            center_to_atom_label.append((Z, Z_counts[Z]))

        center_to_shell = {}
        for ishell in range(self.basis.nshells):
            center = center_given_shell(shell_first_and_size[ishell][0])
            if center not in center_to_shell:
                center_to_shell[center] = []
            center_to_shell[center].append(ishell)

        ibasis = 0
        for iatom in range(self.system.natoms):
            n_count = list(range(1, 11))
            Z, Zidx = center_to_atom_label[iatom]
            for ishell in center_to_shell[iatom]:
                l = self.basis[ishell].l
                size = self.basis[ishell].size
                for i in range(size):
                    label = self._AOLabel(ibasis, iatom, Z, Zidx, n_count[l], l, i)
                    basis_labels.append(label)
                    ibasis += 1
                n_count[l] += 1

        return basis_labels

    def _make_atom_to_ao_map(self):
        atom_to_aos = {}
        for idx, label in enumerate(self.basis_labels):
            if label.Z not in atom_to_aos:
                atom_to_aos[label.Z] = {}
            if label.Zidx not in atom_to_aos[label.Z]:
                atom_to_aos[label.Z][label.Zidx] = []
            atom_to_aos[label.Z][label.Zidx].append(idx)
        return atom_to_aos

    def print_basis_labels(self):
        """
        Pretty print the basis labels.
        """
        width = 30
        logger.log_info1("=" * width)
        logger.log_info1(f"{'AO':<5} {'Atom':<5} {'Label':<5} {'AO label':<10}")
        logger.log_info1("-" * width)
        for label in self.basis_labels:
            logger.log_info1(label)
        logger.log_info1("=" * width)

    def print_ao_composition(self, coeff, idx, nprint=5, thres=1e-3):
        logger.log_info1(f"{'# MO':<5} {'(AO) label : coeff':40}")
        nprint_ = min(nprint, len(self.basis_labels))
        for imo in idx:
            c = coeff[:, imo]
            c_argsort = np.argsort(np.abs(c))[::-1][:nprint_]
            string = f"{imo:<5d}"
            for iao in range(nprint_):
                if np.abs(c[c_argsort[iao]]) < thres:
                    continue
                label = self.basis_labels[c_argsort[iao]]
                abs_ao_idx = "(" + str(label.abs_idx) + ")"
                atom_label = f"{Z_TO_ATOM_SYMBOL[label.Z]}{label.Zidx}"
                shell_label = str(label.n) + forte2.basis_utils.get_shell_label(
                    label.l, label.m
                )
                ao_coeff = f"{c[c_argsort[iao]]:<+6.4f}"
                ao_label = f"{atom_label+' '+shell_label+' '+abs_ao_idx}"
                lc = ao_label + ": " + ao_coeff
                string += f" {lc:<25}"
            logger.log_info1(string)

@dataclass
class BasisInfoHubbard: ...