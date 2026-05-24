from dataclasses import dataclass

import numpy as np

from forte2 import Basis
from forte2.system import System
from forte2.helpers import logger
from forte2.data import Z_TO_ATOM_SYMBOL

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
MAX_L = len(AM_LABELS) - 1


def ml_from_shell_index_cca(l, idx):
    """
    Map Libint Standard/CCA shell index (0..2*l) to signed magnetic quantum number (m_l) value.
    CCA standard is such that Y_{lm}s are listed from Y_{l,-l}, ... Y_{l,0}, ... Y_{l,l}. Thus,
    the 0-based index position is related to the m quantum number via index - l.

    Parameters
    ----------
    l: int
        Angular momentum quantum number.
    idx: int
        Index of the shell within the angular momentum type.

    Returns
    -------
    int:
        Magnetic quantum number m_l.
    """
    if idx < 0 or idx > 2 * l:
        raise ValueError("The shell index must be in within 0 and 2*l")
    return idx - l


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


def get_spinor_label(l, jdouble, mjdouble):
    """
    Get the label for a shell based on its angular momentum quantum number (l),
    total angular momentum (jdouble/2), and magnetic quantum number (mjdouble/2).

    Parameters
    ----------
    l : int
        Angular momentum quantum number.
    jdouble : int
        Total angular momentum quantum number (2*j).
    mjdouble : int
        Magnetic quantum number (2*mj).

    Returns
    -------
    str
        The label for the shell.
    """

    if l < 0 or jdouble < 0:
        raise ValueError(f"Invalid angular momentum index: {l} or jdouble: {jdouble}")

    if l >= len(AM_LABELS):
        raise ValueError(f"Angular momentum {l} exceeds defined labels.")

    return f"{AM_LABELS[l]}{jdouble}/2, {mjdouble:+}/2"


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
    system : System
        The system for which the basis set information is to be generated.
    basis : Basis
        The basis set.

    Attributes
    ----------
    basis_labels : list[_AOLabel]
        `_AOLabel` is a data structure with the following attributes:
        - iatom: int, the index of the atom in the system.
        - Z: int, the atomic number of the atom.
        - Zidx: int, the index of the atom in the system (1-based).
        - n: int, the principal quantum number for the shell.
        - l: int, the angular momentum quantum number.
        - ml: int, the magnetic quantum number m_l.
        - m: int, the index of the basis function within the shell.
    basis_labels_spinor : list[_SpinorAOLabel]
        `_SpinorAOLabel` is a data structure with the following attributes:
        - iatom: int, the index of the atom in the system.
        - Z: int, the atomic number of the atom.
        - Zidx: int, the index of the atom in the system (1-based).
        - n: int, the principal quantum number for the shell.
        - l: int, the angular momentum quantum number.
        - jdouble: int, the total angular momentum quantum number (2*j).
        - mjdouble: int, the magnetic quantum number (2*m_j).
    atom_to_aos : dict[int : dict[int : list[int]]]
        A dict of dict where e.g., ``atom_to_aos[6][2]`` gives a list of absolute indices of all AOs on C2.
    """

    system: System
    basis: Basis

    @dataclass
    class _AOLabel:
        abs_idx: int
        iatom: int
        Z: int
        Zidx: int
        n: int
        l: int
        ml: int
        m: int

        def __str__(self):
            return f"{self.abs_idx:<5} {self.iatom:<5} {Z_TO_ATOM_SYMBOL[self.Z]+str(self.Zidx):<5} {self.label():<10}"

        def label(self):
            return str(self.n) + get_shell_label(self.l, self.m)

    @dataclass
    class _SpinorAOLabel:
        abs_idx: int
        iatom: int
        Z: int
        Zidx: int
        n: int
        l: int
        jdouble: int
        mjdouble: int

        def __str__(self):
            return f"{self.abs_idx:<5} {self.iatom:<5} {Z_TO_ATOM_SYMBOL[self.Z]+str(self.Zidx):<5} {self.label():<10}"

        def label(self):
            return str(self.n) + get_spinor_label(self.l, self.jdouble, self.mjdouble)

    def __post_init__(self):
        self.basis_labels = self._label_basis_functions()
        self.basis_labels_spinor = self._label_basis_functions(spinor=True)
        self.atom_to_aos = self._make_atom_to_ao_map()

    def _label_basis_functions(self, spinor=False):
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
            n_count = list(range(1, MAX_L + 2))
            Z, Zidx = center_to_atom_label[iatom]
            for ishell in center_to_shell[iatom]:
                l = self.basis[ishell].l
                if spinor:
                    # j = l-0.5 case
                    mjdouble = -2 * l + 1
                    for _ in range(2 * l):
                        label = self._SpinorAOLabel(
                            ibasis, iatom, Z, Zidx, n_count[l], l, 2 * l - 1, mjdouble
                        )
                        basis_labels.append(label)
                        ibasis += 1
                        mjdouble += 2
                    # j = l+0.5 case
                    mjdouble = -2 * l - 1
                    for _ in range(2 * l + 2):
                        label = self._SpinorAOLabel(
                            ibasis, iatom, Z, Zidx, n_count[l], l, 2 * l + 1, mjdouble
                        )
                        basis_labels.append(label)
                        ibasis += 1
                        mjdouble += 2
                else:
                    size = self.basis[ishell].size
                    for i in range(size):
                        ml = ml_from_shell_index_cca(l, i)
                        label = self._AOLabel(
                            ibasis, iatom, Z, Zidx, n_count[l], l, ml, i
                        )
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

    def print_basis_labels(self, spinor=False):
        """
        Pretty print the basis labels.
        """
        labels = self.basis_labels_spinor if spinor else self.basis_labels
        width = 30
        logger.log_info1("=" * width)
        logger.log_info1(f"{'AO':<5} {'Atom':<5} {'Label':<5} {'AO label':<10}")
        logger.log_info1("-" * width)
        for label in labels:
            logger.log_info1(label)
        logger.log_info1("=" * width)

    def print_ao_composition(self, coeff, idx, nprint=5, thres=1e-3, spinorbital=False):
        logger.log_info1(f"{'# MO':<5} {'(AO) label : coeff':40}")
        nbf = coeff.shape[0] // 2 if spinorbital else coeff.shape[0]
        nprint_ = min(nprint, len(self.basis_labels))
        for imo in idx:
            c = coeff[:, imo]
            c_argsort = np.argsort(np.abs(c))[::-1][:nprint_]
            string = f"{imo:<5d}"
            for iao in range(nprint_):
                if spinorbital:
                    ao_idx = c_argsort[iao] % nbf
                    spin = " a" if c_argsort[iao] < nbf else " b"
                else:
                    ao_idx = c_argsort[iao]
                    spin = ""
                if np.abs(c[c_argsort[iao]]) < thres:
                    continue
                label = self.basis_labels[ao_idx]
                abs_ao_idx = "(" + str(label.abs_idx) + ")"
                atom_label = f"{Z_TO_ATOM_SYMBOL[label.Z].capitalize()}{label.Zidx}"
                shell_label = str(label.n) + get_shell_label(label.l, label.m)
                ao_coeff = f"{c[c_argsort[iao]]:<+6.4f}"
                ao_label = f"{atom_label} {shell_label} {abs_ao_idx}{spin}"
                lc = ao_label + ": " + ao_coeff
                string += f" {lc:<25}"
            logger.log_info1(string)

    def print_spinor_composition(self, spinor_coeff, idx, nprint=5, thres=1e-3):
        logger.log_info1(f"{'# MO':<5} {'Spinor label : coeff':40}")
        nprint_ = min(nprint, len(self.basis_labels_spinor))
        for imo in idx:
            c = spinor_coeff[:, imo]
            c_argsort = np.argsort(np.abs(c))[::-1][:nprint_]
            string = f"{imo:<5d}"
            for iao in range(nprint_):
                ao_idx = c_argsort[iao]
                if np.abs(c[c_argsort[iao]]) < thres:
                    continue
                label = self.basis_labels_spinor[ao_idx]
                abs_ao_idx = "(" + str(label.abs_idx) + ")"
                atom_label = f"{Z_TO_ATOM_SYMBOL[label.Z].capitalize()}{label.Zidx}"
                spinor_label = str(label.n) + get_spinor_label(
                    label.l, label.jdouble, label.mjdouble
                )
                ao_coeff = f"{c[c_argsort[iao]]:<+6.4f}"
                ao_label = f"{atom_label} {spinor_label} {abs_ao_idx}"
                lc = ao_label + ": " + ao_coeff
                string += f" {lc:<25}"
            logger.log_info1(string)
