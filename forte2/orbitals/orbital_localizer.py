from dataclasses import dataclass
from numbers import Integral

import numpy as np

from forte2.base_classes import Method
from forte2.data import Z_TO_ATOM_SYMBOL
from forte2.helpers import logger
from .iao import IAO
from .pipek_mezey import pipek_mezey_rotation


@dataclass
class Localizer(Method):
    """
    Localizes orbitals as intrinsic bond orbitals (IBOs).

    Each block of orbitals is rotated by the orthogonal transformation that maximizes
    :math:`\\sum_A \\sum_i (Q^A_{ii})^4`, where :math:`Q^A_{ii}` is the intrinsic atomic
    orbital (IAO) charge of orbital :math:`i` on atom :math:`A` [1]_. Rotations mix
    orbitals only within a block.

    Parameters
    ----------
    orbitals : list[int] | list[list[int]], optional
        The MO indices to localize. A list of lists gives blocks that are localized
        separately. By default, the active orbitals are localized one GAS at a time if the
        parent method defines an MO space; otherwise, the occupied orbitals are localized,
        the doubly and singly occupied ones separately.
    split_by_occupation : bool, optional, default=False
        If True, each block is split into sets of orbitals with the same reference
        occupations, so the localized orbitals keep the reference determinant. The parent
        orbitals must carry reference occupations.
    maxiter : int, optional, default=100
        The maximum number of sweeps over all orbital pairs.
    g_tol : float, optional, default=1e-8
        The convergence threshold on the largest element of the orbital-rotation gradient.

    Attributes
    ----------
    blocks : list[list[int]]
        The MO indices of each block that was localized.
    populations : list[NDArray]
        The IAO charge of each localized orbital on each atom, shape (natom, norb), per
        block.
    U : NDArray
        The orthogonal rotation from the parent orbitals to the localized ones, shape
        (nmo, nmo), such that ``C_localized = C_parent @ U``.

    Notes
    -----
    The IAOs span the frozen core, core, and active orbitals if the parent method defines
    an MO space, and the occupied orbitals otherwise, so every block must lie within that
    span. Rotations within an MO space leave CASCI and CASSCF energies unchanged, but not
    the energies of approximate active-space solvers such as selected CI. A block that
    crosses MO spaces, or reference occupations if the parent defines no MO space, also
    changes CASCI and CASSCF energies, so the Localizer warns about it.

    References
    ----------
    .. [1] G. Knizia, J. Chem. Theory Comput. 2013, 9, 4834-4843.
    """

    orbitals: list[int] | list[list[int]] | None = None
    split_by_occupation: bool = False
    maxiter: int = 100
    g_tol: float = 1e-8

    def __post_init__(self):
        if self.orbitals is not None:
            indices = [i for block in self._given_blocks() for i in block]
            if len(set(indices)) != len(indices):
                raise ValueError(
                    f"orbitals lists an MO more than once: {self.orbitals}."
                )
        if self.maxiter < 1:
            raise ValueError(f"maxiter must be positive, got {self.maxiter}.")
        self.requires = {"system", "mos"}
        self.requires_attrs |= {"two_component": False}

    def __call__(self, parent_method):
        self._register_parent_method(parent_method)
        self.provides = {"system", "mos"} | (parent_method.provides & {"mo_space"})
        return self

    def run(self):
        if not self.parent_method.executed:
            self.parent_method.run()
        self.system = self.parent_method.system
        self.mos = self.parent_method.mos.copy()
        self.mo_space = (
            self.parent_method.mo_space if "mo_space" in self.provides else None
        )
        self._check_orbitals()
        iao = IAO.from_mos(self.system, self.mos, self.mo_space)

        self.blocks = self._make_blocks()
        C = self.mos.C[0]
        self.populations = []
        self.U = np.eye(self.mos.nmo)
        logger.log_info1("\nLocalizing orbitals as intrinsic bond orbitals")
        charges = [iao.charge_matrices(C[:, block]) for block in self.blocks]
        for block, Q in zip(self.blocks, charges):
            U_block, Q, nsweep = pipek_mezey_rotation(Q, self.maxiter, self.g_tol)
            C[:, block] = C[:, block] @ U_block
            self.U[np.ix_(block, block)] = U_block
            self.populations.append(np.einsum("aii->ai", Q))
            self._print_block(block, self.populations[-1], nsweep)

        # rotations that mix reference occupations leave no reference determinant
        if self.mos.occupations is not None and any(
            len({self._occupation(i) for i in block}) > 1 for block in self.blocks
        ):
            self.mos.occupations = None
        self.executed = True
        return self

    def _given_blocks(self):
        if all(isinstance(i, Integral) for i in self.orbitals):
            blocks = [self.orbitals]
        else:
            blocks = self.orbitals
        for block in blocks:
            if isinstance(block, Integral) or not all(
                isinstance(i, Integral) for i in block
            ):
                raise ValueError(
                    "orbitals must be a list of MO indices or a list of such lists, "
                    f"got {self.orbitals}."
                )
        return [[int(i) for i in block] for block in blocks]

    def _check_orbitals(self):
        name = self.parent_method.__class__.__name__
        if not self.mos.restricted:
            raise ValueError(
                f"Localizer needs restricted orbitals, but {name} provides unrestricted ones."
            )
        if len(set(self.mos.irrep_indices[0])) > 1:
            raise ValueError(
                "Localization breaks point-group symmetry, but the orbitals of "
                f"{name} span more than one irrep. Build the System with symmetry=False."
            )
        if self.split_by_occupation and self.mos.occupations is None:
            raise ValueError(
                f"split_by_occupation needs reference occupations, but {name} provides "
                "none. Orbitals from a correlated calculation have no reference "
                "determinant."
            )

    def _make_blocks(self):
        if self.orbitals is not None:
            blocks = self._given_blocks()
            for block in blocks:
                if not all(0 <= i < self.mos.nmo for i in block):
                    raise ValueError(
                        f"orbitals {block} are out of range for {self.mos.nmo} MOs."
                    )
        elif self.mo_space is not None:
            blocks = [list(gas) for gas in self.mo_space.active_orbitals]
        else:
            occupied = [i for i in range(self.mos.nmo) if any(self._occupation(i))]
            blocks = self._split_by_occupation(occupied)
        if self.split_by_occupation:
            blocks = [
                part for block in blocks for part in self._split_by_occupation(block)
            ]

        labels = self._space_labels()
        for block in blocks:
            if len({labels[i] for i in block}) > 1:
                logger.log_warning(
                    f"Localizer block {block} mixes MO spaces or reference occupations, "
                    "which changes the energies of the methods downstream."
                )
        return [block for block in blocks if block]

    def _space_labels(self):
        # the label of each MO in the partition that rotations shouldn't cross
        if self.mo_space is None:
            return {i: self._occupation(i) for i in range(self.mos.nmo)}
        ms = self.mo_space
        labels = {}
        for name, indices in (
            ("frozen core", ms.frozen_core_indices),
            ("core", ms.core_indices),
            ("virtual", ms.virtual_indices),
            ("frozen virtual", ms.frozen_virtual_indices),
        ):
            labels.update(dict.fromkeys(indices, name))
        for k, gas in enumerate(ms.active_orbitals):
            labels.update(
                dict.fromkeys(gas, f"GAS{k + 1}" if ms.ngas > 1 else "active")
            )
        return labels

    def _split_by_occupation(self, block):
        groups = {}
        for i in block:
            groups.setdefault(self._occupation(i), []).append(i)
        return list(groups.values())

    def _occupation(self, i):
        return tuple(occ[i] for occ in self.mos.occupations)

    def _print_block(self, block, pops, nsweep):
        logger.log_info1(f"\nLocalized {len(block)} orbitals in {nsweep} sweeps")
        logger.log_info1(f"{'# MO':>6} {'atom':>6} {'charge':>8}")
        for k, i in enumerate(block):
            a = int(pops[:, k].argmax())
            label = f"{Z_TO_ATOM_SYMBOL[int(self.system.atoms[a][0])]}{a + 1}"
            logger.log_info1(f"{i:>6d} {label:>6} {pops[a, k]:>8.4f}")
