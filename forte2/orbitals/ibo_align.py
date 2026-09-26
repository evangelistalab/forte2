from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import qr

from forte2.data import Z_TO_ATOM_SYMBOL
from forte2.helpers import invsqrt_matrix, logger, procrustes_rotation
from forte2.helpers.table import AsciiTable
from forte2.system.basis_utils import BasisInfo, get_shell_label

from .iao import IBO


ATOM_LOCAL_THRESHOLD = 0.9


@dataclass(frozen=True)
class AtomicOrbitalAssignment:
    """Atomic MINAO target assigned to one atomically aligned IBO.

    ``minao_basis_index`` is the absolute basis function index in Forte2's
    native MINAO ordering. ``component_index`` is the function's index within
    its real-spherical shell.
    """

    atom_index: int
    minao_basis_index: int
    n: int
    l: int
    component_index: int
    label: str


@dataclass(frozen=True)
class _AtomicAlignmentSet:
    """Metadata for one accepted set of atom-local IBOs."""

    atom_index: int
    orbital_indices: tuple[int, ...]
    target_minao_rows: tuple[int, ...]
    locality_eigenvalues: NDArray
    target_singular_values: NDArray


class IBOAligner:
    """Align and order IBOs against projected MINAO functions.

    :meth:`align_to_atomic_orbitals` uses IAO populations for rotation-invariant
    atom-locality validation, maximally aligns full-rank projected MINAO
    targets, and orders them by atom and native MINAO basis-function index.

    Parameters
    ----------
    ibo : IBO
        A completed IBO localization.
    """

    def __init__(self, ibo: IBO):
        self.system = ibo.system
        self.C_occ = ibo.C_occ
        self.C_iao = ibo.C_iao
        self.S1 = ibo.S1
        self.nocc = ibo.nocc
        self.C_minao_targets = self._make_minao_targets(ibo.C_minao_projected)
        self._minao_labels = tuple(
            BasisInfo(self.system, self.system.minao_basis).basis_labels
        )

        self.U_ibo = ibo.U_ibo.copy()
        self.C_ibo = ibo.C_ibo.copy()
        self._alignment_sets = ()
        self._atomic_alignment_objective_change = 0.0
        self.atomic_orbital_assignments = (None,) * self.nocc
        self.atomic_orbital_order = tuple(range(self.nocc))

    def _make_minao_targets(self, C_minao_projected):
        """Symmetrically orthonormalize projected MINAO functions per atom."""

        C_targets = np.zeros_like(C_minao_projected)
        for atom_index, (first, last) in enumerate(
            self.system.minao_basis.center_first_and_last
        ):
            C_atom = C_minao_projected[:, first:last]
            atom_metric = C_atom.T.conj() @ self.S1 @ C_atom
            atom_metric_invsqrt, _, info = invsqrt_matrix(atom_metric)
            C_targets[:, first:last] = C_atom @ atom_metric_invsqrt
            if info["n_discarded"]:
                logger.log_warning(
                    f"Projected MINAO functions on atom {atom_index + 1} contain "
                    f"{info['n_discarded']} linear dependence(s); atomic "
                    "alignment may be incomplete."
                )
        return C_targets

    def align_to_atomic_orbitals(self) -> NDArray:
        """Maximally align atom-local IBO sets with projected MINAO targets.

        IAO populations define atom assignment and the rotation-invariant
        locality test. Pivoted QR chooses linearly independent projected MINAO
        targets, and Procrustes alignment fixes the atomic Cartesian gauge.
        Assigned orbitals are placed in atom and native MINAO order; unassigned
        orbitals follow in their original relative order.

        Returns
        -------
        NDArray
            The aligned IBO coefficient matrix.
        """
        S_iao_ibo = self.C_iao.T.conj() @ self.S1 @ self.C_ibo
        S_minao_ibo = self.C_minao_targets.T.conj() @ self.S1 @ self.C_ibo
        _, _, self.U_ibo = self._align_cartesian_atomic_orbitals(
            S_iao_ibo, S_minao_ibo, self.U_ibo.copy()
        )
        self.C_ibo = self.C_occ @ self.U_ibo
        return self.C_ibo

    def _align_cartesian_atomic_orbitals(
        self, S_iao_ibo, S_minao_ibo, U_ibo
    ):
        """Align IAO-local sets to projected MINAO targets."""

        center_ranges = self.system.minao_basis.center_first_and_last
        atom_populations = self._atom_populations(S_iao_ibo)
        dominant_atoms = np.argmax(atom_populations, axis=0)
        minao_labels = self._minao_labels

        alignment_sets = []
        objective_before = self._ibo_objective(atom_populations)

        for iatom, (first, last) in enumerate(center_ranges):
            important_ibos = np.flatnonzero(
                (dominant_atoms == iatom)
                & (atom_populations[iatom] > ATOM_LOCAL_THRESHOLD)
            )
            if important_ibos.size == 0 or important_ibos.size > last - first:
                continue

            alignment = self._align_atom_set(
                S_iao_ibo,
                S_minao_ibo,
                U_ibo,
                atom_index=iatom,
                atom_rows=tuple(range(first, last)),
                orbital_indices=tuple(int(i) for i in important_ibos),
                minao_labels=minao_labels,
            )
            if alignment is not None:
                alignment_sets.append(alignment)

        S_iao_ibo, S_minao_ibo, U_ibo, alignment_sets = (
            self._order_atomic_orbitals(
                S_iao_ibo,
                S_minao_ibo,
                U_ibo,
                alignment_sets,
                minao_labels,
            )
        )
        self._alignment_sets = tuple(alignment_sets)
        objective_after = self._ibo_objective(self._atom_populations(S_iao_ibo))
        self._atomic_alignment_objective_change = objective_after - objective_before
        if alignment_sets:
            logger.log_info1(
                f"Maximally aligned {len(alignment_sets)} rotation-invariant "
                "atom-local IBO set(s) to projected MINAO targets.\n"
                f"Atomic alignment change in IBO objective: "
                f"{self._atomic_alignment_objective_change:+.3e}."
            )
        return S_iao_ibo, S_minao_ibo, U_ibo

    def _align_atom_set(
        self,
        S_iao_ibo,
        S_minao_ibo,
        U_ibo,
        *,
        atom_index,
        atom_rows,
        orbital_indices,
        minao_labels,
    ):
        """Align one atom-local set in place and return its metadata."""

        # The eigenvalues of B_A^H B_A are invariant to rotations within this
        # candidate IBO set. Requiring the smallest one to be large validates
        # the complete subspace rather than only its current diagonal.
        atom_overlap = S_iao_ibo[np.ix_(atom_rows, orbital_indices)]
        locality_eigenvalues = np.linalg.eigvalsh(
            atom_overlap.T.conj() @ atom_overlap
        )
        if locality_eigenvalues[0] < ATOM_LOCAL_THRESHOLD:
            return None

        minao_overlap = S_minao_ibo[np.ix_(atom_rows, orbital_indices)]
        target_minao_rows = self._select_minao_targets(
            minao_overlap, atom_rows, minao_labels
        )
        target_overlap = S_minao_ibo[
            np.ix_(target_minao_rows, orbital_indices)
        ]
        rotation, singular_values = procrustes_rotation(
            target_overlap.T.conj(), return_singular_values=True
        )
        if not self._is_full_rank(singular_values, target_overlap.shape):
            logger.log_warning(
                f"Could not align {len(orbital_indices)} IBO(s) on atom "
                f"{atom_index + 1}: the projected MINAO targets are "
                "rank deficient."
            )
            return None

        S_iao_ibo[:, orbital_indices] = (
            S_iao_ibo[:, orbital_indices] @ rotation
        )
        S_minao_ibo[:, orbital_indices] = (
            S_minao_ibo[:, orbital_indices] @ rotation
        )
        U_ibo[:, orbital_indices] = U_ibo[:, orbital_indices] @ rotation
        return _AtomicAlignmentSet(
            atom_index=atom_index,
            orbital_indices=orbital_indices,
            target_minao_rows=target_minao_rows,
            locality_eigenvalues=locality_eigenvalues,
            target_singular_values=singular_values,
        )

    def _select_minao_targets(self, atom_overlap, atom_rows, minao_labels):
        """Select independent projected MINAO targets in canonical order."""

        # Pivoted QR of B_A^H selects MINAO rows that are both relevant to the
        # IBO set and linearly independent.
        _, _, pivots = qr(atom_overlap.T.conj(), mode="economic", pivoting=True)
        target_rows = [
            atom_rows[pivot] for pivot in pivots[: atom_overlap.shape[1]]
        ]
        target_rows.sort(
            key=lambda row: self._atomic_orbital_order(minao_labels[row])
        )
        return tuple(target_rows)

    @staticmethod
    def _is_full_rank(singular_values, shape):
        """Return whether singular values imply numerical full rank."""

        tolerance = (
            np.finfo(singular_values.dtype).eps
            * max(shape)
            * singular_values[0]
        )
        return singular_values[-1] > tolerance

    def _order_atomic_orbitals(
        self, S_iao_ibo, S_minao_ibo, U_ibo, alignment_sets, minao_labels
    ):
        """Order assigned IBOs by atom and native MINAO function index."""

        assignment_rows = [None] * self.nocc
        for alignment_set in alignment_sets:
            for orbital, row in zip(
                alignment_set.orbital_indices, alignment_set.target_minao_rows
            ):
                assignment_rows[orbital] = row

        assigned = [
            i for i, assignment in enumerate(assignment_rows) if assignment is not None
        ]
        assigned.sort(
            key=lambda i: (
                minao_labels[assignment_rows[i]].iatom,
                minao_labels[assignment_rows[i]].abs_idx,
            )
        )
        unassigned = [
            i for i, assignment in enumerate(assignment_rows) if assignment is None
        ]
        order = assigned + unassigned
        old_to_new = np.empty(self.nocc, dtype=int)
        old_to_new[order] = np.arange(self.nocc)

        assignments = []
        for old_index in order:
            assignment = assignment_rows[old_index]
            if assignment is None:
                assignments.append(None)
                continue
            target = minao_labels[assignment]
            assignments.append(
                AtomicOrbitalAssignment(
                    atom_index=target.iatom,
                    minao_basis_index=target.abs_idx,
                    n=target.n,
                    l=target.l,
                    component_index=target.m,
                    label=f"{target.n}{get_shell_label(target.l, target.m)}",
                )
            )

        reordered_sets = []
        for alignment_set in alignment_sets:
            pairs = sorted(
                (int(old_to_new[orbital]), row)
                for orbital, row in zip(
                    alignment_set.orbital_indices, alignment_set.target_minao_rows
                )
            )
            reordered_sets.append(
                replace(
                    alignment_set,
                    orbital_indices=tuple(index for index, _ in pairs),
                    target_minao_rows=tuple(row for _, row in pairs),
                )
            )
        reordered_sets.sort(key=lambda alignment_set: alignment_set.orbital_indices[0])

        self.atomic_orbital_assignments = tuple(assignments)
        self.atomic_orbital_order = tuple(order)
        return (
            S_iao_ibo[:, order],
            S_minao_ibo[:, order],
            U_ibo[:, order],
            reordered_sets,
        )

    def log_atomic_alignment_summary(
        self,
        *,
        gas_number,
        order,
        mo_indices,
        orbital_energies,
    ):
        """Log atomic assignments and dominant IAO characters.

        Parameters
        ----------
        gas_number : int
            One-based GAS number.
        order : ArrayLike
            Permutation from the aligner's native order to final orbital order.
        mo_indices : ArrayLike
            One-based final MO indices, already in final orbital order.
        orbital_energies : ArrayLike
            Generalized-Fock diagonal elements in final orbital order.
        """

        order = np.asarray(order, dtype=int)
        mo_indices = np.asarray(mo_indices, dtype=int)
        orbital_energies = np.asarray(orbital_energies)
        if not (
            order.shape
            == mo_indices.shape
            == orbital_energies.shape
            == (self.nocc,)
        ):
            raise ValueError(
                "IBO atomic-alignment summary inputs must have one entry per IBO."
            )
        if not np.array_equal(np.sort(order), np.arange(self.nocc)):
            raise ValueError(
                "IBO atomic-alignment summary order must be a permutation."
            )

        assignments = [self.atomic_orbital_assignments[i] for i in order]
        C_ordered = self.C_ibo[:, order]
        iao_overlaps = self.C_iao.T.conj() @ self.S1 @ C_ordered
        iao_populations = np.abs(iao_overlaps) ** 2
        minao_overlaps = self.C_minao_targets.T.conj() @ self.S1 @ C_ordered
        minao_populations = np.abs(minao_overlaps) ** 2
        main_iao_rows = np.argmax(iao_populations, axis=0)
        minao_labels = self._minao_labels
        row_by_abs_idx = {label.abs_idx: row for row, label in enumerate(minao_labels)}

        def format_iao(row):
            label = minao_labels[row]
            atom = f"{Z_TO_ATOM_SYMBOL[label.Z].capitalize()}{label.Zidx}"
            return f"{atom} {label.label()}"

        table = AsciiTable(
            columns=[
                "MO",
                "Energy [Eh]",
                "Atomic target",
                "Target pop.",
                "Main IAO",
                "Main pop.",
            ],
            formats=[
                "{:>4d}",
                "{:>14.8f}",
                "{:<13}",
                "{:>11.4f}",
                "{:<8}",
                "{:>9.4f}",
            ],
            use_color=False,
        )
        lines = [
            f"\nIBO atomic-alignment summary for GAS {gas_number}:",
            table.header(),
        ]
        target_populations = []
        for column, (mo_index, energy, assignment, main_row) in enumerate(
            zip(mo_indices, orbital_energies, assignments, main_iao_rows)
        ):
            if assignment is None:
                target = "unassigned"
                target_population = "-"
                target_populations.append(None)
            else:
                target_row = row_by_abs_idx[assignment.minao_basis_index]
                target = format_iao(target_row)
                target_population = minao_populations[target_row, column]
                target_populations.append(target_population)
            lines.append(
                table.row(
                    mo_index,
                    energy,
                    target,
                    target_population,
                    format_iao(main_row),
                    iao_populations[main_row, column],
                )
            )
        lines.append(table.footer())
        logger.log_info1("\n".join(lines))

        nunassigned = sum(assignment is None for assignment in assignments)
        if nunassigned:
            logger.log_warning(
                f"{nunassigned} of {self.nocc} IBO(s) in GAS {gas_number} "
                "could not be assigned to a projected MINAO target. "
                "They remain converged localized IBOs."
            )

        nweak = sum(
            population is not None and population < ATOM_LOCAL_THRESHOLD
            for population in target_populations
        )
        if nweak:
            logger.log_warning(
                f"{nweak} of {self.nocc} IBO(s) in GAS {gas_number} "
                f"have projected MINAO target populations below "
                f"{ATOM_LOCAL_THRESHOLD:.3f}.\n"
                "The reported targets are maximal-overlap labels, not pure "
                "atomic-orbital assignments.\n"
                "IBO localization itself succeeded for every orbital; this "
                "warning concerns only atomic-label confidence."
            )

    @staticmethod
    def _component_order(angular_momentum, component):
        """Return a stable axis-oriented ordering for shell components."""

        if angular_momentum == 1:
            # Libint stores real p functions as (py, pz, px), but the fixed
            # Cartesian axes convention is more naturally exposed as (px, py,
            # pz). Higher angular momenta retain Forte2's real-spherical order.
            label = get_shell_label(angular_momentum, component)
            return {"px": 0, "py": 1, "pz": 2}[label]
        return component

    @classmethod
    def _atomic_orbital_order(cls, label):
        """Order atomic orbitals by n, l, and axis-oriented component."""

        return (
            label.n,
            label.l,
            cls._component_order(label.l, label.m),
        )

    def _atom_populations(self, S_iao_ibo):
        """Return IAO populations indexed by atom and orbital."""

        return np.vstack(
            [
                np.einsum(
                    "mi,mi->i",
                    S_iao_ibo[first:last].conj(),
                    S_iao_ibo[first:last],
                ).real
                for first, last in self.system.minao_basis.center_first_and_last
            ]
        )

    @staticmethod
    def _ibo_objective(atom_populations):
        """Evaluate the atom-population objective used by the IBO optimizer."""

        return np.sum(atom_populations**4)
