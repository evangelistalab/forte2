from dataclasses import dataclass

import numpy as np

from forte2.helpers import logger, procrustes_rotation
from forte2.system.basis_utils import BasisInfo, get_shell_label

from .iao import IBO


@dataclass(frozen=True)
class AtomicOrbitalAssignment:
    """Atomic MINAO target assigned to one atomically aligned IBO.

    ``atom_index`` and ``minao_basis_index`` are zero-based. The latter is the
    absolute basis-function index in Forte2's native MINAO ordering.
    ``component_index`` is the function's index within its real-spherical
    shell.
    """

    atom_index: int
    minao_basis_index: int
    n: int
    l: int
    component_index: int
    label: str


class IBOAligner:
    """Align and order IBOs against canonical atomic orbitals.

    :meth:`align_to_atomic_orbitals` aligns atom-local blocks with axis-oriented
    IAOs and orders them by atom and native MINAO basis-function index.

    Parameters
    ----------
    ibo : IBO
        A completed IBO localization.
    """

    def __init__(self, ibo: IBO):
        self.ibo = ibo
        self.system = ibo.system
        self.C_occ = ibo.C_occ
        self.C_iao = ibo.C_iao
        self.S1 = ibo.S1
        self.nocc = ibo.nocc

        self.U_ibo = ibo.U_ibo.copy()
        self.C_ibo = ibo.C_ibo.copy()
        self._cartesian_alignment_groups = []
        self._atomic_alignment_objective_change = 0.0
        self.atomic_orbital_assignments = (None,) * self.nocc
        self.atomic_orbital_order = tuple(range(self.nocc))

    def align_to_atomic_orbitals(self):
        """Align localized orbitals with canonical atomic IAO components.

        This post-processes the current IBOs in place, orders assigned orbitals
        by atom and native MINAO basis-function index, and returns the aligned
        coefficients. Every accepted output orbital remains predominantly on
        the same atom, but the explicit atomic gauge may slightly change the IBO
        localization objective. Unassigned orbitals follow the assigned ones in
        their original relative order.

        Returns
        -------
        NDArray
            The aligned IBO coefficient matrix.
        """

        C_ibo_iao = self.C_iao.T @ self.S1 @ self.C_ibo
        _, self.U_ibo = self._align_cartesian_atomic_orbitals(
            C_ibo_iao, self.U_ibo.copy()
        )
        self.C_ibo = self.C_occ @ self.U_ibo
        return self.C_ibo

    def _align_cartesian_atomic_orbitals(self, C_ibo_iao, U_ibo):
        """Fix atom-local blocks to the global axis-oriented IAOs.

        Group all sufficiently atom-local IBOs by their dominant atom. Such a
        block may span several radial and angular-momentum shells. Use an
        orthogonal Procrustes rotation to maximize its overlap with the best
        matching ordered minimal-basis atomic orbitals. This fixes phases and
        orientations without imposing an artificial shell separation. A trial
        rotation is accepted only if every output orbital remains predominantly
        localized on the same atom.
        """

        atom_local_threshold = 0.9
        target_weight_threshold = 0.9

        center_ranges = self.system.minao_basis.center_first_and_last
        atom_populations = np.vstack(
            [
                np.einsum("mi,mi->i", C_ibo_iao[first:last], C_ibo_iao[first:last])
                for first, last in center_ranges
            ]
        )
        minao_labels = BasisInfo(self.system, self.system.minao_basis).basis_labels
        atom_rows = {}
        for row, label in enumerate(minao_labels):
            atom_rows.setdefault(label.iatom, []).append(row)

        alignment_groups = []
        objective_before = self._ibo_objective(C_ibo_iao)
        for iatom, rows in atom_rows.items():
            group = [
                i
                for i in range(self.nocc)
                if np.argmax(atom_populations[:, i]) == iatom
                and atom_populations[iatom, i] > atom_local_threshold
            ]
            if not group or len(group) > len(rows):
                continue

            row_weights = {row: np.sum(C_ibo_iao[row, group] ** 2) for row in rows}
            target_rows = sorted(rows, key=row_weights.get, reverse=True)[: len(group)]
            target_rows.sort(
                key=lambda row: self._atomic_orbital_order(minao_labels[row])
            )
            target_weight = np.sum(C_ibo_iao[np.ix_(target_rows, group)] ** 2) / len(
                group
            )
            if target_weight < target_weight_threshold:
                continue

            # M = L^T S T. Because C_ibo_iao contains IBO coefficients in
            # the orthonormal IAO basis, the selected rows are T^T S L.
            M = C_ibo_iao[np.ix_(target_rows, group)].T
            rotation = procrustes_rotation(M)

            C_trial = C_ibo_iao.copy()
            C_trial[:, group] = C_trial[:, group] @ rotation
            first, last = center_ranges[iatom]
            trial_populations = np.einsum(
                "mi,mi->i", C_trial[first:last, group], C_trial[first:last, group]
            )
            if np.any(trial_populations < atom_local_threshold):
                continue

            C_ibo_iao = C_trial
            U_ibo[:, group] = U_ibo[:, group] @ rotation
            alignment_groups.append((iatom, tuple(group), tuple(target_rows)))

        C_ibo_iao, U_ibo, alignment_groups = self._order_atomic_orbitals(
            C_ibo_iao, U_ibo, alignment_groups, minao_labels
        )
        self._cartesian_alignment_groups = alignment_groups
        objective_after = self._ibo_objective(C_ibo_iao)
        self._atomic_alignment_objective_change = objective_after - objective_before
        if alignment_groups:
            logger.log_info1(
                f"Aligned {len(alignment_groups)} atom-local IBO block(s) "
                "to the global axis-oriented IAOs.\n"
                f"Atomic alignment change in IBO objective: "
                f"{self._atomic_alignment_objective_change:+.3e}."
            )
        return C_ibo_iao, U_ibo

    def _order_atomic_orbitals(self, C_ibo_iao, U_ibo, alignment_groups, minao_labels):
        """Order assigned IBOs by atom and native MINAO function index."""

        assignment_rows = [None] * self.nocc
        for iatom, orbital_indices, target_rows in alignment_groups:
            for orbital, row in zip(orbital_indices, target_rows):
                assignment_rows[orbital] = (iatom, row)

        assigned = [
            i for i, assignment in enumerate(assignment_rows) if assignment is not None
        ]
        assigned.sort(
            key=lambda i: (
                assignment_rows[i][0],
                minao_labels[assignment_rows[i][1]].abs_idx,
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
            _, row = assignment
            target = minao_labels[row]
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

        reordered_groups = []
        for iatom, orbital_indices, target_rows in alignment_groups:
            pairs = sorted(
                (int(old_to_new[orbital]), row)
                for orbital, row in zip(orbital_indices, target_rows)
            )
            reordered_groups.append(
                (
                    iatom,
                    tuple(index for index, _ in pairs),
                    tuple(row for _, row in pairs),
                )
            )
        reordered_groups.sort(key=lambda item: item[1][0] if item[1] else self.nocc)

        self.atomic_orbital_assignments = tuple(assignments)
        self.atomic_orbital_order = tuple(order)
        return C_ibo_iao[:, order], U_ibo[:, order], reordered_groups

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

    def _ibo_objective(self, C_ibo_iao):
        """Evaluate the atom-population objective used by the IBO optimizer."""

        objective = 0.0
        for first, last in self.system.minao_basis.center_first_and_last:
            populations = np.einsum(
                "mi,mi->i", C_ibo_iao[first:last], C_ibo_iao[first:last]
            )
            objective += np.sum(populations**4)
        return objective
