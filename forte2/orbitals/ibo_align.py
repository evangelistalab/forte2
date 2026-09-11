from dataclasses import dataclass

import numpy as np

from forte2.helpers import logger, procrustes_rotation
from forte2.system.basis_utils import BasisInfo, get_shell_label

from .iao import IBO


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
        # compute the IAO/IBO overlap matrix
        S_iao_ibo = self.C_iao.T @ self.S1 @ self.C_ibo

        # align the IBOs with the axis-oriented IAOs
        _, self.U_ibo = self._align_cartesian_atomic_orbitals(
            S_iao_ibo, self.U_ibo.copy()
        )

        # Update the MO coefficients
        self.C_ibo = self.C_occ @ self.U_ibo
        return self.C_ibo

    def _align_cartesian_atomic_orbitals(self, S_iao_ibo, U_ibo):
        """Align atom-local IBOs with the global axis-oriented IAOs.

        Group all IBOs sufficiently localized on an atom by their dominant atom.
        One block may span several radial and angular-momentum shells.

        We then use an orthogonal Procrustes rotation to maximize its overlap with
        a minimal basis of atomic orbitals. This fixes phases and orientations
        without imposing an artificial shell separation. The smallest singular
        value of the target overlap determines whether the block is accepted.
        """

        atom_local_threshold = 0.9

        center_ranges = self.system.minao_basis.center_first_and_last
        # atom_populations[a, i] is the IAO population of IBO_i on atom a.
        atom_populations = np.vstack(
            [
                np.einsum(
                    "mi,mi->i",
                    S_iao_ibo[first:last].conj(),
                    S_iao_ibo[first:last],
                ).real
                for first, last in center_ranges
            ]
        )
        minao_labels = BasisInfo(self.system, self.system.minao_basis).basis_labels
        atom_rows = {}
        for row, label in enumerate(minao_labels):
            atom_rows.setdefault(label.iatom, []).append(row)

        alignment_groups = []
        objective_before = self._ibo_objective(S_iao_ibo)

        for iatom, iaos_indices in atom_rows.items():
            # Find the IBOs with significant population on this atom. If there are more
            # important IBOs than available atomic orbitals, skip this atom.
            important_ibos = [
                i
                for i in range(self.nocc)
                if atom_populations[iatom, i] > atom_local_threshold
            ]
            niibos = len(important_ibos)
            if not important_ibos or niibos > len(iaos_indices):
                continue

            # IAO weights computed from the important IBOs. The IAO with the largest total population
            # from the important IBOs is the best candidate for alignment.
            iaos_weights = {
                iao: np.sum(np.abs(S_iao_ibo[iao, important_ibos]) ** 2)
                for iao in iaos_indices
            }

            # Sort the important IAOs by their total population from the important IBOs,
            # and select the top niibos.
            target_iaos = sorted(iaos_indices, key=iaos_weights.get, reverse=True)[: niibos]
            # Sort the selected IAOs by their atomic orbital order to ensure a
            # consistent ordering of the aligned IBOs.
            target_iaos.sort(
                key=lambda iao: self._atomic_orbital_order(minao_labels[iao])
            )

            # A = T.T @ S @ L is the target-IAO-by-IBO overlap.
            target_overlap = S_iao_ibo[np.ix_(target_iaos, important_ibos)]

            # The rotation acts as L -> L @ Q, so Procrustes requires
            # M = L.T @ S @ T = target_overlap.T.
            M = target_overlap.T
            rotation, singular_values = procrustes_rotation(
                M, return_singular_values=True
            )
            # If the smallest singular value is too small, the IBOs cannot be aligned to
            # the target IAOs, so we skip this atom.
            if singular_values[-1] ** 2 < atom_local_threshold:
                continue

            # Update the IAO/IBO overlap.
            S_iao_ibo[:, important_ibos] = (
                S_iao_ibo[:, important_ibos] @ rotation
            )
            
            # Update the IBO coefficients with the rotation.
            U_ibo[:, important_ibos] = U_ibo[:, important_ibos] @ rotation

            # Store the alignment group for later ordering.
            alignment_groups.append((iatom, tuple(important_ibos), tuple(target_iaos)))

        S_iao_ibo, U_ibo, alignment_groups = self._order_atomic_orbitals(
            S_iao_ibo, U_ibo, alignment_groups, minao_labels
        )
        self._cartesian_alignment_groups = alignment_groups
        objective_after = self._ibo_objective(S_iao_ibo)
        self._atomic_alignment_objective_change = objective_after - objective_before
        if alignment_groups:
            logger.log_info1(
                f"Aligned {len(alignment_groups)} atom-local IBO block(s) "
                "to the global axis-oriented IAOs.\n"
                f"Atomic alignment change in IBO objective: "
                f"{self._atomic_alignment_objective_change:+.3e}."
            )
        return S_iao_ibo, U_ibo

    def _order_atomic_orbitals(self, S_iao_ibo, U_ibo, alignment_groups, minao_labels):
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
        return S_iao_ibo[:, order], U_ibo[:, order], reordered_groups

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

    def _ibo_objective(self, S_iao_ibo):
        """Evaluate the atom-population objective used by the IBO optimizer."""

        objective = 0.0
        for first, last in self.system.minao_basis.center_first_and_last:
            populations = np.einsum(
                "mi,mi->i", S_iao_ibo[first:last], S_iao_ibo[first:last]
            )
            objective += np.sum(populations**4)
        return objective
