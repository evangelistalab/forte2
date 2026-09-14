from dataclasses import dataclass

import numpy as np
from scipy.linalg import qr

from forte2.data import Z_TO_ATOM_SYMBOL
from forte2.helpers import logger, procrustes_rotation
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


class IBOAligner:
    """Align and order IBOs against canonical atomic orbitals.

    :meth:`align_to_atomic_orbitals` uses rotation-invariant atom-locality
    validation, maximally aligns full-rank projected IAO targets, and orders
    them by atom and native MINAO basis-function index.

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
        self._cartesian_alignment_diagnostics = []
        self._atomic_alignment_objective_change = 0.0
        self.atomic_orbital_assignments = (None,) * self.nocc
        self.atomic_orbital_order = tuple(range(self.nocc))

    def align_to_atomic_orbitals(self):
        """Maximally align atom-local subspaces with canonical IAOs.

        Atom locality is verified from the eigenvalues of each block's atomic
        population matrix. Pivoted QR chooses a linearly independent set of
        atomic targets, and the Procrustes rotation is applied whenever that
        target overlap is numerically full-rank. Target purity is retained as a
        diagnostic rather than used as an all-or-nothing acceptance criterion.
        Assigned orbitals are placed in atom and native MINAO order; unassigned
        orbitals follow in their original relative order.

        Returns
        -------
        NDArray
            The aligned IBO coefficient matrix.
        """
        S_iao_ibo = self.C_iao.T @ self.S1 @ self.C_ibo
        _, self.U_ibo = self._align_cartesian_atomic_orbitals(
            S_iao_ibo, self.U_ibo.copy()
        )
        self.C_ibo = self.C_occ @ self.U_ibo
        return self.C_ibo

    def _align_cartesian_atomic_orbitals(self, S_iao_ibo, U_ibo):
        """Align rotation-invariant atom-local blocks to projected IAO targets."""

        center_ranges = self.system.minao_basis.center_first_and_last
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
        dominant_atoms = np.argmax(atom_populations, axis=0)
        minao_labels = BasisInfo(self.system, self.system.minao_basis).basis_labels
        atom_rows = {}
        for row, label in enumerate(minao_labels):
            atom_rows.setdefault(label.iatom, []).append(row)

        alignment_groups = []
        diagnostics = []
        objective_before = self._ibo_objective(S_iao_ibo)

        for iatom, iaos_indices in atom_rows.items():
            important_ibos = [
                i
                for i in range(self.nocc)
                if dominant_atoms[i] == iatom
                and atom_populations[iatom, i] > ATOM_LOCAL_THRESHOLD
            ]
            niibos = len(important_ibos)
            if not important_ibos or niibos > len(iaos_indices):
                continue

            # The eigenvalues of B_A^H B_A are invariant to rotations within
            # this candidate IBO block. Requiring its smallest eigenvalue to be
            # large ensures that the complete subspace, not just the current
            # diagonal representation, is localized on the atom.
            atom_overlap = S_iao_ibo[np.ix_(iaos_indices, important_ibos)]
            atom_metric = atom_overlap.T.conj() @ atom_overlap
            locality_eigenvalues = np.linalg.eigvalsh(atom_metric)
            if locality_eigenvalues[0] < ATOM_LOCAL_THRESHOLD:
                continue

            # Column-pivoted QR of B_A^T chooses IAO rows that are both relevant
            # to the IBO block and linearly independent. Sorting afterward fixes
            # their canonical atom/shell/component order without changing the
            # selected target subspace.
            _, _, pivots = qr(atom_overlap.T, mode="economic", pivoting=True)
            target_iaos = [iaos_indices[pivot] for pivot in pivots[:niibos]]
            target_iaos.sort(
                key=lambda iao: self._atomic_orbital_order(minao_labels[iao])
            )

            target_overlap = S_iao_ibo[np.ix_(target_iaos, important_ibos)]
            rotation, singular_values = procrustes_rotation(
                target_overlap.T, return_singular_values=True
            )
            rank_tolerance = (
                np.finfo(singular_values.dtype).eps
                * max(target_overlap.shape)
                * singular_values[0]
            )
            if singular_values[-1] <= rank_tolerance:
                logger.log_warning(
                    f"Could not align {niibos} IBO(s) on atom {iatom + 1}: "
                    "the projected canonical IAO targets are rank deficient."
                )
                continue

            S_iao_ibo[:, important_ibos] = (
                S_iao_ibo[:, important_ibos] @ rotation
            )
            U_ibo[:, important_ibos] = U_ibo[:, important_ibos] @ rotation
            alignment_groups.append((iatom, tuple(important_ibos), tuple(target_iaos)))
            diagnostics.append(
                (iatom, locality_eigenvalues.copy(), singular_values.copy())
            )

        S_iao_ibo, U_ibo, alignment_groups = self._order_atomic_orbitals(
            S_iao_ibo, U_ibo, alignment_groups, minao_labels
        )
        self._cartesian_alignment_groups = alignment_groups
        self._cartesian_alignment_diagnostics = diagnostics
        objective_after = self._ibo_objective(S_iao_ibo)
        self._atomic_alignment_objective_change = objective_after - objective_before
        if alignment_groups:
            logger.log_info1(
                f"Maximally aligned {len(alignment_groups)} rotation-invariant "
                "atom-local IBO block(s) to projected canonical IAOs.\n"
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

    def log_atomic_alignment_summary(
        self,
        *,
        block_number,
        order,
        mo_indices,
        orbital_energies,
    ):
        """Log atomic assignments and dominant IAO characters.

        Parameters
        ----------
        block_number : int
            One-based GAS block number.
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

        assignments = [self.atomic_orbital_assignments[i] for i in order]
        C_ordered = self.C_ibo[:, order]
        iao_overlaps = self.C_iao.T.conj() @ self.S1 @ C_ordered
        iao_populations = np.abs(iao_overlaps) ** 2
        main_iao_rows = np.argmax(iao_populations, axis=0)
        minao_labels = BasisInfo(self.system, self.system.minao_basis).basis_labels
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
                "{:>14}",
                "{:>11.4f}",
                "{:>12}",
                "{:>9.4f}",
            ],
            use_color=False,
        )
        lines = [
            f"\nIBO atomic-alignment summary for GAS block {block_number}:",
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
                target_population = iao_populations[target_row, column]
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
                f"{nunassigned} of {self.nocc} IBO(s) in GAS block {block_number} "
                "could not be assigned to an atom-local canonical IAO target. "
                "They remain converged localized IBOs."
            )

        nweak = sum(
            population is not None and population < ATOM_LOCAL_THRESHOLD
            for population in target_populations
        )
        if nweak:
            logger.log_warning(
                f"{nweak} of {self.nocc} IBO(s) in GAS block {block_number} "
                f"have best-match canonical IAO target populations below "
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

    def _ibo_objective(self, S_iao_ibo):
        """Evaluate the atom-population objective used by the IBO optimizer."""

        objective = 0.0
        for first, last in self.system.minao_basis.center_first_and_last:
            populations = np.einsum(
                "mi,mi->i", S_iao_ibo[first:last], S_iao_ibo[first:last]
            )
            objective += np.sum(populations**4)
        return objective
