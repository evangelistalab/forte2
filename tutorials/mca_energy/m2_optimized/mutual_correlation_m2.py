"""M2-only mutual-correlation analysis with shared MO integrals.

The 2-cumulant convention and contraction are taken from
``mutual_correlation_energy_mod_fast``.  This module computes just pair
terms, and builds one RestrictedMOIntegrals object for the union of fragments.
Run with ``tutorials/mca_energy`` on ``PYTHONPATH``.
"""

from itertools import combinations

import forte2
import numpy as np

from mutual_correlation_energy_mod_fast import (
    _analysis_basis,
    _spin_summed_2cumulant,
)
from subspaces import subspaces


class MutualCorrelationEnergyAnalysis:
    """Compute only M2 terms while reusing one integral build.

    The constructor intentionally does not build M1, M3, or M4. The accepted
    ``max_body_order`` argument is retained for call-site compatibility; M2
    is always the sole computed order in this focused implementation.
    """

    def __init__(
        self,
        ci,
        fragments=None,
        root=0,
        max_body_order=4,
        core_orbitals=None,
        nat_orbs=False,
    ):
        self.ci = ci
        self.root = root
        self.max_body_order = max_body_order
        self.core_orbitals = tuple(
            ci.core_indices if core_orbitals is None else core_orbitals
        )
        self.nat_orbs = nat_orbs
        self.fragments = (
            [[orb] for orb in ci.mo_space.active_indices]
            if fragments is None
            else [list(fragment) for fragment in fragments]
        )
        self.nfragments = len(self.fragments)

        flat_orbitals, _, local_subspaces = subspaces(
            *self.fragments, rdm_orbitals=ci.mo_space.active_indices
        )
        basis = _analysis_basis(ci, root=root, nat_orbs=nat_orbs)
        self.C = basis["C"]
        self.gamma1 = basis["gamma1"]
        self.no_occupations = basis["occupations"]
        self.orbital_rotation = basis["orbital_rotation"]

        # Build the AO-to-MO transformed integrals once for all pair terms.
        self.integrals = forte2.jkbuilder.RestrictedMOIntegrals(
            system=ci.system,
            C=self.C,
            orbitals=flat_orbitals,
            core_orbitals=list(self.core_orbitals),
        )
        self._V = self.integrals.V

        # The established spin-summed 2-cumulant definition is evaluated once.
        cumulant = _spin_summed_2cumulant(
            ci, root=root, orbital_rotation=self.orbital_rotation
        )
        active_pos = {orb: i for i, orb in enumerate(ci.mo_space.active_indices)}
        rdm_positions = np.asarray([active_pos[orb] for orb in flat_orbitals])
        self._cumulant = cumulant[
            np.ix_(rdm_positions, rdm_positions, rdm_positions, rdm_positions)
        ]
        self._local_subspaces = [np.asarray(indices, dtype=int) for indices in local_subspaces]

        self.M1 = {}
        self.M2 = {}
        self.M3 = {}
        self.M4 = {}
        for i, j in combinations(range(self.nfragments), 2):
            self.M2[(i, j)] = self._pair_energy(i, j)
        self.total_correlation = sum(self.M2.values())

    def _pair_energy(self, i, j):
        local = np.concatenate((self._local_subspaces[i], self._local_subspaces[j]))
        fragment_ids = np.concatenate(
            (
                np.zeros(len(self._local_subspaces[i]), dtype=int),
                np.ones(len(self._local_subspaces[j]), dtype=int),
            )
        )
        indices = np.ix_(local, local, local, local)
        V = self._V[indices]
        cumulant = self._cumulant[indices]

        f0 = fragment_ids[:, None, None, None]
        f1 = fragment_ids[None, :, None, None]
        f2 = fragment_ids[None, None, :, None]
        f3 = fragment_ids[None, None, None, :]
        unique_count = (
            1
            + (f1 != f0)
            + ((f2 != f0) & (f2 != f1))
            + ((f3 != f0) & (f3 != f1) & (f3 != f2))
        )
        mask = unique_count == 2
        return float(0.5 * np.sum(V[mask] * cumulant[mask]))

    @property
    def fragment_labels(self):
        return [fragment[0] if len(fragment) == 1 else tuple(fragment) for fragment in self.fragments]

    def get_M2_matrix(self):
        matrix = np.zeros((self.nfragments, self.nfragments))
        for (i, j), value in self.M2.items():
            matrix[i, j] = matrix[j, i] = value
        return matrix
