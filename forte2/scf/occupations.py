from dataclasses import dataclass
from numbers import Integral

import numpy as np

from forte2.helpers import logger
from forte2.symmetry.sym_utils import COTTON_LABELS


def _count(value):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError("Irrep occupation counts must be non-negative integers.")
    return int(value)


def _spin_counts(value):
    """(alpha, beta) electrons of one irrep; an integer n means n doubly occupied orbitals."""
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError(
                "Irrep occupations must be integers or (alpha, beta) pairs."
            )
        return _count(value[0]), _count(value[1])
    n = _count(value)
    return n, n


@dataclass(frozen=True)
class SpinStructure:
    """
    How the alpha and beta electrons of a one-component SCF method occupy its orbitals.

    Attributes
    ----------
    shared_orbitals : bool
        Whether both spins occupy one set of spatial orbitals (RHF, ROHF).
    nested : bool
        Whether the minority-spin occupied orbitals lie within the majority-spin ones
        (RHF, ROHF, CUHF).
    """

    shared_orbitals: bool
    nested: bool


def validate_occupation_options(target_symmetry, occupations):
    if target_symmetry is not None and (
        isinstance(target_symmetry, bool)
        or not isinstance(target_symmetry, (str, Integral))
    ):
        raise ValueError("target_symmetry must be an irrep label or integer index.")
    if occupations is None:
        return
    if not isinstance(occupations, dict):
        raise ValueError("irrep_occupations must be a dictionary.")
    for key, value in occupations.items():
        if isinstance(key, bool) or not isinstance(key, (str, Integral)):
            raise ValueError("Irrep keys must be labels or integer indices.")
        _spin_counts(value)


class OccupationPolicy:
    """Select HF occupations by irrep, preserving electron counts and determinant symmetry."""

    def __init__(self, structure, point_group, nelec, target_symmetry, occupations):
        self.structure = structure
        self.point_group = point_group
        self.nelec = tuple(nelec)
        self.labels = COTTON_LABELS[point_group]
        self.nirrep = len(self.labels)
        self.target = None if target_symmetry is None else self._irrep(target_symmetry)
        self.counts = self._resolve_counts(occupations)
        major = int(self.nelec[1] > self.nelec[0])
        if self.counts is not None:
            if structure.nested and np.any(self.counts[1 - major] > self.counts[major]):
                raise ValueError(
                    "Minority-spin occupations must be nested within majority-spin occupations."
                )
            if self.target is not None and self._symmetry(self.counts) != self.target:
                raise ValueError(
                    "irrep_occupations are incompatible with target_symmetry."
                )
        if self.target not in (None, 0) and structure.nested and nelec[0] == nelec[1]:
            raise ValueError(
                "A closed-shell restricted determinant is always totally symmetric."
            )
        # Occupation patterns chosen so far, used to detect oscillation.
        self._visited = set()
        self._current = None

    def _resolve_counts(self, occupations):
        if occupations is None:
            return None
        counts = np.zeros((2, self.nirrep), dtype=int)
        seen = set()
        for key, value in occupations.items():
            irrep = self._irrep(key)
            if irrep in seen:
                raise ValueError(f"Irrep {key!r} is specified more than once.")
            seen.add(irrep)
            counts[:, irrep] = _spin_counts(value)
        totals = tuple(counts.sum(axis=1))
        if totals != self.nelec:
            raise ValueError(
                f"irrep_occupations must sum to {self.nelec} (alpha, beta) electrons, "
                f"got {totals}. Unlisted irreps have zero occupation."
            )
        return counts

    def _irrep(self, value):
        if isinstance(value, str):
            value = value.strip().lower()
            if value in self.labels:
                return self.labels[value]
        elif 0 <= value < self.nirrep:
            return int(value)
        raise ValueError(
            f"Unknown irrep {value!r} for {self.point_group}. Valid labels: {list(self.labels)}."
        )

    def validate_capacity(self, irreps):
        capacity = np.bincount(irreps, minlength=self.nirrep)
        if any(n > len(irreps) for n in self.nelec):
            raise ValueError(
                "The orbital basis cannot accommodate the requested electron count."
            )
        if self.counts is not None and np.any(self.counts > capacity):
            raise ValueError(
                "An irrep occupation exceeds the available orbitals in that irrep."
            )
        # Resolve feasibility before the first density or expensive J/K build.
        if self.counts is None:
            nsets = 1 if self.structure.shared_orbitals else 2
            self._lowest_counts([np.zeros(len(irreps))] * nsets, [irreps] * nsets)

    def _symmetry(self, counts):
        """Determinant irrep: the product of the irreps holding an odd number of electrons."""
        symmetry = 0
        for irrep, count in enumerate(np.sum(counts, axis=0)):
            if count % 2:
                symmetry ^= irrep
        return symmetry

    def permutations(self, eps, irreps):
        """Return, for each orbital set, the order that puts the occupied orbitals first."""
        counts = self.counts
        if counts is None:
            counts = self._lowest_counts(eps, irreps)
            self._freeze_if_cycling(counts)

        if self.structure.shared_orbitals:
            major = int(self.nelec[1] > self.nelec[0])
            core, singly = [], []
            for irrep in range(self.nirrep):
                indices = _irrep_order(eps[0], irreps[0], irrep)
                ndocc, nocc = counts[1 - major, irrep], counts[major, irrep]
                core.extend(indices[:ndocc])
                singly.extend(indices[ndocc:nocc])
            return [_partition_order(eps[0], core, singly)]
        return [
            _partition_order(e, _occupied_indices(e, h, row))
            for e, h, row in zip(eps, irreps, counts)
        ]

    def _lowest_counts(self, eps, irreps):
        """Irrep counts with the lowest orbital-energy sum that realize target_symmetry."""
        if self.structure.nested:
            return _nested_counts(eps, irreps, self.nelec, self.nirrep, self.target)
        return _unrestricted_counts(eps, irreps, self.nelec, self.nirrep, self.target)

    def _freeze_if_cycling(self, counts):
        """Fix the counts once the choice returns to a pattern it has already left."""
        key = counts.tobytes()
        if key == self._current:
            return
        if key in self._visited:
            self.counts = counts
            names = {index: label for label, index in self.labels.items()}
            pattern = {
                names[h]: (int(counts[0, h]), int(counts[1, h]))
                for h in range(self.nirrep)
                if counts[:, h].any()
            }
            logger.log_warning(
                f"Irrep occupations are oscillating; fixing them to {pattern} "
                "(alpha, beta). Set irrep_occupations to choose them explicitly."
            )
        self._visited.add(key)
        self._current = key


def _unrestricted_counts(eps, irreps, nelec, nirrep, target):
    """Choose the alpha and beta occupations jointly for the lowest orbital-energy sum."""
    spectra = [
        _occupation_spectrum(e, h, n, nirrep) for e, h, n in zip(eps, irreps, nelec)
    ]
    costs = spectra[0][0] + spectra[1][0][np.arange(nirrep) ^ target]
    symmetry_a = int(np.argmin(costs))
    if not np.isfinite(costs[symmetry_a]):
        raise ValueError("No occupation pattern can realize target_symmetry.")
    selected = [spectra[0][1][symmetry_a], spectra[1][1][symmetry_a ^ target]]
    return np.array(
        [np.bincount(h[idx], minlength=nirrep) for h, idx in zip(irreps, selected)]
    )


def _irrep_order(eps, irreps, irrep):
    indices = np.flatnonzero(irreps == irrep)
    return indices[np.argsort(eps[indices], kind="stable")]


def _occupied_indices(eps, irreps, counts):
    return np.array(
        [
            i
            for irrep, count in enumerate(counts)
            for i in _irrep_order(eps, irreps, irrep)[:count]
        ],
        dtype=int,
    )


def _partition_order(eps, *occupied):
    used = np.zeros(len(eps), dtype=bool)
    partitions = []
    for indices in occupied:
        indices = np.array(indices, dtype=int)
        used[indices] = True
        partitions.append(indices[np.argsort(eps[indices], kind="stable")])
    virtual = np.flatnonzero(~used)
    partitions.append(virtual[np.argsort(eps[virtual], kind="stable")])
    return np.concatenate(partitions)


def _occupation_spectrum(eps, irreps, nocc, nirrep):
    """Minimum orbital-energy sum for each determinant irrep at fixed N."""
    cost = np.full((nocc + 1, nirrep), np.inf)
    cost[0, 0] = 0.0
    chosen = np.zeros((len(eps), nocc + 1, nirrep), dtype=bool)
    predecessors = np.arange(nirrep)[:, None] ^ np.arange(nirrep)
    for i, (energy, irrep) in enumerate(zip(eps, irreps)):
        candidate = cost[:-1, predecessors[irrep]] + energy
        take = candidate < cost[1:]
        chosen[i, 1:] = take
        cost[1:] = np.where(take, candidate, cost[1:])
    selections = []
    for symmetry in range(nirrep):
        count, current = nocc, symmetry
        indices = []
        if np.isfinite(cost[nocc, symmetry]):
            for i in range(len(eps) - 1, -1, -1):
                if chosen[i, count, current]:
                    indices.append(i)
                    count -= 1
                    current = int(predecessors[irreps[i], current])
        selections.append(np.array(indices, dtype=int))
    return cost[nocc], selections


def _nested_counts(eps, irreps, nelec, nirrep, target):
    """Choose core/open occupations for ROHF and CUHF using irrep energy sums."""
    major = int(nelec[1] > nelec[0])
    ncore, nopen = min(nelec), abs(nelec[0] - nelec[1])
    energies = eps if len(eps) == 2 else [eps[0], eps[0]]
    labels = irreps if len(irreps) == 2 else [irreps[0], irreps[0]]
    states = {(0, 0, 0): (0.0, [])}
    for irrep in range(nirrep):
        prefixes = [
            np.concatenate(([0.0], np.cumsum(e[_irrep_order(e, h, irrep)])))
            for e, h in zip(energies, labels)
        ]
        capacity = len(prefixes[0]) - 1
        next_states = {}
        for (core, singly, symmetry), (energy, counts) in states.items():
            for ndocc in range(min(capacity, ncore - core) + 1):
                for nsocc in range(min(capacity - ndocc, nopen - singly) + 1):
                    key = (
                        core + ndocc,
                        singly + nsocc,
                        symmetry ^ (irrep if nsocc % 2 else 0),
                    )
                    value = (
                        energy
                        + prefixes[major][ndocc + nsocc]
                        + prefixes[1 - major][ndocc]
                    )
                    if key not in next_states or value < next_states[key][0]:
                        next_states[key] = (value, counts + [(ndocc, nsocc)])
        states = next_states
    result = states.get((ncore, nopen, target))
    if result is None:
        raise ValueError("No occupation pattern can realize target_symmetry.")
    counts = np.zeros((2, nirrep), dtype=int)
    for irrep, (ndocc, nsocc) in enumerate(result[1]):
        counts[major, irrep] = ndocc + nsocc
        counts[1 - major, irrep] = ndocc
    return counts
