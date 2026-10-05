"""Select HF occupations while preserving electron counts and determinant symmetry."""

from numbers import Integral

import numpy as np

from forte2.symmetry.sym_utils import COTTON_LABELS


def _count(value):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError("Irrep occupation counts must be non-negative integers.")
    return int(value)


def validate_occupation_options(target_symmetry, occupations, mode):
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
        if mode in ("restricted", "generalized"):
            _count(value)
        else:
            if not isinstance(value, (tuple, list)) or len(value) != 2:
                raise ValueError(
                    "Open-shell irrep occupations must be (alpha, beta) pairs."
                )
            for count in value:
                _count(count)


class _OccupationConstraints:
    """Shared input resolution and basis-capacity checks for occupation policies."""

    def __init__(self, mode, point_group, nelec, labels, target_symmetry, occupations):
        self.mode = mode
        self.point_group = point_group
        self.nelec = tuple(nelec)
        self.labels = labels
        self.nirrep = len(self.labels)
        self.target = None if target_symmetry is None else self._irrep(target_symmetry)
        self.counts = self._resolve_counts(occupations)

    def _resolve_counts(self, occupations):
        if occupations is None:
            return None
        counts = np.zeros((len(self.nelec), self.nirrep), dtype=int)
        seen = set()
        for key, value in occupations.items():
            irrep = self._irrep(key)
            if irrep in seen:
                raise ValueError(f"Irrep {key!r} is specified more than once.")
            seen.add(irrep)
            counts[:, irrep] = value
        totals = tuple(counts.sum(axis=1))
        if totals != self.nelec:
            raise ValueError(
                f"irrep_occupations must sum to {self.nelec}, got {totals}. "
                "Unlisted irreps have zero occupation."
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
        self.permutations(
            [np.zeros(len(irreps))]
            * (2 if self.mode in ("unrestricted", "constrained_unrestricted") else 1),
            [irreps]
            * (2 if self.mode in ("unrestricted", "constrained_unrestricted") else 1),
        )


class OccupationPolicy(_OccupationConstraints):
    """Spatial Abelian occupations; constructor arguments remain unchanged."""

    def __init__(self, mode, point_group, nelec, target_symmetry, occupations):
        super().__init__(
            mode,
            point_group,
            nelec,
            COTTON_LABELS[point_group],
            target_symmetry,
            occupations,
        )
        if self.counts is not None:
            if mode in ("restricted_open_shell", "constrained_unrestricted"):
                major = int(nelec[1] > nelec[0])
                if np.any(self.counts[1 - major] > self.counts[major]):
                    raise ValueError(
                        "Minority-spin occupations must be nested within majority-spin occupations."
                    )
            if self.target is not None and self._symmetry(self.counts) != self.target:
                raise ValueError(
                    "irrep_occupations are incompatible with target_symmetry."
                )
        if self.target not in (None, 0) and (
            mode == "restricted"
            or (
                mode in ("restricted_open_shell", "constrained_unrestricted")
                and nelec[0] == nelec[1]
            )
        ):
            raise ValueError(
                "A closed-shell restricted determinant is always totally symmetric."
            )

    def _symmetry(self, counts):
        if self.mode == "restricted":
            return 0
        symmetry = 0
        for irrep, count in enumerate(np.sum(counts, axis=0)):
            if count % 2:
                symmetry ^= irrep
        return symmetry

    def permutations(self, eps, irreps):
        counts = self.counts
        if counts is None:
            if self.mode == "restricted":
                counts = np.array(
                    [
                        np.bincount(
                            irreps[0][np.argsort(eps[0])[: self.nelec[0]]],
                            minlength=self.nirrep,
                        )
                    ]
                )
            elif self.mode == "generalized":
                selected = _target_occupation(
                    eps[0], irreps[0], self.nelec[0], self.nirrep, self.target
                )
                counts = np.array(
                    [np.bincount(irreps[0][selected], minlength=self.nirrep)]
                )
            elif self.mode == "unrestricted":
                spectra = [
                    _occupation_spectrum(e, h, n, self.nirrep)
                    for e, h, n in zip(eps, irreps, self.nelec)
                ]
                costs = (
                    spectra[0][0] + spectra[1][0][np.arange(self.nirrep) ^ self.target]
                )
                symmetry_a = int(np.argmin(costs))
                if not np.isfinite(costs[symmetry_a]):
                    raise ValueError(
                        "No occupation pattern can realize target_symmetry."
                    )
                selected = [
                    spectra[0][1][symmetry_a],
                    spectra[1][1][symmetry_a ^ self.target],
                ]
                counts = np.array(
                    [
                        np.bincount(h[idx], minlength=self.nirrep)
                        for h, idx in zip(irreps, selected)
                    ]
                )
            else:
                counts = _nested_counts(
                    eps, irreps, self.nelec, self.nirrep, self.target
                )

        if self.mode == "restricted_open_shell":
            major = int(self.nelec[1] > self.nelec[0])
            core, singly = [], []
            for irrep in range(self.nirrep):
                indices = _irrep_order(eps[0], irreps[0], irrep)
                ndocc, nocc = counts[1 - major, irrep], counts[major, irrep]
                core.extend(indices[:ndocc])
                singly.extend(indices[ndocc:nocc])
            return [_partition_order(eps[0], core, singly)]
        return [
            _partition_order(
                e,
                _occupied_indices(e, h, row),
            )
            for e, h, row in zip(eps, irreps, counts)
        ]


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


def _occupation_spectrum(eps, irreps, nocc, nirrep, products=None):
    """Minimum orbital-energy sum for each determinant irrep at fixed N."""
    cost = np.full((nocc + 1, nirrep), np.inf)
    cost[0, 0] = 0.0
    chosen = np.zeros((len(eps), nocc + 1, nirrep), dtype=bool)
    predecessors = (
        np.arange(nirrep)[:, None] ^ np.arange(nirrep)
        if products is None
        else np.argsort(products.T, axis=1)
    )
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


def _target_occupation(eps, irreps, nocc, nirrep, target, products=None):
    costs, choices = _occupation_spectrum(eps, irreps, nocc, nirrep, products)
    if not np.isfinite(costs[target]):
        raise ValueError("No occupation pattern can realize target_symmetry.")
    return choices[target]


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


class DoubleGroupOccupationPolicy(_OccupationConstraints):
    """Spinor occupations and genuine total double-group irreps for SO-GHF."""

    def __init__(self, group, nel, target_symmetry, occupations):
        super().__init__(
            "generalized",
            group.point_group,
            (nel,),
            group.labels,
            target_symmetry,
            occupations,
        )
        self.group = group
        self.paired = False
        if self.target is not None:
            fermionic = self.target in group.fermion_parities
            if fermionic != bool(nel % 2):
                raise ValueError(
                    "Odd electron counts require a fermionic double-group irrep; "
                    "even counts require a bosonic irrep."
                )
            if group.nonabelian and not fermionic:
                if self.target != 0:
                    raise ValueError(
                        "This double-group irrep cannot be represented by a single "
                        "HF determinant. A symmetry-pure even-electron determinant "
                        "fills complete 2D spinor multiplets and is totally symmetric."
                    )
                self.paired = True
        if self.counts is not None:
            for irrep, count in enumerate(self.counts[0]):
                if irrep not in group.fermion_parities and count:
                    raise ValueError(
                        "Occupied spinors require fermionic double-group irreps."
                    )
            if self.paired and np.any(self.counts % 2):
                raise ValueError(
                    "A totally symmetric determinant requires even occupations "
                    "of each 2D spinor irrep."
                )
            if self.target is not None and not self.paired:
                if group.nonabelian:
                    symmetry = (
                        sum(
                            self.counts[0, h] * parity
                            for h, parity in group.fermion_parities.items()
                        )
                        % 2
                    )
                    valid = symmetry == group.fermion_parities[self.target]
                else:
                    symmetry = 0
                    for h, count in enumerate(self.counts[0]):
                        for _ in range(count):
                            symmetry = group.abelian_products[symmetry, h]
                    valid = symmetry == self.target
                if not valid:
                    raise ValueError(
                        "irrep_occupations are incompatible with target_symmetry."
                    )

    def permutations(self, eps, irreps):
        e, h = eps[0], irreps[0]
        nel = self.nelec[0]
        if self.paired:
            # Paired diagonalization keeps the two partners adjacent, with
            # identical energies, including at accidental degeneracies.
            if np.any(h[::2] != h[1::2]) or not np.allclose(e[::2], e[1::2]):
                raise RuntimeError("Spinor multiplet partners are missing.")
            if self.counts is None:
                pairs = np.argsort(e[::2], kind="stable")[: nel // 2]
            else:
                pairs = _occupied_indices(e[::2], h[::2], self.counts[0] // 2)
            selected = np.column_stack((2 * pairs, 2 * pairs + 1)).ravel()
        elif self.counts is not None:
            selected = _occupied_indices(e, h, self.counts[0])
        elif self.group.nonabelian:
            parity = np.array([self.group.fermion_parities[int(irrep)] for irrep in h])
            target = self.group.fermion_parities[self.target]
            selected = _target_occupation(e, parity, nel, 2, target)
        else:
            selected = _target_occupation(
                e, h, nel, self.nirrep, self.target, self.group.abelian_products
            )
        return [_partition_order(e, selected)]
