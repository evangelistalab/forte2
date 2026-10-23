"""Focused numerical and work-count comparison against the current M2 path."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import mutual_correlation_energy_mod_fast as reference
import mutual_correlation_m2 as optimized
import plot_correlation_energy
import plot_correlation_energy_v2


class TestM2Optimization(unittest.TestCase):
  def test_m2_matches_pairwise_reference_and_builds_integrals_once(self):
    active = [0, 1, 2, 3]
    ci = SimpleNamespace(
        mo_space=SimpleNamespace(active_indices=active),
        core_indices=[],
        system=object(),
        mos=SimpleNamespace(C=[np.eye(4)]),
        sub_solvers=[object()],
        E=0.0,
    )
    rng = np.random.default_rng(20261001)
    cumulant = rng.normal(size=(4, 4, 4, 4))
    builds = []

    class FakeIntegrals:
        def __init__(self, system, C, orbitals, core_orbitals):
            builds.append(tuple(orbitals))
            # Values depend on the underlying orbital labels, so full-space
            # and pair-space integral builds select identical tensor entries.
            labels = np.asarray(orbitals)
            self.V = (
                labels[:, None, None, None]
                + 2 * labels[None, :, None, None]
                + 3 * labels[None, None, :, None]
                + 5 * labels[None, None, None, :]
            ).astype(float)

    fake_jkbuilder = SimpleNamespace(RestrictedMOIntegrals=FakeIntegrals)
    basis = {
        "C": np.eye(4),
        "gamma1": np.zeros((4, 4)),
        "occupations": None,
        "orbital_rotation": None,
    }
    fragments = [[0], [1], [2], [3]]
    with (
        patch.object(optimized.forte2, "jkbuilder", fake_jkbuilder, create=True),
        patch.object(optimized, "_analysis_basis", return_value=basis),
        patch.object(optimized, "_spin_summed_2cumulant", return_value=cumulant),
        patch.object(reference, "_spin_summed_2cumulant", return_value=cumulant),
    ):
        actual = optimized.MutualCorrelationEnergyAnalysis(ci, fragments=fragments)
        self.assertEqual(builds, [tuple(active)])
        self.assertEqual(actual.M1, {})
        self.assertEqual(actual.M3, {})
        self.assertEqual(actual.M4, {})

        expected = {}
        for i in range(len(fragments)):
            for j in range(i + 1, len(fragments)):
                expected[(i, j)] = reference.twofrag_correlation_energy_enumerated(
                    ci, fragments[i], fragments[j], core_orbitals=[], root=0
                )

        self.assertEqual(len(builds), 1 + len(expected))
        self.assertEqual(actual.M2.keys(), expected.keys())
        for key, value in expected.items():
            self.assertAlmostEqual(actual.M2[key], value, delta=1.0e-12)
        np.testing.assert_allclose(
            actual.get_M2_matrix(),
            reference_matrix(len(fragments), expected),
            rtol=0.0,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            plot_correlation_energy._correlation_matrix(actual),
            actual.get_M2_matrix(),
        )
        np.testing.assert_allclose(
            plot_correlation_energy_v2._correlation_matrix(actual),
            actual.get_M2_matrix(),
        )


def reference_matrix(size, terms):
    matrix = np.zeros((size, size))
    for (i, j), value in terms.items():
        matrix[i, j] = matrix[j, i] = value
    return matrix
