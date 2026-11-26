import numpy as np
import pytest

from forte2.orbitals import molden_string
from forte2.vibemol import (
    atomic_numbers,
    forces,
    orbitals_molden,
    state_dipole,
    transition_dipole,
)


def test_molden_preserves_system_atom_order_and_coordinates(cyclopropene, reference):
    mc, _ = cyclopropene
    name, molden = orbitals_molden(mc, title="Cyclopropene")
    assert name == "Cyclopropene.molden"
    assert molden == molden_string(mc, title="Cyclopropene")
    assert orbitals_molden(mc)[0] == "mcoptimizer.molden"
    lines = molden.splitlines()
    start = lines.index("[Atoms] AU") + 1
    end = lines.index("", start)
    atoms = [line.split() for line in lines[start:end]]
    assert [int(row[1]) for row in atoms] == list(range(1, len(atoms) + 1))
    assert (
        [int(row[2]) for row in atoms]
        == atomic_numbers(mc)
        == reference["atomicNumbers"]
    )
    np.testing.assert_allclose(
        [[float(value) for value in row[3:]] for row in atoms],
        reference["coordinatesBohr"],
        atol=1e-6,
        rtol=0,
    )


@pytest.mark.parametrize("root", [0, 1])
def test_casscf_forces_and_state_dipole(cyclopropene, reference, root):
    mc, _ = cyclopropene
    expected_gradient = reference["gradientEhPerBohr"][f"S{root}"]
    actual = forces(mc, root=root)
    assert actual.shape == (7, 3)
    np.testing.assert_allclose(-actual, mc.gradient(root=root), atol=1e-6, rtol=0)
    np.testing.assert_allclose(-actual, expected_gradient, atol=1e-6, rtol=0)
    np.testing.assert_allclose(
        state_dipole(mc, root), reference["dipoleAu"][f"S{root}"], atol=1e-5, rtol=0
    )


def test_transition_vector_and_strength_in_both_orders(cyclopropene, reference):
    mc, _ = cyclopropene
    expected = np.array(reference["transitionDipoleAu"]["S0-S1"])
    forward, f_forward = transition_dipole(mc, 0, 1)
    backward, f_backward = transition_dipole(mc, 1, 0)
    np.testing.assert_array_equal(backward, forward)
    sign = -1 if np.dot(forward, expected) < 0 else 1
    np.testing.assert_allclose(sign * forward, expected, atol=1e-5, rtol=0)
    strength = reference["oscillatorStrength"]["S0-S1"]
    np.testing.assert_allclose(
        [f_forward, f_backward], [strength, strength], atol=1e-5, rtol=0
    )


def test_transition_pair_uses_energy_order(cyclopropene, monkeypatch):
    mc, _ = cyclopropene
    solver = mc.ci_solver
    stored = np.array([1.0, 2.0, 3.0])
    monkeypatch.setattr(solver, "transition_dipoles", {(1, 0): stored})
    monkeypatch.setattr(solver, "oscillator_strengths", {(1, 0): 0.3})
    for pair in ((0, 1), (1, 0)):
        actual, strength = transition_dipole(mc, *pair)
        np.testing.assert_array_equal(actual, stored)
        assert strength == 0.3


def test_rhf_forces(cyclopropene):
    _, rhf = cyclopropene
    np.testing.assert_allclose(forces(rhf), -rhf.gradient(), atol=1e-6, rtol=0)


def test_unsupported_force_method():
    with pytest.raises(TypeError, match="MCOptimizer, RHF, UHF, or GHF"):
        forces(object())


@pytest.mark.parametrize("value", [1j, np.nan, np.inf])
def test_nonreal_or_nonfinite_gradient_is_refused(cyclopropene, monkeypatch, value):
    _, rhf = cyclopropene
    monkeypatch.setattr(rhf, "gradient", lambda: np.full((7, 3), value))
    with pytest.raises(ValueError, match="real and finite"):
        forces(rhf)


def test_state_dipole_requires_known_charge(cyclopropene, monkeypatch):
    mc, _ = cyclopropene
    monkeypatch.setattr(mc, "parent_method", None)
    with pytest.raises(ValueError, match="Cannot determine the SCF reference charge"):
        state_dipole(mc, 0)
