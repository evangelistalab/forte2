from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from forte2 import CI, CISolver, RHF, State, System
from forte2.vibemol import (
    Shown,
    atomic_numbers,
    orbitals_molden,
    show_dipole,
    show_forces,
    show_orbitals,
    show_transition_dipole,
    state_dipole,
    transition_dipole,
)


class RecordingViewer:
    sequential_commands = True

    def __init__(self):
        self.calls = []
        self.futures = []

    def _record(self, method, *args, **kwargs):
        future = object()
        self.calls.append((method, args, kwargs))
        self.futures.append(future)
        return future

    def load(self, *args, **kwargs):
        return self._record("load", *args, **kwargs)

    def show_orbitals(self, *args, **kwargs):
        return self._record("show_orbitals", *args, **kwargs)

    def set_atom_vectors(self, *args, **kwargs):
        return self._record("set_atom_vectors", *args, **kwargs)

    def set_arrow(self, *args, **kwargs):
        return self._record("set_arrow", *args, **kwargs)


@pytest.mark.parametrize("indices", [None, [11]])
def test_orbital_commands(cyclopropene, indices):
    mc, _ = cyclopropene
    viewer = RecordingViewer()
    shown = show_orbitals(viewer, mc, indices=indices, title="Cyclopropene")
    assert isinstance(shown, Shown)
    expected = [
        ("load", (orbitals_molden(mc, title="Cyclopropene"),), {"clear_first": True})
    ]
    if indices is not None:
        expected.append(("show_orbitals", (indices,), {}))
    assert viewer.calls == expected
    assert shown.future is viewer.futures[-1]
    assert shown.scale is None
    with pytest.raises(FrozenInstanceError):
        shown.scale = 2


def test_force_commands_share_explicit_scale(cyclopropene, reference):
    mc, _ = cyclopropene
    viewer = RecordingViewer()
    s1 = show_forces(viewer, mc, root=1)
    s0 = show_forces(viewer, mc, root=0, scale=s1.scale)
    gradients = reference["gradientEhPerBohr"]
    expected_scale = 1.0 / np.linalg.norm(gradients["S1"], axis=1).max()
    np.testing.assert_allclose(s1.scale, expected_scale, atol=1e-6, rtol=0)
    assert s0.scale == s1.scale
    assert [s1.future, s0.future] == viewer.futures
    for (method, args, kwargs), state in zip(viewer.calls, ("S1", "S0"), strict=True):
        assert method == "set_atom_vectors"
        name, values, color = args
        assert (name, color) == ("forces", "#B22D73")
        np.testing.assert_allclose(
            values, -np.array(gradients[state]), atol=1e-6, rtol=0
        )
        assert kwargs == {
            "scale": s1.scale,
            "atomic_numbers": reference["atomicNumbers"],
        }


def test_rhf_force_command(cyclopropene):
    _, rhf = cyclopropene
    viewer = RecordingViewer()
    shown = show_forces(viewer, rhf)
    ((method, args, kwargs),) = viewer.calls
    assert method == "set_atom_vectors"
    name, values, color = args
    assert (name, color) == ("forces", "#B22D73")
    np.testing.assert_allclose(values, -rhf.gradient(), atol=1e-6, rtol=0)
    np.testing.assert_allclose(np.linalg.norm(values, axis=1).max() * shown.scale, 1.0)
    assert kwargs == {"scale": shown.scale, "atomic_numbers": atomic_numbers(rhf)}
    assert shown.future is viewer.futures[0]


def test_dipole_commands(cyclopropene):
    mc, _ = cyclopropene
    viewer = RecordingViewer()
    s0_vector, s1_vector = (state_dipole(mc, root) for root in (0, 1))
    transition, _ = transition_dipole(mc, 0, 1)
    s1 = show_dipole(viewer, mc, root=1)
    s0 = show_dipole(viewer, mc, root=0, scale=s1.scale)
    tdm = show_transition_dipole(viewer, mc, 1, 0)
    assert s1.scale == 1.5 / np.linalg.norm(s1_vector)
    assert s0.scale == s1.scale
    assert tdm.scale is None
    assert [s1.future, s0.future, tdm.future] == viewer.futures
    assert viewer.calls == [
        ("set_arrow", ("dipole", s1_vector.tolist(), "#2365C6"), {"scale": s1.scale}),
        ("set_arrow", ("dipole", s0_vector.tolist(), "#2365C6"), {"scale": s1.scale}),
        (
            "set_arrow",
            ("transition dipole", transition.tolist(), "#208375"),
            {"length": 2.4, "heads": "both"},
        ),
    ]


@pytest.mark.parametrize(
    ("show", "args"),
    [
        (show_orbitals, ()),
        (show_forces, ()),
        (show_dipole, (0,)),
        (show_transition_dipole, (0, 1)),
    ],
)
def test_viewer_capability_checked_before_calculation(show, args, monkeypatch):
    viewer = RecordingViewer()
    monkeypatch.delattr(RecordingViewer, "sequential_commands")
    with pytest.raises(RuntimeError) as error:
        show(viewer, object(), *args)
    assert str(error.value) == (
        "forte2.vibemol needs vibemol-widget 0.2.0 or later, which runs commands in order"
    )
    assert viewer.calls == []


def test_invalid_force_roots_send_no_commands(cyclopropene):
    mc, rhf = cyclopropene
    viewer = RecordingViewer()
    with pytest.raises(ValueError, match="absolute root must be specified"):
        show_forces(viewer, mc)
    with pytest.raises(ValueError, match="root must be None for SCF"):
        show_forces(viewer, rhf, root=0)
    assert viewer.calls == []


@pytest.fixture
def light_ci():
    def run(charge, do_transition_dipole):
        xyz = (
            "H 0 0 0\nH 0 0 0.74"
            if charge == 0
            else "H 0 0 0\nH 0 0 0.85\nH 0.7361216 0 0.425"
        )
        system = System(
            xyz=xyz, basis_set="sto-3g", auxiliary_basis_set="def2-universal-jkfit"
        )
        rhf = RHF(charge=charge)(system)
        ci = CISolver(
            State(nel=rhf.nel, multiplicity=1, ms=0.0),
            active_orbitals=list(range(system.nbf)),
        )
        return CI(ci, do_transition_dipole=do_transition_dipole)(rhf).run()

    return run


def test_missing_transition_properties_send_no_commands(light_ci):
    viewer = RecordingViewer()
    driver = light_ci(charge=0, do_transition_dipole=False)
    with pytest.raises(ValueError, match="do_transition_dipole=True"):
        show_dipole(viewer, driver, 0)
    with pytest.raises(ValueError, match="do_transition_dipole=True"):
        show_transition_dipole(viewer, driver, 0, 1)
    assert viewer.calls == []


def test_charged_state_dipole_sends_no_commands(light_ci):
    viewer = RecordingViewer()
    driver = light_ci(charge=1, do_transition_dipole=True)
    with pytest.raises(ValueError, match="charged system depends on the origin"):
        show_dipole(viewer, driver, 0)
    assert viewer.calls == []


def test_missing_transition_pair_sends_no_commands(cyclopropene):
    mc, _ = cyclopropene
    viewer = RecordingViewer()
    with pytest.raises(
        ValueError, match=r"Available pairs: \[\(0, 0\), \(0, 1\), \(1, 1\)\]"
    ):
        show_transition_dipole(viewer, mc, 0, 2)
    assert viewer.calls == []


def test_zero_dipole_needs_explicit_scale(cyclopropene, monkeypatch):
    mc, _ = cyclopropene
    viewer = RecordingViewer()
    monkeypatch.setitem(mc.ci_solver.transition_dipoles, (0, 0), np.zeros(3))
    with pytest.raises(ValueError, match="Cannot scale zero dipole"):
        show_dipole(viewer, mc, 0)
    assert viewer.calls == []
    assert show_dipole(viewer, mc, 0, scale=2).scale == 2
    assert viewer.calls == [
        ("set_arrow", ("dipole", [0.0] * 3, "#2365C6"), {"scale": 2})
    ]
