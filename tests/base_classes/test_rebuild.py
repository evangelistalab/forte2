import numpy as np
import pytest

from forte2 import (
    CISolver,
    GHF,
    MCOptimizer,
    RelCISolver,
    RHF,
    SpinorUpcaster,
    State,
    System,
    UHF,
)
from forte2.base_classes.rebuild import (
    list_method_chain,
    rebind_method_chain,
    rebuild_method_chain,
    reset_method_chain,
    project_scf_guess,
    snapshot_orbitals,
)
from forte2.orbitals import mo_overlap

_DISPLACED = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]])


def _h2(bond_length=1.4):
    return System(
        xyz=f"H 0.0 0.0 0.0\nH 0.0 0.0 {bond_length}",
        basis_set="sto-3g",
        auxiliary_basis_set="def2-universal-JKFIT",
        unit="bohr",
    )


def _rhf(system):
    return RHF(charge=0, e_tol=1.0e-12, d_tol=1.0e-10, maxiter=100)(system)


def _casscf(system):
    ci_solver = CISolver(
        State(system=system, multiplicity=1, ms=0.0), active_orbitals=[0, 1]
    )
    return MCOptimizer(ci_solver, e_tol=1.0e-12, g_tol=1.0e-9)(_rhf(system))


def _rel_ci(system):
    return RelCISolver(nel=2, active_orbitals=[0, 1, 2, 3])(
        SpinorUpcaster()(_rhf(system))
    )


def test_rebuild_preserves_method_options():
    system = _h2()
    hf = UHF(
        charge=0,
        ms=0.0,
        e_tol=1.0e-11,
        d_tol=1.0e-9,
        maxiter=77,
    )(system)
    hf.run()

    rebuilt = rebuild_method_chain(hf, system.with_geometry(_DISPLACED))

    assert isinstance(rebuilt, UHF)
    assert rebuilt.e_tol == 1.0e-11
    assert rebuilt.d_tol == 1.0e-9
    assert rebuilt.maxiter == 77
    assert rebuilt.charge == 0


def test_rebuilt_chain_starts_unexecuted_and_carries_no_orbitals():
    method = _casscf(_h2())
    method.run()

    rebuilt = rebuild_method_chain(method, method.system.with_geometry(_DISPLACED))

    for stage in list_method_chain(rebuilt):
        assert not stage.executed
        assert getattr(stage, "C", None) is None


def test_rebuild_leaves_the_original_chain_untouched():
    system = _h2()
    method = _casscf(system)
    method.run()
    energy = method.E
    original_stages = list_method_chain(method)

    rebuilt = rebuild_method_chain(method, system.with_geometry(_DISPLACED))
    rebuilt.run()

    assert method.E == energy
    assert method.executed
    assert method.system is system
    assert list_method_chain(method) == original_stages
    for original, new in zip(original_stages, list_method_chain(rebuilt)):
        assert original is not new


def test_rebuild_replaces_nested_solver_objects():
    # MCOptimizer holds its CI solver as an init field. Carrying it over by
    # reference would hand the new chain a solver that already ran at the old
    # geometry.
    system = _h2()
    method = _casscf(system)
    method.run()

    rebuilt = rebuild_method_chain(method, system.with_geometry(_DISPLACED))

    assert rebuilt.ci_solver is not method.ci_solver
    assert not rebuilt.ci_solver.executed
    assert method.ci_solver.executed


def test_rebuild_reconstructs_mo_space_from_active_orbitals():
    system = _h2()
    method = _casscf(system)
    method.run()

    # mo_space is resolved from active_orbitals and is not an init field, so
    # reconstruction replays active_orbitals rather than the resolved space.
    assert method.ci_solver.mo_space is not None
    assert method.ci_solver.mo_space_override is None

    rebuilt = rebuild_method_chain(method, system.with_geometry(_DISPLACED))

    assert rebuilt.ci_solver.mo_space is None
    assert rebuilt.ci_solver.mo_space_override is None
    assert rebuilt.ci_solver.active_orbitals == [0, 1]


def test_rebuild_preserves_a_user_provided_mo_space_override():
    system = _h2()
    rhf = _rhf(system)
    ci_solver = CISolver(
        State(system=system, multiplicity=1, ms=0.0), active_orbitals=[0, 1]
    )
    mc = MCOptimizer(ci_solver, e_tol=1.0e-12, g_tol=1.0e-9)(rhf)
    mc.run()

    reused = CISolver(
        State(system=system, multiplicity=1, ms=0.0), mo_space_override=mc.mo_space
    )(rhf)

    rebuilt = rebuild_method_chain(reused, system.with_geometry(_DISPLACED))

    assert rebuilt.mo_space_override is mc.mo_space


def test_project_scf_guess():
    system = _h2()
    method = _casscf(system)
    method.run()

    rebuilt = rebuild_method_chain(method, system.with_geometry(_DISPLACED))
    applied = project_scf_guess(method, rebuilt)

    root = list_method_chain(rebuilt)[0]
    assert applied
    assert root.C is not None
    assert getattr(rebuilt, "C", None) is None

    new_system = root.system
    np.testing.assert_allclose(
        mo_overlap(root.C[0], new_system, root.C[0]),
        np.eye(new_system.nmo),
        atol=1.0e-10,
    )


def test_seeded_chain_converges_to_the_same_energy_as_an_unseeded_one():
    system = _h2()
    method = _casscf(system)
    method.run()
    displaced_system = system.with_geometry(_DISPLACED)

    seeded = rebuild_method_chain(method, displaced_system)
    project_scf_guess(method, seeded)
    seeded.run()

    unseeded = rebuild_method_chain(method, displaced_system)
    unseeded.run()

    assert seeded.E == pytest.approx(unseeded.E, abs=1.0e-10)


def test_project_scf_guess_declines_for_a_relativistic_chain():
    # SpinorUpcaster marks the shared System two-component in place, so once a
    # relativistic chain has run, its root reports two_component and the real
    # orbital projection no longer applies. The rebuild still succeeds; it just
    # starts from the default guess.
    system = _h2()
    method = _rel_ci(system)
    method.run()
    assert system.two_component

    rebuilt = rebuild_method_chain(method, system.with_geometry(_DISPLACED))

    assert not project_scf_guess(method, rebuilt)
    rebuilt.run()
    assert rebuilt.executed


def test_project_scf_guess_declines_when_projection_does_not_apply():
    system = _h2()
    source = _rhf(system)
    source.run()

    # GHF orbitals are not partitioned into alpha/beta occupied blocks, so the
    # projection has no defined occupied count to project.
    target = rebuild_method_chain(
        GHF(charge=0)(system), system.with_geometry(_DISPLACED)
    )

    assert not project_scf_guess(source, target)
    assert getattr(target, "C", None) is None


def test_reset_method_chain_invalidates_every_stage():
    system = _h2()
    method = _casscf(system)
    method.run()

    reset_method_chain(method)

    for stage in list_method_chain(method):
        assert not stage.executed


@pytest.mark.parametrize(
    "builder, name",
    [
        pytest.param(_rhf, "RHF", id="rhf"),
        pytest.param(_casscf, "MCOptimizer", id="casscf"),
    ],
)
def test_rebind_reuses_the_same_objects(builder, name):
    method = builder(_h2())
    method.run()

    scratch = rebuild_method_chain(method, method.system.with_geometry(_DISPLACED))
    scratch.run()
    stages_before = list_method_chain(scratch)

    rebound = rebind_method_chain(scratch, method.system.with_geometry(_DISPLACED))
    rebound.run()

    # Same objects, not a fresh copy: contrast with rebuild_method_chain.
    assert rebound is scratch
    assert list_method_chain(rebound) == stages_before

    fresh = builder(_h2(bond_length=1.5))
    fresh.run()
    assert type(rebound).__name__ == name
    assert rebound.E == pytest.approx(fresh.E, abs=1.0e-10)


def test_rebind_can_return_to_a_previously_visited_geometry():
    system = _h2()
    method = _casscf(system)
    method.run()
    first_energy = method.E

    scratch = rebuild_method_chain(method, system.with_geometry(_DISPLACED))
    scratch.run()

    rebind_method_chain(scratch, system)
    scratch.run()

    assert scratch.E == pytest.approx(first_energy, abs=1.0e-10)
    # The original chain stays untouched throughout.
    assert method.E == first_energy
    assert method.system is system


def test_snapshot_orbitals_is_decoupled_from_further_mutation():
    system = _h2()
    method = _rhf(system)
    method.run()

    snapshot = snapshot_orbitals(method)
    original_C = method.C[0].copy()

    rebind_method_chain(method, system.with_geometry(_DISPLACED))
    method.run()

    assert snapshot.system is system
    assert snapshot.mos.C[0] is not method.C[0]
    np.testing.assert_allclose(snapshot.mos.C[0], original_C)
