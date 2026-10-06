from itertools import combinations

import numpy as np
import pytest

from forte2 import CUHF, RHF, ROHF, UHF, System
from forte2.base_classes.rebuild import rebuild_method_chain, rebind_method_chain
from forte2.integrals import LIBCINT_AVAILABLE
from forte2.lib import ints
from forte2.scf.occupations import OccupationPolicy, SpinStructure
from forte2.symmetry.sym_utils import get_symmetry_ops
from forte2.symmetry.sym_utils import CHARACTER_TABLE

UNRESTRICTED = SpinStructure(shared_orbitals=False, nested=False)


def _h2(distance=1.5):
    return System(
        xyz=f"H 0 0 0; H 0 0 {distance}",
        basis_set="sto-3g",
        auxiliary_basis_set="def2-universal-jkfit",
        symmetry=True,
    )


def _assert_real_space_symmetry(hf):
    points = np.random.default_rng(265).uniform(-3, 3, (30, 3))
    for C, labels in zip(hf.mos.C, hf.mos.irrep_labels):
        table = np.array(
            [CHARACTER_TABLE[hf.orbital_point_group][label] for label in labels]
        )
        C = np.ascontiguousarray(C)
        values = ints.orbitals_at_points(hf.system.basis, points, C)
        for j, R in enumerate(get_symmetry_ops(hf.orbital_point_group).values()):
            transformed = ints.orbitals_at_points(hf.system.basis, points @ R.T, C)
            np.testing.assert_allclose(
                transformed, values * table[:, j], atol=1e-9, rtol=0
            )


@pytest.mark.parametrize("method", [ROHF, UHF, CUHF])
def test_target_symmetry(method):
    hf = method(charge=1, ms=0.5, target_symmetry="B1U")(_h2()).run()
    assert hf.state_symmetry == "b1u"
    assert hf.mos.irrep_labels[0][0] == "b1u"
    assert hf.E == pytest.approx(-0.31255213048039077, abs=1e-10)
    _assert_real_space_symmetry(hf)


@pytest.mark.parametrize("method", [RHF, ROHF, UHF, CUHF])
def test_explicit_irrep_occupations(method):
    if method is RHF:
        options = dict(charge=0, irrep_occupations={"b1u": 1})
        expected = "ag"
    else:
        options = dict(charge=1, ms=0.5, irrep_occupations={"b1u": (1, 0)})
        expected = "b1u"
    hf = method(**options)(_h2()).run()
    assert hf.state_symmetry == expected
    assert hf.irrep_labels[0][0] == "b1u"
    _assert_real_space_symmetry(hf)


def test_uhf_independent_spin_occupations():
    hf = UHF(
        charge=0, ms=0, target_symmetry=5, irrep_occupations={0: (1, 0), 5: (0, 1)}
    )(_h2()).run()
    assert hf.irrep_labels[0][0] == "ag"
    assert hf.irrep_labels[1][0] == "b1u"
    assert hf.state_symmetry == "b1u"


@pytest.mark.parametrize("method", [ROHF, CUHF])
def test_core_and_open_shell_ordering(method):
    system = System(
        xyz="Li 0 0 0",
        basis_set="sto-3g",
        auxiliary_basis_set="def2-universal-jkfit",
        symmetry=True,
    )
    hf = method(charge=0, ms=0.5, irrep_occupations={"ag": 1, "b2u": (1, 0)})(
        system
    ).run()
    assert hf.irrep_labels[0][:2] == ["ag", "b2u"]
    assert hf.state_symmetry == "b2u"
    for spin, nocc in enumerate((2, 1)):
        C = hf.C[min(spin, len(hf.C) - 1)]
        np.testing.assert_allclose(hf.D[spin], C[:, :nocc] @ C[:, :nocc].T, atol=1e-12)


@pytest.mark.parametrize(
    "backend",
    [
        "libint2",
        pytest.param(
            "libcint",
            marks=pytest.mark.skipif(
                not LIBCINT_AVAILABLE, reason="Libcint is unavailable"
            ),
        ),
    ],
)
def test_n2_ccpvtz_fixed_configuration(backend):
    system = System(
        xyz="N 0 0 0;N 0 0 2.5",
        basis_set="cc-pvtz",
        auxiliary_basis_set="cc-pvtz-jkfit",
        symmetry=True,
        integral_backend=backend,
    )
    occupations = {"ag": 3, "b1u": 2, "b2u": 1, "b3u": 1}
    hf = RHF(charge=0, irrep_occupations=occupations)(system).run()
    assert hf.E == pytest.approx(-108.14839967816553, abs=1e-9)
    labels = hf.irrep_labels[0][:7]
    assert {label: labels.count(label) for label in set(labels)} == occupations
    _assert_real_space_symmetry(hf)


@pytest.mark.parametrize(
    "options",
    [
        dict(target_symmetry=True),
        dict(target_symmetry=1.5),
        dict(irrep_occupations=[]),
        dict(irrep_occupations={"ag": -1}),
        dict(irrep_occupations={"ag": 1.0}),
        dict(irrep_occupations={"ag": True}),
    ],
)
def test_invalid_constructor_options(options):
    with pytest.raises(ValueError):
        RHF(charge=0, **options)


@pytest.mark.parametrize("method", [RHF, ROHF, UHF, CUHF])
def test_integer_occupation_is_doubly_occupied(method):
    # An integer means doubly occupied orbitals for every method.
    options = {} if method is RHF else {"ms": 0}
    as_integer = method(charge=0, irrep_occupations={"b1u": 1}, **options)(_h2()).run()
    as_pair = method(charge=0, irrep_occupations={"b1u": (1, 1)}, **options)(
        _h2()
    ).run()
    assert as_integer.E == pytest.approx(as_pair.E, abs=1e-12)
    assert all(labels[0] == "b1u" for labels in as_integer.irrep_labels)


@pytest.mark.parametrize(
    "method,options,match",
    [
        (RHF, dict(target_symmetry="b1u"), "always totally symmetric"),
        (RHF, dict(target_symmetry="invalid"), "Unknown irrep"),
        (RHF, dict(target_symmetry=8), "Unknown irrep"),
        (RHF, dict(irrep_occupations={"ag": 2}), "must sum"),
        (RHF, dict(irrep_occupations={"ag": 1, 0: 0}), "more than once"),
        (RHF, dict(irrep_occupations={"ag": (1, 0), "b1u": (0, 1)}), "nested"),
        (UHF, dict(ms=0, irrep_occupations={"ag": (1,)}), "alpha, beta"),
        (
            UHF,
            dict(ms=0, target_symmetry="b1u", irrep_occupations={"ag": (1, 1)}),
            "incompatible",
        ),
        (ROHF, dict(ms=0, irrep_occupations={"ag": (1, 0), "b1u": (0, 1)}), "nested"),
        (CUHF, dict(ms=0, target_symmetry="b1u"), "always totally symmetric"),
    ],
)
def test_invalid_bound_options(method, options, match):
    with pytest.raises(ValueError, match=match):
        method(charge=0, **options)(_h2())


@pytest.mark.parametrize(
    "method,options,match",
    [
        (RHF, dict(charge=0, irrep_occupations={"b2u": 1}), "available orbitals"),
        (UHF, dict(charge=1, ms=0.5, target_symmetry="b2u"), "No occupation pattern"),
    ],
)
def test_unavailable_irrep_raises_before_scf(method, options, match):
    system = _h2()
    hf = method(**options)(system)
    with pytest.raises(ValueError, match=match):
        hf.run()
    assert not hf.executed
    assert "B_Pmn" not in system.fock_builder.__dict__


def test_target_occupation_matches_exhaustive_search():
    eps = np.array([-3.0, -2.0, -1.5, -1.0, 0.0, 0.5, 1.0])
    irreps = np.array([0, 1, 2, 3, 4, 5, 0])
    for target in range(8):
        # With no beta electrons, the alpha selection alone carries the target.
        policy = OccupationPolicy(UNRESTRICTED, "D2H", (2, 0), target, None)
        order = policy.permutations([eps, eps], [irreps, irreps])[0]
        expected = min(
            eps[list(idx)].sum()
            for idx in combinations(range(7), 2)
            if np.bitwise_xor.reduce(irreps[list(idx)]) == target
        )
        assert eps[order[:2]].sum() == pytest.approx(expected)


def test_uhf_target_is_chosen_jointly_for_both_spins():
    # Choosing the alpha Aufbau occupation first would miss the joint minimum.
    eps = [np.array([-5.0, -4.0]), np.array([-100.0, 100.0])]
    irreps = [np.array([0, 1]), np.array([0, 1])]
    policy = OccupationPolicy(UNRESTRICTED, "C2", (1, 1), 1, None)
    orders = policy.permutations(eps, irreps)
    assert orders[0][0] == 1
    assert orders[1][0] == 0


def test_oscillating_occupations_are_frozen():
    # Alternating orbital energies make the lowest-energy pattern flip between the
    # a and b irreps; returning to an earlier pattern fixes it.
    a_lower = [np.array([-1.0, -0.9])] * 2
    b_lower = [np.array([-0.9, -1.0])] * 2
    irreps = [np.array([0, 1])] * 2
    policy = OccupationPolicy(UNRESTRICTED, "C2", (1, 1), 0, None)
    occupied = []
    for eps in (a_lower, b_lower, a_lower, b_lower):
        orders = policy.permutations(eps, irreps)
        occupied.append([int(order[0]) for order in orders])
    assert occupied == [[0, 0], [1, 1], [0, 0], [0, 0]]


def test_rebuild_preserves_raw_symmetry_options():
    occupations = {"B1U": 1}
    hf = RHF(charge=0, target_symmetry="Ag", irrep_occupations=occupations)(_h2()).run()
    rebuilt = rebuild_method_chain(hf, _h2(1.7)).run()
    rebind_method_chain(hf, _h2(1.7)).run()
    for method in (hf, rebuilt):
        assert method.irrep_occupations == {"B1U": 1}
        assert method.target_symmetry == "Ag"
        assert method.irrep_labels[0][0] == "b1u"
    assert occupations == {"B1U": 1}
    assert hf.E == pytest.approx(rebuilt.E, abs=1e-12)
