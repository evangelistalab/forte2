from itertools import combinations

import numpy as np
import pytest

from forte2 import CUHF, GHF, RHF, ROHF, UHF, System, X2CParams
from forte2.base_classes.rebuild import rebuild_method_chain, rebind_method_chain
from forte2.integrals import LIBCINT_AVAILABLE
from forte2.lib import ints
from forte2.scf.occupations import OccupationPolicy
from forte2.symmetry.mo_sym_detect import get_symmetry_ops
from forte2.symmetry.sym_utils import CHARACTER_TABLE
from forte2.symmetry.double_group import spin_rotation


def _h2(distance=1.5, **kwargs):
    return System(
        xyz=f"H 0 0 0; H 0 0 {distance}",
        basis_set="sto-3g",
        auxiliary_basis_set="def2-universal-jkfit",
        symmetry=True,
        **kwargs,
    )


def _assert_real_space_symmetry(hf):
    points = np.random.default_rng(265).uniform(-3, 3, (30, 3))
    for C, labels in zip(hf.mos.C, hf.mos.irrep_labels):
        table = np.array(
            [CHARACTER_TABLE[hf.orbital_point_group][label] for label in labels]
        )
        # Evaluate spin components independently; spin-free spatial operations
        # and inversion act identically on both components.
        components = np.split(C, 2) if hf.two_component else [C]
        for component in components:
            for part in (component.real, component.imag):
                values = ints.orbitals_at_points(
                    hf.system.basis, points, np.ascontiguousarray(part)
                )
                for j, R in enumerate(
                    get_symmetry_ops(hf.orbital_point_group).values()
                ):
                    transformed = ints.orbitals_at_points(
                        hf.system.basis, points @ R.T, np.ascontiguousarray(part)
                    )
                    np.testing.assert_allclose(
                        transformed, values * table[:, j], atol=1e-9, rtol=0
                    )


@pytest.mark.parametrize("method", [ROHF, UHF, CUHF, GHF])
def test_target_symmetry(method):
    options = {} if method is GHF else {"ms": 0.5}
    hf = method(charge=1, target_symmetry="B1U", **options)(_h2()).run()
    assert hf.state_symmetry == "b1u"
    assert hf.mos.irrep_labels[0][0] == "b1u"
    assert hf.E == pytest.approx(-0.31255213048039077, abs=1e-10)
    _assert_real_space_symmetry(hf)


@pytest.mark.parametrize("method", [RHF, ROHF, UHF, CUHF, GHF])
def test_explicit_irrep_occupations(method):
    if method is RHF:
        options = dict(charge=0, irrep_occupations={"b1u": 1})
        expected = "ag"
    elif method is GHF:
        options = dict(charge=0, irrep_occupations={"ag": 1, "b1u": 1})
        expected = "b1u"
    else:
        options = dict(charge=1, ms=0.5, irrep_occupations={"b1u": (1, 0)})
        expected = "b1u"
    hf = method(**options)(_h2()).run()
    assert hf.state_symmetry == expected
    occupied = hf.irrep_labels[0][: 2 if method is GHF else 1]
    assert sorted(occupied) == (["ag", "b1u"] if method is GHF else ["b1u"])
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
    hf = method(charge=0, ms=0.5, irrep_occupations={"ag": (1, 1), "b2u": (1, 0)})(
        system
    ).run()
    assert hf.irrep_labels[0][:2] == ["ag", "b2u"]
    assert hf.state_symmetry == "b2u"
    for spin, nocc in enumerate((2, 1)):
        C = hf.C[min(spin, len(hf.C) - 1)]
        np.testing.assert_allclose(hf.D[spin], C[:, :nocc] @ C[:, :nocc].T, atol=1e-12)


@pytest.mark.parametrize("j_adapt", [False, True])
@pytest.mark.parametrize(
    "option", [dict(target_symmetry="e1/2u"), dict(irrep_occupations={"e1/2u": 1})]
)
def test_spin_orbit_ghf_double_group(j_adapt, option):
    system = _h2(x2c=X2CParams(x2c_type="so", x2c_model="1e"))
    hf = GHF(charge=1, j_adapt=j_adapt, **option)(system).run()
    assert hf.orbital_point_group == "D2H*"
    assert hf.state_symmetry == "e1/2u"
    assert hf.irrep_labels[0][0] == "e1/2u"
    assert hf.state_symmetry_weights["e1/2u"] == pytest.approx(1, abs=1e-10)
    _assert_spinor_projectors_in_real_space(hf)


def _assert_spinor_projectors_in_real_space(hf):
    group = hf._double_group
    C = hf.C[0]
    points = np.random.default_rng(265).uniform(-3, 3, (30, 3))

    def evaluate(points):
        return np.array(
            [
                ints.orbitals_at_points(
                    hf.system.basis, points, np.ascontiguousarray(c.real)
                )
                + 1j
                * ints.orbitals_at_points(
                    hf.system.basis, points, np.ascontiguousarray(c.imag)
                )
                for c in np.split(C, 2)
            ]
        ).transpose(0, 2, 1)

    values = evaluate(points)
    transformed = np.array(
        [
            np.einsum("st,tip->sip", spin_rotation(op), evaluate(points @ R.T))
            for op, R in get_symmetry_ops(group.spatial_group).items()
        ]
    )
    transformed = np.concatenate((transformed, -transformed))
    for irrep in group.fermion_parities:
        projected = (
            group.dimensions[irrep]
            * np.einsum("g,gsip->sip", group.characters[irrep].conj(), transformed)
            / group.order
        )
        expected = values.copy()
        expected[:, np.array(hf.irrep_indices[0]) != irrep] = 0
        np.testing.assert_allclose(projected, expected, atol=1e-9, rtol=0)


def test_spin_orbit_even_totally_symmetric_determinant():
    hf = GHF(charge=0, target_symmetry="ag")(
        _h2(x2c=X2CParams(x2c_type="so", x2c_model="1e"))
    ).run()
    assert hf.state_symmetry == "ag"
    assert hf.state_symmetry_weights["ag"] == pytest.approx(1, abs=1e-10)
    Cocc = hf.C[0][:, :2]
    S = hf.system.ints_overlap()
    for U in hf._symmetry_basis.U_ops.values():
        np.testing.assert_allclose(
            U @ Cocc, Cocc @ (Cocc.conj().T @ S @ U @ Cocc), atol=1e-10
        )


def test_spin_orbit_mixed_determinant_is_not_mislabeled():
    hf = GHF(charge=0, irrep_occupations={"e1/2g": 1, "e1/2u": 1})(
        _h2(x2c=X2CParams(x2c_type="so", x2c_model="1e"))
    ).run()
    assert hf.state_symmetry is None
    assert sum(hf.state_symmetry_weights.values()) == pytest.approx(1, abs=1e-10)
    assert sum(w > 1e-6 for w in hf.state_symmetry_weights.values()) >= 2


@pytest.mark.parametrize(
    "xyz,target",
    [
        ("O 0 0 0;H 0 0.76 0.59;H 0 -0.76 0.59", "e1/2"),
        ("Br 0 0 0;Br 0 0 2.3", "e1/2u"),
    ],
)
def test_double_group_with_nonzero_spin_orbit_coupling(xyz, target):
    system = System(
        xyz=xyz,
        basis_set="sto-3g",
        auxiliary_basis_set="def2-universal-jkfit",
        symmetry=True,
        x2c=X2CParams(x2c_type="so", x2c_model="1e"),
    )
    H = system.ints_hcore()
    assert np.linalg.norm(H[: system.nbf, system.nbf :]) > 1e-6
    hf = GHF(charge=1, target_symmetry=target, guess_type="hcore")(system).run()
    assert hf.state_symmetry == target
    assert hf.state_symmetry_weights[target] == pytest.approx(1, abs=1e-9)
    _assert_spinor_projectors_in_real_space(hf)


def test_rebuild_preserves_double_group_options():
    def system(distance):
        return _h2(distance, x2c=X2CParams(x2c_type="so", x2c_model="1e"))

    options = {"e1/2u": 1}
    hf = GHF(charge=1, target_symmetry="e1/2u", irrep_occupations=options)(
        system(1.5)
    ).run()
    rebuilt = rebuild_method_chain(hf, system(1.7)).run()
    rebind_method_chain(hf, system(1.7)).run()
    for method in (hf, rebuilt):
        assert method.state_symmetry == "e1/2u"
        assert method.irrep_occupations == options
        assert method.target_symmetry == "e1/2u"
    assert hf.E == pytest.approx(rebuilt.E, abs=1e-12)


def test_rebind_clears_double_group_state_metadata():
    hf = GHF(charge=0)(_h2(x2c=X2CParams(x2c_type="so", x2c_model="1e"))).run()
    assert hf.state_symmetry_weights
    rebind_method_chain(hf, _h2())
    assert hf.state_symmetry is None
    assert hf.state_symmetry_weights == {}
    hf.run()
    assert hf.state_symmetry == "ag"
    assert hf.state_symmetry_weights == {}


def test_abelian_double_group_target_with_spin_orbit_coupling():
    # trans-diazene has C2h symmetry: four distinct one-dimensional spinor
    # irreps, rather than just gerade/ungerade labels.
    system = System(
        xyz="N -0.6 0 0;N 0.6 0 0;H -1 0.5 0;H 1 -0.5 0",
        basis_set="sto-3g",
        auxiliary_basis_set="def2-universal-jkfit",
        symmetry=True,
        x2c=X2CParams(x2c_type="so", x2c_model="1e"),
    )
    assert system.point_group == "C2H"
    hf = GHF(charge=1, target_symmetry="1e1/2g")(system).run()
    assert hf.state_symmetry == "1e1/2g"
    assert hf.state_symmetry_weights["1e1/2g"] == pytest.approx(1, abs=1e-9)
    assert len(set(hf.irrep_labels[0])) == 4
    _assert_spinor_projectors_in_real_space(hf)


@pytest.mark.parametrize(
    "options,match",
    [
        (dict(charge=0, target_symmetry="e1/2g"), "even counts"),
        (dict(charge=1, target_symmetry="ag"), "Odd electron"),
        (dict(charge=0, target_symmetry="b1g"), "single HF determinant"),
        (
            dict(
                charge=0,
                target_symmetry="ag",
                irrep_occupations={"e1/2g": 1, "e1/2u": 1},
            ),
            "even occupations",
        ),
        (
            dict(charge=1, target_symmetry="e1/2u", irrep_occupations={"e1/2g": 1}),
            "incompatible",
        ),
        (dict(charge=1, irrep_occupations={"ag": 1}), "fermionic"),
        (dict(charge=1, irrep_occupations={"e1/2g": 1, 8: 0}), "more than once"),
    ],
)
def test_impossible_double_group_requests(options, match):
    system = _h2(x2c=X2CParams(x2c_type="so", x2c_model="1e"))
    with pytest.raises(ValueError, match=match):
        GHF(**options)(system)


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


@pytest.mark.parametrize("method", [ROHF, UHF, CUHF])
def test_spin_pair_required(method):
    with pytest.raises(ValueError, match="alpha, beta"):
        method(charge=0, ms=0, irrep_occupations={"ag": 1})


@pytest.mark.parametrize(
    "method,options,match",
    [
        (RHF, dict(target_symmetry="b1u"), "always totally symmetric"),
        (RHF, dict(target_symmetry="invalid"), "Unknown irrep"),
        (RHF, dict(target_symmetry=8), "Unknown irrep"),
        (RHF, dict(irrep_occupations={"ag": 2}), "must sum"),
        (RHF, dict(irrep_occupations={"ag": 1, 0: 0}), "more than once"),
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
        policy = OccupationPolicy("generalized", "D2H", (2,), target, None)
        order = policy.permutations([eps], [irreps])[0]
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
    policy = OccupationPolicy("unrestricted", "C2", (1, 1), 1, None)
    orders = policy.permutations(eps, irreps)
    assert orders[0][0] == 1
    assert orders[1][0] == 0


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
