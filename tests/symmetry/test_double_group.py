from itertools import combinations

import numpy as np
import pytest

from forte2.scf.occupations import DoubleGroupOccupationPolicy
from forte2.symmetry.double_group import DoubleGroup, DoubleGroupBasis, spin_rotation
from forte2.symmetry.sym_utils import CHARACTER_TABLE, COTTON_LABELS


def _representation(group):
    spatial = np.array(
        [
            CHARACTER_TABLE[group.spatial_group][label]
            for label in COTTON_LABELS[group.spatial_group]
        ]
    )
    unbarred = {
        op: np.kron(spin_rotation(op), np.diag(spatial[:, i]))
        for i, op in enumerate(group.operations)
    }
    return unbarred | {"bar_" + op: -U for op, U in unbarred.items()}


@pytest.mark.parametrize("point_group", list(COTTON_LABELS))
def test_double_group_characters_and_spin_representation(point_group):
    group = DoubleGroup(point_group)
    np.testing.assert_allclose(
        group.characters @ group.characters.conj().T / group.order,
        np.eye(len(group.labels)),
        atol=1e-12,
    )
    assert np.sum(group.dimensions**2) == group.order
    np.testing.assert_array_equal(
        group.products @ group.dimensions,
        group.dimensions[:, None] * group.dimensions[None, :],
    )
    U = np.array(list(_representation(group).values()))
    for left in U:
        for right in U:
            assert np.min(np.linalg.norm(U - left @ right, axis=(1, 2))) < 1e-12
    # The central 2pi rotation changes every single-electron spinor's sign.
    np.testing.assert_allclose(U[len(group.operations)], -U[0], atol=1e-12)


@pytest.mark.parametrize("point_group", list(COTTON_LABELS))
def test_double_group_projectors_resolve_random_complex_orbitals(point_group):
    group = DoubleGroup(point_group)
    U_ops = _representation(group)
    size = next(iter(U_ops.values())).shape[0]
    rng = np.random.default_rng(265)
    X = np.linalg.qr(
        rng.normal(size=(size, size)) + 1j * rng.normal(size=(size, size))
    )[0]
    basis = DoubleGroupBasis(group, np.eye(size), X, U_ops)
    F = rng.normal(size=(size, size)) + 1j * rng.normal(size=(size, size))
    F += F.conj().T
    eps, C, irreps = basis.eigh(F)
    np.testing.assert_allclose(C.conj().T @ C, np.eye(size), atol=1e-12)
    for irrep in group.fermion_parities:
        projector = (
            group.dimensions[irrep]
            * sum(
                char.conjugate() * U
                for char, U in zip(group.characters[irrep], U_ops.values())
            )
            / group.order
        )
        np.testing.assert_allclose(projector @ C, C * (irreps == irrep), atol=1e-12)
    # Each single spinor has a pure full double-group irrep, even for 2D irreps.
    for i, irrep in enumerate(irreps):
        weights = group.determinant_weights(C[:, i : i + 1], np.eye(size), U_ops)
        assert weights[irrep] == pytest.approx(1, abs=1e-12)


@pytest.mark.parametrize("point_group", ["C1", "CI", "C2", "CS", "C2H"])
@pytest.mark.parametrize("nel", [1, 2])
def test_abelian_double_group_occupations_match_exhaustive_search(point_group, nel):
    group = DoubleGroup(point_group)
    irreps = np.repeat(list(group.fermion_parities), 2)
    eps = np.random.default_rng(265).normal(size=len(irreps))
    for label, target in group.labels.items():
        if (target in group.fermion_parities) != bool(nel % 2):
            continue
        candidates = []
        for indices in combinations(range(len(eps)), nel):
            symmetry = 0
            for index in indices:
                symmetry = group.abelian_products[symmetry, irreps[index]]
            if symmetry == target:
                candidates.append(eps[list(indices)].sum())
        policy = DoubleGroupOccupationPolicy(group, nel, label, None)
        if not candidates:
            with pytest.raises(ValueError, match="No occupation pattern"):
                policy.permutations([eps], [irreps])
        else:
            selected = policy.permutations([eps], [irreps])[0][:nel]
            assert eps[selected].sum() == pytest.approx(min(candidates), abs=1e-12)
            operations = {
                str(i): np.diag(group.characters[irreps, i]) for i in range(group.order)
            }
            identity = np.eye(len(eps))
            weights = group.determinant_weights(
                identity[:, selected], identity, operations
            )
            assert weights[target] == pytest.approx(1, abs=1e-12)


@pytest.mark.parametrize("point_group", ["D2", "C2V", "D2H"])
@pytest.mark.parametrize("explicit", [False, True])
def test_complete_spinor_multiplets_give_totally_symmetric_determinants(
    point_group, explicit
):
    group = DoubleGroup(point_group)
    U_ops = _representation(group)
    size = next(iter(U_ops.values())).shape[0]
    rng = np.random.default_rng(265)
    F = rng.normal(size=(size, size)) + 1j * rng.normal(size=(size, size))
    F += F.conj().T
    basis = DoubleGroupBasis(group, np.eye(size), np.eye(size), U_ops, paired=True)
    eps, C, irreps = basis.eigh(F)
    # Independently average every group operation. The cached component trace
    # must give the same spectrum as this full finite-group projection.
    Faverage = sum(U.conj().T @ F @ U for U in U_ops.values()) / group.order
    reference = np.concatenate(
        [np.linalg.eigvalsh(X.conj().T @ Faverage @ X) for X in basis.blocks.values()]
    )
    np.testing.assert_allclose(eps, np.sort(reference), atol=1e-12)
    counts = None
    if explicit:
        labels = [
            label
            for label, irrep in group.labels.items()
            if irrep in group.fermion_parities
        ]
        counts = {label: 4 // len(labels) for label in labels}
    policy = DoubleGroupOccupationPolicy(group, 4, 0, counts)
    selected = policy.permutations([eps], [irreps])[0][:4]
    weights = group.determinant_weights(C[:, selected], np.eye(size), U_ops)
    assert weights[0] == pytest.approx(1, abs=1e-12)
    np.testing.assert_allclose(C.conj().T @ C, np.eye(size), atol=1e-12)


def test_incomplete_spinor_multiplet_is_rejected():
    group = DoubleGroup("D2H")
    operations = _representation(group)
    size = next(iter(operations.values())).shape[0]
    with pytest.raises(RuntimeError, match="not closed"):
        DoubleGroupBasis(group, np.eye(size), np.eye(size)[:, :-1], operations)
