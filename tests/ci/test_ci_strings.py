import pytest
from forte2 import MOSpace, State
from forte2.lib.ci_helpers import CIStrings
from forte2.lib.det import Determinant


def test_ci_strings_input():
    # negative number of electrons
    with pytest.raises(ValueError):
        ci_strings = CIStrings(-1, 0, 1, 0, [[]], [], [])

    with pytest.raises(ValueError):
        ci_strings = CIStrings(-1, -1, 1, 0, [[]], [], [])

    with pytest.raises(ValueError):
        ci_strings = CIStrings(1, -1, 1, 0, [[]], [], [])

    # wrong type (list instead of list[list]) for orbital_symmetry
    with pytest.raises(TypeError):
        ci_strings = CIStrings(1, 1, 1, 0, [0], [], [])

    for orbital_irrep in (-1, 2):
        with pytest.raises(ValueError, match="Orbital irreps must be non-negative"):
            CIStrings(1, 0, 2, 0, [[orbital_irrep]], [], [])


@pytest.mark.parametrize("orbital_symmetry", [[], [[]], [[], []]])
def test_ci_strings_empty_active_space(orbital_symmetry):
    vacuum = CIStrings(0, 0, 1, 0, orbital_symmetry, [], [])
    assert vacuum.gas_size == [0] * max(1, len(orbital_symmetry))
    assert list(vacuum.make_determinants()) == [Determinant.zero()]

    with_electrons = CIStrings(1, 0, 1, 0, orbital_symmetry, [], [])
    assert with_electrons.ndet == 0


@pytest.mark.parametrize("empty_slot", [0, 1, 2])
def test_ci_strings_preserves_empty_gas_slot(empty_slot):
    orbital_symmetry = [[0], [0]]
    orbital_symmetry.insert(empty_slot, [])
    ci_strings = CIStrings(1, 1, 1, 0, orbital_symmetry, [], [])

    assert ci_strings.ngas_spaces == 3
    assert ci_strings.gas_size == [len(space) for space in orbital_symmetry]
    assert set(ci_strings.make_determinants()) == {
        Determinant(occupation) for occupation in ("20", "02", "ab", "ba")
    }


def test_ci_strings_gas_bounds_with_empty_slot():
    after_empty = CIStrings(1, 1, 1, 0, [[0], [], [0]], [0, 0, 2], [0, 0, 2])
    assert list(after_empty.make_determinants()) == [Determinant("02")]

    impossible = CIStrings(1, 1, 1, 0, [[0], []], [0, 1], [])
    assert impossible.ndet == 0


def test_ci_strings_gas_space_limit():
    assert CIStrings(1, 0, 1, 0, [[0]] * 6, [], []).ndet == 6
    with pytest.raises(ValueError, match="At most 6 GAS spaces"):
        CIStrings(1, 0, 1, 0, [[0]] * 7, [], [])


@pytest.mark.parametrize("na, nb, expected", [(1, 1, ("ab", "ba")), (2, 0, ("aa",))])
def test_ci_strings_product_irrep_missing_from_orbitals(na, nb, expected):
    ci_strings = CIStrings(na, nb, 4, 3, [[1], [2]], [], [])

    assert set(ci_strings.make_determinants()) == {Determinant(det) for det in expected}


def test_ci_strings_unreachable_target_irrep():
    assert CIStrings(1, 0, 4, 3, [[0]], [], []).ndet == 0


def test_ci_strings_target_irrep_outside_range():
    with pytest.raises(ValueError, match="Target symmetry is outside the irrep range"):
        CIStrings(1, 0, 2, 3, [[0]], [], [])


def test_ci_strings_empty_active_space():
    ci_strings = CIStrings(0, 0, 0, [[]], [], [])

    assert ci_strings.ngas_spaces == 1
    assert ci_strings.gas_alpha_occupations == [[0] * 6]
    assert ci_strings.gas_beta_occupations == [[0] * 6]
    assert ci_strings.gas_occupations == [(0, 0)]
    assert ci_strings.ndet == 1


def test_ci_strings_singlet():
    # Create a State object
    state = State(nel=10, multiplicity=1, ms=0.0, gas_min=[1, 4, 2], gas_max=[2, 4, 3])
    mospace = MOSpace(
        nmo=10,
        active_orbitals=[[1], [2, 3], [4, 5, 6]],
        core_orbitals=[0],
    )
    orbital_symmetry = [
        [0] * len(mospace.active_orbitals[x]) for x in range(mospace.ngas)
    ]

    # Generate CI strings for the state
    ci_strings = CIStrings(
        state.na - mospace.ncore,
        state.nb - mospace.ncore,
        1,
        state.symmetry,
        orbital_symmetry,
        state.gas_min,
        state.gas_max,
    )

    assert ci_strings.na == 4
    assert ci_strings.nb == 4
    assert ci_strings.ngas_spaces == 3
    assert ci_strings.gas_size == [1, 2, 3]
    assert ci_strings.gas_alpha_occupations == [[0, 2, 2, 0, 0, 0], [1, 2, 1, 0, 0, 0]]
    assert ci_strings.gas_beta_occupations == [[1, 2, 1, 0, 0, 0], [0, 2, 2, 0, 0, 0]]
    assert ci_strings.gas_occupations == [(0, 0), (1, 1), (1, 0)]


def test_ci_strings_triplet():
    # Create a State object
    state = State(nel=10, multiplicity=3, ms=1.0, gas_min=[0, 3])
    mospace = MOSpace(
        nmo=10,
        active_orbitals=[[1], [2, 3], [4, 5, 6]],
        core_orbitals=[0],
    )
    orbital_symmetry = [
        [0] * len(mospace.active_orbitals[x]) for x in range(mospace.ngas)
    ]

    # Generate CI strings for the state
    ci_strings = CIStrings(
        state.na - mospace.ncore,
        state.nb - mospace.ncore,
        1,
        state.symmetry,
        orbital_symmetry,
        state.gas_min,
        state.gas_max,
    )

    assert ci_strings.na == 5
    assert ci_strings.nb == 3
    assert ci_strings.ngas_spaces == 3
    assert ci_strings.gas_size == [1, 2, 3]
    assert ci_strings.gas_alpha_occupations == [
        [0, 2, 3, 0, 0, 0],
        [1, 1, 3, 0, 0, 0],
        [1, 2, 2, 0, 0, 0],
    ]
    assert ci_strings.gas_beta_occupations == [
        [0, 1, 2, 0, 0, 0],
        [0, 2, 1, 0, 0, 0],
        [1, 1, 1, 0, 0, 0],
        [1, 2, 0, 0, 0, 0],
    ]
    assert ci_strings.gas_occupations == [
        (0, 0),
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 1),
        (2, 0),
        (2, 1),
        (1, 3),
        (2, 2),
        (2, 3),
    ]


def test_ci_strings_symmetry_uses_all_set_bits():
    orbital_symmetry = [[i % 4 for i in range(59)], [1], [2], [3], [1], [2]]
    target_symmetry = 3
    ci_strings = CIStrings(
        3,
        3,
        4,
        target_symmetry,
        orbital_symmetry,
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
    )

    mo_symmetry = [sym for gas_symmetry in orbital_symmetry for sym in gas_symmetry]

    def direct_symmetry(det):
        sym = 0
        for p, irrep in enumerate(mo_symmetry):
            if det.na(p):
                sym ^= irrep
            if det.nb(p):
                sym ^= irrep
        return sym

    dets = ci_strings.make_determinants()

    assert dets
    assert all(det.na(63) != det.nb(63) for det in dets)
    assert all(direct_symmetry(det) == target_symmetry for det in dets)
