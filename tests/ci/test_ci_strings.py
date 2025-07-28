from forte2 import CIStrings, State, MOSpace


def test_ci_strings_singlet():
    # Create a State object
    state = State(nel=10, multiplicity=1, ms=0.0, gas_min=[1, 4, 2], gas_max=[2, 4, 3])
    mospace = MOSpace(
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
        state.symmetry,
        orbital_symmetry,
        state.gas_min,
        state.gas_max,
    )

    assert ci_strings.na == 4
    assert ci_strings.nb == 4
    assert ci_strings.ngas_spaces == 3
    assert ci_strings.gas_size == [1, 2, 3]
    assert ci_strings.gas_alfa_occupations == [[0, 2, 2, 0, 0, 0], [1, 2, 1, 0, 0, 0]]
    assert ci_strings.gas_beta_occupations == [[1, 2, 1, 0, 0, 0], [0, 2, 2, 0, 0, 0]]
    assert ci_strings.gas_occupations == [(0, 0), (1, 1), (1, 0)]


def test_ci_strings_triplet():
    # Create a State object
    state = State(nel=10, multiplicity=3, ms=1.0, gas_min=[0, 3])
    mospace = MOSpace(
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
        state.symmetry,
        orbital_symmetry,
        state.gas_min,
        state.gas_max,
    )

    assert ci_strings.na == 5
    assert ci_strings.nb == 3
    assert ci_strings.ngas_spaces == 3
    assert ci_strings.gas_size == [1, 2, 3]
    assert ci_strings.gas_alfa_occupations == [
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
