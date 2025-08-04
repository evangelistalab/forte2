import pytest

import forte2


def test_state_incompatible_spin():
    with pytest.raises(Exception):
        forte2.State(nel=2, multiplicity=1, ms=0.5)
    forte2.State(nel=2, multiplicity=1, ms=0.0)

    with pytest.raises(Exception):
        forte2.State(nel=2, multiplicity=2, ms=0.5)
    forte2.State(nel=2, multiplicity=3, ms=1.0)

    with pytest.raises(Exception):
        forte2.State(nel=2, multiplicity=5, ms=1.5)
    forte2.State(nel=4, multiplicity=5, ms=2.0)


def test_wrong_args():
    with pytest.raises(Exception):
        forte2.State(nel=2, multiplicity=1, ms=1.3)

    with pytest.raises(Exception):
        forte2.State(nel=2, multiplicity=0, ms=0.0)

    with pytest.raises(Exception):
        forte2.State(nel=-2, multiplicity=1, ms=0.0)


def test_state():
    state = forte2.State(nel=2, multiplicity=1, ms=0.0)
    assert state.nel == 2
    assert state.multiplicity == 1
    assert state.ms == 0.0
    assert state.twice_ms == 0
    assert state.na == 1
    assert state.nb == 1

    state = forte2.State(nel=2, multiplicity=3, ms=1.0)
    assert state.nel == 2
    assert state.multiplicity == 3
    assert state.ms == 1.0
    assert state.twice_ms == 2
    assert state.na == 2
    assert state.nb == 0

    state = forte2.State(nel=4, multiplicity=5, ms=2.0)
    assert state.nel == 4
    assert state.multiplicity == 5
    assert state.ms == 2.0
    assert state.twice_ms == 4
    assert state.na == 4
    assert state.nb == 0

    state = forte2.State(nel=4, multiplicity=3, ms=1.0)
    assert state.nel == 4
    assert state.multiplicity == 3
    assert state.ms == 1.0
    assert state.twice_ms == 2
    assert state.na == 3
    assert state.nb == 1
