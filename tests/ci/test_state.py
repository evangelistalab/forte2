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
