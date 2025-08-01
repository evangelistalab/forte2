import pytest
import numpy as np

from forte2.state import State, StateAverageInfo
from forte2.helpers.comparisons import approx


def test_wrong_args():
    with pytest.raises(Exception):
        StateAverageInfo(states="not a state list")

    with pytest.raises(Exception):
        StateAverageInfo(states=["not", "a state list"])

    state1 = State(nel=2, multiplicity=1, ms=0.0)
    state2 = State(nel=2, multiplicity=3, ms=1.0)
    with pytest.raises(Exception):
        # nroots must be a list for multiple states
        StateAverageInfo(states=[state1, state2], nroots=1)
    with pytest.raises(Exception):
        # nroots cannot be 0
        StateAverageInfo(states=[state1, state2], nroots=[0, 1])
    with pytest.raises(Exception):
        # len(weight) must match the the nroot per state
        StateAverageInfo(
            states=[state1, state2], nroots=[2, 1], weights=[[1.0], [2.0, 3]]
        )
    with pytest.raises(Exception):
        # weights must be a list
        StateAverageInfo(states=[state1, state2], nroots=[2, 1], weights=1.0)
    with pytest.raises(Exception):
        # weights must be a list of lists if multiple states
        StateAverageInfo(states=[state1, state2], nroots=[2, 1], weights=[1.0])
    with pytest.raises(Exception):
        # weights cannot be negative
        StateAverageInfo(
            states=[state1, state2], nroots=[1, 2], weights=[[-1.0], [2.0, 3]]
        )


def test_state_average_info():
    state1 = State(nel=2, multiplicity=1, ms=0.0)
    state2 = State(nel=3, multiplicity=2, ms=0.5)
    state3 = State(nel=4, multiplicity=3, ms=1.0)

    sa_info = StateAverageInfo(states=[state1, state2, state3], nroots=[2, 1, 3])
    assert sa_info.ncis == 3
    assert sa_info.nroots == [2, 1, 3]
    assert sa_info.nroots_sum == 6
    assert len(sa_info.weights) == 3
    assert len(sa_info.weights[0]) == 2
    assert len(sa_info.weights[1]) == 1
    assert len(sa_info.weights[2]) == 3
    assert sa_info.weights_flat.sum() == approx(1.0)
    assert sa_info.weights_flat.shape == (6,)
    assert sa_info.weights_flat == approx(np.ones(6) / 6)

    sa_info = StateAverageInfo(
        states=[state1, state2, state3],
        nroots=[2, 1, 3],
        weights=[[1, 0.3], [4], [0.5, 10, 0.3]],
    )
    weights_flat = np.array([1, 0.3, 4, 0.5, 10, 0.3])
    assert sa_info.weights_flat.sum() == approx(1.0)
    assert sa_info.weights_flat == approx(weights_flat / weights_flat.sum())
