import pytest
import forte2


def test_sparse_state():
    # Create a SparseState object
    sparse_state = forte2.SparseState()
    d = forte2.Determinant("2+-02-+0")
    sparse_state.add(d, 0.5)
    print(sparse_state.items())
