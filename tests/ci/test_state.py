import pytest
import forte2


def test_sparse_state():
    # Create a SparseState object
    sparse_state = forte2.State()
    d = forte2.Determinant("2+-02-+0")
    sparse_state.add(d, 0.5)
    print(sparse_state.items())


if __name__ == "__main__":
    test_sparse_state()
