import pytest
import numpy as np

approx_vtight = lambda x: pytest.approx(x, abs=1e-8)
approx = lambda x: pytest.approx(x, abs=5e-8)
approx_loose = lambda x: pytest.approx(x, rel=1e-8, abs=5e-8)
approx_abs = lambda x, atol: pytest.approx(x, abs=atol)
is_diagonal_matrix = lambda x: np.allclose(x, np.diag(np.diag(x)), atol=1e-6)
