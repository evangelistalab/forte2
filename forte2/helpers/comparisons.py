import pytest

approx_vtight = lambda x: pytest.approx(x, abs=1e-8)
approx = lambda x: pytest.approx(x, abs=5e-8)
approx_loose = lambda x: pytest.approx(x, rel=1e-8, abs=5e-8)
