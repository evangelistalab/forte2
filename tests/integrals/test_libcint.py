import pytest
import numpy as np

from forte2 import System, ints
from forte2.helpers.comparisons import approx
from forte2.libcint import LIBCINT_AVAILABLE, get_integral


@pytest.mark.skipif(not LIBCINT_AVAILABLE, reason="libcint not available")
def test_libcint_overlap():
    ref = 4.01289400150967
    xyz = """
    Li 0 0 0
    Li 0 0 1
    """
    system = System(xyz, basis_set="sto-3g")
    s_ref = ints.overlap(system.basis)
    s = get_integral(system, "int1e_ovlp_sph")
    assert np.linalg.norm(s_ref) == approx(ref)
    assert np.linalg.norm(s) == approx(ref)
