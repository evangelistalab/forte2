import contextlib
import io

import numpy as np

from forte2 import System
from forte2.orbitals.extents import orbital_extents


def test_orbital_extents_indices_and_no_debug_print():
    # Regression: the `indices` parameter was declared but never used (subset
    # requests were silently ignored, returning full-size arrays), and the
    # function emitted an unconditional `print(f"C.shape = ...")` on every call.
    system = System(
        xyz="N 0 0 0\nN 0 0 2.0",
        basis_set="sto-3g",
        auxiliary_basis_set="def2-universal-JKFIT",
        unit="bohr",
    )
    nbf = system.nbf
    C = np.eye(nbf)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        coords_all, moments_all = orbital_extents(system.basis, C)
        coords_sub, moments_sub = orbital_extents(system.basis, C, indices=[0, 1, 2])

    # No stray debug output.
    assert buf.getvalue().strip() == ""

    # Subset selection is honored.
    assert coords_all.shape == (nbf, 3)
    assert coords_sub.shape == (3, 3)
    assert moments_sub.shape == (3, 6)
    assert np.allclose(coords_sub, coords_all[[0, 1, 2]])
    assert np.allclose(moments_sub, moments_all[[0, 1, 2]])
