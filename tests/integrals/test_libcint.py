import numpy as np
import pytest

from forte2 import System, integrals


def test_libcint_overlap():
    xyz = """
    Li 0 0 0
    Li 0 0 1.9
    """
    system = System(xyz, basis_set="sto-3g")
    s = integrals.cint_overlap(system)
    assert np.linalg.norm(s) == pytest.approx(3.6556110774906956, rel=1e-6)


def test_libcint_kinetic():
    xyz = """
    Li 0 0 0
    Li 0 0 1.9
    """
    system = System(xyz, basis_set="sto-3g")
    s = integrals.cint_kinetic(system)
    assert np.linalg.norm(s) == pytest.approx(5.128923795496629, rel=1e-6)


def test_libcint_nuclear():
    xyz = """
    Ag 0 0 0
    Ag 0 0 1
    """
    system = System(
        xyz, basis_set="sto-3g", minao_basis_set=None, use_gaussian_charges=True
    )
    s = integrals.cint_nuclear(system)
    # from pyscf, using sto-3g downloaded from bse (the built-in one has different parameters..)
    assert np.linalg.norm(s) == pytest.approx(
        np.linalg.norm(3687.189758783181), rel=1e-6
    )
