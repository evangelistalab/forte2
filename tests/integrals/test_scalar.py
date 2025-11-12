from scipy.linalg import eigh
import pytest

import forte2
from forte2.helpers.comparisons import approx


def test_nuclear_repulsion_1():
    xyz = "H 0.0 0.0 0.0"

    system = forte2.System(xyz=xyz, basis_set="cc-pvdz")

    assert forte2.integrals.nuclear_repulsion(system) == approx(0.0)


def test_nuclear_repulsion_2():
    xyz = """
    No 0.0 0.0 0.0
    Sg 0.1 0.0 0.0
    """

    system = forte2.System(
        xyz=xyz,
        basis_set="sap_helfem_large",
        minao_basis_set=None,
        unit="bohr",
    )

    assert forte2.integrals.nuclear_repulsion(system) == approx(102 * 106 * 10)


def test_nuclear_repulsion_gaussian():
    xyz = """
    No 0.0 0.0 0.0
    Sg 0.1 0.0 0.0
    """

    system = forte2.System(
        xyz=xyz,
        basis_set="sap_helfem_large",
        minao_basis_set=None,
        unit="bohr",
        use_gaussian_charges=True,
    )

    assert forte2.integrals.nuclear_repulsion(system) == approx(108119.99988681375)


def test_nuclear_repulsion_collision():
    xyz = """
    H 0.0 0.0 0.0
    H 0.0 0.0 0.0
    """

    with pytest.raises(Exception):
        system = forte2.System(xyz=xyz, basis_set="cc-pvdz")
