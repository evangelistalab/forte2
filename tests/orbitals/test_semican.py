import pytest


import numpy as np
from forte2 import *
from forte2.helpers.comparisons import approx


def test_semican_rhf():
    # Semicanonicalized RHF eigenvalues should be strictly identical to the RHF eigenvalues
    xyz = f"""
    N 0.0 0.0 -1.0
    N 0.0 0.0 1.0
    """

    system = System(
        xyz=xyz, basis_set="cc-pVDZ", auxiliary_basis_set="cc-pVTZ-JKFIT", unit="bohr"
    )
    rhf = RHF(charge=0, econv=1e-12)(system)
    rhf.run()
    mo_space = MOSpace(
        core_orbitals=[0, 1, 2, 3], active_orbitals=[4, 5, 6, 7, 8, 9], nmo=system.nmo
    )

    semi = orbitals.Semicanonicalizer(
        mo_space=mo_space, g1_sf=np.diag([2, 2, 2, 0, 0, 0]), C=rhf.C[0], system=system
    )
    semi = semi.run()
    assert rhf.eps[0] == approx(semi.eps_semican)


def test_semican_ci():
    xyz = f"""
    N 0.0 0.0 -1.0
    N 0.0 0.0 1.0
    """

    system = System(
        xyz=xyz, basis_set="cc-pVDZ", auxiliary_basis_set="cc-pVTZ-JKFIT", unit="bohr"
    )
    rhf = RHF(charge=0, econv=1e-12)(system)
    ci = CI(
        State(nel=rhf.nel, multiplicity=1, ms=0.0),
        core_orbitals=[0, 1, 2, 3],
        active_orbitals=[4, 5, 6, 7, 8, 9],
    )(rhf)
    ci.run()

    semi = orbitals.Semicanonicalizer(
        mo_space=ci.mo_space, g1_sf=ci.make_average_sf_1rdm(), C=rhf.C[0], system=system
    )
    semi = semi.run()
    print(rhf.eps[0])
    print(semi.eps_semican)


test_semican_ci()
