import pytest


import numpy as np
from forte2 import *
from forte2.helpers.comparisons import approx


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
        State(nel=14, multiplicity=1, ms=0.0),
        core_orbitals=[0, 1, 2, 3],
        active_orbitals=[4, 5, 6, 7, 8, 9],
        nroots=1,
        do_transition_dipole=True,
    )(rhf)
    ci.run()

    semi = orbitals.Semicanonicalizer(
        mo_space=ci.mo_space, g1_sf=ci.make_average_sf_1rdm(), C=rhf.C[0], system=system
    )
    e, C = semi.run()
    print(rhf.eps[0])
    print(e)


test_semican_ci()
