import pytest

from numpy import isclose
from forte2 import *

@pytest.mark.skip(reason="no working mcscf solver yet")
def test_mcscf_1():
    erhf = -1.08928367118043
    emcscf = -1.11873740345286

    xyz = f"""
    H 0.0 0.0 0.0
    H 0.0 0.0 {0.529177210903 * 2}
    """

    system = System(xyz=xyz, basis="cc-pvdz", auxiliary_basis="cc-pVTZ-JKFIT")

    rhf = RHF(charge=0, econv=1e-12)(system)
    ci = CI(
        orbitals=[0, 1],
        state=State(nel=2, multiplicity=1, ms=0.0),
        nroot=1,
    )(rhf)
    oo = OrbitalOptimizer()(ci)
    oo.run()

    assert isclose(rhf.E, erhf)
    assert isclose(oo.E[0], emcscf)

@pytest.mark.skip(reason="no working mcscf solver yet")
def test_mcscf_2():
    erhf = -99.9977252002946
    emcscf = -100.043501894947

    xyz = f"""
    H 0.0 0.0 0.0
    F 0.0 0.0 2.0
    """

    system = System(
        xyz=xyz, basis="cc-pVDZ", auxiliary_basis="cc-pVTZ-JKFIT", unit="bohr"
    )
    rhf = RHF(charge=0, econv=1e-12)(system)
    ci = CI(
        orbitals=[1, 2, 3, 4, 5, 6],
        core_orbitals=[0],
        state=State(nel=10, multiplicity=1, ms=0.0),
        nroot=1,
    )(rhf)
    oo = OrbitalOptimizer()(ci)
    oo.run()

    assert isclose(rhf.E, erhf)
    assert isclose(oo.E[0], emcscf)