from forte2 import *
from forte2.helpers.comparisons import approx


def test_fci_co_o_core_exc():
    efci = -92.133019235463
    xyz = f"""
    C 0 0 0
    O 0 0 1
    """

    system = System(xyz=xyz, basis="sto-6g", auxiliary_basis="cc-pVTZ-JKFIT")

    rhf = RHF(charge=0, econv=1e-12)(system)
    ci = CI(
        orbitals=list(range(system.nbf)),
        state=State(nel=14, multiplicity=1, ms=0.0),
        nroot=1,
        energy_shift=-92,
    )(rhf)
    ci.run()
    assert ci.E[0] == approx(efci)


def test_fci_co_c_core_exc():
    efci = -101.499040903802
    xyz = f"""
    C 0 0 0
    O 0 0 1
    """

    system = System(xyz=xyz, basis="sto-6g", auxiliary_basis="cc-pVTZ-JKFIT")

    rhf = RHF(charge=0, econv=1e-12)(system)
    ci = CI(
        orbitals=list(range(system.nbf)),
        state=State(nel=14, multiplicity=1, ms=0.0),
        nroot=1,
        energy_shift=-102,
    )(rhf)
    ci.run()
    assert ci.E[0] == approx(efci)
