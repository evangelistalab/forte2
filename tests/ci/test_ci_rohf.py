from forte2 import *
from forte2.helpers.comparisons import approx


def test_rohf_ci_1():
    xyz = """
    H 0.0 0.0 0.0
    F 0.0 0.0 2.0
    """

    system = System(
        xyz=xyz, basis_set="cc-pVDZ", auxiliary_basis_set="cc-pVTZ-JKFIT", unit="bohr"
    )
    rhf = ROHF(charge=1, ms=0.5, econv=1e-12)(system)
    ci = CI(
        states=State(system=system, charge=1, multiplicity=2, ms=0.5),
        core_orbitals=[0],
        active_orbitals=[1, 2, 3, 4, 5, 6],
        nroots=2,
    )(rhf)
    ci.run()

    assert ci.E[0] == approx(-99.510706628367)


def test_rohf_ci_2():
    xyz = """
    H 0.0 0.0 0.0
    F 0.0 0.0 2.0
    """

    system = System(
        xyz=xyz, basis_set="cc-pVDZ", auxiliary_basis_set="cc-pVTZ-JKFIT", unit="bohr"
    )
    rhf = ROHF(charge=1, ms=-0.5, econv=1e-12)(system)
    ci = CI(
        active_orbitals=[1, 2, 3, 4, 5, 6],
        core_orbitals=[0],
        states=State(system=system, charge=1, multiplicity=2, ms=-0.5),
        nroots=2,
    )(rhf)
    ci.run()

    assert ci.E[0] == approx(-99.510706628367)
