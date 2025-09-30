from dataclasses import dataclass, field

from forte2 import System, State
from forte2.scf import RHF
from forte2.ci import SelectedCI, CI
from forte2.helpers.comparisons import approx


def test_sci1():
    xyz = f"""
    H 0.0 0.0 0.0
    H 0.0 0.0 1.0
    H 0.0 0.0 2.0
    H 0.0 0.0 3.0
    """

    efci = -2.180967812920

    system = System(xyz=xyz, basis_set="sto-6g", auxiliary_basis_set="cc-pVTZ-JKFIT")

    rhf = RHF(charge=0, econv=1e-14)(system)

    sci = SelectedCI(
        states=State(nel=4, multiplicity=1, ms=0.0),
        active_orbitals=list(range(4)),
        selection_algorithm="hbci",
        threshold=1e-12,
    )(rhf)
    sci.run()

    assert sci.E[0] == approx(efci)


def test_sci2():
    xyz = f"""
    H 0.0 0.0 0.0
    H 0.0 0.0 1.0
    H 0.0 0.0 2.0
    H 0.0 0.0 3.0
    H 0.0 0.0 4.0
    H 0.0 0.0 5.0
    """

    efci = -3.3213221642

    system = System(xyz=xyz, basis_set="cc-pVDZ", auxiliary_basis_set="cc-pVTZ-JKFIT")

    rhf = RHF(charge=0, econv=1e-10)(system)

    ci = CI(
        states=State(nel=6, multiplicity=1, ms=0.0),
        active_orbitals=list(range(12)),
    )(rhf)
    ci.run()

    sci = SelectedCI(
        states=State(nel=6, multiplicity=1, ms=0.0),
        active_orbitals=list(range(12)),
        selection_algorithm="hbci",
        threshold=1e-6,
    )(rhf)

    sci.run()

    assert sci.E[0] == approx(efci)


test_sci2()
