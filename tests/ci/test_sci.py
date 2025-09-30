from dataclasses import dataclass, field

from forte2 import System, State
from forte2.scf import RHF
from forte2.ci import SelectedCI, CI


def test_sci1():
    xyz = f"""
    H 0.0 0.0 0.0
    H 0.0 0.0 1.0
    H 0.0 0.0 2.0
    H 0.0 0.0 3.0
    """

    efci = -2.253991839297

    system = System(xyz=xyz, basis_set="sto-6g", auxiliary_basis_set="cc-pVTZ-JKFIT")

    rhf = RHF(charge=0, econv=1e-14)(system)

    # ci = CI(
    #     states=State(nel=4, multiplicity=1, ms=0.0),
    #     active_orbitals=list(range(4)),
    # )(rhf)
    # ci.run()

    sci = SelectedCI(
        states=State(nel=4, multiplicity=1, ms=0.0),
        active_orbitals=list(range(4)),
        ci_algorithm="exact",
        selection_algorithm="hbci",
        threshold=1e-2,
    )(rhf)

    sci.run()

    # -2.180967812920


test_sci1()
