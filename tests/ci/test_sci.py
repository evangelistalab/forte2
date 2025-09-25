from dataclasses import dataclass, field

from forte2 import System, State
from forte2.scf import RHF
from forte2.ci import SelectedCI


def test_sci1():
    xyz = f"""
    H 0.0 0.0 0.0
    H 0.0 0.0 {0.529177210903 * 2}
    """

    system = System(xyz=xyz, basis_set="cc-pVDZ", auxiliary_basis_set="cc-pVTZ-JKFIT")

    rhf = RHF(charge=0, econv=1e-12)(system)

    sci = SelectedCI(
        states=State(nel=2, multiplicity=1, ms=0.0),
        active_orbitals=list(range(10)),
        ci_algorithm="exact",
        threshold=1e-3,
    )(rhf)

    sci.run()


test_sci1()
