import pytest

from forte2 import *
from forte2.helpers.comparisons import approx


def test_casscf_n2():
    erhf = -108.761639873604
    ecasscf = -108.9800484156

    xyz = f"""
    C 0.0 0.0 0.0
    O 0.0 0.0 1.2
    """

    system = System(xyz=xyz, basis_set="cc-pVDZ", auxiliary_basis_set="cc-pVTZ-JKFIT")
    rhf = RHF(charge=0, econv=1e-12)(system)
    avas = AVAS(
        selection_method="separate",
        subspace=["C(2p)", "O(2p)"],
        num_active_docc=3,
        num_active_uocc=3,
    )(rhf)
    avas.run()
    avas.mo_space = MOSpace(
        frozen_core_orbitals=[0],
        core_orbitals=[1, 2, 3],
        active_orbitals=[4, 5, 6, 7, 8, 9],
        nmo=system.nmo,
    )
    mc = MCOptimizer(
        State(nel=14, multiplicity=1, ms=0.0),
    )(avas)
    mc.run()
