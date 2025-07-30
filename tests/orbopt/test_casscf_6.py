import pytest

from forte2 import *
from forte2.helpers.comparisons import approx


def test_casscf_6():
    """Test CASSCF (frozen core orbital) with HF molecule."""
    emcscf = -99.939295399756

    xyz = """
    F            0.000000000000     0.000000000000    -0.075563346255
    H            0.000000000000     0.000000000000     1.424436653745
    """

    system = System(
        xyz=xyz, basis_set="cc-pvdz", auxiliary_basis_set="def2-universal-jkfit"
    )

    rhf = RHF(charge=0, econv=1e-12)(system)
    ci_state = CIStates(
        active_spaces=[4, 5],
        core_orbitals=[0, 1, 2, 3], # orb 0 is frozen in forte test
        states=State(nel=10, multiplicity=1, ms=0.0),
    )
    mc = MCOptimizer(ci_state)(rhf)
    mc.run()

    assert mc.E == approx(emcscf)