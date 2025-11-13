import numpy as np
import pytest

from forte2 import System, RHF, MCOptimizer, State, AVAS, ROHF
from forte2.dsrg import DSRG_MRPT2
from forte2.helpers.comparisons import approx


def test_sf_mrpt2_n2():
    erhf = -108.954140898736
    emcscf = -109.0811491968

    xyz = """
    N 0.0 0.0 0.0
    N 0.0 0.0 2.0
    """

    system = System(
        xyz=xyz,
        basis_set="cc-pVDZ",
        auxiliary_basis_set="cc-pVTZ-JKFIT",
        unit="bohr",
    )
    rhf = RHF(charge=0)(system)
    rhf.run()

    mc = MCOptimizer(
        states=State(nel=14, multiplicity=1, ms=0.0),
        active_orbitals=6,
        core_orbitals=4,
    )(rhf)
    mc.run()

    assert rhf.E == approx(erhf)
    assert mc.E == approx(emcscf)

    dsrg = DSRG_MRPT2(flow_param=0.5, relax_reference="iterate")(mc)
    dsrg.run()

    assert dsrg.relax_energies[0, 0] == approx(-109.23886074061)
    assert dsrg.relax_energies[0, 1] == approx(-109.23931193044)
    assert dsrg.relax_energies[0, 2] == approx(-109.08114919682)

    assert dsrg.relax_energies[1, 0] == approx(-109.23895207574)
    assert dsrg.relax_energies[1, 1] == approx(-109.23895208449)
    assert dsrg.relax_energies[1, 2] == approx(-109.08065641191)

    assert dsrg.relax_energies[2, 0] == approx(-109.23895388557)
    assert dsrg.relax_energies[2, 1] == approx(-109.23895388557)
    assert dsrg.relax_energies[2, 2] == approx(-109.08065911063)


def test_sf_mrpt2_o2_triplet():
    erhf = -149.598290821387
    emcscf = -149.7432638235

    xyz = """
    O 0.0 0.0 0.0
    O 0.0 0.0 1.251
    """

    system = System(
        xyz=xyz,
        basis_set="cc-pVDZ",
        auxiliary_basis_set="cc-pVTZ-JKFIT",
    )
    mf = ROHF(charge=0, ms=1.0)(system)
    avas = AVAS(
        subspace=["O(2p)"],
        selection_method="separate",
        num_active_docc=3,
        num_active_uocc=1,
    )(mf)
    mc = MCOptimizer(
        states=State(nel=16, multiplicity=3, ms=1.0),
    )(avas)
    dsrg = DSRG_MRPT2(flow_param=1.0, relax_reference="twice")(mc)
    dsrg.run()

    assert dsrg.relax_energies[0, 0] == approx(-149.967302979728)
    assert dsrg.relax_energies[0, 1] == approx(-149.968922391320)
    assert dsrg.relax_energies[0, 2] == approx(-149.707577515943)
    assert dsrg.relax_energies[1, 0] == approx(-149.969319174301)
    assert dsrg.relax_energies[1, 1] == approx(-149.969319899594)
    assert dsrg.relax_energies[1, 2] == approx(-149.705500926314)
