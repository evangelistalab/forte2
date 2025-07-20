import pytest

from forte2 import *
from forte2.helpers.comparisons import approx


def test_mcscf_casscf_2():
    erhf = -99.87284684762975
    emcscf = -99.939295399756

    xyz = f"""
    F            0.000000000000     0.000000000000    -0.075563346255
    H            0.000000000000     0.000000000000     1.424436653745
    """

    system = System(xyz=xyz, basis_set="cc-pvdz", auxiliary_basis_set="def2-universal-jkfit")

    rhf = RHF(charge=0, econv=1e-12)(system)
    ci = CI(
        orbitals=[4,5],
        core_orbitals=[0,1,2,3],
        state=State(nel=10, multiplicity=1, ms=0.0),
        nroot=1
    )(rhf)
    mc = MCOptimizer()(ci)
    mc.maxiter = 400
    mc.run()

    assert rhf.E == approx(erhf)
    assert mc.E == approx(emcscf)


def test_mcscf_casscf_5():
    erhf = -226.40943786499565
    emcscf = -226.575743550979

    xyz = f"""
    C   0.0000000000  -2.5451795941   0.0000000000
    C   0.0000000000   2.5451795941   0.0000000000
    C  -2.2828001669  -1.3508352528   0.0000000000
    C   2.2828001669  -1.3508352528   0.0000000000
    C   2.2828001669   1.3508352528   0.0000000000
    C  -2.2828001669   1.3508352528   0.0000000000
    H  -4.0782187459  -2.3208602146   0.0000000000
    H   4.0782187459  -2.3208602146   0.0000000000
    H   4.0782187459   2.3208602146   0.0000000000
    H  -4.0782187459   2.3208602146   0.0000000000
    """

    system = System(xyz=xyz, basis_set="sto-3g", auxiliary_basis_set="def2-universal-jkfit", unit="bohr")

    rhf = RHF(charge=0, econv=1)(system)
    ci = CI(
        # frozen_core=[0,1,2,3,4,5],
        core_orbitals=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18],
        orbitals=[19,20],
        state=State(nel=40, multiplicity=1, ms=0.0),
        nroot=1,
    )(rhf)
    mc = MCOptimizer()(ci)
    mc.run()

    assert rhf.E == approx(erhf)
    assert mc.E == approx(emcscf)


def test_mcscf_casscf_9():
    erhf = -100.00987356244829
    emcscf_root_1 = -100.038638058262
    emcscf_root_2 = -99.004019251301
    emcscf_root_3 = -98.457781187985
    emcscf_root_4 = -98.427043646164

    xyz = f"""
    H            0.000000000000     0.000000000000    -0.949624435830
    F            0.000000000000     0.000000000000     0.050375564170
    """

    system = System(xyz=xyz, basis_set="cc-pvdz", auxiliary_basis_set="cc-pvtz-jkfit")

    rhf = RHF(charge=0, econv=1e-12)(system)
    ci = CI(
        orbitals=[0,1,2,3,4,5],
        state=State(nel=10, multiplicity=1, ms=0.0),
        nroot=4,
    )(rhf)
    mc = MCOptimizer()(ci)
    mc.run()

    assert rhf.E == approx(erhf)
    assert mc.E == approx(emcscf_root_1)
    # assert mc.E[0] == approx(emcscf_root_1)
    # assert mc.E[1] == approx(emcscf_root_2)
    # assert mc.E[2] == approx(emcscf_root_3)
    # assert mc.E[3] == approx(emcscf_root_4)