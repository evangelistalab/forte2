from logging import fatal
import pytest

from forte2 import *
from forte2.helpers.comparisons import approx


def test_gasscf_1():
    erhf = -76.05702512779526
    emcscf = -76.115688424591

    xyz = """
    O   0.0000000000  -0.0000000000  -0.0662628033
    H   0.0000000000  -0.7540256101   0.5259060578
    H  -0.0000000000   0.7530256101   0.5260060578
    """

    system = System(
        xyz=xyz, basis_set="cc-pVTZ", auxiliary_basis_set="def2-universal-jkfit"
    )

    rhf = RHF(charge=0, econv=1e-12, dconv=1e-12)(system)

    mc = MCOptimizer(
        State(nel=10, multiplicity=1, ms=0.0, gas_min=[3], gas_max=[6]),
        core_orbitals=[0, 1],
        active_orbitals=[[2, 3, 4], [5, 6, 7]],
        econv=1e-8,
        gconv=1e-7,
        maxiter=500,
    )(rhf)
    mc.run()

    assert rhf.E == approx(erhf)
    assert mc.E == approx(emcscf)


def test_gasscf_2():
    erhf = -76.05702512779526
    emcscf = -76.078664204560

    xyz = """
    O   0.0000000000  -0.0000000000  -0.0662628033
    H   0.0000000000  -0.7540256101   0.5259060578
    H  -0.0000000000   0.7530256101   0.5260060578
    """

    system = System(
        xyz=xyz, basis_set="cc-pVTZ", auxiliary_basis_set="def2-universal-jkfit"
    )

    rhf = RHF(charge=0, econv=1e-12, dconv=1e-12)(system)

    mc = MCOptimizer(
        State(nel=10, multiplicity=1, ms=0.0, gas_min=[3], gas_max=[6]),
        core_orbitals=[0, 1],
        active_orbitals=[[2, 3, 4], [5]],
        do_diis=False,
        econv=1e-8,
        gconv=1e-7,
        maxiter=100,
    )(rhf)
    mc.run()

    assert rhf.E == approx(erhf)
    assert mc.E == approx(emcscf)


@pytest.mark.slow
def test_gasscf_3():
    erhf = -40.21254161940163
    emcscf = -29.702820676605

    xyz = """
    C            0.052417904862     0.008091170764     0.039717608738
    H            1.170469902710     0.450548753521     1.737652542463
    H            1.167683656136     0.451231405150    -1.659924552573
    H           -1.690971530149     1.143552334703     0.040997211401
    H           -0.436869468885    -2.014192012525     0.039431679323
    """

    system = System(
        xyz=xyz,
        basis_set="cc-pVTZ",
        auxiliary_basis_set="def2-universal-jkfit",
        unit="bohr",
    )

    rhf = RHF(charge=0, econv=1e-12, dconv=1e-12)(system)

    mc = MCOptimizer(
        State(nel=10, multiplicity=1, ms=0.0, gas_min=[1], gas_max=[1]),
        active_orbitals=[[0], [1, 2, 3, 4, 5, 6, 7, 8]],
        do_diis=True,
        econv=1e-8,
        gconv=1e-7,
        maxiter=100,
    )(rhf)
    mc.run()

    assert rhf.E == approx(erhf)
    assert mc.E == approx(emcscf)


def test_gasscf_5():
    erhf = -76.02146209546578
    emcscf = -76.077753286787

    xyz = """
    O            0.000000000000     0.000000000000    -0.069592187400
    H            0.000000000000    -0.783151105291     0.552239257834
    H            0.000000000000     0.783151105291     0.552239257834
    """

    system = System(
        xyz=xyz, basis_set="cc-pVDZ", auxiliary_basis_set="def2-universal-jkfit"
    )

    rhf = RHF(charge=0, econv=1e-12, dconv=1e-12)(system)

    mc = MCOptimizer(
        states=State(nel=10, multiplicity=1, ms=0.0, gas_min=[6, 0], gas_max=[8, 2]),
        core_orbitals=[0],
        active_orbitals=[[1, 2, 3, 4], [5, 6]],
        maxiter=200,
        econv=1e-10,
        gconv=1e-8,
    )(rhf)
    mc.run()

    assert rhf.E == approx(erhf)
    assert mc.E == approx(emcscf)


test_gasscf_5()
