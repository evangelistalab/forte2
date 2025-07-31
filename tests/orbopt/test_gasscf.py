from logging import fatal
import pytest

from forte2 import *
from forte2.helpers.comparisons import approx


@pytest.mark.skip
def test_gasscf_0():
    erhf = -76.02672338005341
    eci = -76.039405150388
    escf = -76.082673161513

    xyz = f"""
	    O   0.0000000000  -0.0000000000  -0.0662628033
	    H   0.0000000000  -0.7530256101   0.5259060578
	    H  -0.0000000000   0.7530256101   0.5259060578
    """

    system = System(
        xyz=xyz, basis_set="cc-pVDZ", auxiliary_basis_set="def2-universal-jkfit"
    )

    rhf = RHF(charge=0, econv=1e-12)(system)
    mc = MCOptimizer(
        State(nel=10, multiplicity=1, ms=0.0, gas_min=[3], gas_max=[6]),
        core_orbitals=[0, 1],
        active_orbitals=[[2, 3, 4], [5, 6, 12], [15,16,20]]
        )(rhf)
    mc.run()

    assert rhf.E == approx(erhf)
    assert mc.E == approx(escf)


@pytest.mark.skip
def test_gasscf_1():
    erhf = -76.02146209546578
    eci = -76.029273794488
    escf = -76.077753286787

    xyz = f"""
    O            0.000000000000     0.000000000000    -0.069592187400
    H            0.000000000000    -0.783151105291     0.552239257834
    H            0.000000000000     0.783151105291     0.552239257834
    """

    system = System(
        xyz=xyz, basis_set="cc-pVDZ", auxiliary_basis_set="def2-universal-jkfit"
    )

    rhf = RHF(charge=0, econv=1e-12)(system)
    ci = CI(
        core_orbitals=[0],
        orbitals=[[2, 3, 4, 5], [6, 7]],
        state=State(
            nel=10,
            multiplicity=1,
            ms=0.0,
            gas_min=[6, 0],
            gas_max=[8, 2],
        ),
        nroot=1,
    )(rhf)

    mc = MCOptimizer(
        maxiter=200,
        diis_start=1,
        max_rotation=0.5,
        micro_maxiter=1,
        etol=1e-10,
        gradtol=1e-8,
    )(ci)
    mc.run()

    assert rhf.E == approx(erhf)
    assert mc.E == approx(escf)


@pytest.mark.skip
def test_gasscf_2():
    erhf = -76.02146209547571
    eci = -55.819535518942
    escf = -76.063794140801

    xyz = f"""
    O            0.000000000000     0.000000000000    -0.069592187400
    H            0.000000000000    -0.783151105291     0.552239257834
    H            0.000000000000     0.783151105291     0.552239257834
    """

    system = System(
        xyz=xyz, basis_set="cc-pVDZ", auxiliary_basis_set="def2-universal-jkfit"
    )

    rhf = RHF(charge=0, econv=1e-12)(system)
    ci = CI(
        orbitals=[[0], [1, 2, 3, 4, 5, 6]],
        state=State(
            nel=10,
            multiplicity=1,
            ms=0.0,
            gas_min=[0],
            gas_max=[1],
        ),
        nroot=1,
    )(rhf)

    mc = MCOptimizer(etol=1e-10)(ci)
    mc.run()

    assert rhf.E == approx(erhf)
    assert mc.E == approx(escf)


@pytest.mark.skip
def test_gasscf_3():
    erohf = -75.62820784535052
    eci = -75.629550216306
    escf = -75.638998108114

    xyz = f"""
    O            0.000000000000     0.000000000000    -0.069592187400
    H            0.000000000000    -0.783151105291     0.552239257834
    H            0.000000000000     0.783151105291     0.552239257834
    """

    system = System(
        xyz=xyz, basis_set="cc-pVDZ", auxiliary_basis_set="def2-universal-jkfit"
    )

    rohf = ROHF(charge=1, econv=1e-12, dconv=1e-8, ms=0.5, do_diis=False)(system)
    ci = CI(
        core_orbitals=[0],
        orbitals=[[2, 3, 4, 5], [6, 7]],
        state=State(
            nel=9,
            multiplicity=2,
            ms=0.5,
            gas_min=[6, 0],
            gas_max=[8, 2],
        ),
        nroot=1,
    )(rohf)

    mc = MCOptimizer(
        diis_start=1,
        max_rotation=0.5,
        micro_maxiter=1,
        etol=1e-10,
        gradtol=1e-8,
    )(ci)
    mc.run()

    assert rohf.E == approx(erohf)
    assert mc.E == approx(escf)
