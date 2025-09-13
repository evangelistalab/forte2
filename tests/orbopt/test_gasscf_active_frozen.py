import pytest

from forte2 import System, RHF, MCOptimizer, State
from forte2.helpers.comparisons import approx


@pytest.mark.slow
def test_gasscf_ch4_active_frozen_1s():
    erhf = -40.19845141292726
    emcscf = -29.570492567892

    xyz = """
    C            0.052417904862     0.008091170764     0.039717608738
    H            1.170469902710     0.450548753521     1.737652542463
    H            1.167683656136     0.451231405150    -1.659924552573
    H           -1.690971530149     1.143552334703     0.040997211401
    H           -0.436869468885    -2.014192012525     0.039431679323
    """

    system = System(
        xyz=xyz,
        basis_set="cc-pVDZ",
        auxiliary_basis_set="def2-universal-jkfit",
        unit="bohr",
    )

    rhf = RHF(charge=0, econv=1e-12, dconv=1e-12)(system)

    mc = MCOptimizer(
        State(nel=10, multiplicity=1, ms=0.0, gas_min=[1], gas_max=[1]),
        active_orbitals=[[0], [1, 2, 3, 4, 5, 6, 7, 8]],
        active_frozen_orbitals=[0],
        do_diis=True,
        econv=1e-8,
        gconv=1e-7,
        maxiter=100,
    )(rhf)
    mc.run()

    assert rhf.E == approx(erhf)
    assert mc.E == approx(emcscf)


@pytest.mark.slow
def test_gasscf_ch4_active_frozen_1s_highest_active():
    erhf = -40.19845141292726
    emcscf = -29.545257048811

    xyz = """
    C            0.052417904862     0.008091170764     0.039717608738
    H            1.170469902710     0.450548753521     1.737652542463
    H            1.167683656136     0.451231405150    -1.659924552573
    H           -1.690971530149     1.143552334703     0.040997211401
    H           -0.436869468885    -2.014192012525     0.039431679323
    """

    system = System(
        xyz=xyz,
        basis_set="cc-pVDZ",
        auxiliary_basis_set="def2-universal-jkfit",
        unit="bohr",
    )

    rhf = RHF(charge=0, econv=1e-12, dconv=1e-12)(system)

    mc = MCOptimizer(
        State(nel=10, multiplicity=1, ms=0.0, gas_min=[1], gas_max=[1]),
        active_orbitals=[[0], [1, 2, 3, 4, 5, 6, 7, 8]],
        active_frozen_orbitals=[0, 8],
        do_diis=True,
        econv=1e-8,
        gconv=1e-7,
        maxiter=100,
    )(rhf)
    mc.run()

    assert rhf.E == approx(erhf)
    assert mc.E == approx(emcscf)


@pytest.mark.slow
def test_gasscf_ch4_active_frozen_1s_highest_active_noncontiguous():
    erhf = -40.19845141292726
    emcscf = -29.544266400483

    xyz = """
    C            0.052417904862     0.008091170764     0.039717608738
    H            1.170469902710     0.450548753521     1.737652542463
    H            1.167683656136     0.451231405150    -1.659924552573
    H           -1.690971530149     1.143552334703     0.040997211401
    H           -0.436869468885    -2.014192012525     0.039431679323
    """

    system = System(
        xyz=xyz,
        basis_set="cc-pVDZ",
        auxiliary_basis_set="def2-universal-jkfit",
        unit="bohr",
    )

    rhf = RHF(charge=0, econv=1e-12, dconv=1e-12)(system)

    mc = MCOptimizer(
        State(nel=10, multiplicity=1, ms=0.0, gas_min=[1], gas_max=[1]),
        active_orbitals=[[0], [1, 2, 3, 4, 5, 6, 7, 9]],
        active_frozen_orbitals=[0, 9],
        do_diis=True,
        econv=1e-9,
        gconv=1e-9,
        maxiter=200,
    )(rhf)
    mc.run()

    assert rhf.E == approx(erhf)
    assert mc.E == approx(emcscf)
