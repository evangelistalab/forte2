import pytest

from forte2 import *
from forte2.helpers.comparisons import approx


def test_casscf_h2():
    erhf = -1.08928367118043
    emcscf = -1.11873740345286

    xyz = f"""
    H 0.0 0.0 0.0
    H 0.0 0.0 {0.529177210903 * 2}
    """

    system = System(xyz=xyz, basis_set="cc-pvdz", auxiliary_basis_set="cc-pVTZ-JKFIT")

    rhf = RHF(charge=0, econv=1e-12)(system)
    ci_state = CIStates(
        active_spaces=[0, 1], states=State(nel=2, multiplicity=1, ms=0.0)
    )
    mc = MCOptimizer(ci_state)(rhf)
    mc.run()

    assert rhf.E == approx(erhf)
    assert mc.E == approx(emcscf)


def test_casscf_hf():
    erhf = -99.9977252002946
    emcscf = -100.0435018956

    xyz = f"""
    H 0.0 0.0 0.0
    F 0.0 0.0 2.0
    """

    system = System(
        xyz=xyz, basis_set="cc-pVDZ", auxiliary_basis_set="cc-pVTZ-JKFIT", unit="bohr"
    )
    rhf = RHF(charge=0, econv=1e-12)(system)
    ci_state = CIStates(
        active_spaces=[1, 2, 3, 4, 5, 6],
        core_orbitals=[0],
        states=State(nel=10, multiplicity=1, ms=0.0),
    )
    mc = MCOptimizer(ci_state)(rhf)
    mc.run()

    assert rhf.E == approx(erhf)
    assert mc.E == approx(emcscf)


def test_casscf_hf_smaller_active():
    erhf = -99.87284684762975
    emcscf = -99.939295399756

    xyz = f"""
    F            0.000000000000     0.000000000000    -0.075563346255
    H            0.000000000000     0.000000000000     1.424436653745
    """

    system = System(
        xyz=xyz, basis_set="cc-pvdz", auxiliary_basis_set="def2-universal-jkfit"
    )

    rhf = RHF(charge=0, econv=1e-12)(system)
    ci_state = CIStates(
        active_spaces=[4, 5],
        core_orbitals=[0, 1, 2, 3],
        states=State(nel=10, multiplicity=1, ms=0.0),
    )
    mc = MCOptimizer(ci_state)(rhf)
    mc.run()

    assert rhf.E == approx(erhf)
    assert mc.E == approx(emcscf)


def test_casscf_n2():
    erhf = -108.761639873604
    ecasscf = -108.9800484156

    xyz = f"""
    N 0.0 0.0 0.0
    N 0.0 0.0 1.4
    """

    system = System(xyz=xyz, basis_set="cc-pVDZ", auxiliary_basis_set="cc-pVTZ-JKFIT")
    rhf = RHF(charge=0, econv=1e-12)(system)
    ci_state = CIStates(
        active_spaces=[4, 5, 6, 7, 8, 9],
        core_orbitals=[0, 1, 2, 3],
        states=State(nel=14, multiplicity=1, ms=0.0),
    )
    mc = MCOptimizer(ci_state, gconv=1e-7)(rhf)
    mc.run()
    assert rhf.E == approx(erhf)
    assert mc.E == approx(ecasscf)


def test_casscf_n2_cholesky():
    erhf = -108.761717999901
    ecasscf = -108.9801054579

    xyz = f"""
    N 0.0 0.0 0.0
    N 0.0 0.0 1.4
    """

    system = System(xyz=xyz, basis_set="cc-pVDZ", cholesky_tei=True, cholesky_tol=1e-10)
    rhf = RHF(charge=0, econv=1e-12)(system)
    ci_state = CIStates(
        active_spaces=[4, 5, 6, 7, 8, 9],
        core_orbitals=[0, 1, 2, 3],
        states=State(nel=14, multiplicity=1, ms=0.0),
    )
    mc = MCOptimizer(ci_state, gconv=1e-7)(rhf)
    mc.run()
    assert rhf.E == approx(erhf)
    assert mc.E == approx(ecasscf)


def test_mcscf_noncontiguous_spaces():
    # The results of this test should be strictly identical to test_mcscf_3

    erhf = -108.761639873604
    eci = -108.916505576963
    ecasscf = -108.9800484156

    xyz = f"""
    N 0.0 0.0 0.0
    N 0.0 0.0 1.4
    """

    system = System(xyz=xyz, basis_set="cc-pVDZ", auxiliary_basis_set="cc-pVTZ-JKFIT")
    rhf = RHF(charge=0, econv=1e-12)(system)
    rhf.run()
    assert rhf.E == approx(erhf)

    # swap orbitals to make them non-contiguous
    core = [0, 1, 3, 6]
    actv = [2, 4, 5, 7, 8, 11]
    virt = sorted(set(range(system.nbf)) - set(core + actv))
    rhf.C[0][:, core + actv + virt] = rhf.C[0]
    ci_state = CIStates(
        active_spaces=actv,
        core_orbitals=core,
        states=State(nel=14, multiplicity=1, ms=0.0),
    )
    ci = CI(ci_state)(rhf)
    ci.run()
    assert ci.E[0] == approx(eci)

    mc = MCOptimizer(ci_state)(rhf)
    mc.run()
    assert mc.E == approx(ecasscf)


def test_casscf_water():
    erhf = -76.0214620954787819
    emcscf = -76.07856407969193

    xyz = f"""
    O            0.000000000000     0.000000000000    -0.069592187400
    H            0.000000000000    -0.783151105291     0.552239257834
    H            0.000000000000     0.783151105291     0.552239257834
    """

    system = System(
        xyz=xyz,
        basis_set="cc-pvdz",
        auxiliary_basis_set="def2-universal-jkfit",
        unit="angstrom",
    )

    rhf = RHF(charge=0, econv=1e-12, dconv=1e-12)(system)
    ci_state = CIStates(
        active_spaces=[1, 2, 3, 4, 5, 6],
        core_orbitals=[0],
        states=State(nel=10, multiplicity=1, ms=0.0),
    )
    mc = MCOptimizer(ci_state, gconv=1e-6, econv=1e-10)(rhf)
    mc.run()

    assert rhf.E == approx(erhf)
    assert mc.E == approx(emcscf)


def test_casscf_symmetry_breaking():
    """Test CASSCF with BeH2 molecule.
    The solution breaks the symmetry of the molecule.
    """

    erhf = -15.59967761106774
    emcscf = -15.6284020142

    xyz = f"""
    Be        0.000000000000     0.000000000000     0.000000000000
    H         0.000000000000     1.389990000000     2.500000000000
    H         0.000000000000    -1.390010000000     2.500000000000
    """

    system = System(
        xyz=xyz,
        basis_set="cc-pVDZ",
        auxiliary_basis_set="def2-universal-JKFIT",
        unit="bohr",
    )

    rhf = RHF(charge=0, econv=1e-10)(system)
    ci_state = CIStates(
        core_orbitals=[0, 1],
        active_spaces=[2, 3],
        states=State(nel=6, multiplicity=1, ms=0.0),
    )
    mc = MCOptimizer(ci_state, econv=1e-9)(rhf)
    mc.run()

    assert rhf.E == approx(erhf)
    assert mc.E == approx(emcscf)


def test_casscf_singlet_benzyne():
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

    system = System(
        xyz=xyz,
        basis_set="sto-3g",
        auxiliary_basis_set="def2-universal-jkfit",
        unit="bohr",
    )

    rhf = RHF(charge=0, econv=1e-12)(system)
    ci_state = CIStates(
        core_orbitals=list(range(19)),
        active_spaces=[19, 20],
        states=State(nel=40, multiplicity=1, ms=0.0),
    )
    mc = MCOptimizer(ci_state)(rhf)
    mc.run()

    assert rhf.E == approx(erhf)
    assert mc.E == approx(emcscf)
