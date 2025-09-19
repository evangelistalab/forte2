from forte2 import System, RHF, MCOptimizer, State
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
    mc = MCOptimizer(State(nel=2, multiplicity=1, ms=0.0), active_orbitals=[0, 1])(rhf)
    mc.run()

    assert rhf.E == approx(erhf)
    assert mc.E == approx(emcscf)


def test_casscf_n2():
    erhf = -108.761639873604
    ecasscf = -108.9800484156

    xyz = """
    N 0.0 0.0 0.0
    N 0.0 0.0 1.4
    """

    system = System(xyz=xyz, basis_set="cc-pVDZ", auxiliary_basis_set="cc-pVTZ-JKFIT")
    rhf = RHF(charge=0, econv=1e-12)(system)
    mc = MCOptimizer(
        State(nel=14, multiplicity=1, ms=0.0),
        active_orbitals=[4, 5, 6, 7, 8, 9],
        core_orbitals=[0, 1, 2, 3],
        gconv=1e-7,
    )(rhf)
    mc.run()
    assert rhf.E == approx(erhf)
    assert mc.E == approx(ecasscf)


def test_casscf_water():
    erhf = -76.0214620954787819
    emcscf = -76.07856407969193

    xyz = """
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
    mc = MCOptimizer(
        State(nel=10, multiplicity=1, ms=0.0),
        active_orbitals=[1, 2, 3, 4, 5, 6],
        core_orbitals=[0],
        gconv=1e-6,
        econv=1e-10,
    )(rhf)
    mc.run()

    assert rhf.E == approx(erhf)
    assert mc.E == approx(emcscf)
