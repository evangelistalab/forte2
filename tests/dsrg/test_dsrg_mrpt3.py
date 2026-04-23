from forte2 import System, RHF, MCOptimizer, State
from forte2.dsrg import DSRG_MRPT3
from forte2.helpers.comparisons import approx


def test_sf_mrpt3_n2():
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
        core_orbitals=4,
        active_orbitals=6,
    )(rhf)
    mc.run()

    assert rhf.E == approx(erhf)
    assert mc.E == approx(emcscf)

    dsrg = DSRG_MRPT3(flow_param=0.5)(mc)
    dsrg.run()

    assert dsrg.E == approx(-109.25301485037653)


def test_sf_mrpt3_pt2_only():
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
        core_orbitals=4,
        active_orbitals=6,
    )(rhf)
    mc.run()

    assert rhf.E == approx(erhf)
    assert mc.E == approx(emcscf)

    dsrg = DSRG_MRPT3(flow_param=0.5, relax_reference="iterate")(mc)
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
