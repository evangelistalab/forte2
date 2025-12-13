import numpy as np

from forte2 import System
from forte2.props import get_1e_property, mulliken_population
from forte2.scf import RHF, UHF, GHF
from forte2.helpers.comparisons import approx, approx_loose


def test_core_energy():
    erhf = -76.061466407195
    ecore = -124.11336728113105
    xyz = """
    O            0.000000000000     0.000000000000    -0.061664597388
    H            0.000000000000    -0.711620616369     0.489330954643
    H            0.000000000000     0.711620616369     0.489330954643
    """

    system = System(xyz=xyz, basis_set="cc-pVQZ", auxiliary_basis_set="cc-pVQZ-JKFIT")

    scf = RHF(charge=0)(system)
    scf.run()
    assert scf.E == approx(erhf)
    dm1 = scf._build_total_density_matrix()
    ke = get_1e_property(system, dm1, "kinetic_energy")
    ve = get_1e_property(system, dm1, "nuclear_attraction_energy")
    assert ke + ve == approx(ecore)


def test_dipole_rhf():
    erhf = -76.061466407195
    xyz = """
    O            0.000000000000     0.000000000000    -0.061664597388
    H            0.000000000000    -0.711620616369     0.489330954643
    H            0.000000000000     0.711620616369     0.489330954643
    """

    system = System(xyz=xyz, basis_set="cc-pVQZ", auxiliary_basis_set="cc-pVQZ-JKFIT")

    scf = RHF(charge=0)(system)
    scf.run()
    assert scf.E == approx(erhf)
    dm1 = scf._build_total_density_matrix()
    dip = get_1e_property(system, dm1, "dipole")
    assert dip == approx_loose([0.00000, 0.00000, 1.95868013])


def test_dipole_uhf():
    euhf = -75.649277914372
    xyz = """
    O            0.000000000000     0.000000000000    -0.061664597388
    H            0.000000000000    -0.711620616369     0.489330954643
    H            0.000000000000     0.711620616369     0.489330954643
    """

    system = System(
        xyz=xyz,
        basis_set="cc-pVQZ",
        auxiliary_basis_set="cc-pVQZ-JKFIT",
    )

    scf = UHF(charge=1, ms=0.5)(system)
    scf.run()
    assert scf.E == approx(euhf)

    dm1 = scf._build_total_density_matrix()
    e_dip = get_1e_property(
        system, dm1, "electric_dipole", origin=system.center_of_mass
    )

    assert e_dip == approx_loose([0, 0, -2.56784946e-02])
    dip = get_1e_property(system, dm1, "dipole", origin=[1.2, -0.7, 1])
    assert dip == approx([-3.05009553, 1.77922239, -0.23558438])


def test_dipole_ghf():
    euhf = -75.649277914372
    xyz = """
    O            0.000000000000     0.000000000000    -0.061664597388
    H            0.000000000000    -0.711620616369     0.489330954643
    H            0.000000000000     0.711620616369     0.489330954643
    """

    system = System(
        xyz=xyz,
        basis_set="cc-pVQZ",
        auxiliary_basis_set="cc-pVQZ-JKFIT",
    )

    scf = GHF(charge=1)(system)
    scf.run()
    assert scf.E == approx(euhf)

    dm1 = scf._build_total_density_matrix()
    e_dip = get_1e_property(
        system, dm1, "electric_dipole", origin=system.center_of_mass
    )

    assert e_dip == approx_loose([0, 0, -2.56784946e-02])
    dip = get_1e_property(system, dm1, "dipole", origin=[1.2, -0.7, 1])
    assert dip == approx([-3.05009553, 1.77922239, -0.23558438])


def test_quadrupole_rhf():
    erhf = -76.061466407195
    xyz = """
    O            0.000000000000     0.000000000000    -0.061664597388
    H            0.000000000000    -0.711620616369     0.489330954643
    H            0.000000000000     0.711620616369     0.489330954643
    """

    system = System(
        xyz=xyz,
        basis_set="cc-pVQZ",
        auxiliary_basis_set="cc-pVQZ-JKFIT",
    )

    scf = RHF(charge=0)(system)
    scf.run()
    assert scf.E == approx(erhf)
    dm1 = scf._build_total_density_matrix()
    quad = get_1e_property(system, dm1, "quadrupole")
    assert np.trace(quad) == approx_loose(0.0)
    assert np.diag(quad) == approx([-2.16502486e00, 2.28286793e00, -1.17843071e-01])


def test_mulliken_rhf():
    erhf = -76.061466407195
    xyz = """
    O            0.000000000000     0.000000000000    -0.061664597388
    H            0.000000000000    -0.711620616369     0.489330954643
    H            0.000000000000     0.711620616369     0.489330954643
    """

    system = System(xyz=xyz, basis_set="cc-pVQZ", auxiliary_basis_set="cc-pVQZ-JKFIT")

    scf = RHF(charge=0)(system)
    scf.run()
    assert scf.E == approx(erhf)

    dm1 = scf._build_total_density_matrix()
    mp = mulliken_population(system, dm1)
    assert mp[1] == approx([-0.4620044, 0.2310022, 0.2310022])


def test_dipole_sfx2c1e():
    escf_rhf = -2573.01930609129
    escf_sf = -2597.864199663877

    xyz = """
    H 0 0 0
    Br 0 0 1.2
    """
    system = System(xyz=xyz, basis_set="cc-pVQZ", auxiliary_basis_set="cc-pVQZ-JKFIT")

    scf = RHF(charge=0)(system)
    scf.run()
    assert scf.E == approx(escf_rhf)

    dm1 = scf._build_total_density_matrix()
    dip = get_1e_property(system, dm1, "dipole")
    assert dip == approx_loose([0.0, 0.0, -8.60272654e-01])

    # turn on sfx2c1e
    system = System(
        xyz=xyz, basis_set="cc-pVQZ", auxiliary_basis_set="cc-pVQZ-JKFIT", x2c_type="sf"
    )

    scf = RHF(charge=0)(system)
    scf.run()
    assert scf.E == approx(escf_sf)

    dm1 = scf._build_total_density_matrix()

    dip = get_1e_property(system, dm1, "dipole")
    assert dip == approx([0.0, 0.0, -8.34986872284730e-01])
    dip = get_1e_property(system, dm1, "dipole", skip_picture_change=True)
    assert dip == approx([0.0, 0.0, -8.34986872284730e-01])


def test_dipole_sox2c1e():
    escf_ghf = -2597.803003174037

    xyz = """
    H 0 0 0
    Br 0 0 1.2
    """
    system = System(
        xyz=xyz,
        basis_set="cc-pVQZ",
        auxiliary_basis_set="cc-pVQZ-JKFIT",
        x2c_type="so",
        snso_type=None,
    )

    scf = GHF(charge=0)(system)
    scf.run()
    assert scf.E == approx(escf_ghf)

    dm1 = scf._build_total_density_matrix()
    dip = get_1e_property(system, dm1, "dipole")
    assert dip == approx([0.0, 0.0, -8.31845304170969e-01])
