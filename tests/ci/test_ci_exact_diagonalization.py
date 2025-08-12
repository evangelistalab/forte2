from forte2 import System, RHF, CI, State, AVAS, ROHF
from forte2.helpers.comparisons import approx


def test_ci_1():
    xyz = f"""
    H 0.0 0.0 0.0
    H 0.0 0.0 {0.529177210903 * 2}
    """

    system = System(xyz=xyz, basis_set="sto-6g", auxiliary_basis_set="cc-pVTZ-JKFIT")

    rhf = RHF(charge=0, econv=1e-12)(system)
    ci = CI(
        State(system=system, multiplicity=1, ms=0.0),
        active_orbitals=[0, 1],
        ci_algorithm="exact",
    )(rhf)
    ci.run()

    assert rhf.E == approx(-1.05643120731551)
    assert ci.E[0] == approx(-1.096071975854)


def test_ci_2():
    xyz = """
    H 0.0 0.0 0.0
    F 0.0 0.0 2.0
    """

    system = System(
        xyz=xyz, basis_set="cc-pVDZ", auxiliary_basis_set="cc-pVTZ-JKFIT", unit="bohr"
    )
    rhf = RHF(charge=0, econv=1e-12)(system)
    ci = CI(
        states=State(nel=10, multiplicity=1, ms=0.0),
        core_orbitals=[0],
        active_orbitals=[1, 2, 3, 4, 5, 6],
        ci_algorithm="exact",
    )(rhf)
    ci.run()

    assert ci.E[0] == approx(-100.019788438077)


def test_sa_ci_n2():
    xyz = """
    N 0.0 0.0 0.0
    N 0.0 0.0 1.2
    """

    system = System(xyz=xyz, basis_set="cc-pvdz", auxiliary_basis_set="cc-pVTZ-JKFIT")

    rhf = RHF(charge=0, econv=1e-12)(system)
    singlet = State(nel=14, multiplicity=1, ms=0.0)
    triplet = State(nel=14, multiplicity=3, ms=0.0)
    ci = CI(
        states=[singlet, triplet],
        core_orbitals=4,
        active_orbitals=6,
        nroots=[1, 2],
        weights=[[1.0], [0.85, 0.15]],
        ci_algorithm="exact",
    )(rhf)
    ci.run()
    eref_singlet = -109.004622061660
    eref_triplet1 = -108.779926502402
    eref_triplet2 = -108.733907910380
    assert ci.E[0] == approx(eref_singlet)
    assert ci.E[1] == approx(eref_triplet1)
    assert ci.E[2] == approx(eref_triplet2)
    assert ci.compute_average_energy() == approx(
        0.5 * eref_singlet + 0.5 * (0.85 * eref_triplet1 + 0.15 * eref_triplet2)
    )


def test_sa_ci_with_avas():
    # This won't be strictly identical to test_sa_ci_n2 because AVAS will select different orbitals
    eref_singlet = -109.061384781871
    eref_triplet1 = -108.833136404913
    eref_triplet2 = -108.777400848037

    xyz = """
    N 0.0 0.0 0.0
    N 0.0 0.0 1.2
    """

    system = System(xyz=xyz, basis_set="cc-pvdz", auxiliary_basis_set="cc-pVTZ-JKFIT")

    rhf = RHF(charge=0, econv=1e-12)(system)
    avas = AVAS(
        selection_method="separate",
        num_active_docc=3,
        num_active_uocc=3,
        subspace=["N(2p)"],
        diagonalize=True,
    )(rhf)

    singlet = State(nel=14, multiplicity=1, ms=0.0)
    triplet = State(nel=14, multiplicity=3, ms=0.0)

    saci = CI(
        states=[singlet, triplet],
        nroots=[1, 2],
        weights=[[1.0], [0.85, 0.15]],
        ci_algorithm="exact",
    )(avas)
    saci.run()

    assert saci.E[0] == approx(eref_singlet)
    assert saci.E[1] == approx(eref_triplet1)
    assert saci.E[2] == approx(eref_triplet2)
    assert saci.compute_average_energy() == approx(
        0.5 * eref_singlet + 0.5 * (0.85 * eref_triplet1 + 0.15 * eref_triplet2)
    )


def test_ci_tdm():
    xyz = """
    N 0.0 0.0 -1.0
    N 0.0 0.0 1.0
    """

    system = System(
        xyz=xyz, basis_set="cc-pVDZ", auxiliary_basis_set="cc-pVTZ-JKFIT", unit="bohr"
    )
    rhf = RHF(charge=0, econv=1e-12)(system)
    ci = CI(
        State(nel=14, multiplicity=1, ms=0.0),
        core_orbitals=[0, 1, 2, 3],
        active_orbitals=[4, 5, 6, 7, 8, 9],
        nroots=10,
        do_transition_dipole=True,
        ci_algorithm="exact",
    )(rhf)
    ci.run()
    assert abs(ci.tdm_per_solver[0][(0, 6)][2]) == approx(1.5435316739347478)
    assert ci.fosc_per_solver[0][(0, 6)] == approx(1.1589808047738437)


def test_ci_no_active():
    """Test CI with a core orbital and no active orbitals, should return the RHF energy.
                                          _____
    Here we specify the determinant |0123401234|>
                                           core|active

    """

    xyz = """
    H 0.0 0.0 0.0
    F 0.0 0.0 2.0
    """

    system = System(
        xyz=xyz, basis_set="cc-pVDZ", auxiliary_basis_set="cc-pVTZ-JKFIT", unit="bohr"
    )
    state = State(nel=10, multiplicity=1, ms=0.0)
    rhf = RHF(charge=0, econv=1e-12)(system)
    ci = CI(
        states=state,
        core_orbitals=[0, 1, 2, 3, 4],
        active_orbitals=[],
        ci_algorithm="exact",
    )(rhf)
    ci.run()

    assert rhf.E == approx(-99.997725200294)
    assert ci.E[0] == approx(-99.997725200294)


def test_ci_single_determinant1():
    """Test CI with a single determinant, should return the RHF energy.
                                         ____  _
    Here we specify the determinant |01230123|44>
                                         core|active
    """

    xyz = """
    H 0.0 0.0 0.0
    F 0.0 0.0 2.0
    """

    system = System(
        xyz=xyz, basis_set="cc-pVDZ", auxiliary_basis_set="cc-pVTZ-JKFIT", unit="bohr"
    )
    state = State(nel=10, multiplicity=1, ms=0.0)
    rhf = RHF(charge=0, econv=1e-12)(system)
    ci = CI(
        states=state,
        core_orbitals=[0, 1, 2, 3],
        active_orbitals=[4],
        ci_algorithm="exact",
    )(rhf)
    ci.run()

    assert rhf.E == approx(-99.997725200294)
    assert ci.E[0] == approx(-99.997725200294)


def test_ci_single_determinant2():
    """Test CI with a single determinant, should return the RHF energy.
                                          _____
    Here we specify the determinant ||0123401234>
                                 core|active
    """

    xyz = """
    H 0.0 0.0 0.0
    F 0.0 0.0 2.0
    """

    system = System(
        xyz=xyz, basis_set="cc-pVDZ", auxiliary_basis_set="cc-pVTZ-JKFIT", unit="bohr"
    )
    state = State(nel=10, multiplicity=1, ms=0.0)
    rhf = RHF(charge=0, econv=1e-12)(system)
    ci = CI(
        states=state,
        core_orbitals=[],
        active_orbitals=[0, 1, 2, 3, 4],
        ci_algorithm="exact",
    )(rhf)
    ci.run()

    assert rhf.E == approx(-99.997725200294)
    assert ci.E[0] == approx(-99.997725200294)


def test_ci_single_determinant3():
    """Test CI with a high-spin triplet single determinant, should return the ROHF energy.

    Here we specify the determinant ||01>
                                 core|active
    """

    xyz = f"""
    H 0.0 0.0 0.0
    H 0.0 0.0 {0.529177210903 * 2}
    """

    system = System(xyz=xyz, basis_set="cc-pVDZ", auxiliary_basis_set="cc-pVTZ-JKFIT")

    rhf = ROHF(charge=0, ms=1.0, econv=1e-12)(system)
    ci = CI(
        State(nel=2, multiplicity=3, ms=1.0),
        active_orbitals=[0, 1],
        ci_algorithm="exact",
    )(rhf)
    ci.run()

    assert rhf.E == approx(-0.889646913931)
    assert ci.E[0] == approx(-0.889646913931)


def test_ci_single_csf1():
    """Test CI with a high-spin triplet single determinant, should return the ROHF energy.
                                        _           _
    Here we specify the determinants ||01>        ||01>
                                  core|active  core|active
    """

    xyz = f"""
    H 0.0 0.0 0.0
    H 0.0 0.0 {0.529177210903 * 2}
    """

    system = System(xyz=xyz, basis_set="cc-pVDZ", auxiliary_basis_set="cc-pVTZ-JKFIT")

    rhf = ROHF(charge=0, ms=1.0, econv=1e-12)(system)
    ci = CI(
        State(nel=2, multiplicity=3, ms=0.0),
        active_orbitals=[0, 1],
        ci_algorithm="exact",
    )(rhf)
    ci.run()

    assert rhf.E == approx(-0.889646913931)
    assert ci.E[0] == approx(-0.889646913931)
