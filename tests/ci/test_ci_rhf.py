import numpy as np
import pytest

from forte2 import AVAS, CI, CISolver, MOSpace, RHF, ROHF, State, System
from forte2.base_classes import DavidsonLiuParams
from forte2.helpers.comparisons import approx


def test_ci_orbital_invariance_is_true():
    # test that the orbital rotation invariance flag is set to True for CI
    xyz = """H 0.0 0.0 0.0"""

    system = System(
        xyz=xyz, basis_set="sto-6g", auxiliary_basis_set="cc-pVTZ-JKFIT", unit="bohr"
    )
    rhf = RHF(charge=-1, e_tol=1e-12)(system)
    ci = CI(
        CISolver(State(system=system, multiplicity=2, ms=0.5), active_orbitals=[0, 1])
    )(rhf)
    assert ci.ci_solver.orbital_rotation_invariant


def test_ci_1():
    xyz = f"""
    H 0.0 0.0 0.0
    H 0.0 0.0 {0.529177210903 * 2}
    """

    system = System(xyz=xyz, basis_set="sto-6g", auxiliary_basis_set="cc-pVTZ-JKFIT")

    rhf = RHF(charge=0, e_tol=1e-12)(system)
    ci = CI(
        CISolver(State(system=system, multiplicity=1, ms=0.0), active_orbitals=[0, 1])
    )(rhf)
    ci.run()

    assert rhf.E == approx(-1.05643120731551)
    assert ci.E_ci[0] == approx(-1.096071975854)


def test_ci_2():
    xyz = """
    H 0.0 0.0 0.0
    F 0.0 0.0 2.0
    """

    system = System(
        xyz=xyz, basis_set="cc-pVDZ", auxiliary_basis_set="cc-pVTZ-JKFIT", unit="bohr"
    )
    rhf = RHF(charge=0, e_tol=1e-12)(system)
    ci = CI(
        CISolver(
            states=State(nel=10, multiplicity=1, ms=0.0),
            core_orbitals=[0],
            active_orbitals=[1, 2, 3, 4, 5, 6],
        )
    )(rhf)
    ci.run()

    assert ci.E_ci[0] == approx(-100.019788438077)


def test_ci_n2_with_symmetry():
    xyz = """
    N 0.0 0.0 0.0
    N 0.0 0.0 1.2
    """

    system = System(
        xyz=xyz,
        basis_set="cc-pvdz",
        auxiliary_basis_set="cc-pVTZ-JKFIT",
        symmetry=True,
    )

    rhf = RHF(charge=0, e_tol=1e-12)(system)
    ci = CI(
        CISolver(
            states=State(nel=14, multiplicity=1, ms=0.0),
            core_orbitals=4,
            active_orbitals=6,
        )
    )(rhf)
    ci.run()
    eref_singlet = -109.004622061660
    assert ci.E_ci[0] == approx(eref_singlet)


def test_ci_ch4_with_symmetry():
    xyz = """
    C 0.881018195 4.336586688 4.172509116
    H 1.899108274 4.213611337 3.803066093
    H 0.897015796 4.915685304 5.095822368
    H 0.284205956 4.860075328 3.425586572
    H 0.443742755 3.356974782 4.365561431
    """
    system = System(
        xyz=xyz,
        basis_set="cc-pvdz",
        auxiliary_basis_set="cc-pVTZ-JKFIT",
        symmetry=True,
    )
    rhf = RHF(charge=0, e_tol=1e-12)(system)
    ci = CI(
        CISolver(
            states=State(nel=10, multiplicity=1, ms=0.0),
            core_orbitals=1,
            active_orbitals=8,
        )
    )(rhf)
    ci.run()

    # reference energy obtained without symmetry
    assert ci.E_ci[0] == approx(-40.2116319300)


def test_sa_ci_n2():
    xyz = """
    N 0.0 0.0 0.0
    N 0.0 0.0 1.2
    """

    system = System(xyz=xyz, basis_set="cc-pvdz", auxiliary_basis_set="cc-pVTZ-JKFIT")

    rhf = RHF(charge=0, e_tol=1e-12)(system)
    singlet = State(nel=14, multiplicity=1, ms=0.0)
    triplet = State(nel=14, multiplicity=3, ms=0.0)
    ci_solver = CISolver(
        states=[singlet, triplet],
        core_orbitals=4,
        active_orbitals=6,
        nroots=[1, 2],
        weights=[[1.0], [0.85, 0.15]],
    )
    ci = CI(ci_solver)(rhf)
    ci.run()
    eref_singlet = -109.004622061660
    eref_triplet1 = -108.779926502402
    eref_triplet2 = -108.733907910380
    assert ci.E_ci[0] == approx(eref_singlet)
    assert ci.E_ci[1] == approx(eref_triplet1)
    assert ci.E_ci[2] == approx(eref_triplet2)
    assert ci.E_avg == approx(
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

    rhf = RHF(charge=0, e_tol=1e-12)(system)
    avas = AVAS(
        selection_method="separate",
        num_active_docc=3,
        num_active_uocc=3,
        subspace=["N(2p)"],
        diagonalize=True,
    )(rhf)

    singlet = State(nel=14, multiplicity=1, ms=0.0)
    triplet = State(nel=14, multiplicity=3, ms=0.0)

    ci_solver = CISolver(
        [singlet, triplet], nroots=[1, 2], weights=[[1.0], [0.85, 0.15]]
    )
    saci = CI(ci_solver)(avas)
    saci.run()

    assert saci.E_ci[0] == approx(eref_singlet)
    assert saci.E_ci[1] == approx(eref_triplet1)
    assert saci.E_ci[2] == approx(eref_triplet2)
    assert saci.E_avg == approx(
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
    rhf = RHF(charge=0, e_tol=1e-12)(system)
    ci_solver = CISolver(
        State(nel=14, multiplicity=1, ms=0.0),
        core_orbitals=[0, 1, 2, 3],
        active_orbitals=[4, 5, 6, 7, 8, 9],
        nroots=10,
    )
    ci = CI(ci_solver, do_transition_dipole=True)(rhf)
    ci.run()
    assert abs(ci_solver.transition_dipoles[(0, 6)][2]) == pytest.approx(
        1.5435316739347478, abs=1e-4
    )
    assert ci_solver.oscillator_strengths[(0, 6)] == pytest.approx(
        1.1589808047738437, abs=1e-4
    )


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
    rhf = RHF(charge=0, e_tol=1e-12)(system)
    ci = CI(CISolver(states=state, core_orbitals=[0, 1, 2, 3, 4], active_orbitals=[]))(
        rhf
    )
    ci.run()

    assert rhf.E == approx(-99.997725200294)
    assert ci.E_ci[0] == approx(-99.997725200294)


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
    rhf = RHF(charge=0, e_tol=1e-12)(system)
    ci = CI(CISolver(states=state, core_orbitals=[0, 1, 2, 3], active_orbitals=[4]))(
        rhf
    )
    ci.run()

    assert rhf.E == approx(-99.997725200294)
    assert ci.E_ci[0] == approx(-99.997725200294)


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
    rhf = RHF(charge=0, e_tol=1e-12)(system)
    ci = CI(CISolver(states=state, core_orbitals=[], active_orbitals=[0, 1, 2, 3, 4]))(
        rhf
    )
    ci.run()

    assert rhf.E == approx(-99.997725200294)
    assert ci.E_ci[0] == approx(-99.997725200294)


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

    rhf = ROHF(charge=0, ms=1.0, e_tol=1e-12)(system)
    ci = CI(CISolver(State(nel=2, multiplicity=3, ms=1.0), active_orbitals=[0, 1]))(rhf)
    ci.run()

    assert rhf.E == approx(-0.889646913931)
    assert ci.E_ci[0] == approx(-0.889646913931)


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

    rhf = ROHF(charge=0, ms=1.0, e_tol=1e-12)(system)
    ci = CI(CISolver(State(nel=2, multiplicity=3, ms=0.0), active_orbitals=[0, 1]))(rhf)
    ci.run()

    assert rhf.E == approx(-0.889646913931)
    assert ci.E_ci[0] == approx(-0.889646913931)


def _lih_noncontiguous_mo_space(system):
    return MOSpace(
        nmo=system.nmo,
        core_orbitals=[0],
        active_orbitals=[1, 2],
        frozen_virtual_orbitals=[3],
    )


def test_ci_semicanonical_noncontiguous_mo_space():
    xyz = "Li 0.0 0.0 0.0\nH  0.0 0.0 3.0"
    system = System(
        xyz=xyz,
        basis_set="sto-3g",
        auxiliary_basis_set="def2-universal-JKFIT",
        unit="bohr",
    )
    rhf = RHF(charge=0, e_tol=1e-12)(system)
    mo_space = _lih_noncontiguous_mo_space(system)

    ci_original = CI(
        CISolver(State(nel=4, multiplicity=1, ms=0.0), mo_space_override=mo_space)
    )(rhf)
    ci_original.run()
    ci_semicanonical = CI(
        CISolver(State(nel=4, multiplicity=1, ms=0.0), mo_space_override=mo_space),
        final_orbitals="semicanonical",
    )(rhf)
    ci_semicanonical.run()

    # The frozen virtual sits before the regular virtuals in the original
    # ordering but after them in the contiguous [core, active, virt,
    # frozen_virt] ordering, so this permutation is genuinely non-trivial.
    np.testing.assert_array_equal(mo_space.orig_to_contig, [0, 1, 2, 4, 5, 3])
    np.testing.assert_array_equal(mo_space.contig_to_orig, [0, 1, 2, 5, 3, 4])
    assert ci_semicanonical.E_ci[0] == approx(ci_original.E_ci[0])
    np.testing.assert_allclose(
        ci_semicanonical.mos.C[0].T @ system.ints_overlap() @ ci_semicanonical.mos.C[0],
        np.eye(mo_space.nmo),
        atol=1e-10,
    )


def test_ci_natural_noncontiguous_mo_space():
    xyz = "Li 0.0 0.0 0.0\nH  0.0 0.0 3.0"
    system = System(
        xyz=xyz,
        basis_set="sto-3g",
        auxiliary_basis_set="def2-universal-JKFIT",
        unit="bohr",
    )
    rhf = RHF(charge=0, e_tol=1e-12)(system)
    mo_space = _lih_noncontiguous_mo_space(system)

    ci_original = CI(
        CISolver(State(nel=4, multiplicity=1, ms=0.0), mo_space_override=mo_space)
    )(rhf)
    ci_original.run()
    ci_natural = CI(
        CISolver(State(nel=4, multiplicity=1, ms=0.0), mo_space_override=mo_space),
        final_orbitals="natural",
    )(rhf)
    ci_natural.run()

    assert ci_natural.E_ci[0] == approx(ci_original.E_ci[0])
    np.testing.assert_allclose(
        ci_natural.mos.C[0].T @ system.ints_overlap() @ ci_natural.mos.C[0],
        np.eye(mo_space.nmo),
        atol=1e-10,
    )

    g1_act = ci_natural.make_average_rdm(1)
    off_diag = g1_act - np.diag(np.diag(g1_act))
    assert np.max(np.abs(off_diag)) < 1e-8


@pytest.mark.parametrize("final_orbitals", ["original", "semicanonical", "natural"])
def test_ci_final_orbitals(final_orbitals):
    eref = -99.82331087176414
    xyz = """
    H 0.0 0.0 0.0
    F 0.0 0.0 2.0
    """
    system = System(
        xyz=xyz,
        basis_set="cc-pVDZ",
        auxiliary_basis_set="cc-pVTZ-JKFIT",
        unit="bohr",
    )
    rhf = RHF(charge=0, e_tol=1e-8)(system)

    singlet = State(nel=10, multiplicity=1, ms=0.0)
    triplet = State(nel=10, multiplicity=3, ms=1.0)
    ci_solver = CISolver(
        states=[singlet, triplet],
        nroots=[2, 1],
        core_orbitals=[0],
        active_orbitals=[1, 2, 3, 4, 5, 6, 7],
        davidson_liu_params=DavidsonLiuParams(
            e_tol=1e-8,
            r_tol=1e-4,
            ndets_per_guess=10,
        ),
    )
    ci = CI(ci_solver, final_orbitals=final_orbitals)(rhf)
    ci.run()
    assert ci.E_avg == approx(eref)
