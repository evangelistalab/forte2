import numpy as np
import pytest

from forte2 import System, RHF, GHF, SpinorUpcaster, MOSpace, X2CParams
from forte2.helpers.comparisons import approx
from forte2.ci import RelCI


def test_rel_ci_orbital_invariance_is_true():
    # test that the orbital rotation invariance flag is set to True for RelCI
    xyz = """H 0.0 0.0 0.0"""

    system = System(
        xyz=xyz, basis_set="sto-6g", auxiliary_basis_set="cc-pVTZ-JKFIT", unit="bohr"
    )
    scf = GHF(charge=0, e_tol=1e-12)(system)
    conv = SpinorUpcaster(apply_random_phase=True)(scf)
    ci = RelCI(nel=1, active_orbitals=2, do_test_rdms=True)(conv)
    assert ci.orbital_rotation_invariant


def test_rel_ci_h2():
    # equivalent to test_slater_rules::test_slater_rules_1_complex
    xyz = """
    H 0.0 0.0 0.0
    H 0.0 0.0 2.0
    """

    system = System(
        xyz=xyz, basis_set="sto-6g", auxiliary_basis_set="cc-pVTZ-JKFIT", unit="bohr"
    )
    scf = GHF(charge=0, e_tol=1e-12)(system)
    conv = SpinorUpcaster(apply_random_phase=True)(scf)

    ci = RelCI(nel=2, active_orbitals=4, do_test_rdms=True)(conv)

    ci.run()
    assert ci.E[0] == approx(-1.096071975854)


def test_rel_ci_hf():
    # equivalent to test_slater_rules::test_slater_rules_2_complex
    xyz = """
    H 0.0 0.0 0.0
    F 0.0 0.0 2.0
    """

    system = System(
        xyz=xyz, basis_set="cc-pvdz", auxiliary_basis_set="cc-pVTZ-JKFIT", unit="bohr"
    )
    scf = RHF(charge=0, e_tol=1e-10)(system)
    conv = SpinorUpcaster(apply_random_phase=True)(scf)

    ci = RelCI(
        nel=10,
        core_orbitals=2,
        active_orbitals=12,
        do_test_rdms=True,
    )(conv)
    ci.run()
    assert ci.E[0] == approx(-100.019788438077)


def test_rel_ci_hf_ghf():
    # cross-validated with the pyscf fci_dhf_slow solver using integrals from SpinorbitalIntegrals
    eref = -100.10065023157668
    xyz = """
    H 0.0 0.0 0.0
    F 0.0 0.0 2.0
    """

    system = System(
        xyz=xyz,
        basis_set="cc-pvdz",
        auxiliary_basis_set="cc-pVTZ-JKFIT",
        unit="bohr",
        x2c=X2CParams(x2c_type="so", x2c_model="1e"),
    )
    scf = GHF(charge=0)(system)
    ci = RelCI(
        nel=10,
        core_orbitals=2,
        active_orbitals=12,
        do_test_rdms=True,
        final_orbitals="semicanonical",
    )(scf)
    ci.run()
    assert ci.E[0] == approx(eref)


def test_rel_ci_semicanonical_noncontiguous_mo_space():
    xyz = """
    Li 0.0 0.0 0.0
    H  0.0 0.0 3.0
    """

    system = System(
        xyz=xyz,
        basis_set="sto-3g",
        auxiliary_basis_set="def2-universal-JKFIT",
        unit="bohr",
    )
    scf = GHF(charge=0, e_tol=1e-12)(system)
    mo_space = MOSpace(
        nmo=system.nmo * 2,
        core_orbitals=[0, 1],
        active_orbitals=[2, 3, 4, 5],
        frozen_virtual_orbitals=[6, 7],
    )

    ci_original = RelCI(nel=4, mo_space=mo_space)(scf)
    ci_original.run()
    ci_semicanonical = RelCI(
        nel=4,
        mo_space=mo_space,
        final_orbitals="semicanonical",
    )(scf)
    ci_semicanonical.run()

    np.testing.assert_array_equal(
        mo_space.orig_to_contig,
        [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 6, 7],
    )
    np.testing.assert_array_equal(
        mo_space.contig_to_orig,
        [0, 1, 2, 3, 4, 5, 10, 11, 6, 7, 8, 9],
    )
    assert ci_semicanonical.E[0] == approx(ci_original.E[0])
    np.testing.assert_allclose(
        ci_semicanonical.mos.C[0].conj().T
        @ system.ints_overlap()
        @ ci_semicanonical.mos.C[0],
        np.eye(mo_space.nmo),
        atol=1e-10,
    )


def test_rel_ci_natural_noncontiguous_mo_space():
    """RelCI final_orbitals='natural' reproduces the energy and natural
    occupation number spectrum after a genuinely non-trivial contig/orig
    permutation (see test_rel_ci_semicanonical_noncontiguous_mo_space).

    Unlike the semicanonical case, this checks the natural occupation number
    *spectrum* (sorted eigenvalues of the active 1-RDM) rather than the RDM's
    raw matrix diagonality. In a time-reversal-symmetric two-component
    calculation, every natural occupation number is exactly Kramers-doubly-
    degenerate (a single-particle consequence of Kramers' theorem, which
    holds regardless of whether the many-body state itself is degenerate).
    "The" natural spinors are therefore only defined up to an arbitrary
    unitary rotation within each degenerate Kramers pair: the rotation matrix
    that ``NaturalOrbitals`` builds from the pre-rotation 1-RDM diagonalizes
    it to machine precision, but re-diagonalizing RelCI from scratch in that
    rotated basis converges to a different (energy-degenerate, equally valid)
    gauge choice within each pair, leaving the *matrix* far from diagonal
    even though the total energy and the occupation-number spectrum agree
    with the pre-rotation reference to Davidson-Liu convergence precision.
    This was confirmed by direct probing and is not specific to RelCI (the
    same effect appears for the approximate RelDMRG solver).
    """
    xyz = """
    Li 0.0 0.0 0.0
    H  0.0 0.0 3.0
    """

    system = System(
        xyz=xyz,
        basis_set="sto-3g",
        auxiliary_basis_set="def2-universal-JKFIT",
        unit="bohr",
    )
    scf = GHF(charge=0, e_tol=1e-12)(system)
    mo_space = MOSpace(
        nmo=system.nmo * 2,
        core_orbitals=[0, 1],
        active_orbitals=[2, 3, 4, 5],
        frozen_virtual_orbitals=[6, 7],
    )

    ci_original = RelCI(nel=4, mo_space=mo_space)(scf)
    ci_original.run()
    ci_natural = RelCI(
        nel=4,
        mo_space=mo_space,
        final_orbitals="natural",
    )(scf)
    ci_natural.run()

    assert ci_natural.E[0] == approx(ci_original.E[0])
    np.testing.assert_allclose(
        ci_natural.mos.C[0].conj().T @ system.ints_overlap() @ ci_natural.mos.C[0],
        np.eye(mo_space.nmo),
        atol=1e-10,
    )

    original_occs = np.sort(np.linalg.eigvalsh(ci_original.make_average_1rdm()))[::-1]
    natural_occs = np.sort(np.linalg.eigvalsh(ci_natural.make_average_1rdm()))[::-1]
    assert natural_occs == approx(original_occs)


def test_rel_ci_hf_transition_dipole_equivalence_to_rhf():
    xyz = """
    H 0.0 0.0 0.0
    F 0.0 0.0 2.0
    """

    system = System(
        xyz=xyz,
        basis_set="cc-pvdz",
        auxiliary_basis_set="cc-pVTZ-JKFIT",
        unit="bohr",
    )
    scf = GHF(charge=0)(system)
    ci = RelCI(
        nel=10,
        nroots=4,
        core_orbitals=2,
        active_orbitals=12,
        do_transition_dipole=True,
    )(scf)
    ci.run()
    assert np.abs(ci.transition_dipoles[(0, 0)]) == pytest.approx(
        [0.0, 0.0, 0.756780349], abs=1e-6
    )
    assert np.abs(ci.transition_dipoles[(1, 1)]) == pytest.approx(
        [0.0, 0.0, 0.721450697], abs=1e-6
    )


def test_rel_ci_hf_transition_dipole_ghf():
    xyz = """
    H 0.0 0.0 0.0
    F 0.0 0.0 2.0
    """

    system = System(
        xyz=xyz,
        basis_set="cc-pvdz",
        auxiliary_basis_set="cc-pVTZ-JKFIT",
        unit="bohr",
        x2c=X2CParams(x2c_type="so", x2c_model="1e"),
    )
    scf = GHF(charge=0)(system)
    ci = RelCI(
        nel=10,
        nroots=5,
        core_orbitals=2,
        active_orbitals=12,
        do_transition_dipole=True,
        do_test_rdms=True,
    )(scf)
    ci.run()
    assert ci.E[0] == approx(-100.10065023157668)
    assert ci.E[1] == approx(-99.7875319545)
    assert ci.E[3] == approx(-99.7866432345)

    assert np.abs(ci.transition_dipoles[(0, 0)]) == pytest.approx(
        [0.0, 0.0, 7.54972929e-01], abs=1e-4
    )
    assert np.abs(ci.transition_dipoles[(1, 1)]) == pytest.approx(
        [0.0, 0.0, 7.21280467e-01], abs=1e-4
    )
    assert np.abs(ci.transition_dipoles[(3, 3)]) == pytest.approx(
        [0.0, 0.0, 7.21064890e-01], abs=1e-4
    )
    assert np.abs(ci.oscillator_strengths[(0, 3)]) == pytest.approx(
        1.711178808962322e-05, abs=1e-4
    )


@pytest.mark.parametrize("algorithm", ["hz", "sparse", "exact"])
def test_rel_ci_algorithms_agree(algorithm):
    """All three two-component CI algorithms must give the same energy."""
    from forte2.base_classes.params import CIParams

    system = System(
        xyz="H 0.0 0.0 0.0\nH 0.0 0.0 2.0",
        basis_set="sto-6g",
        auxiliary_basis_set="cc-pVTZ-JKFIT",
        unit="bohr",
    )
    scf = GHF(charge=0, e_tol=1e-12)(system)
    conv = SpinorUpcaster(apply_random_phase=True)(scf)

    ci = RelCI(
        nel=2,
        active_orbitals=4,
        ci_params=CIParams(ci_algorithm=algorithm),
    )(conv)
    ci.run()

    assert ci.E[0] == approx(-1.096071975854)
