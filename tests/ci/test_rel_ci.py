import numpy as np
import pytest

from forte2 import CI, GHF, MOSpace, RHF, RelCISolver, SpinorUpcaster, System, X2CParams
from forte2.base_classes import CIParams
from forte2.helpers.comparisons import approx, approx_abs
from forte2 import CI


def test_rel_ci_orbital_invariance_is_true():
    # test that the orbital rotation invariance flag is set to True for CI
    xyz = """H 0.0 0.0 0.0"""

    system = System(
        xyz=xyz, basis_set="sto-6g", auxiliary_basis_set="cc-pVTZ-JKFIT", unit="bohr"
    )
    scf = GHF(charge=0, e_tol=1e-12)(system)
    conv = SpinorUpcaster(apply_random_phase=True)(scf)
    ci = CI(RelCISolver(nel=1, active_orbitals=2, do_test_rdms=True))(conv)
    assert ci.ci_solver.orbital_rotation_invariant


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

    ci = CI(RelCISolver(nel=2, active_orbitals=4, do_test_rdms=True))(conv)

    ci.run()
    assert ci.E_ci[0] == approx(-1.096071975854)


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

    ci = CI(
        RelCISolver(nel=10, core_orbitals=2, active_orbitals=12, do_test_rdms=True)
    )(conv)
    ci.run()
    assert ci.E_ci[0] == approx(-100.019788438077)


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
    ci = CI(
        RelCISolver(nel=10, core_orbitals=2, active_orbitals=12, do_test_rdms=True),
        final_orbitals="semicanonical",
    )(scf)
    ci.run()
    assert ci.E_ci[0] == approx(eref)


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

    ci_original = CI(RelCISolver(nel=4, mo_space_override=mo_space))(scf)
    ci_original.run()
    ci_semicanonical = CI(
        RelCISolver(nel=4, mo_space_override=mo_space), final_orbitals="semicanonical"
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
    assert ci_semicanonical.E_ci[0] == approx(ci_original.E_ci[0])
    np.testing.assert_allclose(
        ci_semicanonical.mos.C[0].conj().T
        @ system.ints_overlap()
        @ ci_semicanonical.mos.C[0],
        np.eye(mo_space.nmo),
        atol=1e-10,
    )


def test_rel_ci_natural_noncontiguous_mo_space():
    """
    Two-component CI final_orbitals='natural' reproduces the energy and natural
    occupation number spectrum after a non-trivial contig/orig
    permutation
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

    ci_original = CI(RelCISolver(nel=4, mo_space_override=mo_space))(scf)
    ci_original.run()
    ci_natural = CI(
        RelCISolver(nel=4, mo_space_override=mo_space), final_orbitals="natural"
    )(scf)
    ci_natural.run()

    assert ci_natural.E_ci[0] == approx(ci_original.E_ci[0])
    np.testing.assert_allclose(
        ci_natural.mos.C[0].conj().T @ system.ints_overlap() @ ci_natural.mos.C[0],
        np.eye(mo_space.nmo),
        atol=1e-10,
    )

    original_occs = np.sort(np.linalg.eigvalsh(ci_original.make_average_rdm(1)))[::-1]
    natural_occs = np.sort(np.linalg.eigvalsh(ci_natural.make_average_rdm(1)))[::-1]
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
    ci = CI(
        RelCISolver(nel=10, nroots=4, core_orbitals=2, active_orbitals=12),
        do_transition_dipole=True,
    )(scf)
    ci.run()
    assert np.abs(ci.ci_solver.transition_dipoles[(0, 0)]) == pytest.approx(
        [0.0, 0.0, 0.756780349], abs=1e-6
    )
    assert np.abs(ci.ci_solver.transition_dipoles[(1, 1)]) == pytest.approx(
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
    ci = CI(
        RelCISolver(
            nel=10, nroots=5, core_orbitals=2, active_orbitals=12, do_test_rdms=True
        ),
        do_transition_dipole=True,
    )(scf)
    ci.run()
    assert ci.E_ci[0] == approx(-100.10065023157668)
    assert ci.E_ci[1] == approx(-99.7875319545)
    assert ci.E_ci[3] == approx(-99.7866432345)

    assert np.abs(ci.ci_solver.transition_dipoles[(0, 0)]) == pytest.approx(
        [0.0, 0.0, 7.54974120895005e-01], abs=1e-6
    )
    assert np.abs(ci.ci_solver.transition_dipoles[(1, 1)]) == pytest.approx(
        [0.0, 0.0, 7.21278331621541e-01], abs=1e-6
    )
    assert np.abs(ci.ci_solver.transition_dipoles[(3, 3)]) == pytest.approx(
        [0.0, 0.0, 7.21062788763614e-01], abs=1e-6
    )
    assert np.abs(ci.ci_solver.oscillator_strengths[(0, 3)]) == pytest.approx(
        1.7104694791515446e-05, abs=1e-6
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

    ci = CI(
        RelCISolver(
            nel=2, active_orbitals=4, ci_params=CIParams(ci_algorithm=algorithm)
        )
    )(conv)
    ci.run()

    assert ci.E_ci[0] == approx(-1.096071975854)


@pytest.mark.parametrize("final_orbitals", ["original", "semicanonical", "natural"])
def test_rel_ci_final_orbitals(final_orbitals):
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
    ci = CI(
        RelCISolver(nel=10, core_orbitals=2, active_orbitals=12, do_test_rdms=True),
        final_orbitals=final_orbitals,
    )(scf)
    ci.run()
    assert ci.E_ci[0] == approx(eref)


def test_rel_ci_24_spinors_matches_exact():
    """HZ and the RDMs match exact diagonalization when spinor pair indices exceed 255."""
    system = System(
        xyz="H 0.0 0.0 0.0\nH 0.0 0.0 1.4",
        basis_set="cc-pVTZ",
        auxiliary_basis_set="cc-pVTZ-JKFIT",
        unit="bohr",
    )
    scf = GHF(charge=0, e_tol=1e-12)(system)

    solvers = {}
    for algorithm in ("exact", "hz"):
        ci = CI(
            RelCISolver(
                nel=2, active_orbitals=24, ci_params=CIParams(ci_algorithm=algorithm)
            )
        )(scf)
        ci.run()
        assert ci.E_ci[0] == approx(-1.165336729106)
        solvers[algorithm] = ci.ci_solver.sub_solvers[0]

    solver = solvers["exact"]
    rdm1 = solver.make_rdm(0, order=1, spin_type="so")
    rdm2 = solver.make_rdm(0, order=2, spin_type="so")
    rdm_energy = (
        solver.ints.E
        + np.einsum("ij,ij", rdm1, solver.ints.H)
        + 0.5 * np.einsum("ijkl,ijkl", rdm2, solver.ints.V)
    )
    assert rdm_energy.real == approx(-1.165336729106)


def _spin2_unfolded(system, C, dets, coefficients, ncore):
    """
    Reference <S^2>: core orbitals are explicitly treated
    """
    from forte2.ci.rel_ci_utils import spin_matrices
    from forte2.lib import rdms
    from forte2.lib.det import Determinant
    from forte2.lib.sparse_ops import SparseState

    nactv = C.shape[1] - ncore
    shifted = {}
    for d, c in zip(dets, coefficients):
        new = Determinant.zero()
        for i in range(ncore):
            new.set_na(i, True)
        for i in range(nactv):
            if d.na(i):
                new.set_na(i + ncore, True)
        shifted[new] = c
    state = SparseState(shifted)

    S_z, S_plus, S_minus, S2_1e = spin_matrices(system, C)
    g1 = rdms.compute_1rdm_2c(state, state, C.shape[1])
    g2 = rdms.compute_2rdm_2c(state, state, C.shape[1])
    value = np.einsum("pq,pq->", S2_1e, g1)
    for A, B in ((S_z, S_z), (S_minus, S_plus)):
        value -= np.einsum("ps,qr,pqrs->", A, B, g2, optimize=True)
    return value.real


def test_rel_ci_spin2_single_determinant():
    """A one-determinant 2c CI reproduces the SCF spin expectation values."""
    system = System(
        xyz="C 0 0 0", basis_set="cc-pVDZ", auxiliary_basis_set="cc-pVTZ-JKFIT"
    )
    scf = GHF(charge=0, ms_guess=1.0)(system)
    ci = CI(RelCISolver(nel=6, core_orbitals=5, active_orbitals=1))(scf)
    ci.run()
    assert scf.S2 == approx(2.0063122057820237)
    assert ci.ci_solver.spin2[0] == approx(scf.S2)
    assert ci.ci_solver.spin_vector[0] == approx([0.0, 0.0, 1.0])


def test_rel_ci_spin2_spin_free_limit():
    """Without spin-orbit coupling the 2c roots are exact spin eigenstates."""
    system = System(
        xyz="""
        H 0.0 0.0 0.0
        H 0.0 0.0 2.0
        """,
        basis_set="sto-6g",
        auxiliary_basis_set="cc-pVTZ-JKFIT",
        unit="bohr",
    )
    scf = RHF(charge=0, e_tol=1e-12)(system)
    conv = SpinorUpcaster(apply_random_phase=True)(scf)
    ci = CI(RelCISolver(nel=2, active_orbitals=4, nroots=6))(conv)
    ci.run()

    # two electrons in two spatial orbitals: three singlets and one triplet
    assert np.sort(ci.ci_solver.spin2) == approx([0.0, 0.0, 0.0, 2.0, 2.0, 2.0])

    ci = CI(RelCISolver(nel=2, active_orbitals=4, nroots=6), do_compute_spin2=False)(
        conv
    )
    ci.run()
    assert not hasattr(ci.ci_solver, "spin2")
