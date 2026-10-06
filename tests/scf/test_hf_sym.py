import numpy as np
import pytest

import forte2
from forte2.scf import RHF, UHF, GHF
from forte2.helpers.comparisons import approx, approx_abs
from forte2.integrals import emultipole1
from forte2.symmetry import SymmetryBasis
from forte2.system import BasisInfo


def test_rhf_h2o_c2v():
    erhf = -76.061466407195
    expected_mo_irreps = [
        "A1",
        "A1",
        "B2",
        "A1",
        "B1",
        "A1",
        "B2",
        "A1",
        "B2",
        "A1",
        "B1",
        "A1",
        "B2",
        "A2",
        "B1",
        "A1",
        "B2",
        "A1",
        "B2",
        "B2",
        "A1",
        "A2",
        "B1",
        "A1",
        "A1",
        "B2",
        "B1",
        "A1",
        "A2",
        "B2",
        "B1",
        "A1",
        "B2",
        "B1",
        "A2",
        "A1",
        "B2",
        "A2",
        "A1",
        "B1",
        "B2",
        "A1",
        "B2",
        "A1",
        "B2",
        "A1",
        "B1",
        "A2",
        "B1",
        "A1",
        "B2",
        "A1",
        "B2",
        "B1",
        "A1",
        "A2",
        "A1",
        "B2",
        "B1",
        "A1",
        "A2",
        "B2",
        "A1",
        "A2",
        "B2",
        "B1",
        "B2",
        "A2",
        "B1",
        "A1",
        "B2",
        "A1",
        "B1",
        "A2",
        "B1",
        "A1",
        "B2",
        "B2",
        "A2",
        "A1",
        "B2",
        "A1",
        "B1",
        "A1",
        "A2",
        "B2",
        "A1",
        "B2",
        "B2",
        "A1",
        "B1",
        "A1",
        "B1",
        "A2",
        "A1",
        "B2",
        "A2",
        "B1",
        "A1",
        "B1",
        "B2",
        "A1",
        "B2",
        "B1",
        "A2",
        "A1",
        "A1",
        "B2",
        "B1",
        "A1",
        "A2",
        "B2",
        "A1",
        "B2",
        "A1",
    ]

    xyz = """
    O            0.000000000000     0.000000000000    -0.061664597388
    H            0.000000000000    -0.711620616369     0.489330954643
    H            0.000000000000     0.711620616369     0.489330954643
    """
    system = forte2.System(
        xyz=xyz,
        basis_set="cc-pVQZ",
        auxiliary_basis_set="cc-pVQZ-JKFIT",
        symmetry=True,
    )

    scf = RHF(charge=0)(system)
    scf.run()
    assert scf.E == approx(erhf)
    assert list(map(str.upper, scf.irrep_labels[0])) == expected_mo_irreps


def test_rhf_cbd_d2h():
    erhf = -153.6511710906
    # obtained from PySCF when CBD is oriented in the x-y plane
    # expected_mo_irreps =["AG", "B2U", "B3U", "B1G", "AG", "B2U", "B3U", "B1G",
    #                      "AG", "B2U", "AG", "B1U", "B3U", "B3G", "B2G", "AG",
    #                      "B2U", "B1G", "B3U", "AU", "B2U", "B3U", "B1G", "AG",
    #                      "AG", "B1G", "B2U", "B1U", "B3G", "B3U", "B2G", "B2U",
    #                      "AG", "AU", "B3U", "B1G", "AG", "B1G", "B2U", "B1U",
    #                      "B3U", "B1U", "B2U", "B1G", "B2G", "B3U", "B3G", "AG",
    #                      "AG", "B3U", "B1G", "B3G", "AU", "B2U", "B2G", "AG",
    #                      "B1U", "B2U", "B3U", "B1G", "AU", "B3G", "B2U", "AG",
    #                      "B3U", "B2G", "B1G", "B1G", "AU", "B2U", "B3U", "AG",
    #                      "B1G", "B2U", "B3U", "B1G"]
    # obtained from PySCF when CBD is oriented in the y-z plane (consistent with principal axis of rotation)
    expected_mo_irreps = [
        "AG",
        "B1U",
        "B2U",
        "B3G",
        "AG",
        "B1U",
        "B2U",
        "B3G",
        "AG",
        "B1U",
        "AG",
        "B3U",
        "B2U",
        "B2G",
        "B1G",
        "AG",
        "B1U",
        "B3G",
        "B2U",
        "AU",
        "B1U",
        "B2U",
        "B3G",
        "AG",
        "AG",
        "B3G",
        "B1U",
        "B3U",
        "B2G",
        "B2U",
        "B1G",
        "B1U",
        "AG",
        "AU",
        "B2U",
        "B3G",
        "AG",
        "B3G",
        "B1U",
        "B3U",
        "B2U",
        "B3U",
        "B1U",
        "B3G",
        "B1G",
        "B2U",
        "B2G",
        "AG",
        "AG",
        "B2U",
        "B3G",
        "B2G",
        "AU",
        "B1U",
        "B1G",
        "AG",
        "B3U",
        "B1U",
        "B2U",
        "B3G",
        "AU",
        "B2G",
        "B1U",
        "AG",
        "B2U",
        "B1G",
        "B3G",
        "B3G",
        "AU",
        "B1U",
        "B2U",
        "AG",
        "B3G",
        "B1U",
        "B2U",
        "B3G",
    ]

    xyz = """
    C    -1.2916277126       -1.4862694893        0.0000000000
    C     1.2916277126       -1.4862694893        0.0000000000
    C    -1.2916277126        1.4862694893        0.0000000000
    C     1.2916277126        1.4862694893       -0.0000000000
    H    -2.7546827497       -2.9442264047        0.0000000000
    H     2.7546827497       -2.9442264047        0.0000000000
    H    -2.7546827497        2.9442264047        0.0000000000
    H     2.7546827497        2.9442264047       -0.0000000000
    """

    system = forte2.System(
        xyz=xyz,
        basis_set="cc-pvdz",
        cholesky_tei=True,
        cholesky_tol=1e-10,
        symmetry=True,
        unit="bohr",
    )

    scf = RHF(charge=0)(system)
    scf.run()
    assert scf.E == approx(erhf)
    assert list(map(str.upper, scf.irrep_labels[0])) == expected_mo_irreps


def test_rhf_h2o_c2v_rot():
    erhf = -76.061466407195
    expected_mo_irreps = [
        "A1",
        "A1",
        "B2",
        "A1",
        "B1",
        "A1",
        "B2",
        "A1",
        "B2",
        "A1",
        "B1",
        "A1",
        "B2",
        "A2",
        "B1",
        "A1",
        "B2",
        "A1",
        "B2",
        "B2",
        "A1",
        "A2",
        "B1",
        "A1",
        "A1",
        "B2",
        "B1",
        "A1",
        "A2",
        "B2",
        "B1",
        "A1",
        "B2",
        "B1",
        "A2",
        "A1",
        "B2",
        "A2",
        "A1",
        "B1",
        "B2",
        "A1",
        "B2",
        "A1",
        "B2",
        "A1",
        "B1",
        "A2",
        "B1",
        "A1",
        "B2",
        "A1",
        "B2",
        "B1",
        "A1",
        "A2",
        "A1",
        "B2",
        "B1",
        "A1",
        "A2",
        "B2",
        "A1",
        "A2",
        "B2",
        "B1",
        "B2",
        "A2",
        "B1",
        "A1",
        "B2",
        "A1",
        "B1",
        "A2",
        "B1",
        "A1",
        "B2",
        "B2",
        "A2",
        "A1",
        "B2",
        "A1",
        "B1",
        "A1",
        "A2",
        "B2",
        "A1",
        "B2",
        "B2",
        "A1",
        "B1",
        "A1",
        "B1",
        "A2",
        "A1",
        "B2",
        "A2",
        "B1",
        "A1",
        "B1",
        "B2",
        "A1",
        "B2",
        "B1",
        "A2",
        "A1",
        "A1",
        "B2",
        "B1",
        "A1",
        "A2",
        "B2",
        "A1",
        "B2",
        "A1",
    ]
    xyz = """
    O   0.000000000000   0.030832298694  -0.053403107852
    H   0.000000000000  -0.860947008954   0.067962729394
    H   0.000000000000   0.371616054311   0.779583345763
    """
    system = forte2.System(
        xyz=xyz,
        basis_set="cc-pVQZ",
        auxiliary_basis_set="cc-pVQZ-JKFIT",
        symmetry=True,
    )

    scf = RHF(charge=0)(system)
    scf.run()
    assert scf.E == approx(erhf)
    assert list(map(str.upper, scf.irrep_labels[0])) == expected_mo_irreps


def test_rhf_n2_d2h_x():
    erhf = -108.94729293307688
    expected_mo_irreps = [
        "AG",
        "B1U",
        "AG",
        "B1U",
        "AG",
        "B3U",
        "B2U",
        "B2G",
        "B3G",
        "B1U",
        "AG",
        "B2U",
        "B3U",
        "AG",
        "B3G",
        "B2G",
        "B1U",
        "B1U",
        "AG",
        "B1G",
        "B3U",
        "B2U",
        "B1U",
        "AU",
        "AG",
        "B3G",
        "B2G",
        "B1U",
    ]
    xyz = """
    N            0.000000000000     0.000000000000     0.000000000000
    N            1.128000000000     0.000000000000     0.000000000000
    """

    system = forte2.System(
        xyz=xyz,
        basis_set="cc-pvdz",
        cholesky_tei=True,
        cholesky_tol=1e-10,
        symmetry=True,
    )

    scf = RHF(charge=0)(system)
    scf.run()
    assert scf.E == approx(erhf)

    try:
        assert list(map(str.upper, scf.irrep_labels[0])) == expected_mo_irreps
    except:
        for e1, e2 in zip(scf.irrep_labels[0], expected_mo_irreps):
            try:
                assert e1.upper() == e2
            except:
                if e1.upper() == "B2G" and e2 == "B3G":
                    continue
                elif e1.upper() == "B3G" and e2 == "B2G":
                    continue
                elif e1.upper() == "B2U" and e2 == "B3U":
                    continue
                elif e1.upper() == "B3U" and e2 == "B2U":
                    continue
                elif e1.upper() == "B1G" and e2 == "AG":
                    continue
                elif e1.upper() == "AG" and e2 == "B1G":
                    continue
                else:
                    raise AssertionError(
                        f"Symmetry assignment wrong beyond ag/b1g, b2g/b3g and b2u/b3u interchanges: {e1} != {e2}."
                    )


def test_rhf_n2_d2h():
    erhf = -108.94729293307688
    expected_mo_irreps = [
        "AG",
        "B1U",
        "AG",
        "B1U",
        "AG",
        "B3U",
        "B2U",
        "B2G",
        "B3G",
        "B1U",
        "AG",
        "B2U",
        "B3U",
        "AG",
        "B3G",
        "B2G",
        "B1U",
        "B1U",
        "AG",
        "B1G",
        "B3U",
        "B2U",
        "B1U",
        "AU",
        "AG",
        "B3G",
        "B2G",
        "B1U",
    ]
    xyz = """
    N            0.000000000000     0.000000000000     0.000000000000
    N            0.000000000000     0.000000000000     1.128000000000
    """

    system = forte2.System(
        xyz=xyz,
        basis_set="cc-pvdz",
        cholesky_tei=True,
        cholesky_tol=1e-10,
        symmetry=True,
    )

    scf = RHF(charge=0)(system)
    scf.run()
    assert scf.E == approx(erhf)

    try:
        assert list(map(str.upper, scf.irrep_labels[0])) == expected_mo_irreps
    except:
        for e1, e2 in zip(scf.irrep_labels[0], expected_mo_irreps):
            try:
                assert e1.upper() == e2
            except:
                if e1.upper() == "B2G" and e2 == "B3G":
                    continue
                elif e1.upper() == "B3G" and e2 == "B2G":
                    continue
                elif e1.upper() == "B2U" and e2 == "B3U":
                    continue
                elif e1.upper() == "B3U" and e2 == "B2U":
                    continue
                elif e1.upper() == "B1G" and e2 == "AG":
                    continue
                elif e1.upper() == "AG" and e2 == "B1G":
                    continue
                else:
                    raise AssertionError(
                        f"Symmetry assignment wrong beyond ag/b1g, b2g/b3g and b2u/b3u interchanges: {e1} != {e2}."
                    )


def test_ghf_runs_in_c1():
    # GHF ignores the point group of a symmetric system and labels its spinors in C1.
    def water():
        return forte2.System(
            xyz="O 0 0 0; H 0 0.76 0.59; H 0 -0.76 0.59",
            basis_set="sto-3g",
            auxiliary_basis_set="def2-universal-jkfit",
            symmetry=True,
        )

    rhf = RHF(charge=0)(water()).run()
    ghf = GHF(charge=0)(water()).run()
    assert ghf.system.point_group == "C2V"
    assert ghf.orbital_point_group == "C1"
    assert set(ghf.irrep_labels[0]) == {"a"}
    assert ghf.E == approx(rhf.E)

    with pytest.raises(ValueError, match="only supported by"):
        GHF(charge=0, target_symmetry="a1")


def test_uhf_guess_mix_with_symmetry():
    # Stretched LiH breaks spin symmetry between orbitals of the same irrep (a1),
    # so mixing a same-irrep pair reaches the C1 broken-symmetry solution.
    system = forte2.System(
        xyz="Li 0 0 0; H 0 0 4.0",
        basis_set="cc-pvdz",
        auxiliary_basis_set="def2-universal-jkfit",
        symmetry=True,
    )
    uhf = UHF(charge=0, ms=0, guess_mix=True)(system).run()
    assert uhf.E == approx(-7.932395483955529)
    assert uhf.S2 == approx(0.9784292338841001)
    assert uhf.irrep_labels[0][:2] == ["a1", "a1"]


def test_rhf_symmetrizes_near_symmetric_geometry():
    # One H is 1e-5 angstrom off the C2v geometry, within symmetry_tol.
    def water(symmetry):
        return forte2.System(
            xyz="O 0 0 0; H 0 0.76 0.59; H 0 -0.76 0.59001",
            basis_set="cc-pvdz",
            auxiliary_basis_set="cc-pvtz-jkfit",
            symmetry=symmetry,
        )

    system = water(True)
    assert system.point_group == "C2V"
    positions = system.prin_atomic_positions
    reflected = positions * [1, -1, 1]
    assert np.allclose(
        np.sort(reflected, axis=0), np.sort(positions, axis=0), atol=1e-14
    )

    # Symmetrization changes the energy only at second order in the displacement.
    rhf = RHF(charge=0)(system).run()
    assert rhf.E == approx_abs(RHF(charge=0)(water(False)).run().E, 1e-9)
    assert rhf.irrep_labels[0][:5] == ["a1", "a1", "b2", "a1", "b1"]


def _water():
    return forte2.System(
        xyz="O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587",
        basis_set="cc-pvdz",
        auxiliary_basis_set="cc-pvtz-jkfit",
        symmetry=True,
    )


def test_symmetry_check_on_core_hamiltonian():
    class FieldRHF(RHF):
        component = 3  # along z, the C2 axis (a1)

        def _get_hcore(self):
            mu = emultipole1(self.system)
            return self.system.ints_hcore() + 1e-3 * mu[self.component]

    class PerpendicularFieldRHF(FieldRHF):
        component = 1  # along x, perpendicular to the molecular plane (b1)

    # A totally symmetric field keeps C2v, so the symmetric solution is stationary.
    rhf = FieldRHF(charge=0)(_water()).run()
    S = rhf.system.ints_overlap()
    F, D = rhf.F[0], rhf.D[0]
    assert np.linalg.norm(F @ D @ S - S @ D @ F) < 1e-6

    with pytest.raises(ValueError, match="core Hamiltonian breaks C2V symmetry"):
        PerpendicularFieldRHF(charge=0)(_water()).run()


def test_supplied_guess_is_symmetry_adapted():
    system = _water()
    rhf = RHF(charge=0)(system).run()
    C = rhf.C[0].copy()
    homo, lumo = rhf.na - 1, rhf.na
    assert rhf.irrep_labels[0][homo] != rhf.irrep_labels[0][lumo]
    # Mixing the HOMO and LUMO, which have different irreps, breaks the symmetry.
    c, s = np.cos(0.3), np.sin(0.3)
    C[:, [homo, lumo]] = C[:, [homo, lumo]] @ np.array([[c, -s], [s, c]])

    # Ranking by occupation recovers the symmetric occupied space of the guess.
    occupations = (np.arange(C.shape[1]) < rhf.na).astype(float)
    basis = SymmetryBasis.build(
        system,
        BasisInfo(system, system.basis),
        system.ints_overlap(),
        system.get_Xorth(),
    )
    _, adapted, _ = basis.adapt(C, -occupations)
    basis.orbital_irreps(adapted)
    occupied, reference = adapted[:, : rhf.na], rhf.C[0][:, : rhf.na]
    np.testing.assert_allclose(
        occupied @ occupied.T, reference @ reference.T, atol=1e-10
    )

    guess = RHF(charge=0)(system)
    guess.C = [C]
    guess.run()
    assert guess.E == approx(rhf.E)
    assert guess.irrep_labels[0] == rhf.irrep_labels[0]
