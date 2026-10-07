import numpy as np
import pytest

import forte2
from forte2.scf import RHF, UHF
from forte2.scf.scf_utils import guess_mix
from forte2.helpers.comparisons import approx, approx_abs
from forte2.integrals import emultipole1
from forte2.lib import ints
from forte2.symmetry.sym_utils import CHARACTER_TABLE, get_symmetry_ops


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


@pytest.mark.parametrize(
    "element,distance", [("N", 2.0), ("N", 3.0), ("C", 2.0), ("C", 3.0)]
)
def test_rhf_stretched_diatomic_irreps(element, distance):
    system = forte2.System(
        xyz=f"{element} 0 0 0; {element} 0 0 {distance}",
        basis_set="cc-pvdz",
        auxiliary_basis_set="cc-pvtz-jkfit",
        symmetry=True,
    )
    scf = RHF(charge=0)(system).run()
    assert system.point_group == "D2H"

    # Each orbital must pick up the character of its irrep under every operation.
    C = np.ascontiguousarray(scf.C[0])
    characters = np.array([CHARACTER_TABLE["D2H"][l] for l in scf.irrep_labels[0]])
    points = np.random.default_rng(265).uniform(-3, 3, (30, 3))
    values = ints.orbitals_at_points(system.basis, points, C)
    for j, R in enumerate(get_symmetry_ops("D2H").values()):
        transformed = ints.orbitals_at_points(system.basis, points @ R.T, C)
        np.testing.assert_allclose(
            transformed, values * characters[:, j], atol=1e-9, rtol=0
        )


def test_rhf_symmetrizes_near_symmetric_geometry():
    # One H is 1e-5 angstrom (1.9e-5 bohr) off the C2v geometry.
    def water(**kwargs):
        return forte2.System(
            xyz="O 0 0 0; H 0 0.76 0.59; H 0 -0.76 0.59001",
            basis_set="cc-pvdz",
            auxiliary_basis_set="cc-pvtz-jkfit",
            **kwargs,
        )

    # With symmetry_tol below the asymmetry, only the molecular plane remains.
    assert water(symmetry=True, symmetry_tol=1e-6).point_group == "CS"

    system = water(symmetry=True)
    assert system.point_group == "C2V"
    # Each operation maps the symmetrized atoms exactly onto their partners.
    positions = system.prin_atomic_positions
    for op, R in get_symmetry_ops("C2V").items():
        np.testing.assert_allclose(
            positions @ R.T, positions[system.atom_map[op]], atol=1e-12
        )

    # Symmetrization changes the energy only at second order in the displacement.
    rhf = RHF(charge=0)(system).run()
    assert rhf.E == approx_abs(RHF(charge=0)(water(symmetry=False)).run().E, 1e-9)
    assert rhf.irrep_labels[0][:5] == ["a1", "a1", "b2", "a1", "b1"]


def _water():
    return forte2.System(
        xyz="O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587",
        basis_set="cc-pvdz",
        auxiliary_basis_set="cc-pvtz-jkfit",
        symmetry=True,
    )


def test_rhf_rejects_symmetry_breaking_hamiltonian():

    def hcore_field(self, comp):
        mu = emultipole1(self.system)
        return self.system.ints_hcore() + 1e-3 * mu[comp]

    system_z = _water()
    system_z.ints_hcore = lambda: hcore_field(comp=3)  # z direction, tot. sym.
    scf_z = RHF(charge=0)(system_z)
    scf_z.run()

    with pytest.raises(ValueError, match="breaks C2V symmetry"):
        system_x = _water()
        system_x.ints_hcore = lambda: hcore_field(comp=1)  # x direction, b1
        scf_x = RHF(charge=0)(system_x)
        scf_x.run()


def test_rhf_adapts_supplied_guess():
    system = _water()
    rhf = RHF(charge=0)(system).run()
    homo, lumo = rhf.na - 1, rhf.na
    assert rhf.irrep_labels[0][homo] != rhf.irrep_labels[0][lumo]
    # Mixing the HOMO and LUMO, which have different irreps, breaks the symmetry.
    C = guess_mix(rhf.C[0], homo, lumo, mixing_parameter=0.3)[0]

    # Adapting by occupation recovers the occupied space of the symmetric solution.
    occupations = 2.0 * (np.arange(C.shape[1]) < rhf.na)
    adapted = system.symmetry_basis.adapt(C, occupations)
    system.symmetry_basis.orbital_irreps(adapted)
    occupied, reference = adapted[:, : rhf.na], rhf.C[0][:, : rhf.na]
    np.testing.assert_allclose(
        occupied @ occupied.T, reference @ reference.T, atol=1e-10
    )

    guess = RHF(charge=0)(system)
    guess.C = [C]
    guess.run()
    assert guess.E == approx(rhf.E)
    assert guess.irrep_labels[0] == rhf.irrep_labels[0]


def test_uhf_guess_mix_with_symmetry():
    def run(xyz, basis_set="cc-pvdz"):
        system = forte2.System(
            xyz=xyz,
            basis_set=basis_set,
            auxiliary_basis_set="def2-universal-jkfit",
            symmetry=True,
        )
        return UHF(charge=0, ms=0, guess_mix=True)(system).run()

    def assert_symmetric(uhf):
        # Each spin's orbitals transform as the irreps they are labeled with.
        basis = uhf.system.symmetry_basis
        for C, indices in zip(uhf.C, uhf.irrep_indices):
            assert basis.orbital_irreps(C).tolist() == indices

    # Stretched LiH: the HOMO and LUMO are both a1, so they are mixed as in C1 and
    # reach the same broken-symmetry solution.
    uhf = run("Li 0 0 0; H 0 0 4.0")
    assert uhf.E == approx(-7.932395483956)
    assert uhf.S2 == approx(0.9784292339)
    assert_symmetric(uhf)

    # Stretched H2: the HOMO (ag) and LUMO (b1u) differ, so a same-irrep pair is mixed
    # instead, or none in a minimal basis. Either way the orbitals stay symmetric.
    for basis_set in ("cc-pvdz", "sto-3g"):
        assert_symmetric(run("H 0 0 0; H 0 0 2.7", basis_set=basis_set))
