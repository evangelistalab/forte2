import numpy as np
import pytest

import forte2
from forte2.scf import GHF, RHF, ROHF, UHF
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
    def water_in_field(component):
        system = _water()
        H = system.ints_hcore() + 1e-3 * emultipole1(system)[component]
        system.ints_hcore = lambda: H
        return system

    # A field along z, the C2 axis, is totally symmetric.
    RHF(charge=0)(water_in_field(3)).run()
    # A field along x, perpendicular to the molecular plane, transforms as b1.
    with pytest.raises(ValueError, match="breaks C2V symmetry"):
        RHF(charge=0)(water_in_field(1)).run()


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


@pytest.mark.parametrize(
    "method, e_a1",
    [(ROHF, -75.5438697705), (UHF, -75.5471570925)],
)
def test_scf_occupation_constraints(method, e_a1):
    # H2O+: aufbau gives the 2B1 ground state, and 2A1 has the hole in 3a1 instead.
    def cation(**kwargs):
        return method(charge=1, ms=0.5, **kwargs)(_water()).run()

    ground = cation()
    target = cation(target_symmetry="a1")
    # also test support for mixed irrep specification
    counts = cation(irrep_occupations={"a1": (3, 2), 2: 2, "B2": 2})
    assert ground.determinant_symmetry == "b1"
    assert target.determinant_symmetry == counts.determinant_symmetry == "a1"
    assert target.E == approx(e_a1)
    assert counts.E == approx(e_a1)
    assert ground.E < e_a1


def test_scf_occupation_constraint_errors():
    water = _water()
    # RHF cannot have a target of non-totally symmetric irrep
    with pytest.raises(ValueError, match="totally symmetric"):
        RHF(charge=0, target_symmetry="b1")(water)
    # target_symmetry is not consistent with irrep_occupations
    with pytest.raises(ValueError, match="target_symmetry"):
        ROHF(
            charge=1,
            ms=0.5,
            target_symmetry="a1",
            irrep_occupations={"a1": 6, "b1": (1, 0), "b2": 2},
        )(water)
    # wrong number of electrons
    with pytest.raises(ValueError, match="electrons"):
        UHF(charge=1, ms=0.5, irrep_occupations={"a1": 6, "b2": 2})(water)
    # unknown irrep label
    with pytest.raises(ValueError, match="Unknown irrep"):
        RHF(charge=0, irrep_occupations={"e": 10})(water)
    # GHF does not support symmetry
    with pytest.raises(ValueError, match="GHF"):
        GHF(charge=0, target_symmetry="a1")
    # In cc-pVDZ, water has only two a2 orbitals.
    with pytest.raises(
        ValueError, match=r"a2 \(3 alpha and 3 beta electrons, 2 orbitals\)"
    ):
        RHF(charge=0, irrep_occupations={"a1": 4, "a2": 6})(water)
    # user is trying to specify low-spin ROHF, i.e., ROHF must have unpair spins all a or b
    with pytest.raises(ValueError, match=r"b1 \(0 alpha, 1 beta\)"):
        ROHF(
            charge=1,
            ms=0.5,
            irrep_occupations={"a1": (3, 2), "b1": (0, 1), "b2": (2, 1)},
        )(water)
    # If integers are given as constraint, they must be even
    with pytest.raises(ValueError, match="odd number of electrons"):
        ROHF(charge=1, ms=0.5, irrep_occupations={"a1": 6, "b1": 1, "b2": 2})


def test_rohf_c2_sym():
    # we want the triplet Sigma_g^- solution (descends to B1g in D2h)
    # at around equilibrium ROHF sometimes converges to a higher B1u solution without constraints
    # at stretched bond lengths B1g is usually obtainable without constraints
    system = forte2.System(
        xyz="C 0 0 0; C 0 0 1.2",
        basis_set="cc-pvdz",
        auxiliary_basis_set="cc-pvtz-jkfit",
        symmetry=True,
    )
    scf = ROHF(
        charge=0,
        ms=1.0,
        irrep_occupations={"ag": 6, "b1u": 4, "b2u": (1, 0), "b3u": (1, 0)},
    )(system)
    scf.run()
    assert scf.E == approx(-75.460179559403)
    assert scf.determinant_symmetry == "b1g"
