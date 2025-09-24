import forte2
from forte2.scf import RHF
from forte2.helpers.comparisons import approx


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
    assert list(map(str.upper, scf.irrep_labels)) == expected_mo_irreps


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
    assert list(map(str.upper, scf.irrep_labels)) == expected_mo_irreps


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
    assert list(map(str.upper, scf.irrep_labels)) == expected_mo_irreps


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
        assert list(map(str.upper, scf.irrep_labels)) == expected_mo_irreps
    except:
        for e1, e2 in zip(scf.irrep_labels, expected_mo_irreps):
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
                else:
                    raise AssertionError(
                        "Symmetry assignment wrong beyond b2g/b3g and b2u/b3u interchanges."
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
        assert list(map(str.upper, scf.irrep_labels)) == expected_mo_irreps
    except:
        for e1, e2 in zip(scf.irrep_labels, expected_mo_irreps):
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
                else:
                    raise AssertionError(
                        "Symmetry assignment wrong beyond b2g/b3g and b2u/b3u interchanges."
                    )
