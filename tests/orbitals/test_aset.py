import pytest
import numpy as np
from pathlib import Path

from forte2 import *
from forte2.helpers.comparisons import approx

# Directory containing *this* file
THIS_DIR = Path(__file__).resolve().parent


def compare_orbital_coefficients(system, aset, filename):
    """
    This function compares the coefficient matrix from an ASET calculation
    with a reference file stored in the folder reference_aset_orbitals.

    Note: this can only handle nondegenerate orbitals.
    """
    C_test = np.load(THIS_DIR / f"reference_aset_orbitals/{filename}")
    S = system.ints_overlap()
    overlap = np.abs(aset.C[0].T @ S @ C_test)
    assert np.allclose(overlap, np.eye(overlap.shape[0]), atol=1e-10, rtol=0.0)


# Ref Energies come from forte1


def test_aset_1():
    """
    test cutoff_method = threshold with a non-default cutoff value.
    """
    eci = -115.699156037836

    xyz = """
    C       -2.2314881720      2.3523969887      0.1565319638                 
    C       -1.1287322054      1.6651786288     -0.1651010551                 
    H       -3.2159664855      1.9109197306      0.0351701750                 
    H       -2.1807424354      3.3645292222      0.5457999612                 
    H       -1.2085033449      0.7043108616     -0.5330598833   
    C        0.2601218384      2.1970946692     -0.0290628762                 
    H        0.7545456004      2.2023392001     -1.0052240245                 
    H        0.8387453665      1.5599644558      0.6466877402                 
    H        0.2749376338      3.2174213526      0.3670138598  
    """

    system = System(
        xyz=xyz, basis_set="sto-3g", auxiliary_basis_set="def2-universal-JKFIT"
    )

    rhf = RHF(charge=0, econv=1e-12)(system)
    mc = MCOptimizer(
        State(nel=24, multiplicity=1, ms=0.0),
        core_orbitals=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        active_orbitals=[11, 12],
    )(rhf)
    aset = ASET(
        fragment=["C1-2", "H1-3"],
        cutoff_method="threshold",
        cutoff=0.99,
    )(mc)
    ci = CI(State(system=system, multiplicity=1, ms=0.0))(aset)
    ci.run()

    # write aset.C[0] to a numpy file
    # np.save("test_aset_1_orbitals.npy", aset.C[0])
    compare_orbital_coefficients(system, aset, "test_aset_1_orbitals.npy")

    assert ci.E == approx(eci)


def test_aset_2():
    eci = -206.084138520360

    xyz = """
    N       -1.1226987119      2.0137160725     -0.0992218410                 
    N       -0.1519067161      1.2402226172     -0.0345618482                 
    H        0.7253474870      1.7181546089     -0.2678695726          
    F       -2.2714806355      1.3880717623      0.2062454513     
    """

    system = System(
        xyz=xyz, basis_set="sto-3g", auxiliary_basis_set="def2-universal-JKFIT"
    )

    rhf = RHF(charge=0, econv=1e-12)(system)
    mc = MCOptimizer(
        State(nel=24, multiplicity=1, ms=0.0),
        frozen_core_orbitals=[0, 1, 2],
        core_orbitals=[3, 4, 5, 6, 7, 8, 9],
        active_orbitals=[10, 11, 12, 13],
    )(rhf)
    aset = ASET(
        fragment=["N", "H"],
        cutoff_method="threshold",
        cutoff=0.99,
    )(mc)
    ci = CI(State(system=system, multiplicity=1, ms=0.0))(aset)
    ci.run()

    # write aset.C[0] to a numpy file
    # np.save("test_aset_2_orbitals.npy", aset.C[0])
    compare_orbital_coefficients(system, aset, "test_aset_2_orbitals.npy")

    assert ci.E == approx(eci)


def test_aset_3():
    """
    test no semicanonicalization.
    """
    eci = -115.699156030288

    xyz = """
    C       -2.2314881720      2.3523969887      0.1565319638
    C       -1.1287322054      1.6651786288     -0.1651010551
    H       -3.2159664855      1.9109197306      0.0351701750
    H       -2.1807424354      3.3645292222      0.5457999612
    H       -1.2085033449      0.7043108616     -0.5330598833
    C        0.2601218384      2.1970946692     -0.0290628762
    H        0.7545456004      2.2023392001     -1.0052240245
    H        0.8387453665      1.5599644558      0.6466877402
    H        0.2749376338      3.2174213526      0.3670138598
"""

    system = System(
        xyz=xyz, basis_set="sto-3g", auxiliary_basis_set="def2-universal-JKFIT"
    )

    rhf = RHF(charge=0, econv=1e-12)(system)
    mc = MCOptimizer(
        State(nel=24, multiplicity=1, ms=0.0),
        frozen_core_orbitals=[0, 1, 2],
        core_orbitals=[3, 4, 5, 6, 7, 8, 9, 10],
        active_orbitals=[11, 12],
    )(rhf)
    aset = ASET(
        fragment=["C1-2", "H1-3"],
        cutoff_method="threshold",
        semicanonicalize_active=False,
        semicanonicalize_frozen=False,
    )(mc)
    ci = CI(State(system=system, multiplicity=1, ms=0.0))(aset)
    ci.run()

    # write aset.C[0] to a numpy file
    # np.save("test_aset_3_orbitals.npy", aset.C[0])
    compare_orbital_coefficients(system, aset, "test_aset_3_orbitals.npy")

    assert ci.E == approx(eci)


def test_aset_4():
    """
    Test cutoff_method = number of orbitals.
    """

    eci = -206.084138520357
    xyz = """
    N       -1.1226987119      2.0137160725     -0.0992218410
    N       -0.1519067161      1.2402226172     -0.0345618482
    H        0.7253474870      1.7181546089     -0.2678695726
    F       -2.2714806355      1.3880717623      0.2062454513
    """
    system = System(
        xyz=xyz, basis_set="sto-3g", auxiliary_basis_set="def2-universal-JKFIT"
    )

    rhf = RHF(charge=0, econv=1e-10)(system)
    mc = MCOptimizer(
        State(nel=24, multiplicity=1, ms=0.0),
        frozen_core_orbitals=[0, 1, 2],
        core_orbitals=[3, 4, 5, 6, 7, 8, 9],
        active_orbitals=[10, 11, 12, 13],
        econv=1e-9,
    )(rhf)
    aset = ASET(
        fragment=["N", "H"],
        cutoff_method="num_of_orbitals",
        num_A_occ=5,
        num_A_vir=1,
    )(mc)
    aset.run()
    ci = CI(State(system=system, multiplicity=1, ms=0.0))(aset)
    ci.run()

    # write aset.C[0] to a numpy file
    # np.save("test_aset_4_orbitals.npy", aset.C[0])
    compare_orbital_coefficients(system, aset, "test_aset_4_orbitals.npy")

    assert ci.E == approx(eci)


def test_aset_5():
    eci = -154.269037292918
    xyz = """
    C            0.736149969259     0.199718340898    -0.207219947401
    C            1.894302493759    -0.319955293970     0.296207387267
    H            0.861933668943     1.105847110317    -0.832928585892
    H            1.842233711006    -1.252567898836     0.893040798768
    H            2.864162955272     0.173377115363     0.186731686072
    C           -1.777918019119     0.526955710902     0.239774606960
    C           -0.669802211906    -0.436809943125    -0.347092635549
    H           -1.538823490089     0.918192642365     1.253716032316
    H           -2.797322479987     0.052951758306     0.328948031715
    H           -1.899218748385     1.428566644507    -0.416125458480
    H           -0.863484663283    -0.665562244675    -1.411954335033
    H           -0.645242334465    -1.402514539204     0.216831010104
"""

    system = System(
        xyz=xyz, basis_set="sto-3g", auxiliary_basis_set="def2-universal-JKFIT"
    )

    rhf = RHF(charge=0, econv=1e-12)(system)
    mc = MCOptimizer(
        State(system=system, multiplicity=1, ms=0.0),
        core_orbitals=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        active_orbitals=[15, 16],
    )(rhf)
    aset = ASET(fragment=["C1-2", "H1-3"], cutoff_method="threshold")(mc)
    ci = CI(State(system=system, multiplicity=1, ms=0.0))(aset)
    ci.run()

    # write aset.C[0] to a numpy file
    # np.save("test_aset_5_orbitals.npy", aset.C[0])
    compare_orbital_coefficients(system, aset, "test_aset_5_orbitals.npy")

    assert ci.E == approx(eci)
