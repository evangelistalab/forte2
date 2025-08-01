import pytest

from forte2 import *
from forte2.helpers.comparisons import approx


def test_aset_1():
    """
    test cutoff_method = threshold with a non-default cutoff value.
    """
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

    system = System(xyz=xyz, basis_set="cc-pvdz", auxiliary_basis_set="cc-pVTZ-JKFIT")

    rhf = RHF(charge=0, econv=1e-12)(system)
    mc = MCOptimizer(
        State(nel=24, multiplicity=1, ms=0.0),
        core_orbitals=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        active_orbitals=[11, 12],
    )(rhf)
    aset = ASET(
        fragment=["C1-2", "H1-3"],
        cutoff_method="threshold",
        cutoff=0.1,
        semicanonicalize_active=False,
        semicanonicalize_frozen=False,
    )(mc)
    aset.run()


test_aset_1()


def test_aset_2():
    """
    Test cutoff_method = cumulative_threshold.
    """
    xyz = """
    N       -1.1226987119      2.0137160725     -0.0992218410                 
    N       -0.1519067161      1.2402226172     -0.0345618482                 
    H        0.7253474870      1.7181546089     -0.2678695726          
    F       -2.2714806355      1.3880717623      0.2062454513     
    """

    system = System(xyz=xyz, basis_set="cc-pvdz", auxiliary_basis_set="cc-pVTZ-JKFIT")

    rhf = RHF(charge=0, econv=1e-12)(system)
    mc = MCOptimizer(
        State(nel=24, multiplicity=1, ms=0.0),
        core_orbitals=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        active_orbitals=[10, 11, 12, 13],
    )(rhf)
    aset = ASET(
        fragment=["N", "H"],
        cutoff_method="cumulative_threshold",
        cutoff=0.99,
        semicanonicalize_active=False,
        semicanonicalize_frozen=False,
    )(mc)
    aset.run()


def test_aset_3():
    """
    This test was used to check for embedding_reference = HF in Forte1.
    The option of choosing HF to be the reference is not included in Forte2.
    Right now we test for cutoff = default = 0.5.
    This test will be updated when semicanonicalization becomes available.
    """
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

    system = System(xyz=xyz, basis_set="cc-pvdz", auxiliary_basis_set="cc-pVTZ-JKFIT")

    rhf = RHF(charge=0, econv=1e-12)(system)
    mc = MCOptimizer(
        State(nel=24, multiplicity=1, ms=0.0),
        core_orbitals=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        active_orbitals=[11, 12],
    )(rhf)
    aset = ASET(
        fragment=["C1-2", "H1-3"],
        cutoff_method="threshold",
        semicanonicalize_active=False,
        semicanonicalize_frozen=False,
    )(mc)
    aset.run()


def test_aset_4():
    """
    Test cutoff_method = number of orbitals.
    """
    xyz = """
    N       -1.1226987119      2.0137160725     -0.0992218410
    N       -0.1519067161      1.2402226172     -0.0345618482
    H        0.7253474870      1.7181546089     -0.2678695726
    F       -2.2714806355      1.3880717623      0.2062454513
    """
    system = System(xyz=xyz, basis_set="cc-pvdz", auxiliary_basis_set="cc-pVTZ-JKFIT")

    rhf = RHF(charge=0, econv=1e-12)(system)
    mc = MCOptimizer(
        State(nel=24, multiplicity=1, ms=0.0),
        core_orbitals=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        active_orbitals=[10, 11, 12, 13],
    )(rhf)
    aset = ASET(
        fragment=["N", "H"],
        cutoff_method="num_of_orbitals",
        num_a_docc=5,
        num_a_uocc=1,
        semicanonicalize_active=False,
        semicanonicalize_frozen=False,
    )(mc)
    aset.run()


# def test_aset_5():
#     """
#     Test PAO for virtual space, not yet a feature.
#     """
#     xyz = """
#     C            0.736149969259     0.199718340898    -0.207219947401
#     C            1.894302493759    -0.319955293970     0.296207387267
#     H            0.861933668943     1.105847110317    -0.832928585892
#     H            1.842233711006    -1.252567898836     0.893040798768
#     H            2.864162955272     0.173377115363     0.186731686072
#     C           -1.777918019119     0.526955710902     0.239774606960
#     C           -0.669802211906    -0.436809943125    -0.347092635549
#     H           -1.538823490089     0.918192642365     1.253716032316
#     H           -2.797322479987     0.052951758306     0.328948031715
#     H           -1.899218748385     1.428566644507    -0.416125458480
#     H           -0.863484663283    -0.665562244675    -1.411954335033
#     H           -0.645242334465    -1.402514539204     0.216831010104
# """

#     system = System(xyz=xyz, basis_set="cc-pvdz", auxiliary_basis_set="cc-pVTZ-JKFIT")

#     rhf = RHF(charge=0, econv=1e-12)(system)
#     mc = MCOptimizer(State(nel=24, multiplicity=1, ms=0.0), core_orbitals=[0,1,2,3,4,5,6,7,8,9,10], active_orbitals=[11,12])(rhf)
#     aset = ASET(
#         fragment=["C1-2", "H1-3"],
#         cutoff_method="threshold",
#         cutoff = 0.5,
#         virtual_space= PAO,
#         semicanonicalize_active=False,
#         semicanonicalize_frozen=False
#     )(mc)
#     aset.run()
