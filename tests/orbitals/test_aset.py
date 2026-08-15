import pytest
import numpy as np
from pathlib import Path

from forte2 import System, RHF, MCOptimizer, ASET, CI, State, CISolver
from forte2.dsrg import DSRG_MRPT2
from forte2.helpers.comparisons import approx, approx_abs
from forte2.state import EmbeddingMOSpace

# Directory containing *this* file
THIS_DIR = Path(__file__).resolve().parent


def test_aset_1_forte_v1_embedding_1():
    """
    Match the Forte v1 embedding-1 reference input with Cholesky integrals.

    This test uses the propene molecule and defines the fragment as the
    vinyl group -CH=CH2 and the environment as the methyl group -CH3.
    """
    emcscf = -115.698779193811518
    edsrgpt2 = -115.778915313387614

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
        xyz=xyz,
        basis_set="sto-3g",
        cholesky_tei=True,
        cholesky_tol=1e-10,
    )

    rhf = RHF(charge=0, e_tol=1e-12)(system)
    ci_solver = CISolver(
        State(nel=24, multiplicity=1, ms=0.0),
        core_orbitals=11,
        active_orbitals=2,
    )
    mc = MCOptimizer(
        ci_solver,
        e_tol=1e-10,
        g_tol=1e-10,
    )(rhf)
    aset = ASET(
        fragment=["C1-2", "H1-3"],
        cutoff_method="threshold",
        cutoff=0.1,
    )(mc)
    ci = CI(
        State(system=system, multiplicity=1, ms=0.0),
        final_orbitals="semicanonical",
    )(aset)
    dsrg = DSRG_MRPT2(flow_param=0.5)(ci)
    dsrg.run()

    assert mc.E == approx(emcscf)
    assert ci.E == approx(emcscf)
    assert dsrg.E_dsrg == approx(edsrgpt2)

    assert aset.mo_space.frozen_core_indices == [0, 1, 2, 3]
    assert aset.mo_space.core_indices == [4, 5, 6, 7, 8, 9, 10]
    assert aset.mo_space.active_indices == [11, 12]
    assert aset.mo_space.virtual_indices == [13, 14, 15, 16, 17]
    assert aset.mo_space.frozen_virtual_indices == [18, 19, 20]
    assert aset.fragment == ["C1-2", "H1-3"]


def test_aset_4_forte_v1_embedding_4():
    """
    Match the Forte v1 embedding-4 reference input with Cholesky integrals.

    This test uses the fluorodiazene (HNNF) molecule and defines the fragment
    as the -NNH group and the environment as the F atom.
    """
    emcscf = -206.083844698525638
    edsrgpt2 = -206.105821145367486

    xyz = """
    N       -1.1226987119      2.0137160725     -0.0992218410
    N       -0.1519067161      1.2402226172     -0.0345618482
    H        0.7253474870      1.7181546089     -0.2678695726
    F       -2.2714806355      1.3880717623      0.2062454513
    """

    system = System(
        xyz=xyz,
        basis_set="sto-3g",
        cholesky_tei=True,
        cholesky_tol=1e-10,
    )

    rhf = RHF(charge=0, e_tol=1e-12)(system)
    ci_solver = CISolver(
        State(nel=24, multiplicity=1, ms=0.0),
        frozen_core_orbitals=3,
        core_orbitals=7,
        active_orbitals=4,
    )
    mc = MCOptimizer(
        ci_solver,
        e_tol=1e-10,
        g_tol=1e-10,
        maxiter=300,
        max_rotation=0.15,
    )(rhf)
    aset = ASET(
        fragment=["N", "H"],
        cutoff_method="num_of_orbitals",
        num_A_occ=5,
        num_A_vir=1,
    )(mc)
    ci = CI(
        State(system=system, multiplicity=1, ms=0.0),
        final_orbitals="semicanonical",
    )(aset)
    dsrg = DSRG_MRPT2(flow_param=0.5)(ci)
    dsrg.run()

    assert mc.E == approx_abs(emcscf, 2e-7)
    assert ci.E == approx_abs(emcscf, 2e-7)
    assert dsrg.E_dsrg == approx_abs(edsrgpt2, 2e-7)

    assert aset.mo_space.frozen_core_indices == [0, 1, 2, 3, 4]
    assert aset.mo_space.core_indices == [5, 6, 7, 8, 9]
    assert aset.mo_space.active_indices == [10, 11, 12, 13]
    assert aset.mo_space.virtual_indices == [14]
    assert aset.mo_space.frozen_virtual_indices == [15]


def compare_orbital_coefficients(system, aset, filename):
    """
    This function compares the coefficient matrix from an ASET calculation
    with a reference file stored in the folder reference_aset_orbitals.

    Note: this can only handle nondegenerate orbitals.
    """
    C_test = np.load(THIS_DIR / f"reference_aset_orbitals/{filename}")
    S = system.ints_overlap()
    overlap = np.abs(aset.C[0].T @ S @ C_test)
    assert np.allclose(overlap, np.eye(overlap.shape[0]), atol=1e-8, rtol=0.0)


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
        xyz=xyz,
        basis_set="sto-3g",
        auxiliary_basis_set="def2-universal-JKFIT",
    )

    rhf = RHF(charge=0, e_tol=1e-12)(system)
    ci_solver = CISolver(
        State(nel=24, multiplicity=1, ms=0.0),
        core_orbitals=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        active_orbitals=[11, 12],
    )
    mc = MCOptimizer(ci_solver)(rhf)
    aset = ASET(
        fragment=["C1-2", "H1-3"],
        cutoff_method="threshold",
        cutoff=0.99,
    )(mc)
    ci = CI(State(system=system, multiplicity=1, ms=0.0))(aset)
    ci.run()

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
        xyz=xyz,
        basis_set="sto-3g",
        auxiliary_basis_set="def2-universal-JKFIT",
    )

    rhf = RHF(charge=0, e_tol=1e-12)(system)
    ci_solver = CISolver(
        State(nel=24, multiplicity=1, ms=0.0),
        core_orbitals=10,
        active_orbitals=4,
    )
    mc = MCOptimizer(ci_solver)(rhf)
    aset = ASET(
        fragment=["N", "H"],
        frozen_core_orbitals=3,
        cutoff_method="threshold",
        cutoff=0.99,
    )(mc)
    ci = CI(State(system=system, multiplicity=1, ms=0.0))(aset)
    ci.run()

    compare_orbital_coefficients(system, aset, "test_aset_2_orbitals.npy")

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
        xyz=xyz,
        basis_set="sto-3g",
        auxiliary_basis_set="def2-universal-JKFIT",
    )

    rhf = RHF(charge=0, e_tol=1e-10)(system)
    ci_solver = CISolver(
        State(nel=24, multiplicity=1, ms=0.0),
        core_orbitals=10,
        active_orbitals=4,
    )
    mc = MCOptimizer(
        ci_solver,
        e_tol=1e-9,
    )(rhf)
    aset = ASET(
        fragment=["N", "H"],
        frozen_core_orbitals=3,
        cutoff_method="num_of_orbitals",
        num_A_occ=5,
        num_A_vir=1,
    )(mc)
    aset.run()
    ci = CI(State(system=system, multiplicity=1, ms=0.0))(aset)
    ci.run()

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
        xyz=xyz,
        basis_set="sto-3g",
        auxiliary_basis_set="def2-universal-JKFIT",
    )

    rhf = RHF(charge=0, e_tol=1e-12)(system)
    ci_solver = CISolver(
        State(system=system, multiplicity=1, ms=0.0),
        core_orbitals=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        active_orbitals=[15, 16],
    )
    mc = MCOptimizer(ci_solver)(rhf)
    aset = ASET(fragment=["C1-2", "H1-3"], cutoff_method="threshold")(mc)
    ci = CI(State(system=system, multiplicity=1, ms=0.0))(aset)
    ci.run()

    compare_orbital_coefficients(system, aset, "test_aset_5_orbitals.npy")

    assert ci.E == approx(eci)


def test_aset_gas():
    xyz = """
    O   0.0000000000  -0.0000000000  -0.0662628033
    H   0.0000000000  -0.7540256101   0.5259060578
    H  -0.0000000000   0.7530256101   0.5260060578
    """

    system = System(
        xyz=xyz,
        basis_set="sto-3g",
        auxiliary_basis_set="def2-universal-JKFIT",
    )

    state = State(
        nel=10,
        multiplicity=1,
        ms=0.0,
        gas_min=[0, 0],
        gas_max=[4, 4],
    )
    rhf = RHF(charge=0, e_tol=1e-10)(system)
    ci_solver = CISolver(
        state,
        core_orbitals=2,
        active_orbitals=(2, 2),
    )
    mc = MCOptimizer(
        ci_solver,
        freeze_inter_gas_rots=True,
        maxiter=1,
        die_if_not_converged=False,
        final_orbitals="original",
    )(rhf)
    aset = ASET(fragment=["O"], cutoff_method="threshold")(mc)
    aset.run()

    assert aset.partition["active_orbitals"] == mc.mo_space.active_orbitals
    assert aset.mo_space.ngas == mc.mo_space.ngas
    assert aset.mo_space.active_orbitals == mc.mo_space.active_orbitals

    ci = CI(state)(aset)
    ci.run()
    assert ci.mo_space.ngas == 2

    # Check that the CASCI energy is preserved
    assert ci.E == approx(mc.E)


def test_aset_gas_semicanonical_noncontiguous_mo_space():
    xyz = """
    O   0.0000000000  -0.0000000000  -0.0662628033
    H   0.0000000000  -0.7540256101   0.5259060578
    H  -0.0000000000   0.7530256101   0.5260060578
    """

    system = System(
        xyz=xyz,
        basis_set="sto-3g",
        auxiliary_basis_set="def2-universal-JKFIT",
    )
    state = State(
        nel=10,
        multiplicity=1,
        ms=0.0,
        gas_min=[0, 0],
        gas_max=[4, 4],
    )
    rhf = RHF(charge=0, e_tol=1e-10)(system)
    ci_solver = CISolver(
        state,
        core_orbitals=3,
        active_orbitals=(2, 2),
    )
    mc = MCOptimizer(
        ci_solver,
        freeze_inter_gas_rots=True,
        maxiter=1,
        die_if_not_converged=False,
        final_orbitals="original",
    )(rhf)
    aset = ASET(
        fragment=["O", "H"],
        frozen_core_orbitals=[2],
        cutoff_method="threshold",
    )(mc)
    aset.run()

    np.testing.assert_array_equal(
        aset.mo_space.orig_to_contig,
        [2, 0, 1, 3, 4, 5, 6],
    )
    np.testing.assert_array_equal(
        aset.mo_space.contig_to_orig,
        [1, 2, 0, 3, 4, 5, 6],
    )

    ci = CI(state, final_orbitals="semicanonical")(aset)
    ci.run()

    assert ci.E == approx(mc.E)
    np.testing.assert_allclose(
        ci.C[0].conj().T @ system.ints_overlap() @ ci.C[0],
        np.eye(system.nmo),
        atol=1e-10,
    )


def test_aset_gas_semicanonical_noninteracting_fragments():
    xyz = """
    F  0.0 0.0    0.0
    H  0.0 0.0    1.7
    He 0.0 0.0 1000.0
    """

    system = System(
        xyz=xyz,
        basis_set="sto-3g",
        auxiliary_basis_set="def2-universal-JKFIT",
        unit="bohr",
    )
    state = State(
        nel=12,
        multiplicity=1,
        ms=0.0,
        gas_min=[0, 0, 0],
        gas_max=[2, 2, 2],
    )
    rhf = RHF(charge=0, e_tol=1e-10)(system)
    # Order the GASes so that converting between MO layouts requires a three-cycle.
    ci_solver = CISolver(
        state,
        core_orbitals=[0, 1, 2, 3],
        active_orbitals=[[6], [4], [5]],
    )
    mc = MCOptimizer(
        ci_solver,
        freeze_inter_gas_rots=True,
        maxiter=1,
        die_if_not_converged=False,
        final_orbitals="original",
    )(rhf)
    aset = ASET(fragment=["F", "H"], cutoff_method="threshold")(mc)
    aset.run()

    assert aset.partition["index_A_occ"] == [1, 2, 3]
    assert aset.partition["index_B_occ"] == [0]
    assert aset.mo_space.active_orbitals == [[6], [4], [5]]
    np.testing.assert_array_equal(
        aset.mo_space.orig_to_contig,
        [0, 1, 2, 3, 6, 4, 5],
    )
    np.testing.assert_array_equal(
        aset.mo_space.contig_to_orig,
        [0, 1, 2, 3, 5, 6, 4],
    )

    hf_system = System(
        xyz="F 0.0 0.0 0.0\nH 0.0 0.0 1.7",
        basis_set="sto-3g",
        auxiliary_basis_set="def2-universal-JKFIT",
        unit="bohr",
    )
    hf_state = State(
        nel=10,
        multiplicity=1,
        ms=0.0,
        gas_min=[0, 0, 0],
        gas_max=[2, 2, 2],
    )
    hf_mc = MCOptimizer(
        CISolver(
            hf_state,
            core_orbitals=[0, 1, 2],
            active_orbitals=[[5], [3], [4]],
        ),
        freeze_inter_gas_rots=True,
        maxiter=1,
        die_if_not_converged=False,
        final_orbitals="original",
    )(RHF(charge=0, e_tol=1e-10)(hf_system))
    hf_mc.run()

    he_system = System(
        xyz="He 0.0 0.0 0.0",
        basis_set="sto-3g",
        auxiliary_basis_set="def2-universal-JKFIT",
        unit="bohr",
    )
    he_rhf = RHF(charge=0, e_tol=1e-10)(he_system)
    he_rhf.run()

    assert mc.E == approx(hf_mc.E + he_rhf.E)

    ci = CI(state, final_orbitals="semicanonical")(aset)
    ci.run()

    assert ci.E == approx(mc.E)
    np.testing.assert_allclose(
        ci.C[0].conj().T @ system.ints_overlap() @ ci.C[0],
        np.eye(system.nmo),
        atol=1e-10,
    )


def spans_same_space(S, C1, C2):
    """
    Check whether the column sets C1 and C2 span the same space.

    Both column sets are assumed orthonormal with respect to the metric S, as
    MO coefficients always are. The singular values of C1^T S C2 are then the
    cosines of the principal angles between the two subspaces, and they are all
    equal to one if and only if the spans coincide.
    """
    assert C1.shape == C2.shape
    sv = np.linalg.svd(C1.conj().T @ S @ C2, compute_uv=False)
    return np.allclose(sv, 1.0, atol=1e-8, rtol=0.0)


def test_aset_noncontiguous_frozen_core_orbital_ordering():
    """
    Check that ASET places the embedding orbitals in the correct MO slots.

    ASET converts between the original and contiguous MO layouts three times:
    when building the fragment projector, when handing the orbitals to the
    semicanonicalizer, and when writing the result back. Choosing frozen core
    orbitals that are not contiguous makes the permutation differ from both the
    identity and its own inverse, so applying the wrong member of the
    orig_to_contig/contig_to_orig pair scrambles the orbitals.

    That scrambling only mixes orbitals within the occupied block, which leaves
    the CASCI energy and the orthonormality of C invariant. The assertions here
    are therefore based on which orbital ends up in which labeled slot: the
    subspaces the user pinned by index must be preserved, and the orbitals
    assigned to fragment A must be the ones localized on the fragment.
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

    system = System(
        xyz=xyz,
        basis_set="sto-3g",
        auxiliary_basis_set="def2-universal-JKFIT",
    )

    rhf = RHF(charge=0, e_tol=1e-12)(system)
    ci_solver = CISolver(
        State(nel=24, multiplicity=1, ms=0.0),
        core_orbitals=11,
        active_orbitals=2,
    )
    mc = MCOptimizer(ci_solver, die_if_not_converged=False)(rhf)

    # Leaving a gap in the frozen core makes the embedding permutation a
    # non-involution, which is what distinguishes the two permutation directions.
    frozen_core = [0, 1, 5]
    aset = ASET(
        fragment=["C1-2", "H1-3"],
        frozen_core_orbitals=frozen_core,
        cutoff_method="threshold",
    )(mc)
    aset.run()

    partition = aset.partition
    index_A_occ = partition["index_A_occ"]
    index_B_occ = partition["index_B_occ"]
    index_A_vir = partition["index_A_vir"]
    index_B_vir = partition["index_B_vir"]

    # Rebuild the space ASET used internally to confirm the premise of the test.
    emb_space = EmbeddingMOSpace(
        nmo=system.nmo,
        frozen_core_orbitals=frozen_core,
        B_core_orbitals=index_B_occ,
        A_core_orbitals=index_A_occ,
        active_orbitals=partition["active_orbitals"],
        A_virtual_orbitals=index_A_vir,
        B_virtual_orbitals=index_B_vir,
        frozen_virtual_orbitals=[],
    )
    orig_to_contig = np.asarray(emb_space.orig_to_contig)
    contig_to_orig = np.asarray(emb_space.contig_to_orig)
    assert not np.array_equal(orig_to_contig, np.arange(system.nmo))
    assert not np.array_equal(orig_to_contig, contig_to_orig)

    S = system.ints_overlap()
    C = aset.C[0]
    np.testing.assert_allclose(C.conj().T @ S @ C, np.eye(system.nmo), atol=1e-10)

    # The orbitals the user pinned by index must still span the same space as
    # in the parent MCSCF, i.e. they must not have been permuted away.
    for indices in (frozen_core, mc.mo_space.active_indices):
        assert len(indices) > 0
        assert spans_same_space(S, mc.C[0][:, indices], C[:, indices])

    # Every orbital assigned to fragment A must be more localized on the
    # fragment than any orbital assigned to environment B.
    diag_P = np.einsum("mi,mn,ni->i", C.conj(), aset.P_ao_frag, C, optimize=True)
    for index_A, index_B in ((index_A_occ, index_B_occ), (index_A_vir, index_B_vir)):
        assert len(index_A) > 0 and len(index_B) > 0
        assert diag_P[index_A].min() > diag_P[index_B].max()
