import numpy as np

from forte2 import System, RHF, CI, MOSpace, orbitals, State, MCOptimizer, integrals
from forte2.helpers.comparisons import approx
from forte2.orbitals import Semicanonicalizer


def test_semican_rhf():
    # Semicanonicalized RHF eigenvalues should be strictly identical to the RHF eigenvalues
    xyz = """
    N 0.0 0.0 -1.0
    N 0.0 0.0 1.0
    """

    system = System(
        xyz=xyz, basis_set="cc-pVDZ", auxiliary_basis_set="cc-pVTZ-JKFIT", unit="bohr"
    )
    rhf = RHF(charge=0, econv=1e-12)(system)
    rhf.run()
    mo_space = MOSpace(
        core_orbitals=[0, 1, 2, 3], active_orbitals=[4, 5, 6, 7, 8, 9], nmo=system.nmo
    )

    semi = orbitals.Semicanonicalizer(mo_space=mo_space, system=system)
    semi.semi_canonicalize(g1=np.diag([2, 2, 2, 0, 0, 0]), C_contig=rhf.C[0])
    assert rhf.eps[0] == approx(semi.eps_semican)


def test_semican_ci():
    # CI energy should be identical using RHF-canonical or semicanonicalized orbitals
    xyz = """
    N 0.0 0.0 -1.0
    N 0.0 0.0 1.0
    """

    system = System(
        xyz=xyz, basis_set="cc-pVDZ", auxiliary_basis_set="cc-pVTZ-JKFIT", unit="bohr"
    )
    rhf = RHF(charge=0, econv=1e-12)(system)
    rhf.run()
    ci = CI(
        State(nel=rhf.nel, multiplicity=1, ms=0.0),
        core_orbitals=[0, 1, 2, 3],
        active_orbitals=[4, 5, 6, 7, 8, 9],
        final_orbital="semicanonical",
    )(rhf)
    ci.run()
    eci_orig = ci.evals_flat[0]
    assert eci_orig == approx(-109.01444624968038)

    rhf.C[0] = ci.C[0].copy()
    ci = CI(
        State(nel=rhf.nel, multiplicity=1, ms=0.0),
        core_orbitals=[0, 1, 2, 3],
        active_orbitals=[4, 5, 6, 7, 8, 9],
    )(rhf)
    ci.run()
    assert ci.evals_flat[0] == approx(eci_orig)


def test_semican_casscf():
    # CI energy should be identical using RHF-canonical or semicanonicalized orbitals
    xyz = """
    N 0.0 0.0 -1.0
    N 0.0 0.0 1.0
    """

    system = System(
        xyz=xyz, basis_set="cc-pVDZ", auxiliary_basis_set="cc-pVTZ-JKFIT", unit="bohr"
    )
    rhf = RHF(charge=0, econv=1e-12)(system)
    rhf.run()
    mc = MCOptimizer(
        State(nel=rhf.nel, multiplicity=1, ms=0.0),
        core_orbitals=[0, 1, 2, 3],
        active_orbitals=[4, 5, 6, 7, 8, 9],
        final_orbital="semicanonical",
    )(rhf)
    mc.run()
    eci_orig = mc.ci_solver.evals_flat[0]
    assert eci_orig == approx(-109.0811491968)

    rhf.C[0] = mc.C[0].copy()
    mc = MCOptimizer(
        State(nel=rhf.nel, multiplicity=1, ms=0.0),
        core_orbitals=[0, 1, 2, 3],
        active_orbitals=[4, 5, 6, 7, 8, 9],
    )(rhf)
    mc.run()
    assert mc.ci_solver.evals_flat[0] == approx(eci_orig)


def test_semican_fock_offdiag():
    # CI energy should be identical using RHF-canonical or semicanonicalized orbitals
    xyz = """
    N 0.0 0.0 -1.0
    N 0.0 0.0 1.0
    """

    system = System(
        xyz=xyz, basis_set="cc-pVDZ", auxiliary_basis_set="cc-pVTZ-JKFIT", unit="bohr"
    )
    rhf = RHF(charge=0, econv=1e-12)(system)
    rhf.run()
    ci = CI(
        State(nel=rhf.nel, multiplicity=1, ms=0.0),
        core_orbitals=[0, 1, 2, 3],
        active_orbitals=[4, 5, 6, 7, 8, 9],
        final_orbital="original",
    )(rhf)
    ci.run()
    assert ci.evals_flat[0] == approx(-109.01444624968038)

    mo_space = ci.mo_space
    semi = orbitals.Semicanonicalizer(mo_space=mo_space, system=system)
    semi.semi_canonicalize(g1=ci.make_average_1rdm(), C_contig=ci.C[0])

    fock = semi.fock
    fock_cc = fock[mo_space.core, mo_space.core]
    fock_aa = fock[mo_space.actv, mo_space.actv]
    fock_vv = fock[mo_space.virt, mo_space.virt]
    assert not np.allclose(fock_cc, np.diag(np.diag(fock_cc)), rtol=0, atol=5e-8)
    # in this case the active fock is already semicanonical
    # assert not np.allclose(fock_aa, np.diag(np.diag(fock_aa)), rtol=0, atol=5e-8)
    assert not np.allclose(fock_vv, np.diag(np.diag(fock_vv)), rtol=0, atol=5e-8)

    # The semicanonicalized Fock should be diagonal in the C/A/V blocks
    fock_semican = semi.fock_semican
    fock_cc = fock_semican[mo_space.core, mo_space.core]
    fock_aa = fock_semican[mo_space.actv, mo_space.actv]
    fock_vv = fock_semican[mo_space.virt, mo_space.virt]
    assert np.allclose(fock_cc, np.diag(np.diag(fock_cc)), rtol=0, atol=5e-8)
    assert np.allclose(fock_aa, np.diag(np.diag(fock_aa)), rtol=0, atol=5e-8)
    assert np.allclose(fock_vv, np.diag(np.diag(fock_vv)), rtol=0, atol=5e-8)

    # The long way to check: recompute the CI in the semicanonical basis,
    # and the generalized Fock should already be diagonal in C/A/V
    ci = CI(
        State(nel=rhf.nel, multiplicity=1, ms=0.0),
        core_orbitals=[0, 1, 2, 3],
        active_orbitals=[4, 5, 6, 7, 8, 9],
        final_orbital="semicanonical",
    )(rhf)
    ci.run()
    assert ci.evals_flat[0] == approx(-109.01444624968038)

    mo_space = ci.mo_space
    semi.semi_canonicalize(g1=ci.make_average_1rdm(), C_contig=ci.C[0])

    fock = semi.fock
    fock_cc = fock[mo_space.core, mo_space.core]
    fock_aa = fock[mo_space.actv, mo_space.actv]
    fock_vv = fock[mo_space.virt, mo_space.virt]
    assert np.allclose(fock_cc, np.diag(np.diag(fock_cc)), rtol=0, atol=5e-8)
    assert np.allclose(fock_aa, np.diag(np.diag(fock_aa)), rtol=0, atol=5e-8)
    assert np.allclose(fock_vv, np.diag(np.diag(fock_vv)), rtol=0, atol=5e-8)


def test_semican_orbitals():
    # Test that repeated semicanonicalization gives the same orbitals
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

    rhf = RHF(charge=0, econv=1e-12)(system)
    mc = MCOptimizer(
        State(nel=24, multiplicity=1, ms=0.0),
        core_orbitals=10,
        active_orbitals=4,
        ci_rconv=1e-10,
        ci_econv=1e-12,
        final_orbital="semicanonical",
    )(rhf)
    mc.run()
    c_mc = mc.C[0].copy()
    assert mc.E == approx(eci)

    semi = Semicanonicalizer(mo_space=mc.mo_space, system=system)
    semi.semi_canonicalize(g1=mc.ci_solver.make_average_1rdm(), C_contig=mc.C[0])
    c_semi = semi.C_semican.copy()
    ovlp = integrals.overlap(system)

    assert np.allclose(
        np.abs(c_mc.T @ ovlp @ c_semi), np.eye(c_mc.shape[1]), rtol=0, atol=1e-8
    )
