import numpy as np

from forte2 import System, RHF, CI, MOSpace, orbitals, State, MCOptimizer
from forte2.helpers.comparisons import approx


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

    semi = orbitals.Semicanonicalizer(
        mo_space=mo_space, g1=np.diag([2, 2, 2, 0, 0, 0]), C=rhf.C[0], system=system
    )
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
    semi = orbitals.Semicanonicalizer(
        mo_space=mo_space,
        g1=ci.make_average_1rdm(),
        C=ci.C[0],
        system=system,
    )

    fock = semi.fock
    fock_cc = fock[mo_space.core, mo_space.core]
    fock_aa = fock[mo_space.actv, mo_space.actv]
    fock_vv = fock[mo_space.virt, mo_space.virt]
    assert not np.allclose(fock_cc, np.diag(np.diag(fock_cc)), rtol=0, atol=5e-8)
    # in this case the active fock is already semicanonical
    # assert not np.allclose(fock_aa, np.diag(np.diag(fock_aa)), rtol=0, atol=5e-8)
    assert not np.allclose(fock_vv, np.diag(np.diag(fock_vv)), rtol=0, atol=5e-8)

    ci = CI(
        State(nel=rhf.nel, multiplicity=1, ms=0.0),
        core_orbitals=[0, 1, 2, 3],
        active_orbitals=[4, 5, 6, 7, 8, 9],
        final_orbital="semicanonical",
    )(rhf)
    ci.run()
    assert ci.evals_flat[0] == approx(-109.01444624968038)

    mo_space = ci.mo_space
    semi = orbitals.Semicanonicalizer(
        mo_space=mo_space,
        g1=ci.make_average_1rdm(),
        C=ci.C[0],
        system=system,
    )

    fock = semi.fock
    fock_cc = fock[mo_space.core, mo_space.core]
    fock_aa = fock[mo_space.actv, mo_space.actv]
    fock_vv = fock[mo_space.virt, mo_space.virt]
    assert np.allclose(fock_cc, np.diag(np.diag(fock_cc)), rtol=0, atol=5e-8)
    assert np.allclose(fock_aa, np.diag(np.diag(fock_aa)), rtol=0, atol=5e-8)
    assert np.allclose(fock_vv, np.diag(np.diag(fock_vv)), rtol=0, atol=5e-8)
