import numpy as np
import pytest

from forte2 import (
    System,
    RHF,
    CI,
    MOSpace,
    orbitals,
    State,
    MCOptimizer,
    integrals,
    CISolver,
)
from forte2.helpers.comparisons import approx
from forte2.orbitals import (
    NaturalOrbital,
    OrbitalBlockBuilder,
    Semicanonicalizer,
)
from forte2.base_classes import DavidsonLiuParams


def test_semican_rhf():
    # Semicanonicalized RHF eigenvalues should be strictly identical to the RHF eigenvalues
    xyz = """
    N 0.0 0.0 -1.0
    N 0.0 0.0 1.0
    """

    system = System(
        xyz=xyz, basis_set="cc-pVDZ", auxiliary_basis_set="cc-pVTZ-JKFIT", unit="bohr"
    )
    rhf = RHF(charge=0, e_tol=1e-12)(system)
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
    rhf = RHF(charge=0, e_tol=1e-12)(system)
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
    rhf = RHF(charge=0, e_tol=1e-12)(system)
    rhf.run()
    ci_solver = CISolver(
        State(nel=rhf.nel, multiplicity=1, ms=0.0),
        core_orbitals=[0, 1, 2, 3],
        active_orbitals=[4, 5, 6, 7, 8, 9],
    )
    mc = MCOptimizer(
        ci_solver,
        final_orbital="semicanonical",
    )(rhf)
    mc.run()
    eci_orig = mc.ci_solver.evals_flat[0]
    assert eci_orig == approx(-109.0811491968)

    rhf.C[0] = mc.C[0].copy()
    ci_solver = CISolver(
        State(nel=rhf.nel, multiplicity=1, ms=0.0),
        core_orbitals=[0, 1, 2, 3],
        active_orbitals=[4, 5, 6, 7, 8, 9],
    )
    mc = MCOptimizer(ci_solver)(rhf)
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
    rhf = RHF(charge=0, e_tol=1e-12)(system)
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


def test_semican_preserves_irrep_blocks():
    class DummySystem:
        two_component = False
        point_group = "D2H"
        fock_builder = None

    mo_space = MOSpace(nmo=4, active_orbitals=[0, 1, 2, 3])
    orbital_blocks = OrbitalBlockBuilder(DummySystem(), mo_space, [0, 2, 0, 2])
    blocks = orbital_blocks.blocks_for_slice(mo_space.actv)
    assert [block.tolist() for block in blocks] == [[0, 2], [], [1, 3]]
    blocks = orbital_blocks.blocks_for_spaces(["gas"])
    assert [block.tolist() for block in blocks] == [[0, 2], [], [1, 3]]
    orbital_blocks = OrbitalBlockBuilder(
        DummySystem(), mo_space, [0, 2, 0, 2], spaces=["gas"]
    )
    blocks = orbital_blocks.blocks_for_spaces()
    assert [block.tolist() for block in blocks] == [[0, 2], [], [1, 3]]

    semi = Semicanonicalizer(
        system=DummySystem(),
        mo_space=mo_space,
        irrep_indices=[0, 2, 0, 2],
    )
    semi._build_fock = lambda g1, C_contig: np.array(
        [
            [1.0, 0.4, 0.2, 0.1],
            [0.4, 2.0, 0.3, 0.5],
            [0.2, 0.3, 3.0, 0.6],
            [0.1, 0.5, 0.6, 4.0],
        ]
    )

    semi.semi_canonicalize(g1=np.eye(4), C_contig=np.eye(4))

    irreps = np.array([0, 2, 0, 2])
    cross_irrep = irreps[:, None] != irreps[None, :]
    assert np.allclose(semi.U[cross_irrep], 0.0)


def test_orbital_block_builder_rejects_unknown_space():
    class DummySystem:
        point_group = "C1"

    mo_space = MOSpace(nmo=2, active_orbitals=[0])
    orbital_blocks = OrbitalBlockBuilder(DummySystem(), mo_space)

    with pytest.raises(ValueError, match="Unknown orbital space"):
        orbital_blocks.blocks_for_spaces(["active"])


def test_semican_validates_input_shapes():
    class DummySystem:
        two_component = False
        point_group = "C1"
        fock_builder = None

    mo_space = MOSpace(nmo=3, core_orbitals=[0], active_orbitals=[1])
    semi = Semicanonicalizer(system=DummySystem(), mo_space=mo_space)

    with pytest.raises(ValueError, match="g1 must have shape"):
        semi.semi_canonicalize(g1=np.eye(2), C_contig=np.eye(3))


def test_natural_orbital_preserves_blocks():
    C_contig = np.eye(4)
    g1_act = np.array(
        [
            [1.0, 0.0, 0.3, 0.0],
            [0.0, 1.8, 0.0, 0.2],
            [0.3, 0.0, 1.2, 0.0],
            [0.0, 0.2, 0.0, 1.6],
        ]
    )

    class DummySystem:
        point_group = "D2H"

    mo_space = MOSpace(nmo=4, active_orbitals=[0, 1, 2, 3])
    natural_orbital = NaturalOrbital(DummySystem(), mo_space, [0, 2, 0, 2])
    natural_orbital.make_natural_orbitals(g1_act=g1_act, C_contig=C_contig)

    U_nat = natural_orbital.Uactv

    irreps = np.array([0, 2, 0, 2])
    cross_irrep = irreps[:, None] != irreps[None, :]
    assert np.allclose(U_nat[cross_irrep], 0.0)
    g1_nat = U_nat.T @ g1_act @ U_nat
    assert np.allclose(
        g1_nat[np.ix_([0, 2], [0, 2])],
        np.diag(np.diag(g1_nat[np.ix_([0, 2], [0, 2])])),
    )
    assert np.allclose(
        g1_nat[np.ix_([1, 3], [1, 3])],
        np.diag(np.diag(g1_nat[np.ix_([1, 3], [1, 3])])),
    )


def test_natural_orbital_requires_complete_active_blocks():
    class DummySystem:
        point_group = "C1"

    mo_space = MOSpace(nmo=3, active_orbitals=[0, 1])
    natural_orbital = NaturalOrbital(DummySystem(), mo_space)
    natural_orbital.orbital_blocks.active_blocks = lambda relative=True: [np.array([0])]

    with pytest.raises(ValueError, match="cover the full active space"):
        natural_orbital.make_natural_orbitals(g1_act=np.eye(2), C_contig=np.eye(3))


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

    rhf = RHF(charge=0, e_tol=1e-12)(system)
    ci_solver = CISolver(
        State(nel=24, multiplicity=1, ms=0.0),
        core_orbitals=10,
        active_orbitals=4,
        davidson_liu_params=DavidsonLiuParams(e_tol=1e-12, r_tol=1e-10),
    )
    mc = MCOptimizer(
        ci_solver,
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
