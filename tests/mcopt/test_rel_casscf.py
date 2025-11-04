from forte2 import System, GHF
from forte2.helpers.comparisons import approx
from forte2.mcopt import RelMCOptimizer
from forte2.orbitals import AVAS


def test_rel_casscf_hf_equivalence_to_nonrel():
    erhf = -99.9977252002946
    emcscf = -100.0435018956

    xyz = """
    H 0.0 0.0 0.0
    F 0.0 0.0 2.0
    """
    system = System(
        xyz=xyz, basis_set="cc-pvdz", auxiliary_basis_set="cc-pVTZ-JKFIT", unit="bohr"
    )
    scf = GHF(charge=0, econv=1e-10)(system)
    mc = RelMCOptimizer(
        nel=10,
        core_orbitals=2,
        active_orbitals=12,
        do_diis=False,
        maxiter=200,
    )(scf)
    mc.run()
    assert scf.E == approx(erhf)
    assert mc.E == approx(emcscf)


def test_rel_casscf_hf_ghf():
    escf = -100.078531285537
    emcscf = -100.1361832608
    xyz = """
    H 0.0 0.0 0.0
    F 0.0 0.0 2.0
    """

    system = System(
        xyz=xyz,
        basis_set="cc-pvdz",
        auxiliary_basis_set="cc-pVTZ-JKFIT",
        unit="bohr",
        x2c_type="so",
    )
    scf = GHF(charge=0)(system)
    mc = RelMCOptimizer(
        nel=10,
        nroots=1,
        core_orbitals=2,
        active_orbitals=12,
        do_diis=False,
        maxiter=200,
    )(scf)
    mc.run()

    assert scf.E == approx(escf)
    assert mc.E == approx(emcscf)


def test_rel_casscf_frozen_co_equivalent_to_nonrel():
    emcscf = -112.8641406910
    emcscf_frz = -112.8633865369

    xyz = """
    C 0.0 0.0 0.0
    O 0.0 0.0 1.2
    """

    system = System(xyz=xyz, basis_set="cc-pvdz", auxiliary_basis_set="cc-pVTZ-JKFIT")

    rhf = GHF(charge=0, econv=1e-12)(system)
    avas = AVAS(
        subspace=["C(2p)", "O(2p)"],
        selection_method="separate",
        num_active_docc=6,
        num_active_uocc=6,
    )(rhf)

    mc = RelMCOptimizer(
        nel=14,
        frozen_core_orbitals=2,
        core_orbitals=6,
        active_orbitals=12,
        frozen_virtual_orbitals=6,
        optimize_frozen_orbs=True,
    )(avas)
    mc.run()
    assert mc.E == approx(emcscf)

    mc = RelMCOptimizer(
        nel=14,
        frozen_core_orbitals=2,
        core_orbitals=6,
        active_orbitals=12,
        frozen_virtual_orbitals=6,
        optimize_frozen_orbs=False,
    )(avas)
    mc.run()
    assert mc.E == approx(emcscf_frz)


def test_rel_casscf_frozen_co_x2c():
    # this energy was obtained without AVAS
    emcscf = -112.9273233729

    xyz = """
    C 0.0 0.0 0.0
    O 0.0 0.0 1.2
    """

    system = System(
        xyz=xyz,
        basis_set="cc-pvdz",
        auxiliary_basis_set="cc-pVTZ-JKFIT",
        x2c_type="so",
        snso_type="row-dependent",
    )

    mf = GHF(charge=0, econv=1e-12)(system)
    avas = AVAS(
        subspace=["C(2p)", "O(2p)"],
        selection_method="separate",
        num_active_docc=6,
        num_active_uocc=6,
    )(mf)

    mc = RelMCOptimizer(
        nel=14,
        core_orbitals=8,
        active_orbitals=12,
    )(avas)
    mc.run()
    assert mc.E == approx(emcscf)


def test_rel_casscf_na_ghf():
    emcscf = -161.9905346837
    xyz = """
    Na 0.0 0.0 0.0
    """

    system = System(
        xyz=xyz,
        basis_set="cc-pvdz",
        auxiliary_basis_set="def2-universal-jkfit",
        unit="bohr",
        x2c_type="so",
    )
    scf = GHF(charge=0)(system)
    mc = RelMCOptimizer(
        nel=11,
        nroots=8,
        core_orbitals=10,
        active_orbitals=8,
        do_diis=False,
        maxiter=500,
    )(scf)
    mc.run()

    assert mc.E == approx(emcscf)


def test_rel_ci_br():
    xyz = """
    Br 0 0 0
    """

    system = System(
        xyz=xyz,
        basis_set="cc-pvtz",
        auxiliary_basis_set="cc-pvtz-jkfit",
        x2c_type="so",
        snso_type="row-dependent",
    )
    scf = GHF(charge=0)(system)
    mc = RelMCOptimizer(
        nel=35,
        nroots=6,
        active_orbitals=8,
        core_orbitals=28,
    )(scf)
    mc.run()
    assert mc.E == approx(-2597.0679040990)
