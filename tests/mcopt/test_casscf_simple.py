from forte2 import System, RHF, MCOptimizer, State, CISolver
from forte2.helpers.comparisons import approx, is_diagonal_matrix


def test_casscf_h2():
    erhf = -1.08928367118043
    emcscf = -1.11873740345286

    xyz = f"""
    H 0.0 0.0 0.0
    H 0.0 0.0 {0.529177210903 * 2}
    """

    system = System(xyz=xyz, basis_set="cc-pvdz", auxiliary_basis_set="cc-pVTZ-JKFIT")

    rhf = RHF(charge=0, e_tol=1e-12)(system)
    ci_solver = CISolver(State(nel=2, multiplicity=1, ms=0.0), active_orbitals=[0, 1])
    mc = MCOptimizer(ci_solver)(rhf)
    mc.run()

    assert rhf.E == approx(erhf)
    assert mc.E == approx(emcscf)


def test_casscf_h2_all_active_orbitals_frozen():
    xyz = f"""
    H 0.0 0.0 0.0
    H 0.0 0.0 {0.529177210903 * 2}
    """

    system = System(xyz=xyz, basis_set="cc-pvdz", auxiliary_basis_set="cc-pVTZ-JKFIT")

    rhf = RHF(charge=0, e_tol=1e-12)(system)
    ci_solver = CISolver(State(nel=2, multiplicity=1, ms=0.0), active_orbitals=[0, 1])
    mc = MCOptimizer(
        ci_solver,
        active_frozen_orbitals=[0, 1],
        maxiter=0,
        final_orbital="original",
    )(rhf)
    mc.run()

    assert mc.orb_opt.nrot == 0
    assert mc.iter == 0
    assert mc.converged
    assert mc.E == approx(mc.E_ci[0])


def test_casscf_h2_all_orbitals_active_positive_maxiter():
    xyz = f"""
    H 0.0 0.0 0.0
    H 0.0 0.0 {0.529177210903 * 2}
    """

    system = System(xyz=xyz, basis_set="sto-6g", auxiliary_basis_set="cc-pVTZ-JKFIT")

    rhf = RHF(charge=0, e_tol=1e-12)(system)
    ci_solver = CISolver(State(nel=2, multiplicity=1, ms=0.0), active_orbitals=2)
    mc = MCOptimizer(
        ci_solver,
        maxiter=5,
        final_orbital="original",
    )(rhf)
    mc.run()

    assert mc.maxiter != 0
    assert mc.mo_space.nactv == mc.mo_space.nmo
    assert mc.mo_space.ncore == 0
    assert mc.mo_space.nvirt == 0
    assert mc.orb_opt.nrot == 0
    assert mc.iter == 0
    assert mc.converged
    assert mc.E == approx(-1.096071975854)


def test_casscf_n2():
    erhf = -108.761639873604
    ecasscf = -108.9800484156

    xyz = """
    N 0.0 0.0 0.0
    N 0.0 0.0 1.4
    """

    system = System(xyz=xyz, basis_set="cc-pVDZ", auxiliary_basis_set="cc-pVTZ-JKFIT")
    rhf = RHF(charge=0, e_tol=1e-12)(system)
    ci_solver = CISolver(
        State(nel=14, multiplicity=1, ms=0.0),
        active_orbitals=[4, 5, 6, 7, 8, 9],
        core_orbitals=[0, 1, 2, 3],
    )
    mc = MCOptimizer(
        ci_solver,
        g_tol=1e-7,
    )(rhf)
    mc.run()
    assert rhf.E == approx(erhf)
    assert mc.E == approx(ecasscf)


def test_casscf_water():
    erhf = -76.0214620954787819
    emcscf = -76.07856407969193

    xyz = """
    O            0.000000000000     0.000000000000    -0.069592187400
    H            0.000000000000    -0.783151105291     0.552239257834
    H            0.000000000000     0.783151105291     0.552239257834
    """

    system = System(
        xyz=xyz,
        basis_set="cc-pvdz",
        auxiliary_basis_set="def2-universal-jkfit",
        unit="angstrom",
    )

    rhf = RHF(charge=0, e_tol=1e-12, d_tol=1e-12)(system)
    ci_solver = CISolver(
        State(nel=10, multiplicity=1, ms=0.0),
        active_orbitals=[1, 2, 3, 4, 5, 6],
        core_orbitals=[0],
    )
    mc = MCOptimizer(
        ci_solver,
        g_tol=1e-6,
        e_tol=1e-10,
    )(rhf)
    mc.run()

    assert rhf.E == approx(erhf)
    assert mc.E == approx(emcscf)


def test_casscf_water_nos():
    erhf = -76.0214620954787819
    emcscf = -76.07856407969193

    xyz = """
    O            0.000000000000     0.000000000000    -0.069592187400
    H            0.000000000000    -0.783151105291     0.552239257834
    H            0.000000000000     0.783151105291     0.552239257834
    """

    system = System(
        xyz=xyz,
        basis_set="cc-pvdz",
        auxiliary_basis_set="def2-universal-jkfit",
        unit="angstrom",
    )

    rhf = RHF(charge=0, e_tol=1e-10, d_tol=1e-5)(system)
    ci_solver = CISolver(
        State(nel=10, multiplicity=1, ms=0.0),
        active_orbitals=[1, 2, 3, 4, 5, 6],
        core_orbitals=[0],
    )
    mc = MCOptimizer(
        ci_solver,
        g_tol=1e-12,
        e_tol=1e-11,
        final_orbital="natural",
    )(rhf)
    mc.run()
    assert rhf.E == approx(erhf)
    assert mc.E == approx(emcscf)

    # Check that the 1-RDM is block diagonal in the active space
    g1 = mc.make_average_1rdm()
    assert is_diagonal_matrix(g1)
