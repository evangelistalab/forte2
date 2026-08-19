import pytest

from forte2 import System, RHF, MCOptimizer, State, CISolver
from forte2.base_classes import DavidsonLiuParams
from forte2.helpers.comparisons import approx


def test_casscf_hf():
    erhf = -99.9977252002946
    emcscf = -100.0435018956

    xyz = """
    H 0.0 0.0 0.0
    F 0.0 0.0 2.0
    """

    system = System(
        xyz=xyz, basis_set="cc-pVDZ", auxiliary_basis_set="cc-pVTZ-JKFIT", unit="bohr"
    )
    rhf = RHF(charge=0, e_tol=1e-12)(system)
    ci_solver = CISolver(
        State(nel=10, multiplicity=1, ms=0.0),
        active_orbitals=6,
        core_orbitals=1,
    )
    mc = MCOptimizer(ci_solver)(rhf)
    mc.run()

    assert rhf.E == approx(erhf)
    assert mc.E == approx(emcscf)


def test_casscf_hf_smaller_active():
    erhf = -99.87284684762975
    emcscf = -99.939295399756

    xyz = """
    F            0.000000000000     0.000000000000    -0.075563346255
    H            0.000000000000     0.000000000000     1.424436653745
    """

    system = System(
        xyz=xyz, basis_set="cc-pvdz", auxiliary_basis_set="def2-universal-jkfit"
    )

    rhf = RHF(charge=0, e_tol=1e-12)(system)
    ci_solver = CISolver(
        State(nel=10, multiplicity=1, ms=0.0),
        active_orbitals=[4, 5],
        core_orbitals=[0, 1, 2, 3],
    )
    mc = MCOptimizer(ci_solver)(rhf)
    mc.run()

    assert rhf.E == approx(erhf)
    assert mc.E == approx(emcscf)


@pytest.mark.parametrize("final_orbitals", ["semicanonical", "natural"])
def test_mcoptimizer_final_orbitals_wrong_root(final_orbitals):
    xyz = """
    H 0.0 0.0 0.0
    F 0.0 0.0 2.0
    """
    system = System(
        xyz=xyz,
        basis_set="cc-pVDZ",
        auxiliary_basis_set="cc-pVTZ-JKFIT",
        unit="bohr",
    )
    rhf = RHF(charge=0, e_tol=1e-8)(system)

    singlet = State(nel=10, multiplicity=1, ms=0.0)
    triplet = State(nel=10, multiplicity=3, ms=1.0)
    ci_solver = CISolver(
        states=[singlet, triplet],
        nroots=[2, 1],
        core_orbitals=[0],
        active_orbitals=[1, 2, 3, 4, 5, 6, 7],
        davidson_liu_params=DavidsonLiuParams(
            e_tol=1e-8,
            r_tol=1e-4,
            ndets_per_guess=10,
        ),
    )
    mc = MCOptimizer(ci_solver, final_orbitals=final_orbitals)(rhf)
    with pytest.raises(Exception, match="converged to different roots."):
        mc.run()
