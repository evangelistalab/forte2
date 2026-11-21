import pytest

from forte2 import CI, CISolver, RHF, ROHF, State, System
from forte2.helpers.comparisons import approx
from forte2.base_classes import CIParams


def test_fci_co_o_core_exc():
    efci = -92.133019235463
    xyz = """
    C 0 0 0
    O 0 0 1
    """

    system = System(xyz=xyz, basis_set="sto-6g", auxiliary_basis_set="cc-pVTZ-JKFIT")

    rhf = RHF(charge=0, e_tol=1e-12)(system)
    ci = CI(
        CISolver(
            State(nel=14, multiplicity=1, ms=0.0),
            active_orbitals=list(range(system.nbf)),
            ci_params=CIParams(energy_shift=-92),
        )
    )(rhf)
    ci.run()
    assert ci.E_ci[0] == approx(efci)


def test_fci_co_c_core_exc():
    efci = -101.499040903802
    xyz = """
    C 0 0 0
    O 0 0 1
    """

    system = System(xyz=xyz, basis_set="sto-6g", auxiliary_basis_set="cc-pVTZ-JKFIT")

    rhf = RHF(charge=0, e_tol=1e-12)(system)
    ci = CI(
        CISolver(
            State(nel=14, multiplicity=1, ms=0.0),
            active_orbitals=list(range(system.nbf)),
            ci_params=CIParams(energy_shift=-102),
        )
    )(rhf)
    ci.run()
    assert ci.E_ci[0] == approx(efci)


@pytest.mark.slow
def test_fci_o_core_exc_triplet():
    """The O 1s -> 2p core-excited 3B1u state of the oxygen atom."""
    efci = -55.4197469054
    system = System(
        xyz="O 0 0 0",
        basis_set="cc-pvdz",
        auxiliary_basis_set="cc-pvtz-jkfit",
        symmetry=True,
    )
    rohf = ROHF(charge=0, ms=1.0, e_tol=1e-10)(system)
    ci_solver = CISolver(
        State(nel=8, multiplicity=3, ms=1.0, symmetry=5),  # B1u
        active_orbitals=list(range(system.nbf)),
        ci_params=CIParams(energy_shift=-55.42),
    )
    ci = CI(ci_solver)(rohf)
    ci.run()
    assert ci.E_ci[0] == approx(efci)
    assert ci_solver.sub_solvers[0].eigensolver.iter < 30
