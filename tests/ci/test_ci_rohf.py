import pytest

from forte2 import CI, CISolver, ROHF, State, System
from forte2.helpers.comparisons import approx
from forte2.base_classes import CIParams


def test_rohf_ci_1():
    xyz = """
    H 0.0 0.0 0.0
    F 0.0 0.0 2.0
    """

    system = System(
        xyz=xyz, basis_set="cc-pVDZ", auxiliary_basis_set="cc-pVTZ-JKFIT", unit="bohr"
    )
    rhf = ROHF(charge=1, ms=0.5, e_tol=1e-12)(system)
    ci = CI(
        CISolver(
            states=State(system=system, charge=1, multiplicity=2, ms=0.5),
            core_orbitals=[0],
            active_orbitals=[1, 2, 3, 4, 5, 6],
            nroots=2,
        )
    )(rhf)
    ci.run()

    assert ci.E_ci[0] == approx(-99.510706628367)


def test_rohf_ci_2():
    from forte2.base_classes import CIParams

    xyz = """
    H 0.0 0.0 0.0
    F 0.0 0.0 2.0
    """

    system = System(
        xyz=xyz, basis_set="cc-pVDZ", auxiliary_basis_set="cc-pVTZ-JKFIT", unit="bohr"
    )
    rhf = ROHF(charge=1, ms=-0.5, e_tol=1e-12)(system)
    ci = CI(
        CISolver(
            active_orbitals=[1, 2, 3, 4, 5, 6],
            core_orbitals=[0],
            states=State(system=system, charge=1, multiplicity=2, ms=-0.5),
            nroots=2,
            ci_params=CIParams(ci_algorithm="exact"),
        )
    )(rhf)
    ci.run()

    assert ci.E_ci[0] == approx(-99.510706628367)


@pytest.mark.parametrize("alg", ["kh", "hz", "exact"])
def test_ci_rohf_high_spin_h2(alg):
    system = System(
        xyz="H 0.0 0.0 0.0\nH 0.0 0.0 1.0",
        basis_set="cc-pvdz",
        auxiliary_basis_set="cc-pVTZ-JKFIT",
    )
    rohf = ROHF(charge=0, ms=1.0, e_tol=1e-12, d_tol=1e-8)(system)
    ci_solver = CISolver(
        State(nel=2, multiplicity=3, ms=1.0),
        core_orbitals=0,
        active_orbitals=4,
        ci_params=CIParams(ci_algorithm=alg),
    )
    ci = CI(ci_solver)(rohf)
    ci.run()
    assert ci.E_ci[0] == approx(-0.8746928467)
