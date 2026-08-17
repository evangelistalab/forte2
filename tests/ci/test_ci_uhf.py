from forte2 import System, CI, State, UHF
from forte2.helpers.comparisons import approx


def test_uhf_ci_1():
    xyz = """
    H 0.0 0.0 0.0
    H 0.0 0.0 2.0
    """

    system = System(
        xyz=xyz, basis_set="cc-pVDZ", auxiliary_basis_set="cc-pVTZ-JKFIT", unit="bohr"
    )
    hf = UHF(charge=0, ms=1.0)(system)
    # CI only takes the alpha MOs from UHF, but since we're doing FCI, it shouldn't matter
    ci = CI(
        states=State(nel=2, system=system, multiplicity=1, ms=0.0),
        active_orbitals=system.nmo,
        nroots=1,
    )(hf)
    ci.run()
    assert ci.E[0] == approx(-1.1306920385)
