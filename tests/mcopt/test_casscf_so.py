import pytest
from forte2 import (
    System,
    AVAS,
    ROHF,
    MCOptimizer,
    State,
    CISolver,
    RelCI,
)
from forte2.scf.scf_utils import convert_coeff_spatial_to_spinor
from forte2.data import EH_TO_WN


def test_casscf_so():
    xyz = """
    Br 0 0 0
    """

    system = System(
        xyz=xyz,
        basis_set="decon-cc-pVTZ",
        auxiliary_basis_set="cc-pVQZ-JKFIT",
        x2c_type="sf",
    )

    rhf = ROHF(charge=0, ms=0.5)(system)
    avas = AVAS(
        selection_method="separate",
        num_active_docc=3,
        num_active_uocc=0,
        subspace=["Br(4s)", "Br(4p)"],
    )(rhf)
    ci_solver = CISolver(
        states=State(nel=35, multiplicity=2, ms=0.5),
        nroots=3,
    )
    mc = MCOptimizer(ci_solver)(avas)
    mc.run()

    system.two_component = True

    C_2c = convert_coeff_spatial_to_spinor(mc.C)
    rhf.C = C_2c
    system.x2c_type = "so"
    system.snso_type = "row-dependent"
    ci = RelCI(nel=35, nroots=6, core_orbitals=28, active_orbitals=8)(rhf)
    ci.run()

    # corresponds to ~ 4.6e-8 Eh
    assert (ci.E[4] - ci.E[3]) * EH_TO_WN == pytest.approx(3416.391762052979, abs=1e-2)
