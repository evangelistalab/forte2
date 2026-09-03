import pytest
from forte2 import (
    AVAS,
    CI,
    CISolver,
    MCOptimizer,
    ROHF,
    RelCISolver,
    SpinorUpcaster,
    State,
    System,
    UHF,
    X2CParams,
)
from forte2.data import EH_TO_WN


def test_casscf_so():
    xyz = """
    F 0 0 0
    """

    system = System(
        xyz=xyz,
        basis_set="decon-cc-pVTZ",
        auxiliary_basis_set="cc-pVQZ-JKFIT",
        x2c=X2CParams(x2c_type="sf", x2c_model="1e"),
    )

    rhf = ROHF(charge=0, ms=0.5)(system)
    avas = AVAS(
        selection_method="separate",
        num_active_docc=3,
        num_active_uocc=0,
        subspace=["F(2s)", "F(2p)"],
    )(rhf)
    ci_solver = CISolver(
        states=State(nel=9, multiplicity=2, ms=0.5),
        nroots=3,
    )
    mc = MCOptimizer(ci_solver)(avas)
    conv = SpinorUpcaster(
        x2c_override=X2CParams(
            x2c_type="so", x2c_model="1e", snso_type="row-dependent"
        ),
    )(mc)
    ci = CI(RelCISolver(nel=9, nroots=6, core_orbitals=2, active_orbitals=8))(conv)
    ci.run()

    # corresponds to ~ 4.6e-8 Eh
    assert (ci.E_ci[4] - ci.E_ci[3]) * EH_TO_WN == pytest.approx(
        401.04241215150705, abs=1e-2
    )


def test_2c_casscf_with_rohf():
    # Here we use ROHF orbitals to feed directly into a relativistic CASSCF calculation.
    # The result should be identical to using a GHF reference.
    xyz = """
    F 0 0 0
    """

    system = System(
        xyz=xyz,
        basis_set="decon-cc-pVTZ",
        auxiliary_basis_set="cc-pVQZ-JKFIT",
        x2c=None,
    )

    rhf = ROHF(charge=0, ms=0.5)(system)
    avas = AVAS(
        selection_method="separate",
        num_active_docc=3,
        num_active_uocc=0,
        subspace=["F(2s)", "F(2p)"],
    )(rhf)
    conv = SpinorUpcaster(
        x2c_override=X2CParams(
            x2c_type="so", x2c_model="1e", snso_type="row-dependent"
        ),
    )(avas)
    ci_solver = RelCISolver(
        nel=9,
        nroots=6,
    )
    mc = MCOptimizer(ci_solver)(conv)
    mc.run()
    # corresponds to ~ 4.6e-8 Eh
    assert (ci_solver.E[4] - ci_solver.E[3]) * EH_TO_WN == pytest.approx(
        393.7085978290238, abs=1e-2
    )


def test_2c_casscf_with_uhf():
    # Here we use ROHF orbitals to feed directly into a relativistic CASSCF calculation.
    # The result should be identical to using a GHF reference.
    xyz = """
    O 0 0 0
    H 0 0 1.1
    """

    system = System(
        xyz=xyz,
        basis_set="decon-cc-pVTZ",
        auxiliary_basis_set="cc-pVQZ-JKFIT",
        x2c=X2CParams(x2c_type="sf", x2c_model="1e"),
    )

    uhf = UHF(charge=0, ms=0.5)(system)
    conv = SpinorUpcaster(
        x2c_override=X2CParams(
            x2c_type="so", x2c_model="1e", snso_type="row-dependent"
        ),
    )(uhf)
    ci_solver = RelCISolver(
        nel=9,
        core_orbitals=2,
        active_orbitals=10,
        nroots=4,
    )
    mc = MCOptimizer(ci_solver)(conv)
    mc.run()
    assert (ci_solver.E[2] - ci_solver.E[1]) * EH_TO_WN == pytest.approx(
        134.77771444071902, abs=1e-2
    )
