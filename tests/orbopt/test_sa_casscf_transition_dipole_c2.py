import pytest
import numpy as np

from forte2 import System, RHF, MCOptimizer, AVAS, State
from forte2.helpers.comparisons import approx


def test_sa_casscf_c2_transition_dipole():
    """Test CASSCF with C2 molecule."""

    erhf = -75.382486499716
    ecasscf = -75.5580517997

    xyz = """
    C 0.0 0.0 0.0
    C 0.0 0.0 1.2
    """

    system = System(
        xyz=xyz,
        basis_set="cc-pVDZ",
        auxiliary_basis_set="cc-pVTZ-JKFIT",
    )

    rhf = RHF(charge=0, econv=1e-12)(system)
    singlet = State(nel=rhf.nel, multiplicity=1, ms=0.0)
    triplet = State(nel=rhf.nel, multiplicity=3, ms=0.0)
    avas = AVAS(
        selection_method="separate",
        num_active_docc=4,
        num_active_uocc=4,
        subspace=["C(2s)", "C(2p)"],
    )(rhf)
    mc = MCOptimizer([singlet, triplet], nroots=[2, 2], do_transition_dipole=True)(avas)
    mc.run()
    assert mc.ci_solver.evals_per_solver[0][0] == approx(-75.6107427252)
    assert mc.ci_solver.evals_per_solver[0][1] == approx(-75.5314535451)
    assert mc.ci_solver.evals_per_solver[1][0] == approx(-75.5792187010)
    assert mc.ci_solver.evals_per_solver[1][1] == approx(-75.5789867708)
    assert mc.ci_solver.fosc_per_solver[0][(0, 1)] == approx(0.006525243082121279)
