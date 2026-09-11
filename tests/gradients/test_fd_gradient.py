import pytest
import numpy as np

from forte2 import (
    CISolver,
    MCOptimizer,
    RHF,
    State,
    System,
    GHF,
)
from forte2.base_classes import X2CParams
from forte2.gradients import FDGradient


def _rhf(x2c=None):
    sys = System(
        xyz="""
        O 0 0 0
        H 0 0 1.9
        H 1.6 0.3 0.0
        """,
        basis_set="sto-3g",
        auxiliary_basis_set="def2-universal-JKFIT",
        unit="bohr",
        x2c=x2c,
    )
    _HF = RHF if x2c is None else GHF
    hf = _HF(
        charge=0,
        e_tol=1.0e-12,
        d_tol=1.0e-10,
        maxiter=100,
    )(sys)
    hf.run()
    return hf


def _ghf():
    return _rhf(x2c=X2CParams(x2c_type="so", x2c_model="1e", snso_type="row-dependent"))


def _casscf():
    system = System(
        xyz="H 0 0 0\nH 0 0 1.6",
        basis_set="sto-3g",
        auxiliary_basis_set="def2-universal-JKFIT",
        unit="bohr",
    )
    rhf = RHF(charge=0, e_tol=1.0e-12, d_tol=1.0e-10, maxiter=100)(system)
    ci_solver = CISolver(
        State(system=system, multiplicity=1, ms=0.0), active_orbitals=[0, 1]
    )
    mc = MCOptimizer(ci_solver, e_tol=1.0e-12, g_tol=1.0e-10, final_orbitals="original")
    return mc(rhf)


@pytest.mark.parametrize("_hf", [_rhf, _ghf])
def test_fd_gradient_matches_analytic_hf_gradient(_hf):
    hf = _hf()
    analytic = hf.gradient()

    fd = FDGradient(step=1.0e-3, npoints=4)(_hf())
    numeric = fd.gradient()

    np.testing.assert_allclose(numeric, analytic, atol=1.0e-7)


def test_fd_gradient_matches_analytic_casscf_gradient():
    mc = _casscf()
    analytic = mc.gradient()

    fd = FDGradient(step=1.0e-3, npoints=4)(_casscf())
    numeric = fd.gradient()

    np.testing.assert_allclose(numeric, analytic, atol=1.0e-6)
