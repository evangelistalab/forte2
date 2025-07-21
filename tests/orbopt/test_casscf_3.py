import pytest

from forte2 import *
from forte2.helpers.comparisons import approx


def test_mcscf_casscf_3():
    erhf = -76.0214620954787819
    emcscf = -76.07856407969193

    xyz = f"""
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

    rhf = RHF(charge=0)(system)
    rhf.econv = 1e-12
    rhf.dconv = 1e-12
    ci = CI(
        orbitals=[1, 2, 3, 4, 5, 6],
        core_orbitals=[0],
        state=State(nel=10, multiplicity=1, ms=0.0),
        nroot=1,
    )(rhf)
    mc = MCOptimizer()(ci)
    mc.maxiter = 400
    mc.gradtol = 1e-6
    mc.etol = 1e-10
    mc.run()

    assert rhf.E == approx(erhf)
    assert mc.E == approx(emcscf)
