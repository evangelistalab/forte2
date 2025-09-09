import pytest
import numpy as np

from forte2 import System, GHF
from forte2.helpers.comparisons import approx
from forte2.orbopt import RelMCOptimizer


@pytest.mark.slow
def test_rel_casscf_hf_equivalence_to_nonrel():
    erhf = -99.9977252002946
    emcscf = -100.0435018956

    xyz = """
    H 0.0 0.0 0.0
    F 0.0 0.0 2.0
    """
    system = System(
        xyz=xyz, basis_set="cc-pvdz", auxiliary_basis_set="cc-pVTZ-JKFIT", unit="bohr"
    )
    scf = GHF(charge=0, econv=1e-10)(system)
    mc = RelMCOptimizer(
        nel=10,
        core_orbitals=2,
        active_orbitals=12,
        do_diis=False,
        maxiter=200,
    )(scf)
    mc.run()
    assert scf.E == approx(erhf)
    assert mc.E == approx(emcscf)


@pytest.mark.slow
def test_rel_casscf_hf_ghf():
    escf = -100.078531285537
    emcscf = -100.1361832608
    xyz = """
    H 0.0 0.0 0.0
    F 0.0 0.0 2.0
    """

    system = System(
        xyz=xyz,
        basis_set="cc-pvdz",
        auxiliary_basis_set="cc-pVTZ-JKFIT",
        unit="bohr",
        x2c_type="so",
    )
    scf = GHF(charge=0)(system)
    mc = RelMCOptimizer(
        nel=10,
        nroots=1,
        core_orbitals=2,
        active_orbitals=12,
        do_diis=False,
        maxiter=200,
    )(scf)
    mc.run()

    assert scf.E == approx(escf)
    assert mc.E == approx(emcscf)


def test_rel_casscf_na_ghf():
    emcscf = -161.9905346837
    xyz = """
    Na 0.0 0.0 0.0
    """

    system = System(
        xyz=xyz,
        basis_set="cc-pvdz",
        auxiliary_basis_set="def2-universal-jkfit",
        unit="bohr",
        x2c_type="so",
    )
    scf = GHF(charge=0)(system)
    mc = RelMCOptimizer(
        nel=11,
        nroots=8,
        core_orbitals=10,
        active_orbitals=8,
        do_diis=False,
        maxiter=500,
    )(scf)
    mc.run()

    assert mc.E == approx(emcscf)
