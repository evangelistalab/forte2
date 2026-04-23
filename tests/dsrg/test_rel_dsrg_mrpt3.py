import numpy as np
import pytest

from forte2 import System, GHF, RelMCOptimizer, AVAS
from forte2.dsrg import RelDSRG_MRPT3
from forte2.helpers.comparisons import approx
from forte2.data.atom_data import EH_TO_WN


def test_mrpt2_n2_nonrel():
    erhf = -108.954140898736
    emcscf = -109.0811491968

    xyz = """
    N 0.0 0.0 0.0
    N 0.0 0.0 2.0
    """

    system = System(
        xyz=xyz,
        basis_set="cc-pVDZ",
        auxiliary_basis_set="cc-pVTZ-JKFIT",
        unit="bohr",
    )
    rhf = GHF(charge=0)(system)
    rhf.run()
    rng = np.random.default_rng(1234)
    random_phase = np.diag(np.exp(1j * rng.uniform(-np.pi, np.pi, size=rhf.nmo * 2)))
    rhf.C[0] = rhf.C[0] @ random_phase

    mc = RelMCOptimizer(
        nel=14,
        core_orbitals=8,
        active_orbitals=12,
    )(rhf)
    mc.run()

    assert rhf.E == approx(erhf)
    assert mc.E == approx(emcscf)

    dsrg = RelDSRG_MRPT3(
        frozen_core_orbitals=4,
        flow_param=0.5,
        relax_reference="iterate",
    )(mc)
    dsrg.run()

    assert dsrg.relax_energies[0] == approx(
        [-109.25301485009223, -109.2538362628393, -109.08114919682387]
    )
    assert dsrg.relax_energies[1] == approx(
        [-109.25344887585058, -109.25344888535047, -109.0802678007682]
    )
    assert dsrg.relax_energies[2] == approx(
        [-109.25344824472272, -109.25344824472299, -109.08026606599341]
    )


def test_mrpt3_n_atom_rel_sa():
    xyz = """
    N 0 0 0
    """

    system = System(
        xyz=xyz,
        basis_set="decon-cc-pVTZ",
        auxiliary_basis_set="cc-pVQZ-JKFIT",
        x2c_type="so",
        snso_type="row-dependent",
    )
    mf = GHF(charge=-1, die_if_not_converged=False)(system)
    mc = RelMCOptimizer(
        nel=7,
        nroots=14,
        active_orbitals=8,
        core_orbitals=2,
        econv=1e-8,
        gconv=1e-6,
    )(mf)
    dsrg = RelDSRG_MRPT3(flow_param=0.35, relax_reference="once")(mc)
    dsrg.run()
    assert (dsrg.relax_eigvals[4] - dsrg.relax_eigvals[3]) * EH_TO_WN == pytest.approx(
        19959.81596, abs=1e-2
    )
