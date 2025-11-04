import numpy as np
import pytest

from forte2 import System, GHF, RelMCOptimizer
from forte2.dsrg import DSRG_MRPT2
from forte2.helpers.comparisons import approx


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
        active_orbitals=12,
        core_orbitals=8,
    )(rhf)
    mc.run()

    assert rhf.E == approx(erhf)
    assert mc.E == approx(emcscf)

    dsrg = DSRG_MRPT2(flow_param=0.5, relax_reference="iterate")(mc)
    dsrg.run()

    assert dsrg.relax_energies[0, 0] == approx(-109.238860710091)
    assert dsrg.relax_energies[0, 1] == approx(-109.239311979963)
    assert dsrg.relax_energies[0, 2] == approx(-109.081149196818)

    assert dsrg.relax_energies[1, 0] == approx(-109.238952001270)
    assert dsrg.relax_energies[1, 1] == approx(-109.238952010521)
    assert dsrg.relax_energies[1, 2] == approx(-109.080656324868)

    assert dsrg.relax_energies[2, 0] == approx(-109.238953941310)
    assert dsrg.relax_energies[2, 1] == approx(-109.238953941321)
    assert dsrg.relax_energies[2, 2] == approx(-109.080659175782)


@pytest.mark.slow
def test_mrpt2_br_sa():
    xyz = """
    Br 0 0 0
    """

    system = System(
        xyz=xyz,
        basis_set="decon-cc-pVTZ",
        auxiliary_basis_set="cc-pVQZ-JKFIT",
        x2c_type="so",
        snso_type="row-dependent",
    )
    mf = GHF(charge=0, die_if_not_converged=False)(system)
    mc = RelMCOptimizer(
        nel=34,
        nroots=10,
        active_orbitals=8,
        core_orbitals=28,
        econv=1e-8,
        gconv=1e-6,
        do_diis=False,
    )(mf)
    dsrg = DSRG_MRPT2(flow_param=0.5, relax_reference="once")(mc)
    dsrg.run()
