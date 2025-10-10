import numpy as np

from forte2 import System, GHF, RelMCOptimizer, RelCI
from forte2.dsrg import DSRG_MRPT2, DSRG_MRPT2_DF
from forte2.helpers.comparisons import approx


def test_mrpt2_n2_nonrel():
    erhf = -108.954140898736
    emcscf = -109.0811491968
    ept2 = -0.15771153424163914

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
    rhf = GHF(charge=0, econv=1e-12)(system)
    rhf.run()
    rng = np.random.default_rng(1234)
    random_phase = np.diag(np.exp(1j * rng.uniform(-np.pi, np.pi, size=rhf.nmo * 2)))
    rhf.C[0] = rhf.C[0] @ random_phase

    mc = RelMCOptimizer(
        nel=14,
        active_orbitals=12,
        core_orbitals=8,
        econv=1e-12,
        gconv=1e-10,
    )(rhf)
    mc.run()

    assert rhf.E == approx(erhf)
    assert mc.E == approx(emcscf)

    dsrg = DSRG_MRPT2_DF(flow_param=0.5)(mc)
    dsrg.run()
    print(dsrg.E)
    assert dsrg.E == approx(ept2)


def test_mrpt2_n2_so():
    erhf = -108.954140898736
    emcscf = -109.0811491968
    ept2 = -0.1577115347604592

    xyz = """
    N 0.0 0.0 0.0
    N 0.0 0.0 2.0
    """

    system = System(
        xyz=xyz,
        basis_set="cc-pVDZ",
        auxiliary_basis_set="cc-pVTZ-JKFIT",
        unit="bohr",
        x2c_type="so",
        snso_type="row-dependent",
    )
    rhf = GHF(charge=0, econv=1e-12)(system)
    rhf.run()

    mc = RelMCOptimizer(
        nel=14,
        active_orbitals=12,
        core_orbitals=8,
    )(rhf)
    mc.run()

    # assert rhf.E == approx(erhf)
    # assert mc.E == approx(emcscf)

    dsrg = DSRG_MRPT2()(mc)
    dsrg.run()
    print(dsrg.E)
    # assert dsrg.E == approx(ept2)


test_mrpt2_n2_nonrel()
