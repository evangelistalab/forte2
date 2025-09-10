from logging import fatal
import pytest

from forte2 import System, GHF, RelState
from forte2.orbopt import RelMCOptimizer
from forte2.helpers.comparisons import approx


@pytest.mark.slow
def test_rel_gasscf_equivalence_to_nonrel():
    erhf = -76.05702512779526
    emcscf = -76.1156924702

    xyz = """
    O   0.0000000000  -0.0000000000  -0.0662628033
    H   0.0000000000  -0.7540256101   0.5259060578
    H  -0.0000000000   0.7530256101   0.5260060578
    """

    system = System(
        xyz=xyz, basis_set="cc-pVTZ", auxiliary_basis_set="def2-universal-jkfit"
    )

    hf = GHF(charge=0, econv=1e-12, dconv=1e-12)(system)

    mc = RelMCOptimizer(
        RelState(nel=10, gas_min=[3], gas_max=[6]),
        core_orbitals=4,
        active_orbitals=(6, 6),
        do_diis=False,
        freeze_inter_gas_rots=True,
    )(hf)
    mc.run()

    assert hf.E == approx(erhf)
    assert mc.E == approx(emcscf)
