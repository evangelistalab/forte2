import numpy as np

from forte2 import CISolver, MCOptimizer, RHF, State, System
from forte2.helpers.comparisons import approx


def test_mcscf_noncontiguous_spaces():
    # The results of this test should be strictly identical to test_casscf_n2

    erhf = -108.761639873604
    ecasscf = -108.9800484156

    xyz = """
    N 0.0 0.0 0.0
    N 0.0 0.0 1.4
    """

    system = System(
        xyz=xyz,
        basis_set="cc-pVDZ",
        auxiliary_basis_set="cc-pVTZ-JKFIT",
        symmetry=True,
    )
    rhf = RHF(charge=0, e_tol=1e-12)(system)
    rhf.run()
    assert rhf.E == approx(erhf)

    # swap orbitals to make them non-contiguous
    core = [0, 1, 3, 6]
    actv = [2, 4, 5, 7, 8, 11]
    virt = sorted(set(range(system.nbf)) - set(core + actv))
    perm = np.argsort(core + actv + virt)
    rhf.mos = rhf.mos.permute(perm)

    ci_solver = CISolver(
        State(nel=14, multiplicity=1, ms=0.0),
        core_orbitals=core,
        active_orbitals=actv,
    )
    mc = MCOptimizer(
        ci_solver,
        final_orbitals="natural",
    )(rhf)
    mc.run()
    assert mc.E == approx(ecasscf)
