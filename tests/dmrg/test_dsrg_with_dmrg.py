"""
DSRG-MRPT2 interfacing with DMRG-CASSCF (spin-free, real).

DSRG-MRPT2 does not talk to a CI-type active-space solver directly: it calls
``ci_solver.make_average_cumulants()`` (see ``DSRG_MRPT2.get_integrals``),
which in turn calls ``make_average_{1,2,3}rdm`` on ``CIBase``. This exercises
DMRGSolver's 3-RDM machinery through that generic path (as opposed to the
direct ``make_3rdm`` calls in test_dmrg_rdms.py), plus the ``set_ints`` /
``reset_eigensolver`` / rerun contract used for reference relaxation. The
DMRG-CASSCF + DSRG-MRPT2 energy must match the exact CI-CASSCF + DSRG-MRPT2
energy computed in-test.
"""

import pytest

from forte2 import System, RHF, State, MCOptimizer, CISolver
from forte2.dmrg import DMRGSolver
from forte2.dsrg import DSRG_MRPT2
from forte2.base_classes.params import DMRGParams
from forte2.helpers.comparisons import approx

from conftest import requires_block2

# A tight DMRG schedule: on this small active space this is effectively exact,
# so the RDM/cumulant truncation noise stays well below the DSRG-MRPT2
# comparison tolerance (mirrors test_casscf_with_dmrg.py).
TIGHT = lambda scratch=None: DMRGParams(
    bond_dims=[200] * 4 + [400] * 4,
    noises=[1e-4] * 4 + [1e-6] * 2 + [0.0],
    thrds=[1e-12] * 8,
    n_sweeps=12,
    n_threads=1,
    iprint=0,
    scratch=scratch,
)


@requires_block2
@pytest.mark.slow
def test_dsrg_mrpt2_with_dmrg_casscf(tmp_path):
    """DMRG-CASSCF + DSRG-MRPT2 == CI-CASSCF + DSRG-MRPT2 for N2, CAS(6,6)."""
    xyz = "N 0.0 0.0 0.0\nN 0.0 0.0 2.0"
    system = System(
        xyz=xyz,
        basis_set="cc-pVDZ",
        auxiliary_basis_set="cc-pVTZ-JKFIT",
        unit="bohr",
    )

    rhf = RHF(charge=0)(system)
    ci_solver = CISolver(
        states=State(nel=14, multiplicity=1, ms=0.0),
        core_orbitals=4,
        active_orbitals=6,
    )
    mc_ref = MCOptimizer(ci_solver)(rhf)
    dsrg_ref = DSRG_MRPT2(flow_param=0.5, relax_reference="once")(mc_ref)
    dsrg_ref.run()

    rhf2 = RHF(charge=0)(system)
    dmrg_solver = DMRGSolver(
        states=State(nel=14, multiplicity=1, ms=0.0),
        active_orbitals=[4, 5, 6, 7, 8, 9],
        core_orbitals=[0, 1, 2, 3],
        dmrg_params=TIGHT(str(tmp_path)),
    )
    mc = MCOptimizer(dmrg_solver)(rhf2)
    dsrg = DSRG_MRPT2(flow_param=0.5, relax_reference="once")(mc)
    dsrg.run()

    assert dsrg.relax_energies[0, 0] == approx(dsrg_ref.relax_energies[0, 0])
    assert dsrg.relax_energies[0, 1] == approx(dsrg_ref.relax_energies[0, 1])
    assert dsrg.relax_energies[0, 2] == approx(dsrg_ref.relax_energies[0, 2])
