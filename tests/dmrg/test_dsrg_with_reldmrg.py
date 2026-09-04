"""
RelDSRG-MRPT2 interfacing with 2C-DMRG-CASSCF (complex, spin-orbit).

Same contract as test_dsrg_with_dmrg.py, but through the complex/SGF path:
RelDSRG_MRPT2.get_integrals calls ``ci_solver.make_average_cumulants()``,
exercising RelDMRGSolver's spin-orbital 3-RDM machinery (test_reldmrg_rdms.py
tests make_3rdm directly; here it is reached through the generic DSRG path).

Unlike the CASSCF-only comparison in test_casscf_with_reldmrg.py (which only
needs energy invariance under active-active rotations to hold to high
precision), DSRG-MRPT2 is *not* invariant to the specific active-active
orbital choice fixed by semicanonicalizing on the (DMRG- vs exact-RelCI-)
converged 1-RDM. The two DMRG-CASSCF and RelCI-CASSCF trajectories reach
numerically distinct (but physically equivalent) semicanonical orbitals, so a
small residual survives even as the DMRG bond dimension grows -- this was
checked explicitly and does not shrink between bond_dims=800 and 1200, so it
is a real gauge effect and not truncation noise. Hence ``approx_loose`` (which
carries a relative tolerance) rather than the tighter ``approx`` used for the
CASSCF energy comparisons.
"""

import pytest

from forte2 import System, GHF, MCOptimizer, RelCISolver, X2CParams
from forte2.dmrg import RelDMRGSolver
from forte2.dsrg import RelDSRG_MRPT2
from forte2.base_classes.params import DMRGParams
from forte2.helpers.comparisons import approx, approx_loose

from conftest import requires_block2_complex

# Mirrors the TIGHT schedule in test_casscf_with_reldmrg.py.
TIGHT = lambda scratch=None: DMRGParams(
    bond_dims=[400] * 4 + [800] * 4,
    noises=[1e-4] * 4 + [1e-6] * 2 + [0.0],
    thrds=[1e-14] * 8,
    n_sweeps=16,
    n_threads=1,
    iprint=0,
    scratch=scratch,
)

DMRG_MC_KWARGS = dict(g_tol=1e-6, maxiter=100)


@requires_block2_complex
@pytest.mark.slow
def test_reldsrg_mrpt2_with_reldmrg_casscf(tmp_path):
    """2C-DMRG-CASSCF + RelDSRG-MRPT2 == 2C-RelCI-CASSCF + RelDSRG-MRPT2.

    HF/cc-pVDZ with spin-orbit X2C, CAS(10,8) (same system as
    test_reldmrg_casscf_hf_small_cas).
    """
    system = System(
        xyz="H 0.0 0.0 0.0\nF 0.0 0.0 2.0",
        basis_set="cc-pvdz",
        auxiliary_basis_set="cc-pVTZ-JKFIT",
        unit="bohr",
        x2c=X2CParams(x2c_type="so"),
    )
    scf = GHF(charge=0)(system)

    ci_ref = RelCISolver(nel=10, core_orbitals=4, active_orbitals=8)
    mc_ref = MCOptimizer(ci_ref, g_tol=1e-7)(scf)
    dsrg_ref = RelDSRG_MRPT2(flow_param=0.5, relax_reference="once")(mc_ref)
    dsrg_ref.run()

    scf2 = GHF(charge=0)(system)
    dmrg_solver = RelDMRGSolver(
        nel=10, core_orbitals=4, active_orbitals=8, dmrg_params=TIGHT(str(tmp_path))
    )
    mc = MCOptimizer(dmrg_solver, **DMRG_MC_KWARGS)(scf2)
    dsrg = RelDSRG_MRPT2(flow_param=0.5, relax_reference="once")(mc)
    dsrg.run()

    # The un-relaxed reference energy (column 2) is just <Psi|bare H|Psi> at
    # the converged CASSCF orbitals/reference, which is invariant to
    # active-active rotations, so it still matches tightly.
    assert dsrg.relax_energies[0, 2] == approx(dsrg_ref.relax_energies[0, 2])
    assert dsrg.relax_energies[0, 0] == approx_loose(dsrg_ref.relax_energies[0, 0])
    assert dsrg.relax_energies[0, 1] == approx_loose(dsrg_ref.relax_energies[0, 1])
