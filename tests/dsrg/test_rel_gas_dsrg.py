import numpy as np

from forte2 import CISolver, MCOptimizer, RHF, RelCISolver, RelState, State, System
from forte2.dsrg import DSRG_MRPT2, DSRG_MRPT3, RelDSRG_MRPT2, RelDSRG_MRPT3
from forte2.orbitals import SpinorUpcaster
from forte2.helpers.comparisons import approx_abs

# On a nonrelativistic Hamiltonian a randomly phased spinor reference is the same
# wave function as the spin-adapted one, so the two-component GAS solvers must
# reproduce the spin-adapted GAS energies. That holds only where the Fock
# coupling between GASes vanishes: the two families disagree about whether that
# coupling belongs in H(0) (see DSRGBase._fock_actv_0th), and with a coupling of
# 0.042 Eh they differ by 3e-6 Eh at PT2 and 4e-4 Eh at PT3. The N2 partition
# below puts the two GASes in different irreps, so the coupling is zero by
# symmetry and the comparison is exact -- which is what makes it a test of the
# GAS bookkeeping rather than of the convention.

XYZ_N2 = """
N 0.0 0.0 0.0
N 0.0 0.0 2.0
"""
CORE = [0, 1, 2, 3]
GAS = [[4, 5], [6, 7, 8, 9]]


def _spinor(orbitals):
    return [s for i in orbitals for s in (2 * i, 2 * i + 1)]


def _build_system():
    return System(
        xyz=XYZ_N2,
        basis_set="cc-pVDZ",
        auxiliary_basis_set="cc-pVTZ-JKFIT",
        unit="bohr",
    )


def _run_nonrel(dsrg_cls):
    rhf = RHF(charge=0, e_tol=1e-12, d_tol=1e-10)(_build_system())
    rhf.run()
    ci_solver = CISolver(
        states=State(nel=14, multiplicity=1, ms=0.0, gas_min=[0], gas_max=[4]),
        core_orbitals=CORE,
        active_orbitals=GAS,
    )
    mc = MCOptimizer(ci_solver, maxiter=500, e_tol=1e-12, g_tol=1e-10)(rhf)
    mc.run()
    dsrg = dsrg_cls(flow_param=1.0, relax_reference="once")(mc)
    dsrg.run()
    return mc, dsrg


def _run_rel(dsrg_cls):
    rhf = RHF(charge=0, e_tol=1e-12, d_tol=1e-10)(_build_system())
    rhf.run()
    # The random phase makes the spinor 1-RDM genuinely complex, so a convention
    # error in the active mean field cannot pass unnoticed.
    conv = SpinorUpcaster(apply_random_phase=True, rng=1234)(rhf)
    conv.run()
    ci_solver = RelCISolver(
        states=RelState(nel=14, gas_min=[0], gas_max=[4]),
        core_orbitals=_spinor(CORE),
        active_orbitals=[_spinor(g) for g in GAS],
    )
    mc = MCOptimizer(ci_solver, maxiter=500, e_tol=1e-12, g_tol=1e-10)(conv)
    mc.run()
    dsrg = dsrg_cls(flow_param=1.0, relax_reference="once")(mc)
    dsrg.run()
    return mc, dsrg


def _compare(nonrel_cls, rel_cls):
    mc_nr, nr = _run_nonrel(nonrel_cls)
    mc_rel, rel = _run_rel(rel_cls)

    # The precondition: no coupling between GASes, so the convention cannot enter.
    for dsrg in (nr, rel):
        gas = dsrg.mo_space.gas_corr
        assert len(gas) == 2
        assert np.abs(dsrg.fock[gas[0], gas[1]]).max() < 1e-10

    assert np.real(mc_rel.E) == approx_abs(np.real(mc_nr.E), 1e-10)
    for i in (0, 1, 2):
        assert np.real(rel.relax_energies[0, i]) == approx_abs(
            np.real(nr.relax_energies[0, i]), 1e-7
        )


def test_rel_gas_dsrg_mrpt2_matches_nonrel():
    """Two-component GAS-DSRG-MRPT2 on a nonrelativistic Hamiltonian."""
    _compare(DSRG_MRPT2, RelDSRG_MRPT2)


def test_rel_gas_dsrg_mrpt3_matches_nonrel():
    """Two-component GAS-DSRG-MRPT3 on a nonrelativistic Hamiltonian."""
    _compare(DSRG_MRPT3, RelDSRG_MRPT3)
