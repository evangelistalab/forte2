import numpy as np

from forte2 import CISolver, MCOptimizer, RHF, RelCISolver, RelState, State, System
from forte2.dsrg import DSRG_MRPT2, DSRG_MRPT3, RelDSRG_MRPT2, RelDSRG_MRPT3
from forte2.orbitals import SpinorUpcaster
from forte2.helpers.comparisons import approx_abs

# On a nonrelativistic Hamiltonian a randomly phased spinor reference is the same
# wave function as the spin-adapted one, so the two-component GAS solvers must
# reproduce the spin-adapted GAS energies.
#
# At second order they do so for any inter-GAS Fock coupling, since both families
# drop it from H(0) (see DSRGBase._fock_actv_0th) -- hence the H2O case below,
# whose coupling is 0.042 Eh. Third order keeps that coupling in the spin-free
# H(0) and drops it in the two-component one, a deliberate split recorded in
# _build_fock_0th_1st, so the two differ by ~1e-4 Eh once the coupling is
# nonzero. PT3 is therefore compared on the N2 partition, whose two GASes sit in
# different irreps so the coupling vanishes by symmetry and both conventions
# coincide.

# (xyz, unit, nel, core, GASes, gas_max, freeze_inter_gas_rots)
N2_ZERO_COUPLING = (
    "N 0.0 0.0 0.0\nN 0.0 0.0 2.0\n",
    "bohr",
    14,
    [0, 1, 2, 3],
    [[4, 5], [6, 7, 8, 9]],
    [4],
    False,
)
H2O_LARGE_COUPLING = (
    "O\nH  1 1.00\nH  1 1.00 2 103.1\n",
    "angstrom",
    10,
    [0],
    [[1, 2], [3, 4, 5, 6]],
    [4],
    True,
)


def _spinor(orbitals):
    return [s for i in orbitals for s in (2 * i, 2 * i + 1)]


def _build_system(case):
    xyz, unit = case[0], case[1]
    return System(
        xyz=xyz,
        basis_set="cc-pVDZ",
        auxiliary_basis_set="cc-pVTZ-JKFIT",
        unit=unit,
    )


def _run(case, dsrg_cls, relativistic):
    _, _, nel, core, gas, gas_max, freeze = case
    rhf = RHF(charge=0, e_tol=1e-12, d_tol=1e-10)(_build_system(case))
    rhf.run()
    if relativistic:
        # The random phase makes the spinor 1-RDM genuinely complex, so a
        # convention error in the active mean field cannot pass unnoticed.
        parent = SpinorUpcaster(apply_random_phase=True, rng=1234)(rhf)
        parent.run()
        ci_solver = RelCISolver(
            states=RelState(nel=nel, gas_min=[0], gas_max=gas_max),
            core_orbitals=_spinor(core),
            active_orbitals=[_spinor(g) for g in gas],
        )
    else:
        parent = rhf
        ci_solver = CISolver(
            states=State(nel=nel, multiplicity=1, ms=0.0, gas_min=[0], gas_max=gas_max),
            core_orbitals=core,
            active_orbitals=gas,
        )
    mc = MCOptimizer(
        ci_solver,
        maxiter=500,
        e_tol=1e-12,
        g_tol=1e-10,
        freeze_inter_gas_rots=freeze,
    )(parent)
    mc.run()
    dsrg = dsrg_cls(flow_param=1.0, relax_reference="once")(mc)
    dsrg.run()
    return mc, dsrg


def _compare(case, nonrel_cls, rel_cls, coupling_is_zero):
    mc_nr, nr = _run(case, nonrel_cls, False)
    mc_rel, rel = _run(case, rel_cls, True)

    for dsrg in (nr, rel):
        gas = dsrg.mo_space.gas_corr
        assert len(gas) == 2
        coupling = np.abs(dsrg.fock[gas[0], gas[1]]).max()
        assert (coupling < 1e-10) == coupling_is_zero

    assert np.real(mc_rel.E) == approx_abs(np.real(mc_nr.E), 1e-10)
    for i in (0, 1, 2):
        assert np.real(rel.relax_energies[0, i]) == approx_abs(
            np.real(nr.relax_energies[0, i]), 1e-7
        )


def test_rel_gas_dsrg_mrpt2_matches_nonrel():
    """Two-component GAS-DSRG-MRPT2, with the GASes genuinely coupled."""
    _compare(H2O_LARGE_COUPLING, DSRG_MRPT2, RelDSRG_MRPT2, coupling_is_zero=False)


def test_rel_gas_dsrg_mrpt3_matches_nonrel():
    """Two-component GAS-DSRG-MRPT3, restricted to zero inter-GAS coupling."""
    _compare(N2_ZERO_COUPLING, DSRG_MRPT3, RelDSRG_MRPT3, coupling_is_zero=True)
