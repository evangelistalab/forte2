"""
DMRG (block2) tight-interface tests: reduced density matrices.

The 1- and 2-RDMs returned by the DMRG solver must follow the same convention
as the CI solver, i.e. spin-free RDMs with the energy round-trip

    E = E_core + sum_pq H_pq gamma1_pq + 0.5 * sum_pqrs V_pqrs gamma2_pqrs

where V is in physicist's notation (V[p,q,r,s] = <pq|rs>). This is the same
identity checked for the CI solver in tests/ci/test_ci_rdms.py.

We also cross-check the DMRG RDMs against the FCI RDMs from the CI solver on the
same active space and orbitals.
"""

import numpy as np

from forte2 import System, RHF, State, CI
from forte2.dmrg import DMRG
from forte2.base_classes.params import DMRGParams
from forte2.helpers.comparisons import approx

from conftest import requires_block2

TIGHT_DMRG_PARAMS = lambda scratch=None: DMRGParams(
    bond_dims=[200] * 4 + [400] * 4,
    noises=[1e-4] * 4 + [1e-6] * 2 + [0.0],
    thrds=[1e-12] * 8,
    n_sweeps=12,
    n_threads=1,
    iprint=0,
    scratch=scratch,
)


def _build_dmrg_n2(system, scratch):
    rhf = RHF(charge=0, e_tol=1e-12)(system)
    dmrg = DMRG(
        states=State(nel=14, multiplicity=1, ms=0.0),
        core_orbitals=[0, 1, 2, 3],
        active_orbitals=[4, 5, 6, 7, 8, 9],
        dmrg_params=TIGHT_DMRG_PARAMS(scratch),
    )(rhf)
    dmrg.run()
    return dmrg


def _build_ci_n2(system):
    rhf = RHF(charge=0, e_tol=1e-12)(system)
    ci = CI(
        states=State(nel=14, multiplicity=1, ms=0.0),
        core_orbitals=[0, 1, 2, 3],
        active_orbitals=[4, 5, 6, 7, 8, 9],
    )(rhf)
    ci.run()
    return ci


# H2O near equilibrium: the three lowest singlets are non-degenerate, so their
# transition RDMs are well-defined (up to an overall phase) and can be matched
# against FCI. (Degenerate manifolds, e.g. the N2 pi states, would leave the
# transition RDM basis-dependent within the manifold.)
H2O_XYZ = "O 0 0 0.117\nH 0 0.757 -0.469\nH 0 -0.757 -0.469"


# CAS(6,6) has only 6 sites, so a bond dimension of a few hundred is effectively
# exact; the transition RDMs between state-averaged excited roots nonetheless
# need a longer, noise-free tail to converge the individual (split) root MPS to
# the eigenstates. This dedicated schedule drives them to ~machine precision so
# the FCI comparison is not truncation-limited.
CONVERGED_DMRG_PARAMS = lambda scratch=None: DMRGParams(
    bond_dims=[200] * 4 + [400] * 12,
    noises=[1e-4] * 4 + [1e-6] * 4 + [0.0] * 8,
    thrds=[1e-14] * 16,
    n_sweeps=16,
    n_threads=1,
    iprint=0,
    scratch=scratch,
)


def _build_dmrg_h2o(system, scratch, nroots, dmrg_params=None):
    rhf = RHF(charge=0, e_tol=1e-12)(system)
    dmrg = DMRG(
        states=State(nel=10, multiplicity=1, ms=0.0),
        core_orbitals=[0, 1],
        active_orbitals=[2, 3, 4, 5, 6, 7],
        nroots=nroots,
        dmrg_params=dmrg_params or TIGHT_DMRG_PARAMS(scratch),
    )(rhf)
    dmrg.run()
    return dmrg


def _build_ci_h2o(system, nroots):
    rhf = RHF(charge=0, e_tol=1e-12)(system)
    ci = CI(
        states=State(nel=10, multiplicity=1, ms=0.0),
        core_orbitals=[0, 1],
        active_orbitals=[2, 3, 4, 5, 6, 7],
        nroots=nroots,
    )(rhf)
    ci.run()
    return ci


def _match_up_to_sign(a, b):
    """Smallest of ||a - b|| and ||a + b|| (transition RDMs carry a phase)."""
    return min(np.linalg.norm(a - b), np.linalg.norm(a + b))


@requires_block2
def test_dmrg_1rdm_trace(tmp_path):
    """Trace of the spin-free 1-RDM equals the number of active electrons."""
    system = System(
        xyz="N 0.0 0.0 0.0\nN 0.0 0.0 1.2",
        basis_set="cc-pvdz",
        auxiliary_basis_set="cc-pVTZ-JKFIT",
    )
    dmrg = _build_dmrg_n2(system, str(tmp_path))
    g1 = dmrg.make_sf_1rdm(0)
    assert g1.shape == (6, 6)
    # CAS(6,6): 6 active electrons.
    assert np.trace(g1) == approx(6.0)
    # Spin-free 1-RDM is symmetric.
    assert np.linalg.norm(g1 - g1.T) < 1e-8


@requires_block2
def test_dmrg_rdm_energy_roundtrip(tmp_path):
    """
    Energy reconstructed from the DMRG 1- and 2-RDMs matches the DMRG energy,
    fixing the RDM/integral convention (physicist V, spin-free RDMs).
    """
    system = System(
        xyz="N 0.0 0.0 0.0\nN 0.0 0.0 1.2",
        basis_set="cc-pvdz",
        auxiliary_basis_set="cc-pVTZ-JKFIT",
    )
    dmrg = _build_dmrg_n2(system, str(tmp_path))

    g1 = dmrg.make_sf_1rdm(0)
    g2 = dmrg.make_sf_2rdm(0)
    ints = dmrg.sub_solvers[0].ints

    e = ints.E
    e += np.einsum("pq,pq->", ints.H, g1)
    e += 0.5 * np.einsum("pqrs,pqrs->", ints.V, g2)

    assert e == approx(dmrg.E[0])


@requires_block2
def test_dmrg_rdms_match_fci(tmp_path):
    """DMRG spin-free 1- and 2-RDMs match the FCI RDMs on the same active space."""
    system = System(
        xyz="N 0.0 0.0 0.0\nN 0.0 0.0 1.2",
        basis_set="cc-pvdz",
        auxiliary_basis_set="cc-pVTZ-JKFIT",
    )
    dmrg = _build_dmrg_n2(system, str(tmp_path))
    ci = _build_ci_n2(system)

    # The RDMs are limited by the DMRG truncation error (the energy converges
    # far tighter than the density matrices at a given bond dimension).
    g1_dmrg = dmrg.make_sf_1rdm(0)
    g1_ci = ci.make_sf_1rdm(0)
    assert np.linalg.norm(g1_dmrg - g1_ci) < 1e-5

    g2_dmrg = dmrg.make_sf_2rdm(0)
    g2_ci = ci.make_sf_2rdm(0)
    assert np.linalg.norm(g2_dmrg - g2_ci) < 1e-5


@requires_block2
def test_dmrg_average_rdms_roundtrip(tmp_path):
    """
    State-averaged RDM machinery (inherited from CIBase) works with DMRG:
    the average energy reconstructs from the average 1-RDM and 2-RDM.
    """
    system = System(
        xyz="N 0.0 0.0 0.0\nN 0.0 0.0 1.2",
        basis_set="cc-pvdz",
        auxiliary_basis_set="cc-pVTZ-JKFIT",
    )
    rhf = RHF(charge=0, e_tol=1e-12)(system)
    dmrg = DMRG(
        states=State(nel=14, multiplicity=1, ms=0.0),
        core_orbitals=[0, 1, 2, 3],
        active_orbitals=[4, 5, 6, 7, 8, 9],
        nroots=2,
        dmrg_params=TIGHT_DMRG_PARAMS(str(tmp_path)),
    )(rhf)
    dmrg.run()

    g1 = dmrg.make_average_1rdm()
    g2 = dmrg.make_average_2rdm()
    ints = dmrg.sub_solvers[0].ints

    e = ints.E
    e += np.einsum("pq,pq->", ints.H, g1)
    e += 0.5 * np.einsum("pqrs,pqrs->", ints.V, g2)

    # equal-weight average over the two roots
    assert e == approx(dmrg.compute_average_energy())


@requires_block2
def test_dmrg_3rdm_matches_fci(tmp_path):
    """DMRG spin-free 3-RDM matches the FCI 3-RDM on the same active space."""
    system = System(
        xyz="N 0.0 0.0 0.0\nN 0.0 0.0 1.2",
        basis_set="cc-pvdz",
        auxiliary_basis_set="cc-pVTZ-JKFIT",
    )
    dmrg = _build_dmrg_n2(system, str(tmp_path))
    ci = _build_ci_n2(system)

    g3_dmrg = dmrg.make_sf_3rdm(0)
    g3_ci = ci.sub_solvers[0].make_sf_3rdm(0)
    assert g3_dmrg.shape == (6, 6, 6, 6, 6, 6)
    # The 3-RDM is limited by the DMRG truncation error at this bond dimension.
    assert np.linalg.norm(g3_dmrg - g3_ci) < 1e-4


@requires_block2
def test_dmrg_3rdm_partial_trace(tmp_path):
    """
    Partial trace of the spin-free 3-RDM gives the 2-RDM:
    sum_r gamma3[p,q,r,s,t,r] = (N_act - 2) * gamma2[p,q,s,t].
    """
    system = System(
        xyz="N 0.0 0.0 0.0\nN 0.0 0.0 1.2",
        basis_set="cc-pvdz",
        auxiliary_basis_set="cc-pVTZ-JKFIT",
    )
    dmrg = _build_dmrg_n2(system, str(tmp_path))

    g2 = dmrg.make_sf_2rdm(0)
    g3 = dmrg.make_sf_3rdm(0)
    # CAS(6,6): 6 active electrons.
    g2_from_g3 = np.einsum("pqrstr->pqst", g3) / (6 - 2)
    assert np.linalg.norm(g2_from_g3 - g2) < 1e-4


@requires_block2
def test_dmrg_transition_rdms_match_fci(tmp_path):
    """
    DMRG spin-free 1-, 2-, and 3-transition RDMs match the FCI transition RDMs
    (up to an overall phase) between non-degenerate roots.
    """
    system = System(
        xyz=H2O_XYZ,
        basis_set="cc-pVDZ",
        auxiliary_basis_set="cc-pVTZ-JKFIT",
    )
    dmrg = _build_dmrg_h2o(
        system, str(tmp_path), nroots=3, dmrg_params=CONVERGED_DMRG_PARAMS(str(tmp_path))
    )
    ci = _build_ci_h2o(system, nroots=3)

    # sanity: the roots are the ones we expect (and non-degenerate)
    for r in range(3):
        assert dmrg.E[r] == approx(ci.E[r])
    assert abs(ci.E[1] - ci.E[0]) > 1e-2
    assert abs(ci.E[2] - ci.E[1]) > 1e-2

    ci_solver = ci.sub_solvers[0]
    for left, right in [(0, 1), (0, 2), (1, 2)]:
        t1_dmrg = dmrg.make_sf_1rdm(left, right)
        t1_ci = ci_solver.make_sf_1rdm(left, right)
        assert _match_up_to_sign(t1_dmrg, t1_ci) < 1e-5

        t2_dmrg = dmrg.make_sf_2rdm(left, right)
        t2_ci = ci_solver.make_sf_2rdm(left, right)
        assert _match_up_to_sign(t2_dmrg, t2_ci) < 1e-5

        t3_dmrg = dmrg.make_sf_3rdm(left, right)
        t3_ci = ci_solver.make_sf_3rdm(left, right)
        assert _match_up_to_sign(t3_dmrg, t3_ci) < 1e-5


@requires_block2
def test_dmrg_diagonal_rdm_via_transition_api(tmp_path):
    """
    Passing right_root == left_root returns the ordinary (diagonal) RDM,
    identical to passing right_root=None.
    """
    system = System(
        xyz=H2O_XYZ,
        basis_set="cc-pVDZ",
        auxiliary_basis_set="cc-pVTZ-JKFIT",
    )
    dmrg = _build_dmrg_h2o(system, str(tmp_path), nroots=2)

    for maker in (dmrg.make_sf_1rdm, dmrg.make_sf_2rdm, dmrg.make_sf_3rdm):
        diag = maker(1)
        diag_explicit = maker(1, 1)
        assert np.linalg.norm(diag - diag_explicit) < 1e-10
