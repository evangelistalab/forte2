"""
Relativistic (complex, two-component) DMRG tight-interface tests: RDMs.

The 1- and 2-RDMs from the relativistic DMRG solver must follow the same
convention as the RelCI solver: complex spin-orbital RDMs with the energy
round-trip

    E = E_core + sum_pq H_pq gamma1_pq + 0.5 * sum_pqrs V_pqrs gamma2_pqrs

where V is in physicist's notation (V[p,q,r,s] = <pq|rs>). This is the same
identity checked for RelCI in tests/ci (see the 2C branch of _test_rdms and
tests/ci/test_rel_ci_rdms.py). RDMs are also cross-checked against 2C-FCI.
"""

import numpy as np

from forte2 import System, GHF
from forte2.ci import RelCI
from forte2.dmrg import RelDMRG
from forte2.base_classes.params import DMRGParams, X2CParams
from forte2.helpers.comparisons import approx

from conftest import requires_block2_complex

TIGHT = lambda scratch=None: DMRGParams(
    bond_dims=[250] * 4 + [500] * 4,
    noises=[1e-4] * 4 + [1e-6] * 2 + [0.0],
    thrds=[1e-12] * 8,
    n_sweeps=14,
    n_threads=1,
    iprint=0,
    scratch=scratch,
)


def _hf_x2c_system():
    return System(
        xyz="H 0.0 0.0 0.0\nF 0.0 0.0 2.0",
        basis_set="cc-pvdz",
        auxiliary_basis_set="cc-pVTZ-JKFIT",
        unit="bohr",
        x2c=X2CParams(x2c_type="so"),
    )


# CAS with 6 active electrons in 12 active spinors: genuinely correlated (so the
# MPS is non-trivial) and matches the reference active space in tests/ci.
NEL = 10
NCORE = 2
NACT_SPINORS = 12
NACTEL = NEL - NCORE  # each spinor holds one electron in the 2C representation


def _build_reldmrg(system, scratch, nroots=1):
    scf = GHF(charge=0)(system)
    dmrg = RelDMRG(
        nel=NEL,
        core_orbitals=NCORE,
        active_orbitals=NACT_SPINORS,
        nroots=nroots,
        dmrg_params=TIGHT(scratch),
    )(scf)
    dmrg.run()
    return dmrg


@requires_block2_complex
def test_reldmrg_1rdm_hermitian_and_trace(tmp_path):
    """The spin-orbital 1-RDM is complex Hermitian with trace = n active electrons."""
    dmrg = _build_reldmrg(_hf_x2c_system(), str(tmp_path))
    g1 = dmrg.make_1rdm(0)
    assert g1.shape == (NACT_SPINORS, NACT_SPINORS)
    assert np.iscomplexobj(g1)
    # In the 2C spin-orbital representation each core spinor holds one electron,
    # so the active electron count is nel - ncore.
    assert np.trace(g1).real == approx(float(NACTEL))
    assert abs(np.trace(g1).imag) < 1e-8
    assert np.linalg.norm(g1 - g1.conj().T) < 1e-6


@requires_block2_complex
def test_reldmrg_rdm_energy_roundtrip(tmp_path):
    """
    Energy reconstructed from the complex 1- and 2-RDMs matches the DMRG energy,
    fixing the relativistic RDM/integral convention.
    """
    dmrg = _build_reldmrg(_hf_x2c_system(), str(tmp_path))

    g1 = dmrg.make_1rdm(0)
    g2 = dmrg.make_2rdm(0)
    ints = dmrg.sub_solvers[0].ints

    e = ints.E
    e += np.einsum("pq,pq->", ints.H, g1)
    e += 0.5 * np.einsum("pqrs,pqrs->", ints.V, g2)

    assert e.real == approx(dmrg.E[0])
    assert abs(e.imag) < 1e-8


@requires_block2_complex
def test_reldmrg_rdms_match_relci(tmp_path):
    """RelDMRG complex 1- and 2-RDMs match the 2C-FCI RDMs on the same space."""
    system = _hf_x2c_system()
    dmrg = _build_reldmrg(system, str(tmp_path))

    scf = GHF(charge=0)(system)
    ci = RelCI(nel=NEL, core_orbitals=NCORE, active_orbitals=NACT_SPINORS)(scf)
    ci.run()

    # RDMs are limited by DMRG truncation error, looser than the energy.
    g1_dmrg = dmrg.make_1rdm(0)
    g1_ci = ci.make_1rdm(0)
    assert np.linalg.norm(g1_dmrg - g1_ci) < 1e-5

    g2_dmrg = dmrg.make_2rdm(0)
    g2_ci = ci.make_2rdm(0)
    assert np.linalg.norm(g2_dmrg - g2_ci) < 1e-5


@requires_block2_complex
def test_reldmrg_average_rdm_roundtrip(tmp_path):
    """
    State-averaged RDM machinery (inherited from RelCIBase/CIBase) works with
    RelDMRG: the average energy reconstructs from the average 1- and 2-RDMs.
    """
    dmrg = _build_reldmrg(_hf_x2c_system(), str(tmp_path), nroots=2)

    g1 = dmrg.make_average_1rdm()
    g2 = dmrg.make_average_2rdm()
    ints = dmrg.sub_solvers[0].ints

    e = ints.E
    e += np.einsum("pq,pq->", ints.H, g1)
    e += 0.5 * np.einsum("pqrs,pqrs->", ints.V, g2)

    assert e.real == approx(dmrg.compute_average_energy())


@requires_block2_complex
def test_reldmrg_3rdm_matches_relci(tmp_path):
    """RelDMRG complex 3-RDM matches the 2C-FCI 3-RDM on the same active space."""
    system = _hf_x2c_system()
    dmrg = _build_reldmrg(system, str(tmp_path))

    scf = GHF(charge=0)(system)
    ci = RelCI(nel=NEL, core_orbitals=NCORE, active_orbitals=NACT_SPINORS)(scf)
    ci.run()

    g3_dmrg = dmrg.make_3rdm(0)
    g3_ci = ci.sub_solvers[0].make_3rdm(0)
    assert g3_dmrg.shape == (NACT_SPINORS,) * 6
    assert np.iscomplexobj(g3_dmrg)
    # RDMs are limited by DMRG truncation error, looser than the energy.
    assert np.linalg.norm(g3_dmrg - g3_ci) < 1e-4


@requires_block2_complex
def test_reldmrg_3rdm_partial_trace(tmp_path):
    """
    Partial trace of the spin-orbital 3-RDM gives the 2-RDM:
    sum_r gamma3[p,q,r,s,t,r] = (N_act - 2) * gamma2[p,q,s,t].
    """
    dmrg = _build_reldmrg(_hf_x2c_system(), str(tmp_path))

    g2 = dmrg.make_2rdm(0)
    g3 = dmrg.make_3rdm(0)
    g2_from_g3 = np.einsum("pqrstr->pqst", g3) / (NACTEL - 2)
    assert np.linalg.norm(g2_from_g3 - g2) < 1e-4


@requires_block2_complex
def test_reldmrg_transition_rdms_hermiticity(tmp_path):
    """
    RelDMRG cross-root transition RDMs obey the Hermitian-conjugate relations

        gamma1(1,0)[p,q]         = conj(gamma1(0,1)[q,p])
        gamma2(1,0)[p,q,r,s]     = conj(gamma2(0,1)[r,s,p,q])
        gamma3(1,0)[p,q,r,s,t,u] = conj(gamma3(0,1)[s,t,u,p,q,r]).

    These follow from gamma(m,n) = <m| E... |n> = <n| E... |m>^* and are
    independent of the (basis-arbitrary) phase and of any degeneracy, so they
    validate the bra/ket transition-RDM machinery directly. The transform from
    block2's layout to forte2's is separately pinned to ground truth by
    test_reldmrg_3rdm_matches_relci (diagonal, complex/SGF) and by the
    non-relativistic transition-RDM test (cross-root).
    """
    dmrg = _build_reldmrg(_hf_x2c_system(), str(tmp_path), nroots=2)

    t1_01 = dmrg.make_1rdm(0, 1)
    t1_10 = dmrg.make_1rdm(1, 0)
    assert np.linalg.norm(t1_10 - t1_01.conj().transpose(1, 0)) < 1e-6

    t2_01 = dmrg.make_2rdm(0, 1)
    t2_10 = dmrg.make_2rdm(1, 0)
    assert np.linalg.norm(t2_10 - t2_01.conj().transpose(2, 3, 0, 1)) < 1e-6

    t3_01 = dmrg.make_3rdm(0, 1)
    t3_10 = dmrg.make_3rdm(1, 0)
    assert (
        np.linalg.norm(t3_10 - t3_01.conj().transpose(3, 4, 5, 0, 1, 2)) < 1e-6
    )
