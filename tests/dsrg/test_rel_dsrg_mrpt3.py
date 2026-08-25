import numpy as np
import pytest

from forte2 import System, GHF, MCOptimizer, RelCISolver, X2CParams
from forte2.dsrg import RelDSRG_MRPT2, RelDSRG_MRPT3
from forte2.helpers.comparisons import approx
from forte2.data.atom_data import EH_TO_WN


def test_mrpt3_n2_nonrel():
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

    ci_solver = RelCISolver(
        nel=14,
        core_orbitals=8,
        active_orbitals=12,
    )
    mc = MCOptimizer(ci_solver)(rhf)
    mc.run()

    assert rhf.E == approx(erhf)
    assert mc.E == approx(emcscf)

    dsrg = RelDSRG_MRPT3(
        frozen_core_orbitals=4,
        flow_param=0.5,
        relax_reference="iterate",
    )(mc)
    dsrg.run()

    assert dsrg.relax_energies[0] == approx(
        [-109.25301485009223, -109.2538362628393, -109.08114919682387]
    )
    assert dsrg.relax_energies[1] == approx(
        [-109.25344887585058, -109.25344888535047, -109.0802678007682]
    )
    assert dsrg.relax_energies[2] == approx(
        [-109.25344824472272, -109.25344824472299, -109.08026606599341]
    )


def test_mrpt3_f_atom_rel_sa():
    xyz = """
    F 0 0 0
    """

    system = System(
        xyz=xyz,
        basis_set="decon-cc-pVTZ",
        auxiliary_basis_set="cc-pVQZ-JKFIT",
        x2c=X2CParams(x2c_type="so", snso_type="row-dependent"),
        use_gaussian_charges=True,
    )
    mf = GHF(charge=-1, die_if_not_converged=False)(system)
    ci_solver = RelCISolver(
        nel=9,
        nroots=6,
        active_orbitals=8,
        core_orbitals=2,
    )
    mc = MCOptimizer(
        ci_solver,
        e_tol=1e-8,
        g_tol=1e-6,
    )(mf)
    dsrg = RelDSRG_MRPT3(flow_param=0.35, relax_reference="once")(mc)
    dsrg.run()
    assert (dsrg.relax_eigvals[4] - dsrg.relax_eigvals[3]) * EH_TO_WN == pytest.approx(
        400.1722015310902, abs=1e-2
    )


def test_mrpt3_fno():
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

    ci_solver = RelCISolver(
        nel=14,
        core_orbitals=8,
        active_orbitals=12,
    )
    mc = MCOptimizer(ci_solver)(rhf)
    mc.run()

    pt3_ref = RelDSRG_MRPT3(flow_param=0.5, frozen_core_orbitals=4)(mc)
    pt3_ref.run()

    # p_o=1.0 retains every virtual orbital, so the FNO correction should
    # vanish and the chained PT3 energy must reproduce the untruncated result
    # to near machine precision -- a real correctness check on the
    # correction/detection logic, not an accuracy statement about truncation.
    pt2_full_100 = RelDSRG_MRPT2(flow_param=0.5, frozen_core_orbitals=4, fno_p_o=1.0)(
        mc
    )
    pt2_full_100.run()
    pt2_fno_100 = RelDSRG_MRPT2(flow_param=0.5)(pt2_full_100)
    pt2_fno_100.run()
    assert pt2_fno_100.fno_e == approx(0.0)
    assert pt2_fno_100.fno_hbar1 == approx(np.zeros_like(pt2_fno_100.fno_hbar1))
    assert pt2_fno_100.fno_hbar2 == approx(np.zeros_like(pt2_fno_100.fno_hbar2))
    pt3_fno_100 = RelDSRG_MRPT3(flow_param=0.5)(pt2_fno_100)
    pt3_fno_100.run()
    assert pt3_fno_100.E_dsrg == approx(pt3_ref.E_dsrg)

    # a genuinely truncated case: the FNO correction is non-zero and brings
    # the FNO-space PT3 energy within a bounded, physically sane distance of
    # the untruncated reference for this small basis.
    pt2_full = RelDSRG_MRPT2(flow_param=0.5, frozen_core_orbitals=4, fno_p_o=0.9)(mc)
    pt2_full.run()
    assert pt2_full.mo_space.nvirt < pt2_full_100.mo_space.nvirt

    pt2_fno = RelDSRG_MRPT2(flow_param=0.5)(pt2_full)
    pt2_fno.run()
    assert abs(pt2_fno.fno_e) > 1e-6
    pt3_fno = RelDSRG_MRPT3(flow_param=0.5)(pt2_fno)
    pt3_fno.run()
    assert abs(pt3_fno.E_dsrg - pt3_ref.E_dsrg) < 0.05

    # with reference relaxation, the correction is folded directly into the
    # effective Hamiltonian handed to the CI solver (not a post-hoc shift),
    # so the relaxed (single, here) eigenvalue must stay consistent with the
    # state-averaged relaxed energy it comes from.
    pt3_fno_relaxed = RelDSRG_MRPT3(flow_param=0.5, relax_reference="once")(pt2_fno)
    pt3_fno_relaxed.run()
    assert pt3_fno_relaxed.relax_eigvals[0] == approx(pt3_fno_relaxed.E_relaxed_ref)
