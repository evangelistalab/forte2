import numpy as np
import pytest

from forte2 import System, GHF, MCOptimizer, RelCISolver, X2CParams, AVAS
from forte2.dsrg import RelDSRG_MRPT3
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


def test_mrpt3_ch_snso_kramers():
    """CH has an odd number of electrons, so its roots must be Kramers degenerate.

    The reference is a 2-Pi state, so the four roots form two pairs that are exactly
    degenerate by time-reversal symmetry. That makes this a zero-tolerance check
    needing no external reference. It only has power when the state-averaged 1-RDM is
    genuinely complex in the semicanonical basis, which is why the assertion on
    max|Im(gamma1)| is here: an atomic reference has a real 1-RDM and would let a
    convention error in the active mean field pass unnoticed.
    """
    escf = -38.286254865078
    emcscf = -38.320115307064
    edsrg = -38.433774865372
    erelaxed = -38.434549231635
    soc_wn = 27.424588

    xyz = """
    C 0.0 0.0 0.0
    H 0.0 0.0 1.1199
    """

    system = System(
        xyz=xyz,
        basis_set="decon-cc-pVDZ",
        auxiliary_basis_set="cc-pVTZ-JKFIT",
        minao_basis_set="ano-r0",
        x2c=X2CParams(x2c_type="so", snso_type="row-dependent"),
        use_gaussian_charges=True,
    )
    mf = GHF(charge=0, die_if_not_converged=False)(system)
    avas = AVAS(
        subspace=["H(1s)", "C(2s)", "C(2p)"],
        selection_method="separate",
        num_active_docc=5,
        num_active_uocc=5,
    )(mf)
    mc = MCOptimizer(
        RelCISolver(nel=7, nroots=4), e_tol=1e-9, g_tol=1e-7, die_if_not_converged=False
    )(avas)
    mc.run()

    assert mf.E == approx(escf)
    assert mc.E.real == approx(emcscf)
    # Without a complex 1-RDM this test cannot see a conjugation error.
    assert np.abs(np.imag(mc.make_average_rdm(1))).max() > 1e-4

    dsrg = RelDSRG_MRPT3(flow_param=0.5, relax_reference="once")(mc)
    dsrg.run()

    assert dsrg.E_dsrg.real == approx(edsrg)
    assert dsrg.E_relaxed_ref.real == approx(erelaxed)

    levels = np.sort(np.asarray(dsrg.relax_eigvals).ravel().real)
    levels = (levels - levels[0]) * EH_TO_WN
    kramers = [levels[1] - levels[0], levels[3] - levels[2]]
    # Conjugating gamma1 in the active mean field splits these by ~16 cm^-1.
    assert max(kramers) < 1e-2, f"Kramers pairs split by {max(kramers):.4f} cm^-1"
    assert levels[2] == pytest.approx(soc_wn, abs=1e-2)
