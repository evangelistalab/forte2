import numpy as np
import pytest

from forte2 import System, GHF, MCOptimizer, RelCISolver, AVAS, X2CParams
from forte2.dsrg import RelDSRG_MRPT2, RelDSRG_MRPT2_Slow
from forte2.helpers.comparisons import approx
from forte2.data.atom_data import EH_TO_WN


def test_mrpt2_n2_nonrel():
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

    dsrg = RelDSRG_MRPT2(
        flow_param=0.5,
        relax_reference="iterate",
        frozen_core_orbitals=4,
    )(mc)
    dsrg.run()

    assert dsrg.relax_energies[0, 0] == approx(-109.23447641615361)
    assert dsrg.relax_energies[0, 1] == approx(-109.23492912085933)
    assert dsrg.relax_energies[0, 2] == approx(-109.0811491968237)

    assert dsrg.relax_energies[1, 0] == approx(-109.23456979285112)
    assert dsrg.relax_energies[1, 1] == approx(-109.23456980167653)
    assert dsrg.relax_energies[1, 2] == approx(-109.08065516005186)

    assert dsrg.relax_energies[2, 0] == approx(-109.2345716278556)
    assert dsrg.relax_energies[2, 1] == approx(-109.23457162785648)
    assert dsrg.relax_energies[2, 2] == approx(-109.08065784569052)


def test_mrpt2_n2_sa_nonrel():
    xyz = """
    N 0.0 0.0 0.0
    N 0.0 0.0 1.2
    """

    system = System(xyz=xyz, basis_set="cc-pvdz", auxiliary_basis_set="cc-pVTZ-JKFIT")

    rhf = GHF(charge=0)(system)
    rhf.run()
    rng = np.random.default_rng(1234)
    random_phase = np.diag(np.exp(1j * rng.uniform(-np.pi, np.pi, size=rhf.nmo * 2)))
    rhf.C[0] = rhf.C[0] @ random_phase

    avas = AVAS(
        selection_method="separate",
        num_active_docc=6,
        num_active_uocc=6,
        subspace=["N(2p)"],
        diagonalize=True,
    )(rhf)
    ci_solver = RelCISolver(
        nel=14,
        nroots=4,
        weights=[3, 1, 1, 1],
    )
    mc = MCOptimizer(ci_solver)(avas)

    dsrg = RelDSRG_MRPT2(flow_param=0.5, relax_reference="once")(mc)
    dsrg.run()
    assert dsrg.relax_energies[0, 2] == approx(-108.956246895213)
    assert dsrg.relax_energies[0, 0] == approx(-109.134006255948)
    assert dsrg.relax_energies[0, 1] == approx(-109.135319188567)
    assert dsrg.relax_eigvals.real == approx(
        [
            -109.23881806,
            -109.03182032,
            -109.03182032,
            -109.03182032,
        ]
    )


def test_mrpt2_carbon_rel_sa(tmp_path):
    xyz = """
    C 0 0 0
    """

    system_0 = System(
        xyz=xyz,
        basis_set="decon-cc-pVTZ",
        auxiliary_basis_set="cc-pVQZ-JKFIT",
        x2c=X2CParams(x2c_type="so", x2c_model="1e", snso_type="row-dependent"),
    )

    system_0.save(tmp_path / "carbon_rel_sa")
    system = System.load(tmp_path / "carbon_rel_sa")
    mf = GHF(charge=0, die_if_not_converged=False)(system)
    ci_solver = RelCISolver(
        nel=6,
        nroots=9,
        active_orbitals=8,
        core_orbitals=2,
    )
    mc = MCOptimizer(
        ci_solver,
        e_tol=1e-8,
        g_tol=1e-6,
    )(mf)
    dsrg = RelDSRG_MRPT2(flow_param=0.24, relax_reference="once")(mc)
    dsrg.run()
    assert dsrg.relax_energies[0, 2] == approx(-37.718966923804714)
    assert dsrg.relax_energies[0, 0] == approx(-37.822217257745514)
    assert dsrg.relax_energies[0, 1] == approx(-37.82225918040399)
    assert dsrg.relax_eigvals.real == approx(
        [
            -37.822405824369625,
            -37.822332213565495,
            -37.822332213565424,
            -37.82233221356541,
            -37.82218603171408,
            -37.82218603171406,
            -37.822186031713976,
            -37.82218603171396,
            -37.822186031713905,
        ]
    )


@pytest.mark.slow
def test_mrpt2_se_rel_sa_gauss_nuc_jk_otf():
    # Test the zero-field splitting of Se atom with Gaussian nuclear charges
    # Freezing all non-4s/4p orbitals (zero correlated core orbitals)
    xyz = """
    Se 0 0 0
    """

    from forte2.jkbuilder import FockBuilderOTF

    system = System(
        xyz=xyz,
        basis_set="decon-cc-pVTZ",
        auxiliary_basis_set="cc-pVQZ-JKFIT",
        x2c=X2CParams(x2c_type="so", x2c_model="1e", snso_type="row-dependent"),
        use_gaussian_charges=True,
    )
    system.fock_builder = FockBuilderOTF(system, jk_mem_thres_mb=20, backend="libcint")

    mf = GHF(
        charge=-1,
        die_if_not_converged=False,
        maxiter=50,
    )(system)
    ci_solver = RelCISolver(
        nel=34,
        nroots=9,
        core_orbitals=28,
        active_orbitals=8,
    )
    mc = MCOptimizer(ci_solver)(mf)
    dsrg = RelDSRG_MRPT2(
        flow_param=0.24,
        relax_reference="once",
        frozen_core_orbitals=28,
    )(mc)
    dsrg.run()
    assert (dsrg.relax_eigvals[5] - dsrg.relax_eigvals[4]) * EH_TO_WN == pytest.approx(
        1916.780369353602, rel=1e-4
    )


def test_mrpt2_s_rel_sa_gauss_nuc():
    xyz = """
    S 0 0 0
    """

    system = System(
        xyz=xyz,
        basis_set="decon-cc-pVTZ",
        auxiliary_basis_set="cc-pVQZ-JKFIT",
        x2c=X2CParams(x2c_type="so", x2c_model="1e", snso_type="row-dependent"),
        use_gaussian_charges=True,
    )
    mf = GHF(
        charge=0,
        die_if_not_converged=False,
        maxiter=50,
    )(system)
    ci_solver = RelCISolver(
        nel=16,
        nroots=9,
        core_orbitals=10,
        active_orbitals=8,
    )
    mc = MCOptimizer(
        ci_solver,
        e_tol=1e-11,
        g_tol=1e-10,
    )(mf)
    dsrg = RelDSRG_MRPT2(flow_param=0.24, relax_reference="once")(mc)
    dsrg.run()
    assert (dsrg.relax_eigvals[5] - dsrg.relax_eigvals[4]) * EH_TO_WN == pytest.approx(
        387.5233440732472, rel=1e-4
    )

    # diagonalizing hbar should reproduce most recent relaxed energy
    hbar0 = dsrg.hbar0
    hbar1 = dsrg.hbar1_canon
    hbar2 = dsrg.hbar2_canon
    ci_solver.set_ints(hbar0, hbar1, hbar2)
    ci_solver.run()
    assert (ci_solver.E[5] - ci_solver.E[4]) * EH_TO_WN == pytest.approx(
        387.5233440732472, rel=1e-4
    )


@pytest.mark.slow
def test_mrpt2_sh_with_slow():
    xyz = """
    S 0 0 0
    H 0 0 1.4
    """

    system = System(
        xyz=xyz,
        basis_set="cc-pvtz",
        auxiliary_basis_set="cc-pVTZ-JKFIT",
        x2c=X2CParams(x2c_type="so", x2c_model="1e", snso_type="row-dependent"),
        use_gaussian_charges=True,
    )
    mf = GHF(
        charge=0,
        die_if_not_converged=False,
        maxiter=50,
    )(system)
    ci_solver = RelCISolver(
        nel=17,
        nroots=4,
        core_orbitals=10,
        active_orbitals=10,
    )
    mc = MCOptimizer(ci_solver)(mf)
    dsrg = RelDSRG_MRPT2(flow_param=0.5, relax_reference="iterate")(mc)
    dsrg.run()
    assert np.abs(dsrg.E_dsrg.imag) < 1e-12

    ci_solver = RelCISolver(
        nel=17,
        nroots=4,
        core_orbitals=10,
        active_orbitals=10,
    )
    mc = MCOptimizer(ci_solver)(mf)
    dsrg_slow = RelDSRG_MRPT2_Slow(flow_param=0.5, relax_reference="iterate")(mc)
    dsrg_slow.run()
    assert np.abs(dsrg_slow.E_dsrg.imag) < 1e-12

    ref_relax_energies = np.array(
        [
            [-399.255354806219, -399.255873658049, -399.075511243649],
            [-399.255767878431, -399.255767913076, -399.074949441528],
            [-399.255767038688, -399.255767038695, -399.074948675386],
        ]
    )
    ref_relax_eigvals = np.array(
        [
            -399.256583004776 + 0.0j,
            -399.256583004623 + 0.0j,
            -399.254951072768 + 0.0j,
            -399.254951072612 + 0.0j,
        ]
    )
    ref_relax_eigvals_history = np.array(
        [
            [
                -399.25668945232,
                -399.256689452199,
                -399.255057863899,
                -399.255057863776,
            ],
            [
                -399.25658388134,
                -399.256583881185,
                -399.254951944968,
                -399.254951944811,
            ],
            [
                -399.256583004776,
                -399.256583004623,
                -399.254951072768,
                -399.254951072612,
            ],
        ]
    )
    ref_E = -399.2557670386876

    assert dsrg.relax_energies[:3, :] == approx(ref_relax_energies)
    assert dsrg.relax_eigvals == approx(ref_relax_eigvals)
    assert dsrg.relax_eigvals_history == approx(ref_relax_eigvals_history)
    assert dsrg.E_dsrg == approx(ref_E)

    assert dsrg_slow.relax_energies[:3, :] == approx(ref_relax_energies)
    assert dsrg_slow.relax_eigvals == approx(ref_relax_eigvals)
    assert dsrg_slow.relax_eigvals_history == approx(ref_relax_eigvals_history)
    assert dsrg_slow.E_dsrg == approx(ref_E)

    assert dsrg.relax_energies == approx(dsrg_slow.relax_energies)
    assert dsrg.relax_eigvals == approx(dsrg_slow.relax_eigvals)
    assert dsrg.relax_eigvals_history == approx(dsrg_slow.relax_eigvals_history)
    assert dsrg.E_dsrg == approx(dsrg_slow.E_dsrg)


def test_mrpt2_gamma_vv_fast_vs_slow():
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

    dsrg = RelDSRG_MRPT2(flow_param=0.5, frozen_core_orbitals=4)(mc)
    dsrg.run()
    dsrg_slow = RelDSRG_MRPT2_Slow(flow_param=0.5, frozen_core_orbitals=4)(mc)
    dsrg_slow.run()

    assert dsrg.E_dsrg == approx(dsrg_slow.E_dsrg)

    Gamma_vv = dsrg.compute_unrelaxed_gamma_vv()
    Gamma_vv_slow = dsrg_slow.compute_unrelaxed_gamma_vv()
    assert Gamma_vv == approx(Gamma_vv_slow)
    assert Gamma_vv == approx(Gamma_vv.conj().T)

    evals = np.linalg.eigvalsh(Gamma_vv)
    assert np.all(evals > -1e-10)


def test_mrpt2_fno():
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

    pt2_ref = RelDSRG_MRPT2(flow_param=0.5, frozen_core_orbitals=4)(mc)
    pt2_ref.run()

    # p_o=1.0 retains every virtual orbital, so this is a pure orbital
    # rotation (natural-orbital basis instead of semicanonical) with no actual
    # truncation. The chained (fno-less) second pass must reproduce the
    # untruncated energy to near machine precision -- this is a real
    # correctness check on the rotation/write-back, not an accuracy statement
    # about truncation.
    pt2_full_100 = RelDSRG_MRPT2(flow_param=0.5, frozen_core_orbitals=4, fno_p_o=1.0)(
        mc
    )
    pt2_full_100.run()
    assert pt2_full_100.fno_active
    assert pt2_full_100.mo_space.nvirt == pt2_ref.mo_space.nvirt
    assert pt2_full_100.mo_space.nfrozen_core == pt2_ref.mo_space.nfrozen_core
    # large (full-space) integrals/amplitudes are released once no longer needed
    assert pt2_full_100.ints is None
    assert pt2_full_100.T1 is None

    pt2_fno_100 = RelDSRG_MRPT2(flow_param=0.5)(pt2_full_100)
    pt2_fno_100.run()
    # a plain instance chained onto an FNO pass does not itself set fno_active
    assert not pt2_fno_100.fno_active
    assert pt2_fno_100.E_dsrg == approx(pt2_ref.E_dsrg)

    # a genuinely truncated case: fewer virtuals retained, energy deviates by
    # a bounded, physically sane amount for this small basis.
    pt2_full = RelDSRG_MRPT2(flow_param=0.5, frozen_core_orbitals=4, fno_p_o=0.9)(mc)
    pt2_full.run()
    assert pt2_full.mo_space.nvirt < pt2_ref.mo_space.nvirt

    pt2_fno = RelDSRG_MRPT2(flow_param=0.5)(pt2_full)
    pt2_fno.run()
    assert abs(pt2_fno.E_dsrg - pt2_ref.E_dsrg) < 0.05


def test_rel_mrpt2_all_active():
    xyz = f"""
    H 0.0 0.0 0.0
    H 0.0 0.0 {0.529177210903 * 2}
    """

    system = System(xyz=xyz, basis_set="sto-6g", auxiliary_basis_set="cc-pVTZ-JKFIT")

    rhf = GHF(charge=0, e_tol=1e-12)(system)
    ci_solver = RelCISolver(nel=2, active_orbitals=4)
    mc = MCOptimizer(
        ci_solver,
        maxiter=5,
    )(rhf)
    pt = RelDSRG_MRPT2(flow_param=0.5)(mc)
    pt.run()
    assert pt.E_dsrg == approx(-1.096071975854)
