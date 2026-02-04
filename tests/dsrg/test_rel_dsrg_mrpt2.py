import numpy as np
import pytest

from forte2 import System, GHF, RelMCOptimizer, AVAS
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

    mc = RelMCOptimizer(
        nel=14,
        core_orbitals=8,
        active_orbitals=12,
    )(rhf)
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
    mc = RelMCOptimizer(
        nel=14,
        nroots=4,
        weights=[3, 1, 1, 1],
    )(avas)

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


def test_mrpt2_carbon_rel_sa():
    xyz = """
    C 0 0 0
    """

    system = System(
        xyz=xyz,
        basis_set="decon-cc-pVTZ",
        auxiliary_basis_set="cc-pVQZ-JKFIT",
        x2c_type="so",
        snso_type="row-dependent",
    )
    mf = GHF(charge=0, die_if_not_converged=False)(system)
    mc = RelMCOptimizer(
        nel=6,
        nroots=9,
        active_orbitals=8,
        core_orbitals=2,
        econv=1e-8,
        gconv=1e-6,
        do_diis=False,
    )(mf)
    dsrg = RelDSRG_MRPT2(flow_param=0.24, relax_reference="once")(mc)
    dsrg.run()
    assert dsrg.relax_energies[0, 2] == approx(-37.718966923805)
    assert dsrg.relax_energies[0, 0] == approx(-37.822217257747)
    assert dsrg.relax_energies[0, 1] == approx(-37.822259180404)
    assert dsrg.relax_eigvals.real == approx(
        [
            -37.82240582,
            -37.82233221,
            -37.82233221,
            -37.82233221,
            -37.82218603,
            -37.82218603,
            -37.82218603,
            -37.82218603,
            -37.82218603,
        ]
    )


def test_mrpt2_se_rel_sa_gauss_nuc():
    # Test the zero-field splitting of Se atom with Gaussian nuclear charges
    # Freezing all non-4s/4p orbitals (zero correlated core orbitals)
    xyz = """
    Se 0 0 0
    """

    system = System(
        xyz=xyz,
        basis_set="decon-cc-pVTZ",
        auxiliary_basis_set="cc-pVQZ-JKFIT",
        x2c_type="so",
        snso_type="row-dependent",
        use_gaussian_charges=True,
    )
    mf = GHF(
        charge=-1,
        die_if_not_converged=False,
        maxiter=50,
    )(system)
    mc = RelMCOptimizer(
        nel=34,
        nroots=9,
        do_diis=False,
        core_orbitals=28,
        active_orbitals=8,
    )(mf)
    dsrg = RelDSRG_MRPT2(
        flow_param=0.24,
        relax_reference="once",
        frozen_core_orbitals=28,
    )(mc)
    dsrg.run()
    assert (dsrg.relax_eigvals[5] - dsrg.relax_eigvals[4]) * EH_TO_WN == pytest.approx(
        1916.780243598663, rel=1e-4
    )


def test_mrpt2_s_rel_sa_gauss_nuc():
    xyz = """
    S 0 0 0
    """

    system = System(
        xyz=xyz,
        basis_set="decon-cc-pVTZ",
        auxiliary_basis_set="cc-pVQZ-JKFIT",
        x2c_type="so",
        snso_type="row-dependent",
        use_gaussian_charges=True,
    )
    mf = GHF(
        charge=0,
        die_if_not_converged=False,
        maxiter=50,
    )(system)
    mc = RelMCOptimizer(
        nel=16,
        nroots=9,
        do_diis=False,
        econv=1e-11,
        gconv=1e-10,
        core_orbitals=10,
        active_orbitals=8,
    )(mf)
    dsrg = RelDSRG_MRPT2(flow_param=0.24, relax_reference="once")(mc)
    dsrg.run()
    assert (dsrg.relax_eigvals[5] - dsrg.relax_eigvals[4]) * EH_TO_WN == pytest.approx(
        387.5234521376601, rel=1e-4
    )


def test_mrpt2_sh_with_slow():
    xyz = """
    S 0 0 0
    H 0 0 1.4
    """

    system = System(
        xyz=xyz,
        basis_set="cc-pvtz",
        auxiliary_basis_set="cc-pVTZ-JKFIT",
        x2c_type="so",
        snso_type="row-dependent",
        use_gaussian_charges=True,
    )
    mf = GHF(
        charge=0,
        die_if_not_converged=False,
        maxiter=50,
    )(system)
    mc = RelMCOptimizer(
        nel=17,
        nroots=4,
        core_orbitals=10,
        active_orbitals=10,
    )(mf)
    dsrg = RelDSRG_MRPT2(flow_param=0.5, relax_reference="iterate")(mc)
    dsrg.run()

    mc = RelMCOptimizer(
        nel=17,
        nroots=4,
        core_orbitals=10,
        active_orbitals=10,
    )(mf)
    dsrg_slow = RelDSRG_MRPT2_Slow(flow_param=0.5, relax_reference="iterate")(mc)
    dsrg_slow.run()

    assert dsrg.relax_energies == approx(dsrg_slow.relax_energies)
    assert dsrg.relax_eigvals == approx(dsrg_slow.relax_eigvals)
    assert dsrg.relax_eigvals_history == approx(dsrg_slow.relax_eigvals_history)
    assert dsrg.E_dsrg == approx(dsrg_slow.E_dsrg)
