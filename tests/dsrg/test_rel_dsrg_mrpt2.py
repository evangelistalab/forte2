import numpy as np
import pytest

from forte2 import System, GHF, RelMCOptimizer, AVAS
from forte2.dsrg import RelDSRG_MRPT2
from forte2.helpers.comparisons import approx
from forte2.data.atom_data import EH_TO_WN


@pytest.mark.slow
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
        active_orbitals=12,
        core_orbitals=8,
    )(rhf)
    mc.run()

    assert rhf.E == approx(erhf)
    assert mc.E == approx(emcscf)

    dsrg = RelDSRG_MRPT2(flow_param=0.5, relax_reference="iterate")(mc)
    dsrg.run()

    assert dsrg.relax_energies[0, 0] == approx(-109.238860710091)
    assert dsrg.relax_energies[0, 1] == approx(-109.239311979963)
    assert dsrg.relax_energies[0, 2] == approx(-109.081149196818)

    assert dsrg.relax_energies[1, 0] == approx(-109.238952001270)
    assert dsrg.relax_energies[1, 1] == approx(-109.238952010521)
    assert dsrg.relax_energies[1, 2] == approx(-109.080656324868)

    assert dsrg.relax_energies[2, 0] == approx(-109.238953941310)
    assert dsrg.relax_energies[2, 1] == approx(-109.238953941321)
    assert dsrg.relax_energies[2, 2] == approx(-109.080659175782)


def test_mrpt2_n2_sa_nonrel():
    # This should be strictly identical to test_mcscf_sa_diff_mult given a sufficiently robust MCSCF solver.
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
        nel=14, nroots=4, final_orbital="original", weights=[3, 1, 1, 1]
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
        charge=0,
        die_if_not_converged=False,
        maxiter=50,
    )(system)
    mc = RelMCOptimizer(
        nel=34,
        nroots=9,
        do_diis=False,
        econv=1e-11,
        gconv=1e-10,
        core_orbitals=28,
        active_orbitals=8,
    )(mf)
    dsrg = RelDSRG_MRPT2(flow_param=0.24, relax_reference="once")(mc)
    dsrg.run()
    assert (dsrg.relax_eigvals[5] - dsrg.relax_eigvals[4]) * EH_TO_WN == pytest.approx(
        1934.7036712902677, rel=1e-4
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
        387.52343852668406, rel=1e-4
    )
