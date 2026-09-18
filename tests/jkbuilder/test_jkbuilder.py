import numpy as np
import pytest

from forte2 import System, X2CParams, integrals, jkbuilder
from forte2.lib.ints import Basis, Shell
from forte2.jkbuilder.mointegrals import RestrictedMOIntegrals, SpinorbitalIntegrals


@pytest.mark.parametrize("df_ortho_rtol", [None, 1.0e-8])
def test_metric_inverted_three_center_reuses_in_core_builder(
    monkeypatch, df_ortho_rtol
):
    """Reuse the stored metric factor and B tensor without integral calls."""
    system = System(
        xyz="H 0 0 0\nH 0 0 1.7",
        basis_set="cc-pvdz",
        auxiliary_basis_set="cc-pvtz-jkfit",
        unit="bohr",
        df_ortho_rtol=df_ortho_rtol,
    )
    M = integrals.coulomb_2c(system)
    J = integrals.coulomb_3c(system)
    Z = np.linalg.solve(M, J.reshape(system.naux, -1)).reshape(J.shape)
    _ = system.fock_builder.B_Pmn

    rng = np.random.default_rng(0)
    D = rng.standard_normal((system.nbf, system.nbf))
    C1 = rng.standard_normal((system.nbf, 3))
    C2 = rng.standard_normal((system.nbf, 3))
    expected_Z = Z
    expected_rho = np.einsum("mn,Pmn->P", D, Z, optimize=True)
    expected_mo_block = np.einsum(
        "mi,Pmn,nj->Pij", C1, Z, C1, optimize=True
    ) + np.einsum("mi,Pmn,nj->Pij", C2, Z, C2, optimize=True)

    def unexpected_integral_call(*args, **kwargs):
        raise AssertionError("The in-core Fock builder must reuse its DF tensors.")

    monkeypatch.setattr(integrals, "coulomb_2c", unexpected_integral_call)
    monkeypatch.setattr(integrals, "coulomb_3c", unexpected_integral_call)
    fock_builder = system.fock_builder
    actual_Z = np.einsum(
        "QP,Qmn->Pmn", fock_builder.Mm12, fock_builder.B_Pmn, optimize=True
    )
    actual_rho = fock_builder.build_metric_inverted_density_contraction(D)
    actual_mo_block = fock_builder.build_metric_inverted_mo_block((C1, C1), (C2, C2))

    assert actual_Z == pytest.approx(expected_Z, abs=1.0e-10)
    assert actual_rho == pytest.approx(expected_rho, abs=1.0e-10)
    assert actual_mo_block == pytest.approx(expected_mo_block, abs=1.0e-10)


@pytest.mark.parametrize("df_ortho_rtol", [None, 1.0e-8])
def test_metric_inverted_otf_matches_in_core_without_full_tensor(df_ortho_rtol):
    """The on-the-fly builder never materializes the full AO tensor, but its
    density-contraction and MO-block methods must match the in-core builder,
    which does form it as a reference."""
    system = System(
        xyz="H 0 0 0\nH 0 0 1.7",
        basis_set="cc-pvdz",
        auxiliary_basis_set="cc-pvtz-jkfit",
        unit="bohr",
        df_ortho_rtol=df_ortho_rtol,
    )
    rng = np.random.default_rng(1)
    D = rng.standard_normal((system.nbf, system.nbf))
    C1 = rng.standard_normal((system.nbf, 3))
    C2 = rng.standard_normal((system.nbf, 3))

    in_core = system.fock_builder
    expected_rho = in_core.build_metric_inverted_density_contraction(D)
    expected_mo_block = in_core.build_metric_inverted_mo_block((C1, C1), (C2, C2))

    otf = jkbuilder.FockBuilderOTF(system, jk_mem_thres_mb=10)
    actual_rho = otf.build_metric_inverted_density_contraction(D)
    actual_mo_block = otf.build_metric_inverted_mo_block((C1, C1), (C2, C2))

    assert actual_rho == pytest.approx(expected_rho, abs=1.0e-10)
    assert actual_mo_block == pytest.approx(expected_mo_block, abs=1.0e-10)


@pytest.mark.parametrize(
    "builder_cls", [jkbuilder.FockBuilder, jkbuilder.FockBuilderOTF]
)
@pytest.mark.parametrize("two_component", [False, True])
def test_generalized_fock_assembly(builder_cls, two_component):
    hcore = np.array([[1.0, 0.1], [0.1, 0.8]])
    Jcore = np.array([[0.4, 0.2], [0.2, 0.3]])
    Kcore = np.array([[0.1, 0.05], [0.05, 0.2]])
    Jact = np.array([[0.3, 0.04], [0.04, 0.2]])
    Kact = np.array([[0.08, 0.03], [0.03, 0.06]])

    class DummySystem:
        def __init__(self):
            self.two_component = two_component

        def ints_hcore(self):
            return hcore

    fock_builder = builder_cls.__new__(builder_cls)
    fock_builder.system = DummySystem()
    fock_builder.build_JK = lambda C: ([Jcore], [Kcore])
    fock_builder.build_JK_generalized = lambda C, g1: (Jact, Kact)

    C_core = np.eye(2)
    C_act = np.eye(2)
    g1 = np.eye(2)

    if two_component:
        core_ref = hcore + Jcore - Kcore
        active_ref = Jact - Kact
    else:
        core_ref = hcore + 2.0 * Jcore - Kcore
        active_ref = Jact - 0.5 * Kact

    core_fock = fock_builder.build_core_fock(C_core)
    active_fock = fock_builder.build_active_fock(C_act, g1)
    generalized_fock = fock_builder.build_generalized_fock(C_core, C_act, g1)

    assert np.allclose(core_fock, core_ref)
    assert np.allclose(active_fock, active_ref)
    assert np.allclose(generalized_fock, core_ref + active_ref)

    shifted_hcore = hcore + np.eye(2)
    shifted_core_fock = fock_builder.build_core_fock(C_core, hcore=shifted_hcore)
    assert np.allclose(shifted_core_fock, core_ref + np.eye(2))


def test_jkbuilder():
    xyz = """
    H 0.0 0.0 0.0
    F 0.0 0.0 2.0
    """

    system = System(
        xyz=xyz,
        basis_set="cc-pvdz",
        auxiliary_basis_set="cc-pVTZ-JKFIT",
        unit="bohr",
    )

    nmo = system.nbf
    C = np.random.rand(nmo, nmo)
    occ = slice(0, 5)
    Cocc = C[:, occ]

    fock_builder = system.fock_builder
    ints = RestrictedMOIntegrals(system, C, orbitals=list(range(C.shape[1])))

    J_ref = np.einsum("piqi->pq", ints.V[:, occ, :, occ], optimize=True)
    K_ref = np.einsum("piiq->pq", ints.V[:, occ, occ, :], optimize=True)

    J, K = fock_builder.build_JK([Cocc])
    J = np.einsum("mp,nq,mn->pq", C.conj(), C, J[0], optimize=True)
    K = np.einsum("mp,nq,mn->pq", C.conj(), C, K[0], optimize=True)

    assert np.allclose(J, J_ref), np.linalg.norm(J - J_ref)
    assert np.allclose(K, K_ref), np.linalg.norm(K - K_ref)


def test_jkbuilder_general():
    xyz = """
    H 0.0 0.0 0.0
    F 0.0 0.0 2.0
    """

    system = System(
        xyz=xyz,
        basis_set="cc-pvdz",
        auxiliary_basis_set="cc-pVTZ-JKFIT",
        unit="bohr",
    )
    nmo = system.nbf
    C = np.random.rand(nmo, nmo)
    actv = slice(1, 7)
    Cact = C[:, 1:7]
    fock_builder = system.fock_builder
    ints = RestrictedMOIntegrals(system, C, orbitals=list(range(C.shape[1])))
    nact = 6
    rdm1 = np.random.rand(nact, nact)
    # make rdm1 hermitian
    rdm1 += rdm1.conj().T
    # make rdm1 positive semi-definite
    rdm1 = rdm1 @ rdm1.T.conj()
    Jact_ref = np.einsum("ptqu,tu->pq", ints.V[:, actv, :, actv], rdm1, optimize=True)
    Kact_ref = np.einsum("ptuq,tu->pq", ints.V[:, actv, actv, :], rdm1, optimize=True)

    Jact, Kact = fock_builder.build_JK_generalized(Cact, rdm1)
    Jact = np.einsum("mp,nq,mn->pq", C.conj(), C, Jact, optimize=True)
    Kact = np.einsum("mp,nq,mn->pq", C.conj(), C, Kact, optimize=True)

    assert np.allclose(Jact, Jact_ref), np.linalg.norm(Jact - Jact_ref)
    assert np.allclose(Kact, Kact_ref), np.linalg.norm(Kact - Kact_ref)


def test_jkbuilder_complex():
    xyz = """
    H 0.0 0.0 0.0
    F 0.0 0.0 2.0
    """

    system = System(
        xyz=xyz,
        basis_set="cc-pvdz",
        auxiliary_basis_set="cc-pVTZ-JKFIT",
        unit="bohr",
        x2c=X2CParams(x2c_type="so", x2c_model="1e"),
    )

    nmo = system.nbf * 2
    C = np.random.rand(nmo, nmo) + 1j * np.random.rand(nmo, nmo)
    occ = slice(0, 10)
    Cocc = C[:, occ]

    fock_builder = system.fock_builder
    ints = SpinorbitalIntegrals(system, C, spinorbitals=list(range(C.shape[1])))

    J_ref = np.einsum("piqi->pq", ints.V[:, occ, :, occ], optimize=True)
    K_ref = np.einsum("piiq->pq", ints.V[:, occ, occ, :], optimize=True)

    J, K = fock_builder.build_JK([Cocc])
    J = np.einsum("mp,nq,mn->pq", C.conj(), C, J[0], optimize=True)
    K = np.einsum("mp,nq,mn->pq", C.conj(), C, K[0], optimize=True)

    assert np.allclose(J, J_ref), np.linalg.norm(J - J_ref)
    assert np.allclose(K, K_ref), np.linalg.norm(K - K_ref)


def test_jkbuilder_general_complex():
    xyz = """
    H 0.0 0.0 0.0
    F 0.0 0.0 2.0
    """

    system = System(
        xyz=xyz,
        basis_set="cc-pvdz",
        auxiliary_basis_set="cc-pVTZ-JKFIT",
        unit="bohr",
        x2c=X2CParams(x2c_type="so", x2c_model="1e"),
    )
    nmo = system.nbf * 2
    C = np.random.rand(nmo, nmo) + 1j * np.random.rand(nmo, nmo)
    actv = slice(2, 14)
    Cact = C[:, 2:14]
    fock_builder = system.fock_builder
    ints = SpinorbitalIntegrals(system, C, spinorbitals=list(range(C.shape[1])))
    nact = 12
    rdm1 = np.random.rand(nact, nact) + 1j * np.random.rand(nact, nact)
    # make rdm1 hermitian
    rdm1 += rdm1.conj().T
    # make rdm1 positive semi-definite
    rdm1 = rdm1 @ rdm1.T.conj()
    Jact_ref = np.einsum("ptqu,tu->pq", ints.V[:, actv, :, actv], rdm1, optimize=True)
    Kact_ref = np.einsum("ptuq,tu->pq", ints.V[:, actv, actv, :], rdm1, optimize=True)

    Jact, Kact = fock_builder.build_JK_generalized(Cact, rdm1)
    Jact = np.einsum("mp,nq,mn->pq", C.conj(), C, Jact, optimize=True)
    Kact = np.einsum("mp,nq,mn->pq", C.conj(), C, Kact, optimize=True)

    assert np.allclose(Jact, Jact_ref), np.linalg.norm(Jact - Jact_ref)
    assert np.allclose(Kact, Kact_ref), np.linalg.norm(Kact - Kact_ref)


def test_jkbuilder_on_the_fly():
    xyz = """
    N 0.0 0.0 0.0
    N 0.0 0.0 2.0
    """

    system = System(
        xyz=xyz,
        basis_set="cc-pvqz",
        auxiliary_basis_set="cc-pvqz-jkfit",
        unit="bohr",
    )

    nmo = system.nbf
    rng = np.random.default_rng(12345)
    C = rng.standard_normal((nmo, nmo))
    occ = slice(0, 50)
    Cocc = C[:, occ]
    D = [Cocc @ Cocc.T.conj()]

    fb = system.fock_builder
    J_ref = fb.build_J(D)
    K_ref = fb.build_K([Cocc])

    fb_otf = jkbuilder.FockBuilderOTF(system, jk_mem_thres_mb=4.5)
    J_otf = fb_otf.build_J(D)[0]
    K_otf = fb_otf.build_K([Cocc])[0]

    assert np.allclose(J_otf, J_ref), np.linalg.norm(J_otf - J_ref)
    assert np.allclose(K_otf, K_ref), np.linalg.norm(K_otf - K_ref)

    # separately test the combined JK builder, since the algorithm is different for the combined builder
    J_otf, K_otf = fb_otf.build_JK([Cocc])
    assert np.allclose(J_otf[0], J_ref[0]), np.linalg.norm(J_otf[0] - J_ref[0])
    assert np.allclose(K_otf[0], K_ref[0]), np.linalg.norm(K_otf[0] - K_ref[0])


def test_jkbuilder_on_the_fly_general():
    xyz = """
    N 0.0 0.0 0.0
    N 0.0 0.0 2.0
    """

    system = System(
        xyz=xyz,
        basis_set="cc-pvqz",
        auxiliary_basis_set="cc-pvqz-jkfit",
        unit="bohr",
    )

    nmo = system.nbf
    rng = np.random.default_rng(12345)
    C = rng.standard_normal((nmo, nmo))
    actv = slice(12, 24)
    Cact = C[:, actv]
    nact = actv.stop - actv.start
    rdm1 = rng.standard_normal((nact, nact))
    rdm1 += rdm1.T.conj()
    rdm1 = rdm1 @ rdm1.T.conj()

    fb = system.fock_builder
    J_ref, K_ref = fb.build_JK_generalized(Cact, rdm1)

    fb_otf = jkbuilder.FockBuilderOTF(system, jk_mem_thres_mb=4.5)
    J_otf, K_otf = fb_otf.build_JK_generalized(Cact, rdm1)

    assert np.allclose(J_otf, J_ref), np.linalg.norm(J_otf - J_ref)
    assert np.allclose(K_otf, K_ref), np.linalg.norm(K_otf - K_ref)


def test_otf_gen_block_matches_incore():
    system = System(
        xyz="N 0.0 0.0 0.0\nN 0.0 0.0 2.0",
        basis_set="cc-pvdz",
        auxiliary_basis_set="cc-pVTZ-JKFIT",
        unit="bohr",
    )
    nmo = system.nbf
    rng = np.random.default_rng(2024)
    C1 = rng.standard_normal((nmo, 2))
    C2 = rng.standard_normal((nmo, 3))
    C3 = rng.standard_normal((nmo, 4))
    C4 = rng.standard_normal((nmo, 5))

    fb = jkbuilder.FockBuilder(system)
    fb_otf = jkbuilder.FockBuilderOTF(system, jk_mem_thres_mb=4.5)

    V_ref = fb.two_electron_integrals_gen_block(C1, C2, C3, C4)
    V_otf = fb_otf.two_electron_integrals_gen_block(C1, C2, C3, C4)
    assert V_otf.shape == V_ref.shape
    assert np.allclose(V_otf, V_ref), np.max(np.abs(V_otf - V_ref))


def test_otf_gen_block_spinor_matches_incore():
    system = System(
        xyz="N 0.0 0.0 0.0\nN 0.0 0.0 2.0",
        basis_set="cc-pvdz",
        auxiliary_basis_set="cc-pVTZ-JKFIT",
        unit="bohr",
    )
    nso = 2 * system.nbf
    rng = np.random.default_rng(99)

    def cmplx(shape):
        return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)

    C1 = cmplx((nso, 2))
    C2 = cmplx((nso, 3))
    C3 = cmplx((nso, 4))
    C4 = cmplx((nso, 5))

    fb = jkbuilder.FockBuilder(system)
    fb_otf = jkbuilder.FockBuilderOTF(system, jk_mem_thres_mb=4.5)

    V_ref = fb.two_electron_integrals_gen_block_spinor(C1, C2, C3, C4)
    V_otf = fb_otf.two_electron_integrals_gen_block_spinor(C1, C2, C3, C4)
    assert V_otf.shape == V_ref.shape
    assert np.allclose(V_otf, V_ref), np.max(np.abs(V_otf - V_ref))


def test_jkbuilder_on_the_fly_complex():
    xyz = """
    N 0.0 0.0 0.0
    N 0.0 0.0 2.0
    """

    system = System(
        xyz=xyz,
        basis_set="cc-pvqz",
        auxiliary_basis_set="cc-pvqz-jkfit",
        unit="bohr",
        x2c=X2CParams(x2c_type="so", x2c_model="1e"),
    )

    nmo = system.nbf * 2
    rng = np.random.default_rng(12345)
    C = rng.standard_normal((nmo, nmo)) + 1j * rng.standard_normal((nmo, nmo))
    occ = slice(0, 100)
    Cocc = C[:, occ]
    D = [Cocc @ Cocc.T.conj()]
    nbf = system.nbf
    D = [D[0][:nbf, :nbf], D[0][nbf:, nbf:]]

    fb = system.fock_builder
    Jaa_ref, Jbb_ref = fb.build_J(D)
    Kaa_ref, Kab_ref, Kba_ref, Kbb_ref = fb.build_K([Cocc])

    fb_otf = jkbuilder.FockBuilderOTF(system, jk_mem_thres_mb=30)
    Jaa_otf, Jbb_otf = fb_otf.build_J(D)
    Kaa_otf, Kab_otf, Kba_otf, Kbb_otf = fb_otf.build_K([Cocc])

    assert np.allclose(Jaa_otf, Jaa_ref), np.linalg.norm(Jaa_otf - Jaa_ref)
    assert np.allclose(Jbb_otf, Jbb_ref), np.linalg.norm(Jbb_otf - Jbb_ref)

    assert np.allclose(Kaa_otf, Kaa_ref), np.linalg.norm(Kaa_otf - Kaa_ref)
    assert np.allclose(Kab_otf, Kab_ref), np.linalg.norm(Kab_otf - Kab_ref)
    assert np.allclose(Kba_otf, Kba_ref), np.linalg.norm(Kba_otf - Kba_ref)
    assert np.allclose(Kbb_otf, Kbb_ref), np.linalg.norm(Kbb_otf - Kbb_ref)

    [J_ref], [K_ref] = fb.build_JK([Cocc])
    [J_otf], [K_otf] = fb_otf.build_JK([Cocc])
    assert np.allclose(J_otf, J_ref), np.linalg.norm(J_otf - J_ref)
    assert np.allclose(K_otf, K_ref), np.linalg.norm(K_otf - K_ref)


def test_jkbuilder_on_the_fly_large():
    xyz = """
    Cl 0.0 0.0 0.0
    Cl 0.0 0.0 2.0
    """

    system = System(
        xyz=xyz,
        basis_set="cc-pvqz",
        auxiliary_basis_set="cc-pvqz-autoaux",
        df_ortho_rtol=1e-8,
    )

    nmo = system.nbf
    rng = np.random.default_rng(12345)
    Cocc = rng.standard_normal((nmo, 24))
    D = [Cocc @ Cocc.T.conj()]

    fb = system.fock_builder
    J_ref = fb.build_J(D)
    K_ref = fb.build_K([Cocc])

    fb_otf = jkbuilder.FockBuilderOTF(system, jk_mem_thres_mb=15)
    J_otf = fb_otf.build_J(D)[0]
    K_otf = fb_otf.build_K([Cocc])[0]

    assert np.linalg.norm(J_otf - J_ref) < 1e-8
    assert np.linalg.norm(K_otf - K_ref) < 1e-8

    # separately test the combined JK builder, since the algorithm is different for the combined builder
    J_otf, K_otf = fb_otf.build_JK([Cocc])
    assert np.linalg.norm(J_otf[0] - J_ref[0]) < 1e-8
    assert np.linalg.norm(K_otf[0] - K_ref[0]) < 1e-8


def test_jkbuilder_lindep_metric():
    xyz = """
    H 0.0 0.0 0.0
    H 0.0 0.0 1.0
    """

    system = System(
        xyz=xyz,
        basis_set="cc-pvdz",
        auxiliary_basis_set="cc-pvtz-jkfit",
        unit="bohr",
    )
    fakeaux = Basis()
    # two identical auxiliary functions, this forces the Coulomb metric to be linearly dependent
    fakeaux.add(Shell(0, [1.0], [1.0], [0.0, 0.0, 0.0]))
    fakeaux.add(Shell(0, [1.0], [1.0], [0.0, 0.0, 0.0]))
    system.auxiliary_basis = fakeaux
    with pytest.raises(ValueError, match="positive definite"):
        system.fock_builder.B_Pmn

    with pytest.raises(ValueError, match="positive definite"):
        system.fock_builder = jkbuilder.FockBuilderOTF(
            system, jk_mem_thres_mb=10, backend="libcint"
        )
