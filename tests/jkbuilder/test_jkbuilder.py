import numpy as np
import pytest

from forte2 import System, jkbuilder
from forte2.jkbuilder.mointegrals import RestrictedMOIntegrals, SpinorbitalIntegrals


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
        x2c_type="so",
        snso_type=None,
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
        x2c_type="so",
        snso_type=None,
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

    fb_otf = jkbuilder.FockBuilderOTF(system, memory_threshold_mb=4.5)
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

    fb_otf = jkbuilder.FockBuilderOTF(system, memory_threshold_mb=4.5)
    J_otf, K_otf = fb_otf.build_JK_generalized(Cact, rdm1)

    assert np.allclose(J_otf, J_ref), np.linalg.norm(J_otf - J_ref)
    assert np.allclose(K_otf, K_ref), np.linalg.norm(K_otf - K_ref)


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
        x2c_type="so",
        snso_type=None,
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

    fb_otf = jkbuilder.FockBuilderOTF(system, memory_threshold_mb=30)
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

    fb_otf = jkbuilder.FockBuilderOTF(system, memory_threshold_mb=15)
    J_otf = fb_otf.build_J(D)[0]
    K_otf = fb_otf.build_K([Cocc])[0]

    assert np.linalg.norm(J_otf - J_ref) < 1e-8
    assert np.linalg.norm(K_otf - K_ref) < 1e-8

    # separately test the combined JK builder, since the algorithm is different for the combined builder
    J_otf, K_otf = fb_otf.build_JK([Cocc])
    assert np.linalg.norm(J_otf[0] - J_ref[0]) < 1e-8
    assert np.linalg.norm(K_otf[0] - K_ref[0]) < 1e-8
