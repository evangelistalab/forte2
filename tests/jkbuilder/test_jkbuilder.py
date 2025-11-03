import numpy as np

from forte2 import System
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
