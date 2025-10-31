import numpy as np

from forte2 import System
from forte2.jkbuilder.mointegrals import RestrictedMOIntegrals, SpinorbitalIntegrals


def test_restricted_mo_integrals():
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

    fock_builder = system.fock_builder
    C = np.random.rand(system.nbf, system.nbf)
    ints = RestrictedMOIntegrals(system, C, orbitals=list(range(C.shape[1])))
    B_mo = np.einsum("Bpq,pi,qj->Bij", fock_builder.B_Qpq, C.conj(), C, optimize=True)
    V_ref = np.einsum("Bij,Bkl->ikjl", B_mo, B_mo, optimize=True)
    assert np.allclose(ints.V, V_ref), np.linalg.norm(ints.V - V_ref)

    ints_antisym = RestrictedMOIntegrals(
        system,
        C,
        orbitals=list(range(C.shape[1])),
        antisymmetrize=True,
    )
    V_ref_antisym = V_ref - V_ref.swapaxes(2, 3)
    assert np.allclose(ints_antisym.V, V_ref_antisym), np.linalg.norm(
        ints_antisym.V - V_ref_antisym
    )


def test_spinorbital_integrals():
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
    system.two_component = True

    nbf = system.nbf * 2
    fock_builder = system.fock_builder
    C = np.random.rand(nbf, nbf) + 1j * np.random.rand(nbf, nbf)
    ints = SpinorbitalIntegrals(system, C, spinorbitals=list(range(C.shape[1])))
    B_spinorbital = np.zeros((fock_builder.B_Qpq.shape[0], nbf, nbf), dtype=complex)
    B_spinorbital[:, : nbf // 2, : nbf // 2] = fock_builder.B_Qpq
    B_spinorbital[:, nbf // 2 :, nbf // 2 :] = fock_builder.B_Qpq
    B_mo = np.einsum("Bpq,pi,qj->Bij", B_spinorbital, C.conj(), C, optimize=True)
    V_ref = np.einsum("Bij,Bkl->ikjl", B_mo, B_mo, optimize=True)
    assert np.allclose(ints.V, V_ref), np.linalg.norm(ints.V - V_ref)

    ints_antisym = SpinorbitalIntegrals(
        system, C, spinorbitals=list(range(C.shape[1])), antisymmetrize=True
    )
    V_ref_antisym = V_ref - V_ref.swapaxes(2, 3)
    assert np.allclose(ints_antisym.V, V_ref_antisym), np.linalg.norm(
        ints_antisym.V - V_ref_antisym
    )
