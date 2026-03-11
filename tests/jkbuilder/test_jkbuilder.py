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


@pytest.mark.slow
def test_jkbuilder_timing():
    rng = np.random.default_rng(12345)
    xyz = """
C       -4.1589713086     -0.5186115349      0.3160200342                 
C       -5.0814955444     -0.3344236419     -0.6366808615                 
C       -2.9966275704     -1.4613439204      0.1915004691                 
C       -1.6640967733     -0.7130602960      0.2640639983                 
C       -0.4800713386     -1.6753670427      0.1493887365                 
C        0.8528810407     -0.9284549887      0.2203322205                 
C        2.0365732446     -1.8901611656      0.1037025056                 
C        3.3692975111     -1.1426086589      0.1715960082                 
C        4.5532154282     -2.1037518119      0.0526883976                 
C        5.8857253488     -1.3555110496      0.1173781401                 
C        7.0698564462     -2.3161746871     -0.0034615130                 
C        8.4022173266     -1.5674927103      0.0588100193                 
C        9.5855238862     -2.5272655309     -0.0629082490                 
H       -4.2470729164      0.0341380257      1.2493378758                 
H       -5.0397999159     -0.8652596995     -1.5826741610                 
H       -5.9035740665      0.3587510603     -0.4867732131                 
H        7.0015476730     -2.8661654031     -0.9499564940                 
H        7.0262660568     -3.0564793114      0.8046523056                 
H        8.4709469327     -1.0176682416      1.0054025601                 
H        8.4455360517     -0.8266770260     -0.7488994953                 
H        4.4863345167     -2.6535288439     -0.8940254784                 
H        4.5079403596     -2.8441823506      0.8605853766                 
H        5.9531920821     -0.8065264993      1.0645089568                 
H        5.9302681239     -0.6143495378     -0.6899011321                 
H        1.9714915503     -2.4395596861     -0.8433569102                 
H        1.9891621553     -2.6308603523      0.9112288992                 
H        3.4350726520     -0.5941542321      1.1191523092                 
H        3.4158618055     -0.4010422212     -0.6351954144                 
H       -0.5433520659     -2.2242668036     -0.7980851023                 
H       -0.5294951762     -2.4163175504      0.9565669767                 
H        0.9169285536     -0.3803129448      1.1681847868                 
H        0.9015160934     -0.1866829700     -0.5861467857                 
H       -3.0534680087     -2.0235170797     -0.7486685079                 
H       -3.0562233557     -2.1945025711      1.0047550944                 
H       -1.5975674014     -0.1627214725      1.2112101778                 
H       -1.6119429394      0.0294843429     -0.5423397107                 
C       10.9119492057     -1.7866776831     -0.0025410272                 
H        9.5234901330     -3.0769912423     -1.0093017594                 
H        9.5497047995     -3.2671492813      0.7451678969                 
H       10.9933587365     -1.0606567425     -0.8177776658                 
H       11.7442035730     -2.4919182653     -0.0914131049                 
H       11.0197822000     -1.2517330135      0.9464255705                 
"""
    import time

    system = System(
        xyz=xyz,
        basis_set="cc-pvdz",
        auxiliary_basis_set="cc-pvtz-jkfit",
        unit="bohr",
    )
    nbf = system.nbf

    occ = slice(0, 56)
    C = rng.standard_normal((nbf, nbf))
    Cocc = C[:, occ]
    D = Cocc @ Cocc.T.conj()

    fb = system.fock_builder
    fb_otf = jkbuilder.FockBuilderOTF(system, memory_threshold_mb=1000)
    # trigger lazy evaluation
    a = fb.B_Pmn
    b = fb.B_nPm

    t0 = time.time()
    K_ref = fb.build_K([Cocc])[0]
    t1 = time.time()
    print(f"incore K builder took {t1 - t0:.2f} seconds")

    t0 = time.time()
    K_ref = fb._build_K_Pmn([Cocc])[0]
    t1 = time.time()
    print(f"incore K builder took {t1 - t0:.2f} seconds")

    t0 = time.time()
    K_ref = fb._build_K_nPm([Cocc])[0]
    t1 = time.time()
    print(f"incore K builder took {t1 - t0:.2f} seconds")

    t0 = time.time()
    K_otf = fb_otf.build_K([Cocc])[0]
    t1 = time.time()
    print(f"on-the-fly K builder took {t1 - t0:.2f} seconds")

    t0 = time.time()
    J_ref = fb.build_J([D])[0]
    t1 = time.time()
    print(f"incore J builder took {t1 - t0:.2f} seconds")

    t0 = time.time()
    J_otf = fb_otf.build_J([D])[0]
    t1 = time.time()
    print(f"on-the-fly J builder took {t1 - t0:.2f} seconds")

    t0 = time.time()
    res = fb.build_JK([Cocc])
    t1 = time.time()
    print(f"incore JK builder took {t1 - t0:.2f} seconds")

    t0 = time.time()
    res = fb_otf.build_JK([Cocc])
    t1 = time.time()
    print(f"on-the-fly JK builder took {t1 - t0:.2f} seconds")
