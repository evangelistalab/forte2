import time
import numpy as np
from forte2 import System
from forte2.scf import RHF, ROHF
from forte2.helpers.comparisons import approx
from forte2.mp import RMP2, ROMP2


def test_mp2():
    # reference values from Psi4 using the cc-pVQZ basis set and the cc-pVQZ-JKFIT auxiliary basis set
    energy_scf = -76.0614664072629836
    energy_mp2 = -76.3710978841482984

    xyz = """
    O            0.000000000000     0.000000000000    -0.061664597388
    H            0.000000000000    -0.711620616369     0.489330954643
    H            0.000000000000     0.711620616369     0.489330954643
    """

    system = System(xyz=xyz, basis_set="cc-pVQZ", auxiliary_basis_set="cc-pVQZ-JKFIT")

    scf = RHF(charge=0)(system)
    scf.run()

    print(f"RHF energy: {scf.E:.10f} [Eh]")

    assert scf.E == approx(energy_scf)

    jkbuilder = system.fock_builder
    nocc = scf.na
    nvir = scf.nbf - nocc
    Co = scf.C[0][:, :nocc]
    Cv = scf.C[0][:, nocc:]
    V = jkbuilder.two_electron_integrals_gen_block(Co, Co, Cv, Cv)
    epso = scf.eps[0][:nocc]
    epsv = scf.eps[0][nocc:]

    # Compute the MP2 energy
    start = time.monotonic()
    Emp2 = scf.E
    for i in range(nocc):
        for j in range(nocc):
            for a in range(nvir):
                for b in range(nvir):
                    den = 1.0 / (epso[i] + epso[j] - epsv[a] - epsv[b])
                    Emp2 += V[i, j, a, b] * (2 * V[i, j, a, b] - V[i, j, b, a]) * den
    end = time.monotonic()

    print(f"MP2 energy: {Emp2:.10f} [Eh]")
    print(f"Time taken: {end - start:.4f} seconds")

    assert Emp2 == approx(energy_mp2)


def test_rhf_mp2():
    erhf = -76.0614664072629
    emp2 = -76.3710978833093
    t2 = 0.21136887525049314
    xyz = """
    O            0.000000000000     0.000000000000    -0.061664597388
    H            0.000000000000    -0.711620616369     0.489330954643
    H            0.000000000000     0.711620616369     0.489330954643
    """
    system = System(xyz=xyz, basis_set="cc-pVQZ", auxiliary_basis_set="cc-pVQZ-JKFIT")
    scf = RHF(charge=0)(system)
    mp2 = RMP2(compute_1rdm=True, compute_2rdm=True, compute_cumulants=True)(scf)
    mp2.run()

    assert scf.E == approx(erhf)
    assert mp2.E_total == approx(emp2)
    assert np.linalg.norm(mp2.t2) == approx(t2)


def test_h4_rhf_mp2():
    erhf = -1.998839903161
    emp2 = -2.0915387810627
    t2 = 0.2374525413818509
    xyz = """
  H   -2.7270878    1.9884277    1.0000000
  H   -1.8074993    2.0159410    -1.0000000
  H   -1.8213175    1.0960448    0.0000000
  H   -2.7409060    1.0685315    0.0000000
    """
    system = System(xyz=xyz, basis_set="cc-pVQZ", auxiliary_basis_set="cc-pVQZ-JKFIT")
    scf = RHF(charge=0)(system)
    mp2 = RMP2(compute_1rdm=True)(scf)
    mp2.run()

    assert scf.E == approx(erhf)
    assert mp2.E_total == approx(emp2)
    assert np.linalg.norm(mp2.t2) == approx(t2)


def test_rohf_mp2():
    erhf = -76.0614664072629
    emp2 = -76.3710978833093
    t2 = 0.21136887525048242
    xyz = """
    O            0.000000000000     0.000000000000    -0.061664597388
    H            0.000000000000    -0.711620616369     0.489330954643
    H            0.000000000000     0.711620616369     0.489330954643
    """
    system = System(xyz=xyz, basis_set="cc-pVQZ", auxiliary_basis_set="cc-pVQZ-JKFIT")
    scf = ROHF(charge=0, ms=0.0)(system)
    mp2 = ROMP2(compute_1rdm=True, compute_2rdm=True, compute_cumulants=True)(scf)
    mp2.run()

    assert scf.E == approx(erhf)
    assert mp2.E_total == approx(emp2)
    assert np.linalg.norm(mp2.t2) == approx(t2)


def test_triplet_h2o_rohf_mp2():
    erohf = -75.805109024040
    emp2 = -76.1326710229210
    # t2 unstable in this example

    xyz = """
    O            0.000000000000     0.000000000000    -0.061664597388
    H            0.000000000000    -0.711620616369     0.489330954643
    H            0.000000000000     0.711620616369     0.489330954643
    """
    system = System(xyz=xyz, basis_set="cc-pVQZ", auxiliary_basis_set="cc-pVQZ-JKFIT")

    scf = ROHF(charge=0, ms=1)(system)
    mp2 = ROMP2()(scf)
    mp2.run()

    assert scf.E == approx(erohf)
    assert mp2.E_total == approx(emp2)


def test_singlet_rohf_mp2():
    # Test the ROHF implementation with a simple example (this is equivalent to RHF)
    erohf = -76.061466407194
    emp2 = -76.3710978838554
    t2 = 0.21136887525048242
    xyz = """
    O            0.000000000000     0.000000000000    -0.061664597388
    H            0.000000000000    -0.711620616369     0.489330954643
    H            0.000000000000     0.711620616369     0.489330954643
    """
    system = System(xyz=xyz, basis_set="cc-pVQZ", auxiliary_basis_set="cc-pVQZ-JKFIT")

    scf = ROHF(charge=0, ms=0)(system)
    mp2 = ROMP2()(scf)
    mp2.run()

    assert scf.E == approx(erohf)
    assert mp2.E_total == approx(emp2)
    assert np.linalg.norm(mp2.t2) == approx(t2)
