import time
import numpy as np
from forte2 import System
from forte2.scf import RHF
from forte2.helpers.comparisons import approx
from forte2.dsrg.df_mp2 import DFRHFMP2, MP2MCASolverLike


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


def test_rhf_df_rmpt2():
    erhf = -76.0614664072629
    emp2 = -76.3710978833093
    xyz = """
    O            0.000000000000     0.000000000000    -0.061664597388
    H            0.000000000000    -0.711620616369     0.489330954643
    H            0.000000000000     0.711620616369     0.489330954643
    """
    system = System(xyz=xyz, basis_set="cc-pVQZ", auxiliary_basis_set="cc-pVQZ-JKFIT")
    scf = RHF(charge=0)(system)
    mp2 = DFRHFMP2(compute_1rdm=True, compute_2rdm=True, compute_cumulants=True)(scf)
    mp2.run()
    np.save("forte2_g1.npy", mp2.gamma1_sf)
    np.save("forte2_g2.npy", mp2.gamma2_sf)
    Emp2 = mp2.E_total

    assert scf.E == approx(erhf)
    assert Emp2 == approx(emp2)


test_rhf_df_rmpt2()


def test_h4_rhf_df_rmpt2():
    xyz = """
  H   -2.7270878    1.9884277    1.0000000
  H   -1.8074993    2.0159410    -1.0000000
  H   -1.8213175    1.0960448    0.0000000
  H   -2.7409060    1.0685315    0.0000000
    """
    system = System(xyz=xyz, basis_set="cc-pVQZ", auxiliary_basis_set="cc-pVQZ-JKFIT")
    scf = RHF(charge=0)(system)
    mp2 = DFRHFMP2(compute_1rdm=True)(scf)
    mp2.run()
    np.save(
        "forte2_gamma1_h4.npy",
        mp2.gamma1_sf,
    )
    # Emp2 = mp2.E_total

    # assert scf.E == approx(erhf)
    # assert Emp2 == approx(emp2)


# test_h2o_rhf_df_rmpt2()
