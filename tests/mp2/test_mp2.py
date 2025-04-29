import forte2
import numpy as np
import scipy as sp
import time

from forte2.jkbuilder.jkbuilder import DFFockBuilder
from forte2.scf import RHF


def test_mp2():
    # reference values from Psi4 using the cc-pVQZ basis set and the cc-pVQZ-JKFIT auxiliary basis set
    energy_scf = -76.0614664072629836
    energy_mp2 = -76.3710978841482984

    xyz = """
    O            0.000000000000     0.000000000000    -0.061664597388
    H            0.000000000000    -0.711620616369     0.489330954643
    H            0.000000000000     0.711620616369     0.489330954643
    """

    system = forte2.System(xyz=xyz, basis="cc-pVQZ", auxiliary_basis="cc-pVQZ-JKFIT")

    scf = RHF(charge=0)
    scf.run(system)

    print(f"RHF energy: {scf.E:.10f} [Eh]")

    assert np.isclose(
        scf.E, energy_scf, atol=1e-10
    ), f"SCF energy {scf.E} is not close to expected value {energy_scf}"

    jkbuilder = DFFockBuilder(system)
    nocc = scf.na
    nvir = scf.nbasis - nocc
    nbasis = scf.nbasis
    Co = scf.C[:, :nocc]
    Cv = scf.C[:, nocc:]
    V = jkbuilder.two_electron_integrals_gen_block(Co, Co, Cv, Cv)
    epso = scf.eps[:nocc]
    epsv = scf.eps[nocc:]

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

    assert np.isclose(
        Emp2, energy_mp2, atol=1e-10
    ), f"MP2 energy {Emp2} is not close to expected value {energy_mp2}"


if __name__ == "__main__":
    test_mp2()
