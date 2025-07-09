import forte2
import numpy as np
import time

from forte2.jkbuilder import FockBuilder
from forte2.scf import RHF
from forte2.helpers.comparisons import approx


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

    scf = RHF(charge=0)(system)
    scf.run()

    print(f"RHF energy: {scf.E:.10f} [Eh]")

    assert scf.E == approx(energy_scf)

    jkbuilder = FockBuilder(system)
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
