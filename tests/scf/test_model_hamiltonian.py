import forte2
import numpy as np

from forte2.scf import RHF, UHF


def test_hubbard_rhf():
    erhf = -2.944271909999

    nsites = 10
    na = nb = 5
    nel = na + nb
    t = 1.0
    u = 4.0
    hcore = np.zeros((nsites, nsites))
    for i in range(nsites - 1):
        hcore[i, i + 1] = hcore[i + 1, i] = -t
    # periodic boundary conditions
    hcore[0, nsites - 1] = hcore[nsites - 1, 0] = -t
    ovlp = np.eye(nsites)

    eri = np.zeros((nsites,) * 4)
    for i in range(nsites):
        eri[i, i, i, i] = u

    system = forte2.ModelSystem("hubbard", hcore, ovlp, eri)

    scf = RHF(system, charge=-nel, mult=1)
    scf.guess_type = "hcore"
    scf = scf.run()
    assert np.isclose(
        scf.E, erhf, atol=1e-10
    ), f"SCF energy {scf.E} is not close to expected value {erhf}"


def test_hubbard_uhf():
    euhf = -3.225795887373
    s2uhf = 3.040944953960

    nsites = 10
    na = nb = 5
    nel = na + nb
    t = 1.0
    u = 4.0
    hcore = np.zeros((nsites, nsites))
    for i in range(nsites - 1):
        hcore[i, i + 1] = hcore[i + 1, i] = -t
    # periodic boundary conditions
    hcore[0, nsites - 1] = hcore[nsites - 1, 0] = -t
    ovlp = np.eye(nsites)

    eri = np.zeros((nsites,) * 4)
    for i in range(nsites):
        eri[i, i, i, i] = u

    system = forte2.ModelSystem("hubbard", hcore, ovlp, eri)

    scf = UHF(system, charge=-nel, mult=3)
    scf.guess_type = "hcore"
    scf = scf.run()
    assert np.isclose(
        scf.E, euhf, atol=1e-10
    ), f"SCF energy {scf.E} is not close to expected value {euhf}"
    assert np.isclose(
        scf.S2, s2uhf, atol=1e-10
    ), f"SCF S2 {scf.S2} is not close to expected value {s2uhf}"



if __name__ == "__main__":
    test_hubbard_rhf()
    test_hubbard_uhf()
