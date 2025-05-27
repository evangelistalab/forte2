import forte2
import numpy as np

from forte2.scf import RHF, UHF


def test_hubbard_rhf():
    erhf = -2.944271909999

    system = forte2.system.HubbardModel1D(t=1.0, U=4.0, nsites=10, pbc=True)

    scf = RHF(system, charge=-10)
    scf.guess_type = "hcore"
    scf = scf.run()
    assert np.isclose(
        scf.E, erhf, atol=1e-10
    ), f"SCF energy {scf.E} is not close to expected value {erhf}"


def test_hubbard_uhf():
    euhf = -3.225795887373
    s2uhf = 3.040944953960

    system = forte2.system.HubbardModel1D(t=1.0, U=4.0, nsites=10, pbc=True)

    scf = UHF(system, charge=-10, ms=1)
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
