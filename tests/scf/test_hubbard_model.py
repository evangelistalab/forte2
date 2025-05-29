import forte2
from forte2.scf import RHF, UHF
import pytest

# assuming default scf tolerance of 1e-9
approx = lambda x: pytest.approx(x, rel=0.0, abs=5e-8)


def test_hubbard_rhf():
    erhf = -2.944271909999

    system = forte2.system.HubbardModel1D(t=1.0, U=4.0, nsites=10, pbc=True)

    scf = RHF(charge=-10)(system)
    scf.guess_type = "hcore"
    scf = scf.run()
    assert scf.E == approx(erhf)


def test_hubbard_uhf():
    euhf = -3.225795894806
    s2uhf = 3.040944954030

    system = forte2.system.HubbardModel1D(t=1.0, U=4.0, nsites=10, pbc=True)

    scf = UHF(charge=-10, ms=1)(system)
    scf.guess_type = "hcore"
    scf = scf.run()
    assert scf.E == approx(euhf)
    assert scf.S2 == approx(s2uhf)


if __name__ == "__main__":
    test_hubbard_rhf()
    test_hubbard_uhf()
