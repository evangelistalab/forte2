import forte2
from forte2.scf import RHF, ROHF, UHF
from forte2.helpers.comparisons import approx


def test_hubbard_rhf():
    erhf = -2.944271909999

    system = forte2.system.HubbardModel1D(t=1.0, U=4.0, nsites=10, pbc=True)

    scf = RHF(charge=-10)(system)
    scf.guess_type = "hcore"
    scf = scf.run()
    assert scf.E == approx(erhf)


def test_hubbard_rohf():
    erohf = -2.940938175283
    s2rohf = 0.750000000000

    system = forte2.system.HubbardModel1D(t=1.0, U=2.5, nsites=8, pbc=False)

    scf = ROHF(charge=-9, ms=0.5)(system)
    scf.guess_type = "hcore"
    scf = scf.run()
    assert scf.E == approx(erohf)
    assert scf.S2 == approx(s2rohf)


def test_hubbard_uhf():
    euhf = -3.870340669207
    s2uhf = 4.287968149173

    system = forte2.system.HubbardModel1D(t=1.0, U=4.0, nsites=10, pbc=False)

    scf = UHF(charge=-10, ms=1.0)(system)
    scf.guess_type = "hcore"
    scf = scf.run()
    assert scf.E == approx(euhf)
    assert scf.S2 == approx(s2uhf)
