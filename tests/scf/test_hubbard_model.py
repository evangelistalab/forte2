from forte2 import HubbardModel, State
from forte2.scf import RHF, ROHF, UHF
from forte2.ci import CI
from forte2.helpers.comparisons import approx


def test_1d_hubbard_rhf():
    erhf = -2.944271909999

    system = HubbardModel(t=1.0, U=4.0, nsites=10, pbc=True)

    scf = RHF(charge=-10)(system)
    scf.guess_type = "hcore"
    scf = scf.run()
    assert scf.E == approx(erhf)


def test_1d_hubbard_rhf_fci():
    erhf = -1.517540966287
    efci = -4.235806999124

    system = HubbardModel(t=1.0, U=4.0, nsites=8, pbc=False)

    scf = RHF(charge=-8)(system)
    scf.guess_type = "hcore"
    ci = CI(
        active_orbitals=list(range(8)),
        states=State(nel=8, multiplicity=1, ms=0.0),
    )(scf)
    ci.run()
    assert scf.E == approx(erhf)
    assert ci.E[0] == approx(efci)


def test_1d_hubbard_rohf():
    erohf = -2.940938175283
    s2rohf = 0.750000000000

    system = HubbardModel(t=1.0, U=2.5, nsites=8, pbc=False)

    scf = ROHF(charge=-9, ms=0.5)(system)
    scf.guess_type = "hcore"
    scf = scf.run()
    assert scf.E == approx(erohf)
    assert scf.S2 == approx(s2rohf)


def test_1d_hubbard_uhf():
    euhf = -3.8703406692066187
    s2uhf = 4.287967519056591

    system = HubbardModel(t=1.0, U=4.0, nsites=10, pbc=False)

    scf = UHF(charge=-10, ms=1.0, do_diis=False)(system)
    scf.guess_type = "hcore"
    scf = scf.run()
    assert scf.E == approx(euhf)
    assert scf.S2 == approx(s2uhf)


def test_2d_hubbard_rhf():
    erhf = -8.944271909999152

    system = HubbardModel(t=1.0, U=4.0, nsites=(10, 2), pbc=True)

    scf = RHF(charge=-20)(system)
    scf.guess_type = "hcore"
    scf = scf.run()
    assert scf.E == approx(erhf)


def test_2d_hubbard_equivalence_to_1d():
    erhf = -2.944271909999

    system = HubbardModel(t=1.0, U=4.0, nsites=(10, 1), pbc=(True, False))

    scf = RHF(charge=-10)(system)
    scf.guess_type = "hcore"
    scf = scf.run()
    assert scf.E == approx(erhf)


def test_2d_hubbard_rhf_fci():
    erhf = -2.472135955000
    efci = -5.012503152630

    system = HubbardModel(t=1.0, U=4.0, nsites=(2, 4), pbc=False)
    scf = RHF(charge=-8)(system)
    scf.guess_type = "hcore"
    ci = CI(
        active_orbitals=list(range(8)),
        states=State(nel=8, multiplicity=1, ms=0.0),
    )(scf)
    ci.run()
    assert scf.E == approx(erhf)
    assert ci.E[0] == approx(efci)


def test_2d_hubbard_rohf():
    erohf = -2.252765000467

    system = HubbardModel(t=1.0, U=2.5, nsites=(3, 2), pbc=False)

    scf = ROHF(charge=-7, ms=0.5, diis_start=2)(system)
    scf.guess_type = "hcore"
    scf = scf.run()
    assert scf.E == approx(erohf)


def test_2d_hubbard_uhf():
    euhf = -3.9293383471710914

    system = HubbardModel(t=1.0, U=4.0, nsites=(5, 2), pbc=False)

    scf = UHF(charge=-10, ms=1.0)(system)
    scf.guess_type = "hcore"
    scf = scf.run()
    assert scf.E == approx(euhf)
