import numpy as np
import pytest

import forte2, forte2.integrals
from forte2.system import BSE_AVAILABLE
from forte2.helpers.comparisons import approx


def test_two_electron_integrals():
    xyz = """
    O 0.000000000000     0.000000000000    -0.061664597388
    H 0.000000000000    -0.711620616370     0.489330954643
    H 0.000000000000     0.711620616370     0.489330954643
    """

    system = forte2.System(xyz=xyz, basis_set="sto-3g")

    V = forte2.integrals.coulomb_4c(system)

    # Test random integrals against the reference values
    assert V[4, 5, 0, 4] == approx(0.011203183573992602)
    assert V[6, 3, 0, 1] == approx(0.043937066018280665)
    assert V[6, 1, 0, 1] == approx(0.10316476705668452)
    assert V[1, 3, 1, 5] == approx(0.0347130867494285)
    assert V[3, 2, 2, 5] == approx(0.012205381062317456)
    assert V[0, 4, 4, 5] == approx(0.011203183573992604)
    assert V[0, 4, 5, 4] == approx(0.011203183573992604)
    assert V[2, 5, 2, 1] == approx(0.088959722367807)
    assert V[1, 1, 6, 0] == approx(0.0669877231256632)
    assert V[2, 3, 3, 5] == approx(-0.013249429680865767)
    assert V[2, 3, 6, 3] == approx(0.013249429680865767)
    assert V[3, 5, 1, 6] == approx(0.08488308941086266)
    assert V[5, 0, 0, 5] == approx(0.009602115579028458)
    assert V[1, 2, 3, 5] == approx(-0.01955131963900513)
    assert V[0, 2, 5, 2] == approx(0.014154072956092055)
    assert V[5, 5, 5, 1] == approx(0.33820230329917145)
    assert V[1, 3, 0, 5] == approx(0.0018694062463856086)
    assert V[0, 0, 0, 0] == approx(4.785065751815717)
    assert V[0, 5, 3, 0] == approx(0.0010038501294464502)
    assert V[3, 6, 1, 1] == approx(0.163693540728617)
    assert V[3, 3, 2, 2] == approx(0.785270203138278)
    assert V[3, 2, 3, 6] == approx(0.013249429680865767)
    assert V[0, 3, 1, 5] == approx(0.004687753192919011)
    assert V[1, 0, 6, 2] == approx(0.056745507084677765)
    assert V[0, 5, 5, 3] == approx(0.012314281459575225)


def test_two_electron_integrals_by_shell_slices():
    xyz = """
    O   0.000000000000     0.000000000000    -0.061664597388
    H   0.000000000000    -0.711620616370     0.489330954643
    H   0.000000000000     0.711620616370     0.489330954643
    """

    system = forte2.System(xyz=xyz, basis_set="sto-3g")

    Vref = forte2.integrals.coulomb_4c(system)

    # generate random slices for each of the four shells
    lo = 0
    hi = system.basis.nshells
    rng = np.random.default_rng(42)

    for _ in range(20):
        shell_slices = []
        slobjs = []
        for _ in range(4):
            start = rng.integers(lo, hi - 1)
            end = rng.integers(start + 1, hi)
            shell_slices.append((start, end))
            slobjs.append(slice(*system.basis.shell_slice_to_basis_slice((start, end))))

        Vslice = forte2.ints.coulomb_4c_by_shell_slices(system.basis, shell_slices)
        assert np.allclose(
            Vslice,
            Vref[slobjs[0], slobjs[1], slobjs[2], slobjs[3]],
            atol=1e-8,
            rtol=0,
        )


def test_two_electron_integral_diagonal():
    xyz = """
    O   0.000000000000     0.000000000000    -0.061664597388
    H   0.000000000000    -0.711620616370     0.489330954643
    H   0.000000000000     0.711620616370     0.489330954643
    """
    system = forte2.System(xyz=xyz, basis_set="sto-3g")

    Vref = forte2.integrals.coulomb_4c(system)
    nbf = system.nbf
    Vref = np.diag(Vref.reshape((nbf**2,) * 2))

    Vdiag = forte2.ints.coulomb_4c_diagonal(system.basis)
    assert np.allclose(Vdiag, Vref, atol=1e-8, rtol=0)


def test_two_electron_integral_row():
    xyz = """
    O   0.000000000000     0.000000000000    -0.061664597388
    H   0.000000000000    -0.711620616370     0.489330954643
    H   0.000000000000     0.711620616370     0.489330954643
    """
    system = forte2.System(xyz=xyz, basis_set="sto-3g")

    Vref = forte2.ints.coulomb_4c(system.basis)
    nbf = system.nbf
    Vref = Vref.reshape((nbf**2,) * 2)
    rng = np.random.default_rng(42)

    for _ in range(20):
        row = rng.integers(0, nbf**2)
        Vref_row = Vref[row, :]
        Vrow = forte2.ints.coulomb_4c_row(system.basis, row)
        assert np.allclose(Vrow, Vref_row, atol=1e-8, rtol=0)


def test_two_electron_integral_cholesky():
    xyz = "Pd 0 0 0"
    system = forte2.System(xyz=xyz, basis_set="sto-3g", minao_basis_set=None)

    Vref = forte2.integrals.coulomb_4c(system)
    nbf = system.nbf

    chol = forte2.integrals.CholeskyIntegrals(system.basis, memory=2000, delta=1e-4)
    chol.compute()
    B = chol.B
    Vchol = np.einsum("Bpq,Brs->pqrs", B, B)
    assert np.linalg.norm(Vchol - Vref) < 1e-3


@pytest.mark.skipif(not BSE_AVAILABLE, reason="BSE module is not available")
def test_3c2e():
    xyz = "Au 0 0 0"
    system = forte2.System(
        xyz=xyz,
        basis_set="ano-rcc",
        minao_basis_set=None,
        auxiliary_basis_set="def2-universal-jkfit",
    )
    ref = forte2.integrals.coulomb_3c(system)
    assert np.linalg.norm(ref) == approx(239.55734891969408)
