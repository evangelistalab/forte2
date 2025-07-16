import forte2
from forte2.helpers.comparisons import approx


def test_two_electron_integrals():
    xyz = """
O                     0.000000000000     0.000000000000    -0.061664597388
H                     0.000000000000    -0.711620616370     0.489330954643
H                     0.000000000000     0.711620616370     0.489330954643
"""

    system = forte2.System(xyz=xyz, basis_set="sto-3g")

    V = forte2.ints.coulomb_4c(system.basis)

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


def test_two_electron_integrals_timing():
    xyz = """
O  0.000000000000  0.000000000000 -0.116529200700
H  0.000000000000 -1.344768070168  0.924701488984
H  0.000000000000  1.344768070168  0.924701488984
"""

    system = forte2.System(xyz=xyz, basis_set="cc-pvqz")
    V = forte2.ints.coulomb_4c(system.basis)
