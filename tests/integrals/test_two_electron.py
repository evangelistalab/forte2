import forte2
from numpy import isclose


def test_two_electron_integrals():
    xyz = """
O  0.000000000000  0.000000000000 -0.116529200700
H  0.000000000000 -1.344768070168  0.924701488984
H  0.000000000000  1.344768070168  0.924701488984
"""

    system = forte2.System(xyz=xyz, basis="sto-3g")

    V = forte2.ints.coulomb_4c(system.basis)

    # Test random integrals against the reference values
    assert isclose(V[4, 5, 0, 4], 0.011203183573992602, atol=1e-10)
    assert isclose(V[6, 3, 0, 1], 0.043937066018280665, atol=1e-10)
    assert isclose(V[6, 1, 0, 1], 0.10316476705668452, atol=1e-10)
    assert isclose(V[1, 3, 1, 5], 0.0347130867494285, atol=1e-10)
    assert isclose(V[3, 2, 2, 5], 0.012205381062317456, atol=1e-10)
    assert isclose(V[0, 4, 4, 5], 0.011203183573992604, atol=1e-10)
    assert isclose(V[0, 4, 5, 4], 0.011203183573992604, atol=1e-10)
    assert isclose(V[2, 5, 2, 1], 0.088959722367807, atol=1e-10)
    assert isclose(V[1, 1, 6, 0], 0.0669877231256632, atol=1e-10)
    assert isclose(V[2, 3, 3, 5], -0.013249429680865767, atol=1e-10)
    assert isclose(V[2, 3, 6, 3], 0.013249429680865767, atol=1e-10)
    assert isclose(V[3, 5, 1, 6], 0.08488308941086266, atol=1e-10)
    assert isclose(V[5, 0, 0, 5], 0.009602115579028458, atol=1e-10)
    assert isclose(V[1, 2, 3, 5], -0.01955131963900513, atol=1e-10)
    assert isclose(V[0, 2, 5, 2], 0.014154072956092055, atol=1e-10)
    assert isclose(V[5, 5, 5, 1], 0.33820230329917145, atol=1e-10)
    assert isclose(V[1, 3, 0, 5], 0.0018694062463856086, atol=1e-10)
    assert isclose(V[0, 0, 0, 0], 4.785065404705505, atol=1e-10)
    assert isclose(V[0, 5, 3, 0], 0.0010038501294464502, atol=1e-10)
    assert isclose(V[3, 6, 1, 1], 0.163693540728617, atol=1e-10)
    assert isclose(V[3, 3, 2, 2], 0.785270203138278, atol=1e-10)
    assert isclose(V[3, 2, 3, 6], 0.013249429680865767, atol=1e-10)
    assert isclose(V[0, 3, 1, 5], 0.004687753192919011, atol=1e-10)
    assert isclose(V[1, 0, 6, 2], 0.056745507084677765, atol=1e-10)
    assert isclose(V[0, 5, 5, 3], 0.012314281459575225, atol=1e-10)


if __name__ == "__main__":
    test_two_electron_integrals()
