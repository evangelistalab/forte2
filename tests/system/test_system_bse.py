import pytest
import forte2
import numpy as np


def test_1e_ovlp():
    xyz = """

    O            0.000000000000     0.000000000000    -0.061664597388
    H            0.000000000000    -0.711620616369     0.489330954643
    H            0.000000000000     0.711620616369     0.489330954643
    """

    # Create system in Forte2
    system = forte2.System(xyz=xyz, basis="6-31g*", auxiliary_basis=None)
    # Forte2 integrals
    S = forte2.ints.overlap(system.basis)
    assert np.allclose(np.linalg.norm(S), 5.550074629204314, atol=1.0e-09)
    print("Passed overlap test.")


def test_1e_hcore():
    xyz = """

    O            0.000000000000     0.000000000000    -0.061664597388
    H            0.000000000000    -0.711620616369     0.489330954643
    H            0.000000000000     0.711620616369     0.489330954643
    """

    # Create system in Forte2
    system = forte2.System(xyz=xyz, basis="6-31g*", auxiliary_basis=None)
    # Forte2 integrals
    T = forte2.ints.kinetic(system.basis)
    V = forte2.ints.nuclear(system.basis, system.atoms)
    H = T + V
    assert np.allclose(np.linalg.norm(H), 50.50852825093756, atol=1.0e-09)
    print("Passed Hcore test.")


def test_2e_eri():
    xyz = """

    O            0.000000000000     0.000000000000    -0.061664597388
    H            0.000000000000    -0.711620616369     0.489330954643
    H            0.000000000000     0.711620616369     0.489330954643
    """

    # Create system in Forte2
    system = forte2.System(xyz=xyz, basis="6-31g*", auxiliary_basis=None)
    # Forte2 integrals
    eri = forte2.ints.coulomb_4c(system.basis)
    assert np.allclose(np.linalg.norm(eri.flatten()), 20.21663398506663, atol=1.0e-09)
    print("Passed ERI test.")


if __name__ == "__main__":
    # Run the tests
    test_1e_ovlp()
    test_1e_hcore()
    test_2e_eri()
