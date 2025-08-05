from scipy.linalg import eigh

import forte2
from forte2.helpers.comparisons import approx


def test_one_electron_integrals():
    xyz = "H  0.0 0.0 0.0"

    system = forte2.System(xyz=xyz, basis_set="cc-pvdz")

    S = forte2.ints.overlap(system.basis)
    T = forte2.ints.kinetic(system.basis)
    V = forte2.ints.nuclear(system.basis, system.atoms)

    H = T + V

    # Solve the generalized eigenvalue problem H C = S C ε
    ε, _ = eigh(H, S)
    assert ε[0] == approx(-0.4992784)
