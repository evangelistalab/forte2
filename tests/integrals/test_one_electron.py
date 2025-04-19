import forte2


def test_one_electron_integrals():
    xyz = "H  0.0 0.0 0.0"

    system = forte2.System(xyz=xyz, basis="cc-pvdz")

    S = forte2.ints.overlap(system.basis)
    T = forte2.ints.kinetic(system.basis)
    V = forte2.ints.nuclear(system.basis, system.atoms)

    H = T + V

    # Solve the generalized eigenvalue problem H C = S C ε
    from scipy.linalg import eigh
    from numpy import isclose

    ε, _ = eigh(H, S)
    assert isclose(ε[0], -0.4992784, atol=1e-7)


if __name__ == "__main__":
    test_one_electron_integrals()
