import forte2
import numpy as np


def test_df():
    xyz = """
C 0.000000000000  0.000000000000  0.000000000000
O 2.500000000000  0.000000000000  0.000000000000
"""

    system = forte2.System(xyz=xyz, basis="cc-pVDZ")
    jkfit_basis = forte2.system.build_basis("cc-pVQZ-jkfit", system.atoms)

    # def2-universal-jkfit
    # Compute the density fitted integrals as
    #   (mn|rs) = (P|mn) (P|Q)^{-1} (Q|rs)

    # Compute the two-electron integrals in the JKfit basis AA = (P|Q)
    AA = forte2.ints.coulomb_2c(jkfit_basis)

    S_AA = forte2.ints.overlap(jkfit_basis)

    # Compute the two-electron integrals in the JKfit/computational basis Acc = (P|mn)
    Acc = forte2.ints.coulomb_3c(jkfit_basis, system.basis, system.basis)

    # Compute the pseudoinverse: (P|Q)^{-1} and check it this raises LinAlgError
    try:
        AA_inv = np.linalg.pinv(AA)
    except np.linalg.LinAlgError:
        print("The SVD computation in the pseudoinverse did not converge.")

    # Compute the four-center two-electron integrals (mn|rs) = (P|mn) (P|Q)^{-1} (Q|rs)
    import time

    start = time.monotonic_ns()
    Vdf = np.einsum("Pmn,PQ,Qrs->mnrs", Acc, AA_inv, Acc, optimize=True)
    end = time.monotonic_ns()
    print(f"[forte2] Einsum timing:                 {int(end - start) // 1000000:d} ms")

    V = forte2.ints.coulomb_4c(system.basis)

    dV = np.linalg.norm(V - Vdf)
    print(f"||V - Vdf|| = {dV:.3e}")

    print(f"{V[0, 0, 0, 0]   = }")
    print(f"{Vdf[0, 0, 0, 0] = }")
