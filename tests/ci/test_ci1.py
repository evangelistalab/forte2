from numpy import isclose

from forte2 import *


def test_ci1():
    xyz = f"""
    H 0.0 0.0 0.0
    H 0.0 0.0 {0.529177210903 * 2}
    """

    system = System(xyz=xyz, basis="sto-6g", auxiliary_basis="cc-pVTZ-JKFIT")

    # scf = RHF(charge=0, econv=1e-12)(system)
    # scf.run()
    # ci = CI(
    #     orbitals=[0, 1],
    #     state=State(nel=2, multiplicity=1, ms=0.0),
    #     nroot=1,
    # )(scf)
    workflow = [
        RHF(charge=0, econv=1e-12),
        CI(
            orbitals=[0, 1],
            state=State(nel=2, multiplicity=1, ms=0.0),
            nroot=1,
        ),
    ]

    x = system
    for method in workflow:
        x = method(x)
    x.run()

    assert isclose(workflow[0].E, -1.05643120731551)


if __name__ == "__main__":
    test_ci1()
