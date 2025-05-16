import forte2

# import forte2.ints
import numpy as np
import scipy as sp

from forte2.scf import RHF, GHF


def test_sfx2c1e():
    escf = -5192.021044046336
    xyz = """
    Br 0 0 0
    Br 0 0 1.2
    """

    system = forte2.System(
        xyz=xyz, basis="cc-pVQZ", auxiliary_basis="cc-pVQZ-JKFIT", x2c_type="sf"
    )

    scf = RHF(system, charge=0, mult=1)
    scf.econv = 1e-10
    scf.dconv = 1e-8
    scf.run()
    assert np.isclose(
        scf.E, escf, atol=1e-10
    ), f"SCF energy {scf.E} is not close to expected value escf"


def test_sox2c1e():
    eghf = -128.61570430672734
    xyz = "Ne 0 0 0"

    system = forte2.System(
        xyz=xyz, basis="cc-pvdz", auxiliary_basis="def2-universal-jkfit", x2c_type="so"
    )
    scf = GHF(system, charge=0, mult=1)
    scf.econv = 1e-10
    scf.dconv = 1e-8
    scf.run()
    assert np.isclose(
        scf.E, eghf, atol=1e-8, rtol=1e-6
    ), f"SCF energy {scf.E} is not close to expected value eghf"


if __name__ == "__main__":
    test_sfx2c1e()
    test_sox2c1e()
