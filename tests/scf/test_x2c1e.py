import forte2
from forte2.scf import RHF, GHF
import numpy as np
import pytest

# assuming default scf tolerance of 1e-9
approx = lambda x: pytest.approx(x, rel=1e-8, abs=5e-8)


def test_sfx2c1e():
    escf = -5192.021044046336
    xyz = """
    Br 0 0 0
    Br 0 0 1.2
    """

    system = forte2.System(
        xyz=xyz, basis="cc-pVQZ", auxiliary_basis="cc-pVQZ-JKFIT", x2c_type="sf"
    )

    scf = RHF(charge=0)(system)
    scf.run()
    assert scf.E == approx(escf)


def test_sox2c1e():
    eghf = -128.61570430672734
    xyz = "Ne 0 0 0"

    system = forte2.System(
        xyz=xyz, basis="cc-pvdz", auxiliary_basis="def2-universal-jkfit", x2c_type="so"
    )
    scf = GHF(charge=0)(system)
    scf.run()
    assert scf.E == approx(eghf)


if __name__ == "__main__":
    test_sfx2c1e()
    test_sox2c1e()
