import forte2
import numpy as np
import scipy as sp
import time

from forte2.scf import RHF, ROHF, UHF, CUHF, GHF


def test_ghf():
    # Test the RHF implementation with a simple example
    e_ghf = -128.48875618899837
    xyz = """
    Ne 0 0 0
    """

    system = forte2.System(
        xyz=xyz, basis="cc-pvdz", auxiliary_basis="def2-universal-jkfit"
    )

    scf = GHF(charge=0, mult=1, econv=1e-8, dconv=1e-6)
    scf.run(system)
    assert np.isclose(
        scf.E, e_ghf, atol=1e-6
    ), f"RHF energy mismatch: {scf.E} vs {e_ghf}"


if __name__ == "__main__":
    test_ghf()
