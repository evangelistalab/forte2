import forte2

# import forte2.ints
import numpy as np
import scipy as sp
import time

from forte2.scf import RHF, get_hcore_x2c


def test_sfx2c1e():
    xyz = """
    O            0.000000000000     0.000000000000    -0.061664597388
    H            0.000000000000    -0.711620616369     0.489330954643
    H            0.000000000000     0.711620616369     0.489330954643
    """

    system = forte2.System(
        xyz=xyz, basis_name="cc-pVQZ", auxiliary_basis_name="cc-pVQZ-JKFIT"
    )

    scf = RHF(charge=0)
    scf._get_hcore = lambda x: get_hcore_x2c(x, x2c_type="sf")
    scf.run(system)
    assert np.isclose(
        scf.E, -76.110651355917, atol=1e-10
    ), f"SCF energy {scf.E} is not close to expected value -76.110651355917"


def test_sox2c1e():
    xyz = "Ne 0 0 0"

    system = forte2.System(xyz=xyz, basis_name="cc-pvdz")

    hcore = get_hcore_x2c(system, x2c_type="so")
    assert np.isclose(np.linalg.eigvalsh(hcore)[1], -51.969286062806205, atol=1e-10)


if __name__ == "__main__":
    test_sfx2c1e()
    test_sox2c1e()
