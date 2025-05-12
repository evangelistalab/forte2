import forte2
import numpy as np
import scipy as sp
import time
import os
import glob

from forte2.scf import RHF
from forte2.orbitals.cube import Cube


def test_cube():
    escf = -76.021766174866
    # Test the SCF implementation with a simple example
    xyz = """
    O            0.000000000000     0.000000000000    -0.061664597388
    H            0.000000000000    -0.711620616369     0.489330954643
    H            0.000000000000     0.711620616369     0.489330954643
    """

    system = forte2.System(xyz=xyz, basis="cc-pVDZ", auxiliary_basis="cc-pVTZ-JKFIT")

    scf = RHF(charge=0)
    scf.run(system)
    assert np.isclose(scf.E, escf, atol=1e-10), f"SCF energy {scf.E} is not close to expected value {escf}"

    cube = Cube()
    cube.run(system, scf.C)
    # assert if 24 cube files are created using glob
    assert len(glob.glob("*.cube")) == 24
    # clean up the cube files
    for file in glob.glob("*.cube"):
        os.remove(file)

if __name__ == "__main__":
    test_cube()
