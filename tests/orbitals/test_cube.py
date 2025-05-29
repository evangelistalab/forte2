import forte2
import numpy as np
import scipy as sp
import time
import os
import glob

from forte2.scf import RHF
from forte2.orbitals.cube import Cube

import pytest

# assuming default scf tolerance of 1e-9
approx = lambda x: pytest.approx(x, rel=0.0, abs=5e-8)



def test_cube():
    escf = -76.02176598836786
    # Test the SCF implementation with a simple example
    xyz = """
    O            0.000000000000     0.000000000000    -0.061664597388
    H            0.000000000000    -0.711620616369     0.489330954643
    H            0.000000000000     0.711620616369     0.489330954643
    """

    system = forte2.System(xyz=xyz, basis="cc-pVDZ", auxiliary_basis="cc-pVTZ-JKFIT")

    scf = RHF(charge=0)(system)
    scf.run()
    assert scf.E == approx(escf)

    cube = Cube()
    cube.run(system, scf.C[0])
    # assert if 24 cube files are created using glob
    assert len(glob.glob("*.cube")) == 24
    # clean up the cube files
    for file in glob.glob("*.cube"):
        os.remove(file)

if __name__ == "__main__":
    test_cube()
