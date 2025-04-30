import forte2
import numpy as np
import scipy as sp
import time

from forte2.scf import RHF
from forte2.orbitals.cube import Cube


def test_cube():
    # Test the SCF implementation with a simple example
    escf = -76.0535428512802127
    xyz = """
    O            0.000000000000     0.000000000000    -0.061664597388
    H            0.000000000000    -0.711620616369     0.489330954643
    H            0.000000000000     0.711620616369     0.489330954643
    """

    system = forte2.System(xyz=xyz, basis="cc-pVDZ", auxiliary_basis="cc-pVTZ-JKFIT")

    scf = RHF(charge=0)
    scf.run(system)

    cube = Cube()
    cube.run(system, scf.C)


if __name__ == "__main__":
    test_cube()
