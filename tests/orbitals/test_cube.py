import os
import glob

from forte2 import System
from forte2.scf import RHF, GHF
from forte2.orbitals.cube import Cube
from forte2.helpers.comparisons import approx


def test_cube():
    """
    Test cube generation for RHF orbitals.
    """

    escf = -76.02176598836786

    xyz = """
    O            0.000000000000     0.000000000000    -0.061664597388
    H            0.000000000000    -0.711620616369     0.489330954643
    H            0.000000000000     0.711620616369     0.489330954643
    """

    system = System(xyz=xyz, basis_set="cc-pVDZ", auxiliary_basis_set="cc-pVTZ-JKFIT")

    scf = RHF(charge=0)(system)
    scf.run()
    assert scf.E == approx(escf)

    cube = Cube()
    cube.run(system, scf.C[0])
    # assert if 24 cube files are created using glob
    assert len(glob.glob("*.cube")) == 24

    # check that the orbitals are indexed from 0 to 23
    assert os.path.isfile("orbital_00.cube")
    assert os.path.isfile("orbital_23.cube")

    # clean up the cube files
    for file in glob.glob("*.cube"):
        os.remove(file)


def test_cube_ghf():
    """
    Test cube generation for GHF orbitals.
    """

    eref = -75.427367675651
    s2ref = 0.7525463566917241

    xyz = """
    O 0 0 0
    H 0 0 1.1"""

    system = System(
        xyz=xyz,
        basis_set="cc-pVDZ",
        auxiliary_basis_set="cc-pVTZ-JKFIT",
        x2c_type="so",
    )

    scf = GHF(charge=0, j_adapt=True)(system)
    scf.run()
    assert scf.E == approx(eref)
    assert scf.S2 == approx(s2ref)

    cube = Cube()
    cube.run(system, scf.C[0], indices=list(range(12)))
    # assert if 24 cube files are created using glob
    assert len(glob.glob("*.cube")) == 24
    # clean up the cube files
    for file in glob.glob("*.cube"):
        os.remove(file)
