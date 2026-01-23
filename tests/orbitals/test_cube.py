import os
import glob

from forte2 import System
from forte2.scf import RHF, GHF
from forte2.orbitals import write_orbital_cubes
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

    # generate cube files for all 24 orbitals
    write_orbital_cubes(system, scf.C[0])

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

    xyz = """
    O 0 0 0
    H 0 0 1.1"""

    system = System(
        xyz=xyz,
        basis_set="decon-cc-pVDZ",
        auxiliary_basis_set="cc-pVTZ-JKFIT",
        x2c_type="so",
        snso_type=None,
    )

    scf = GHF(charge=0, j_adapt=True)(system)
    scf.run()

    # generate cube files for first 12 orbitals only
    write_orbital_cubes(system, scf.C[0], indices=list(range(12)))

    # assert if 24 cube files are created using glob
    assert len(glob.glob("*.cube")) == 24
    # clean up the cube files
    for file in glob.glob("*.cube"):
        os.remove(file)


def test_2ccube_ghf():
    """
    Test two-component cube generation (four fields per orbital).
    """
    xyz = """
    O 0 0 0
    H 0 0 1.1"""

    system = System(
        xyz=xyz,
        basis_set="cc-pVDZ",
        auxiliary_basis_set="cc-pVTZ-JKFIT",
        x2c_type="so",
        snso_type=None,
    )

    scf = GHF(charge=0, j_adapt=True)(system)
    scf.run()

    indices = list(range(9))
    write_orbital_cubes(system, scf.C[0], format=("cube", "2ccube"), indices=indices)
    # expect one .2ccube file per requested orbital
    files = sorted(glob.glob("*.2ccube"))

    assert len(files) == 9
    assert os.path.isfile("orbital_0.2ccube")

    # sanity check data length: should be 4 * (nx*ny*nz)
    with open("orbital_0.2ccube", "r") as f:
        lines = f.read().splitlines()

    natoms = int(lines[2].split()[0])
    nx = abs(int(float(lines[3].split()[0])))
    ny = abs(int(float(lines[4].split()[0])))
    nz = abs(int(float(lines[5].split()[0])))
    start_data = 6 + natoms
    tokens = " ".join(lines[start_data:]).split()
    assert len(tokens) == 4 * nx * ny * nz

    # clean up the 2ccube files
    for file in glob.glob("*.2ccube"):
        os.remove(file)
    # clean up the cube files
    for file in glob.glob("*.cube"):
        os.remove(file)
