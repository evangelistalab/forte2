import os
import glob

from forte2 import System
from forte2.scf import RHF, GHF
from forte2.orbitals import write_orbital_cubes
from forte2.helpers.comparisons import approx


def test_cube_zero_padding_power_of_ten_boundary(tmp_path):
    # Regression: number_of_digits = int(log10(max_index + 1)) + 1 over-pads by
    # one digit when max_index + 1 is an exact power of 10 (max_index = 9). For
    # orbitals 0..9 the highest index is 9, which needs a single digit.
    xyz = """
    O            0.000000000000     0.000000000000    -0.061664597388
    H            0.000000000000    -0.711620616369     0.489330954643
    H            0.000000000000     0.711620616369     0.489330954643
    """
    system = System(xyz=xyz, basis_set="cc-pVDZ", auxiliary_basis_set="cc-pVTZ-JKFIT")
    scf = RHF(charge=0)(system)
    scf.run()

    outdir = tmp_path / "cubes"
    write_orbital_cubes(
        system, scf.C[0], indices=list(range(10)), filepath=str(outdir) + "/"
    )

    files = sorted(os.path.basename(f) for f in glob.glob(str(outdir / "*.cube")))
    assert len(files) == 10
    # Single-digit padding: orbital_0.cube ... orbital_9.cube (not orbital_00).
    assert "orbital_0.cube" in files
    assert "orbital_9.cube" in files
    assert "orbital_00.cube" not in files


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
