import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from forte2 import System
from forte2.orbitals import write_molden, write_orbital_cubes

THIS_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = THIS_DIR / "reference_cubes" / "water_ccpvdz_identity"


@pytest.mark.slow
def test_write_water_ccpvdz_identity_basis_artifacts():
    """
    Generate basis-function cubes and a Molden file for visual inspection.

    This test intentionally writes artifacts to
    ``tests/orbitals/reference_cubes/water_ccpvdz_identity`` so the AO ordering
    can be inspected in external viewers.
    """

    xyz = """
    O            0.000000000000     0.000000000000    -0.061664597388
    H            0.000000000000    -0.711620616369     0.489330954643
    H            0.000000000000     0.711620616369     0.489330954643
    """

    system = System(xyz=xyz, basis_set="cc-pVDZ", auxiliary_basis_set="cc-pVTZ-JKFIT")

    coeff = np.eye(system.nbf)
    molden_obj = SimpleNamespace(
        system=system,
        C=[coeff],
        eps=[np.arange(system.nbf, dtype=float)],
        ndocc=0,
        irrep_labels=[f"ao_{i}" for i in range(system.nbf)],
    )

    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    write_orbital_cubes(
        system,
        coeff,
        filepath=OUTPUT_DIR,
        prefix="basis_function",
        indices=list(range(system.nbf)),
    )
    write_molden(molden_obj, OUTPUT_DIR / "water_ccpvdz_identity_basis.molden")

    cube_files = sorted(OUTPUT_DIR.glob("basis_function_*.cube"))
    assert len(cube_files) == system.nbf
    assert (OUTPUT_DIR / "basis_function_00.cube").is_file()
    assert (OUTPUT_DIR / f"basis_function_{system.nbf - 1:02d}.cube").is_file()
    assert (OUTPUT_DIR / "water_ccpvdz_identity_basis.molden").is_file()

test_write_water_ccpvdz_identity_basis_artifacts()
