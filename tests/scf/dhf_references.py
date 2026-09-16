# Regenerates the reference energies asserted in `test_dhf.py`.
#
# Run this with a Python that has PySCF installed (it is not a Forte2 dependency and
# this file is not collected by pytest):
#
#     python tests/scf/dhf_references.py
#
# Two details make the comparison exact rather than approximate:
#
# 1. PySCF reads the speed of light from `pyscf.lib.param.LIGHT_SPEED` at call time,
#    so overriding it with Forte2's value removes an otherwise ~1e-7 Eh offset for
#    heavy elements.
# 2. PySCF's own basis library and the Basis Set Exchange disagree in the last digits
#    of some heavy-element contractions (Kr/cc-pVDZ, for one). Loading Forte2's
#    bundled BSE JSON into PySCF makes both codes see identical basis data; without
#    it the Kr reference is off by ~3e-6 Eh.

import json
import pathlib

import numpy as np
from pyscf import gto, lib
from pyscf.data.elements import _std_symbol

lib.param.LIGHT_SPEED = 137.03599917697  # forte2.x2c.x2c.LIGHT_SPEED

from pyscf.scf import dhf  # noqa: E402  (must follow the LIGHT_SPEED override)

BASIS_DIR = pathlib.Path(__file__).resolve().parents[2] / "forte2" / "data" / "basis"
AUX = "def2-universal-jkfit"
H2O = """O 0.0 0.0 -0.061664597388
H 0.0 -0.711620616369 0.489330954643
H 0.0 0.711620616369 0.489330954643"""


def load_basis(name, elements):
    """Convert one of Forte2's bundled BSE JSON basis sets to PySCF's format."""
    data = json.loads((BASIS_DIR / f"{name}.json").read_text())["elements"]
    basis = {}
    for Z in elements:
        shells = []
        for shell in data[str(Z)]["electron_shells"]:
            exponents = [float(e) for e in shell["exponents"]]
            ams = shell["angular_momentum"]
            if len(ams) == 1:
                ams = ams * len(shell["coefficients"])
            for am, coefficients in zip(ams, shell["coefficients"]):
                shells.append(
                    [am] + [[e, float(c)] for e, c in zip(exponents, coefficients)]
                )
        basis[_std_symbol(Z)] = shells
    return basis


def dirac_hf(atom, elements, charge=0, spin=0, nucmod=None):
    basis = load_basis("cc-pvdz", elements)
    auxbasis = load_basis(AUX, elements)
    mol = gto.M(
        atom=atom,
        basis=basis,
        charge=charge,
        spin=spin,
        nucmod=nucmod or {},
        verbose=0,
    )
    mf = dhf.DHF(mol).density_fit(auxbasis=auxbasis)
    mf.conv_tol = 1e-12
    return mf.kernel()


if __name__ == "__main__":
    for label, atom, elements in [
        ("Ne", "Ne 0 0 0", [10]),
        ("Ar", "Ar 0 0 0", [18]),
        ("Kr", "Kr 0 0 0", [36]),
        ("HF", "H 0 0 0; F 0 0 0.91693", [1, 9]),
        ("H2O", H2O, [1, 8]),
    ]:
        print(f"{label:5s} {dirac_hf(atom, elements):20.12f}")

    print(f"{'Ne/G':5s} {dirac_hf('Ne 0 0 0', [10], nucmod={'Ne': 1}):20.12f}")
