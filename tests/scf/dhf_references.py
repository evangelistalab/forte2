# Regenerates the reference values asserted in `test_dhf.py`.
#
# Run this with a Python that has PySCF installed (it is not a Forte2 dependency and
# this file is not collected by pytest):
#
#     python tests/scf/dhf_references.py
#
# Three details make the comparison exact rather than approximate:
#
# 1. PySCF reads the speed of light from `pyscf.lib.param.LIGHT_SPEED` at call time,
#    so overriding it with Forte2's value removes an otherwise ~1e-7 Eh offset for
#    heavy elements.
# 2. PySCF's own basis library and the Basis Set Exchange disagree in the last digits
#    of some heavy-element contractions. Loading Forte2's bundled BSE JSON into PySCF
#    makes both codes see identical basis data.
# 3. Both sides decontract the basis, matching Forte2's `decon-` prefix. Restricted
#    kinetic balance generates the small component from the large-component basis, so
#    a contracted basis describes it poorly: the relativistic correction to Ar shifts
#    by 30 mEh on decontraction, and Kr by 2.4 Eh.

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


def build_mol(atom, elements, charge=0, spin=0, nucmod=None):
    mol = gto.M(
        atom=atom,
        basis=load_basis("cc-pvdz", elements),
        charge=charge,
        spin=spin,
        nucmod=nucmod or {},
        verbose=0,
    )
    decontracted = mol.decontract_basis(aggregate=True)
    if isinstance(decontracted, tuple):
        decontracted = decontracted[0]
    return decontracted


def dirac_hf(atom, elements, **kwargs):
    mol = build_mol(atom, elements, **kwargs)
    mf = dhf.DHF(mol).density_fit(auxbasis=load_basis(AUX, elements))
    mf.conv_tol = 1e-12
    energy = mf.kernel()
    return mf, energy


def dhf_ci_from_forte2_integrals(path, norb, nelec, nroots=1):
    """Re-solve a Forte2 active-space CI with PySCF's four-component FCI.

    Export the active-space integrals from Forte2 first, for example::

        ints = mc.ci_solver.sub_solvers[0].ints
        np.savez("carbon.npz", E=ints.E, H=ints.H, V=ints.V)

    Forte2 stores the two-electron integrals in physicist ordering,
    ``V[p, q, r, s] = <pq|rs>``, while `fci_dhf_slow` expects chemist ordering.
    """
    from pyscf.fci import fci_dhf_slow

    data = np.load(path)
    eri = np.ascontiguousarray(data["V"].transpose(0, 2, 1, 3))
    energies, _ = fci_dhf_slow.kernel(
        data["H"],
        eri,
        norb,
        nelec,
        ecore=complex(data["E"]).real,
        nroots=nroots,
        verbose=0,
    )
    return np.asarray(energies).real


def report_negative_branch(mf):
    """Compare the negative-energy count PySCF assumes against the one it produces.

    PySCF occupies from index `nao_2c()` onwards. Its orthogonalization of the
    assembled four-component metric can retain fewer small-component functions than
    that, in which case electronic states fall below the boundary and the occupation
    skips them.
    """
    assumed = mf.mol.nao_2c()
    found = int(np.sum(mf.mo_energy < -(lib.param.LIGHT_SPEED**2)))
    return assumed, found


if __name__ == "__main__":
    for label, atom, elements in [
        ("Ne", "Ne 0 0 0", [10]),
        ("Ar", "Ar 0 0 0", [18]),
        ("HF", "H 0 0 0; F 0 0 0.91693", [1, 9]),
    ]:
        print(f"{label:5s} {dirac_hf(atom, elements)[1]:20.12f}")

    print(f"{'Ne/G':5s} {dirac_hf('Ne 0 0 0', [10], nucmod={'Ne': 1})[1]:20.12f}")

    mf, _ = dirac_hf("Ne 0 0 0", [10])
    occupied = mf.mo_energy[mf.mol.nao_2c() :][:10]
    print("Ne occupied spinor energies:")
    print(np.array2string(occupied, precision=9))
    print(f"Ne 2p splitting: {occupied[6] - occupied[4]:.9f}")

    # The carbon CASSCF roots in `tests/ci/test_dhf_ci.py` were reproduced with
    # `dhf_ci_from_forte2_integrals("carbon.npz", 8, 4, nroots=9)`, which agreed to
    # every printed digit.

    # Water has no usable reference here. Every PySCF initial guess fails to
    # converge and lands ~19.4 Eh high, so `test_dhf.py` asserts Forte2's own value.
    mf, energy = dirac_hf(H2O, [1, 8])
    assumed, found = report_negative_branch(mf)
    print(
        f"{'H2O':5s} {energy:20.12f}  converged={mf.converged}  "
        f"negative-energy states assumed={assumed} found={found}"
    )
