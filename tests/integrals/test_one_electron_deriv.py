from scipy.linalg import eigh
import numpy as np

import forte2
from forte2.helpers.comparisons import approx
from forte2.data.atom_data import ANGSTROM_TO_BOHR


def _set_up_tests(xyz0, xyzp, xyzm, basis_set):
    system0 = forte2.System(xyz=xyz0, basis_set=basis_set)
    systemp = forte2.System(xyz=xyzp, basis_set=basis_set)
    systemm = forte2.System(xyz=xyzm, basis_set=basis_set)
    nbasis = len(system0.basis)
    dm = np.ones((nbasis, nbasis))
    return system0, systemp, systemm, dm


def test_one_electron_deriv_h2_minbas():
    dz = 1e-6

    xyz0 = f"H 0.0 0.0 0.0 \n H 0.0 0.0 1.0"
    xyzp = f"H 0.0 0.0 0.0 \n H 0.0 0.0 {1.0 + dz}"
    xyzm = f"H 0.0 0.0 0.0 \n H 0.0 0.0 {1.0 - dz}"
    system0, systemp, systemm, dm = _set_up_tests(xyz0, xyzp, xyzm, basis_set="sto-3g")

    s_deriv_ref = forte2.ints.overlap_deriv(
        system0.basis, system0.basis, dm, system0.atoms
    )

    sp = forte2.integrals.overlap(systemp)
    sm = forte2.integrals.overlap(systemm)
    s_deriv_num = np.einsum("ij,ij->", sp - sm, dm) / (2 * dz * ANGSTROM_TO_BOHR)

    assert s_deriv_ref[5] == approx(s_deriv_num)


def test_one_electron_deriv_h2o_dz():
    delta = 5e-7

    xyz0 = f"O 0 0 0\n H 0 0 1.0\n H 0 1.0 0"
    xyzp = f"O 0 0 0\n H 0 0 {1.0 + delta}\n H 0 1.0 0"
    xyzm = f"O 0 0 0\n H 0 0 {1.0 - delta}\n H 0 1.0 0"
    system0, systemp, systemm, dm = _set_up_tests(xyz0, xyzp, xyzm, basis_set="cc-pvdz")

    s_deriv_ref = forte2.ints.overlap_deriv(
        system0.basis, system0.basis, dm, system0.atoms
    )

    sp = forte2.integrals.overlap(systemp)
    sm = forte2.integrals.overlap(systemm)
    s_deriv_num = np.einsum("ij,ij->", sp - sm, dm) / (2 * delta * ANGSTROM_TO_BOHR)

    assert s_deriv_ref[5] == approx(s_deriv_num)

    xyz0 = f"O 0 0 0\n H 0 0 1.0\n H 0 1.0 0"
    xyzp = f"O 0 {delta:.10f} 0\n H 0 0 1.0\n H 0 1.0 0"
    xyzm = f"O 0 {-delta:.10f} 0\n H 0 0 1.0\n H 0 1.0 0"
    system0, systemp, systemm, dm = _set_up_tests(xyz0, xyzp, xyzm, basis_set="cc-pvdz")

    s_deriv_ref = forte2.ints.overlap_deriv(
        system0.basis, system0.basis, dm, system0.atoms
    )

    sp = forte2.integrals.overlap(systemp)
    sm = forte2.integrals.overlap(systemm)
    s_deriv_num = np.einsum("ij,ij->", sp - sm, dm) / (2 * delta * ANGSTROM_TO_BOHR)

    assert s_deriv_ref[1] == approx(s_deriv_num)

    xyz0 = f"O 0 0 0\n H 0 0 1.0\n H 0 1.0 0"
    xyzp = f"O 0 0 0\n H 0 0 1.0\n H {delta:.10f} 1.0 0"
    xyzm = f"O 0 0 0\n H 0 0 1.0\n H {-delta:.10f} 1.0 0"
    system0, systemp, systemm, dm = _set_up_tests(xyz0, xyzp, xyzm, basis_set="cc-pvdz")

    s_deriv_ref = forte2.ints.overlap_deriv(
        system0.basis, system0.basis, dm, system0.atoms
    )

    sp = forte2.integrals.overlap(systemp)
    sm = forte2.integrals.overlap(systemm)
    s_deriv_num = np.einsum("ij,ij->", sp - sm, dm) / (2 * delta * ANGSTROM_TO_BOHR)

    assert s_deriv_ref[6] == approx(s_deriv_num)
