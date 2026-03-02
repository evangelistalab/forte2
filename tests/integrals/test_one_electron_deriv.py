import numpy as np

import forte2
from forte2.helpers.comparisons import approx_abs
from forte2.data.atom_data import ANGSTROM_TO_BOHR

rng = np.random.default_rng(seed=42)


def _set_up_tests(xyz0, xyzp, xyzm, basis_set):
    system0 = forte2.System(xyz=xyz0, basis_set=basis_set)
    systemp = forte2.System(xyz=xyzp, basis_set=basis_set)
    systemm = forte2.System(xyz=xyzm, basis_set=basis_set)
    nbasis = len(system0.basis)
    # set up a random symmetric positive definite density matrix
    dm = rng.standard_normal(size=(nbasis, nbasis))
    dm = dm @ dm.T
    return system0, systemp, systemm, dm


def test_overlap_deriv_h2_minbas():
    dz = 1e-5

    xyz0 = f"H 0.0 0.0 0.0 \n H 0.0 0.0 1.0"
    xyzp = f"H 0.0 0.0 0.0 \n H 0.0 0.0 {1.0 + dz}"
    xyzm = f"H 0.0 0.0 0.0 \n H 0.0 0.0 {1.0 - dz}"
    system0, systemp, systemm, dm = _set_up_tests(xyz0, xyzp, xyzm, basis_set="sto-3g")

    s_deriv_analytical = forte2.ints.overlap_deriv(
        system0.basis, system0.basis, dm, system0.atoms
    )

    sp = forte2.integrals.overlap(systemp)
    sm = forte2.integrals.overlap(systemm)
    s_deriv_num = np.einsum("ij,ij->", sp - sm, dm) / (2 * dz * ANGSTROM_TO_BOHR)

    assert s_deriv_analytical[5] == approx_abs(s_deriv_num, atol=1e-7)


def test_kinetic_deriv_h2_minbas():
    dz = 1e-5

    xyz0 = f"H 0.0 0.0 0.0 \n H 0.0 0.0 1.0"
    xyzp = f"H 0.0 0.0 0.0 \n H 0.0 0.0 {1.0 + dz}"
    xyzm = f"H 0.0 0.0 0.0 \n H 0.0 0.0 {1.0 - dz}"
    system0, systemp, systemm, dm = _set_up_tests(xyz0, xyzp, xyzm, basis_set="sto-3g")

    s_deriv_analytical = forte2.ints.kinetic_deriv(
        system0.basis, system0.basis, dm, system0.atoms
    )

    sp = forte2.integrals.kinetic(systemp)
    sm = forte2.integrals.kinetic(systemm)
    s_deriv_num = np.einsum("ij,ij->", sp - sm, dm) / (2 * dz * ANGSTROM_TO_BOHR)

    assert s_deriv_analytical[5] == approx_abs(s_deriv_num, atol=1e-7)


def test_nuclear_deriv_h2_minbas():
    dz = 1e-5

    xyz0 = f"H 0.0 0.0 0.0 \n H 0.0 0.0 1.0"
    xyzp = f"H 0.0 0.0 0.0 \n H 0.0 0.0 {1.0 + dz}"
    xyzm = f"H 0.0 0.0 0.0 \n H 0.0 0.0 {1.0 - dz}"
    system0, systemp, systemm, dm = _set_up_tests(xyz0, xyzp, xyzm, basis_set="sto-3g")

    s_deriv_analytical = forte2.ints.nuclear_deriv(
        system0.basis, system0.basis, dm, system0.atoms
    )

    sp = forte2.integrals.nuclear(systemp)
    sm = forte2.integrals.nuclear(systemm)
    s_deriv_num = np.einsum("ij,ij->", sp - sm, dm) / (2 * dz * ANGSTROM_TO_BOHR)
    assert s_deriv_analytical[5] == approx_abs(s_deriv_num, atol=1e-7)


def test_overlap_deriv_h2o_dz():
    delta = 1e-5

    xyz0 = f"O 0 0 0\n H 0 0 1.0\n H 0 1.0 0"
    xyzp = f"O 0 0 0\n H 0 0 {1.0 + delta}\n H 0 1.0 0"
    xyzm = f"O 0 0 0\n H 0 0 {1.0 - delta}\n H 0 1.0 0"
    system0, systemp, systemm, dm = _set_up_tests(xyz0, xyzp, xyzm, basis_set="cc-pvdz")

    s_deriv_analytical = forte2.ints.overlap_deriv(
        system0.basis, system0.basis, dm, system0.atoms
    )

    sp = forte2.integrals.overlap(systemp)
    sm = forte2.integrals.overlap(systemm)
    s_deriv_num = np.einsum("ij,ij->", sp - sm, dm) / (2 * delta * ANGSTROM_TO_BOHR)

    assert s_deriv_analytical[5] == approx_abs(s_deriv_num, atol=1e-7)

    xyz0 = f"O 0 0 0\n H 0 0 1.0\n H 0 1.0 0"
    xyzp = f"O 0 {delta:.10f} 0\n H 0 0 1.0\n H 0 1.0 0"
    xyzm = f"O 0 {-delta:.10f} 0\n H 0 0 1.0\n H 0 1.0 0"
    system0, systemp, systemm, dm = _set_up_tests(xyz0, xyzp, xyzm, basis_set="cc-pvdz")

    s_deriv_analytical = forte2.ints.overlap_deriv(
        system0.basis, system0.basis, dm, system0.atoms
    )

    sp = forte2.integrals.overlap(systemp)
    sm = forte2.integrals.overlap(systemm)
    s_deriv_num = np.einsum("ij,ij->", sp - sm, dm) / (2 * delta * ANGSTROM_TO_BOHR)

    assert s_deriv_analytical[1] == approx_abs(s_deriv_num, atol=1e-7)

    xyz0 = f"O 0 0 0\n H 0 0 1.0\n H 0 1.0 0"
    xyzp = f"O 0 0 0\n H 0 0 1.0\n H {delta:.10f} 1.0 0"
    xyzm = f"O 0 0 0\n H 0 0 1.0\n H {-delta:.10f} 1.0 0"
    system0, systemp, systemm, dm = _set_up_tests(xyz0, xyzp, xyzm, basis_set="cc-pvdz")

    s_deriv_analytical = forte2.ints.overlap_deriv(
        system0.basis, system0.basis, dm, system0.atoms
    )

    sp = forte2.integrals.overlap(systemp)
    sm = forte2.integrals.overlap(systemm)
    s_deriv_num = np.einsum("ij,ij->", sp - sm, dm) / (2 * delta * ANGSTROM_TO_BOHR)

    assert s_deriv_analytical[6] == approx_abs(s_deriv_num, atol=1e-7)


def test_kinetic_deriv_h2o_dz():
    delta = 1e-5

    xyz0 = f"O 0 0 0\n H 0 0 1.0\n H 0 1.0 0"
    xyzp = f"O 0 0 0\n H 0 0 {1.0 + delta}\n H 0 1.0 0"
    xyzm = f"O 0 0 0\n H 0 0 {1.0 - delta}\n H 0 1.0 0"
    system0, systemp, systemm, dm = _set_up_tests(xyz0, xyzp, xyzm, basis_set="cc-pvdz")

    s_deriv_analytical = forte2.ints.kinetic_deriv(
        system0.basis, system0.basis, dm, system0.atoms
    )

    sp = forte2.integrals.kinetic(systemp)
    sm = forte2.integrals.kinetic(systemm)
    s_deriv_num = np.einsum("ij,ij->", sp - sm, dm) / (2 * delta * ANGSTROM_TO_BOHR)

    assert s_deriv_analytical[5] == approx_abs(s_deriv_num, atol=1e-7)

    xyz0 = f"O 0 0 0\n H 0 0 1.0\n H 0 1.0 0"
    xyzp = f"O 0 {delta:.10f} 0\n H 0 0 1.0\n H 0 1.0 0"
    xyzm = f"O 0 {-delta:.10f} 0\n H 0 0 1.0\n H 0 1.0 0"
    system0, systemp, systemm, dm = _set_up_tests(xyz0, xyzp, xyzm, basis_set="cc-pvdz")

    s_deriv_analytical = forte2.ints.kinetic_deriv(
        system0.basis, system0.basis, dm, system0.atoms
    )

    sp = forte2.integrals.kinetic(systemp)
    sm = forte2.integrals.kinetic(systemm)
    s_deriv_num = np.einsum("ij,ij->", sp - sm, dm) / (2 * delta * ANGSTROM_TO_BOHR)

    assert s_deriv_analytical[1] == approx_abs(s_deriv_num, atol=1e-7)

    xyz0 = f"O 0 0 0\n H 0 0 1.0\n H 0 1.0 0"
    xyzp = f"O 0 0 0\n H 0 0 1.0\n H {delta:.10f} 1.0 0"
    xyzm = f"O 0 0 0\n H 0 0 1.0\n H {-delta:.10f} 1.0 0"
    system0, systemp, systemm, dm = _set_up_tests(xyz0, xyzp, xyzm, basis_set="cc-pvdz")

    s_deriv_analytical = forte2.ints.kinetic_deriv(
        system0.basis, system0.basis, dm, system0.atoms
    )

    sp = forte2.integrals.kinetic(systemp)
    sm = forte2.integrals.kinetic(systemm)
    s_deriv_num = np.einsum("ij,ij->", sp - sm, dm) / (2 * delta * ANGSTROM_TO_BOHR)

    assert s_deriv_analytical[6] == approx_abs(s_deriv_num, atol=1e-7)


def test_nuclear_deriv_h2o_dz():
    delta = 1e-5

    xyz0 = f"O 0 0 0\n H 0 0 1.0\n H 0 1.0 0"
    xyzp = f"O 0 0 0\n H 0 0 {1.0 + delta:.10f}\n H 0 1.0 0"
    xyzm = f"O 0 0 0\n H 0 0 {1.0 - delta:.10f}\n H 0 1.0 0"
    system0, systemp, systemm, dm = _set_up_tests(xyz0, xyzp, xyzm, basis_set="cc-pvdz")

    s_deriv_analytical = forte2.ints.nuclear_deriv(
        system0.basis, system0.basis, dm, system0.atoms
    )

    sp = forte2.integrals.nuclear(systemp)
    sm = forte2.integrals.nuclear(systemm)
    s_deriv_num = np.einsum("ij,ij->", sp - sm, dm) / (2 * delta * ANGSTROM_TO_BOHR)
    assert s_deriv_analytical[5] == approx_abs(s_deriv_num, atol=1e-7)

    xyz0 = f"O 0 0 0\n H 0 0 1.0\n H 0 1.0 0"
    xyzp = f"O 0 {delta:.10f} 0\n H 0 0 1.0\n H 0 1.0 0"
    xyzm = f"O 0 {-delta:.10f} 0\n H 0 0 1.0\n H 0 1.0 0"
    system0, systemp, systemm, dm = _set_up_tests(xyz0, xyzp, xyzm, basis_set="cc-pvdz")

    s_deriv_analytical = forte2.ints.nuclear_deriv(
        system0.basis, system0.basis, dm, system0.atoms
    )

    sp = forte2.integrals.nuclear(systemp)
    sm = forte2.integrals.nuclear(systemm)
    s_deriv_num = np.einsum("ij,ij->", sp - sm, dm) / (2 * delta * ANGSTROM_TO_BOHR)

    assert s_deriv_analytical[1] == approx_abs(s_deriv_num, atol=1e-7)

    xyz0 = f"O 0 0 0\n H 0 0 1.0\n H 0 1.0 0"
    xyzp = f"O 0 0 0\n H 0 0 1.0\n H {delta:.10f} 1.0 0"
    xyzm = f"O 0 0 0\n H 0 0 1.0\n H {-delta:.10f} 1.0 0"
    system0, systemp, systemm, dm = _set_up_tests(xyz0, xyzp, xyzm, basis_set="cc-pvdz")

    s_deriv_analytical = forte2.ints.nuclear_deriv(
        system0.basis, system0.basis, dm, system0.atoms
    )

    sp = forte2.integrals.nuclear(systemp)
    sm = forte2.integrals.nuclear(systemm)
    s_deriv_num = np.einsum("ij,ij->", sp - sm, dm) / (2 * delta * ANGSTROM_TO_BOHR)

    assert s_deriv_analytical[6] == approx_abs(s_deriv_num, atol=1e-7)
