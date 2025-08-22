import numpy as np

import forte2
from forte2 import System, RHF, State
from forte2.helpers.comparisons import approx
from forte2.siso import RelCI


def test_slater_rules_1_complex():
    xyz = """
    H 0.0 0.0 0.0
    H 0.0 0.0 2.0
    """

    system = System(
        xyz=xyz, basis_set="sto-6g", auxiliary_basis_set="cc-pVTZ-JKFIT", unit="bohr"
    )
    scf = RHF(charge=0, econv=1e-12)(system)

    state = State(nel=2, multiplicity=1, ms=0.0)
    ci = RelCI(state=state, active_spinorbitals=[0, 1, 2, 3])(scf)
    ci.run()

    assert ci.evals[0] == approx(-1.096071975854)


def test_slater_rules_2_complex():
    xyz = """
    H 0.0 0.0 0.0
    F 0.0 0.0 2.0
    """

    system = System(
        xyz=xyz, basis_set="cc-pvdz", auxiliary_basis_set="cc-pVTZ-JKFIT", unit="bohr"
    )
    scf = RHF(charge=0, econv=1e-10)(system)
    ci = RelCI(
        state=State(nel=10, multiplicity=1, ms=0.0),
        core_spinorbitals=[0, 1],
        active_spinorbitals=list(range(2, 14)),
    )(scf)
    ci.run()
    assert ci.evals[0] == approx(-100.019788438077)


def test_slater_rules_3_complex():
    # reference energy from pyscf:
    # np.random.seed(12)
    # norb = 12
    # h1 = np.random.random((norb,norb)) + 1j * np.random.random((norb,norb))
    # h2 = np.random.random((norb,norb,norb,norb)) + 1j * np.random.random((norb,norb,norb,norb))
    # print(np.linalg.norm(h1), np.linalg.norm(h2))
    # # Restore permutation symmetry
    # h1 = h1 + h1.T.conj()
    # h2 = h2 + h2.transpose(2, 3, 0, 1)
    # h2 = h2 + h2.transpose(1, 0, 3, 2).conj()
    # h2 = h2 + h2.transpose(3, 2, 1, 0).conj()
    # cisolver = pyscf.fci.fci_dhf_slow.FCI()
    # cisolver.max_cycle = 100
    # cisolver.conv_tol = 1e-12
    # e, fcivec = cisolver.kernel(h1, h2, norb, nelec=8, verbose=5)

    eref = -80.43551643145948
    rng = np.random.default_rng(12)
    norb = 12
    h1 = rng.random((norb, norb)) + 1j * rng.random((norb, norb))
    h2 = rng.random((norb, norb, norb, norb)) + 1j * rng.random(
        (norb, norb, norb, norb)
    )
    # Restore permutation symmetry
    h1 = h1 + h1.T.conj()
    # pyscf uses chemist's notation, forte2 uses physicist's notation
    h2 = h2.swapaxes(1, 2)
    h2 = h2 + h2.transpose(2, 3, 0, 1).conj()
    h2 = h2 + h2.transpose(1, 0, 3, 2)
    h2 = h2 + h2.transpose(3, 2, 1, 0).conj()

    slater_rules = forte2.RelSlaterRules(norb, 0.0, h1, h2)
    dets = forte2.hilbert_space(norb, 8, 0)
    H = np.zeros((len(dets), len(dets)), dtype=complex)
    for i in range(len(dets)):
        # no triangular loop: explicitly construct both i,j and j,i to check Hermiticity
        for j in range(len(dets)):
            H[i, j] = slater_rules.slater_rules(dets[i], dets[j])
    assert np.allclose(H, H.T.conj()), "Slater rules matrix is not Hermitian"
    E = np.linalg.eigvalsh(H)[0]
    assert E == approx(eref), E
