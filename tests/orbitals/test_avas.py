import pytest

from forte2 import *
from forte2.helpers.comparisons import approx


def test_avas_inputs():
    xyz = f"""
    N 0.0 0.0 0.0
    N 0.0 0.0 1.2
    """

    system = System(xyz=xyz, basis_set="cc-pvdz", auxiliary_basis_set="cc-pVTZ-JKFIT")

    rhf = RHF(charge=0, econv=1e-12)(system)

    # raise if num_active_occ/vir <= 0
    with pytest.raises(Exception):
        avas = AVAS(
            selection_method="separate",
            subspace=["N(2p)"],
        )(rhf)

    # raise if 1-cutoff < evals_threshold
    with pytest.raises(Exception):
        avas = AVAS(
            selection_method="cutoff",
            evals_threshold=0.1,
            cutoff=0.92,
            subspace=["N(2p)"],
        )(rhf)

    # raise if sigma < 0 or sigma > 1
    with pytest.raises(Exception):
        avas = AVAS(
            selection_method="cumulative",
            sigma=1.2,
            subspace=["N(2p)"],
        )(rhf)


def test_avas_1():
    eref_casci = -109.00462206150347
    eref_casci_avas = -109.005019207444
    eref_casci_avas_diagonalize = -109.061384781471

    xyz = f"""
    N 0.0 0.0 0.0
    N 0.0 0.0 1.2
    """

    system = System(xyz=xyz, basis_set="cc-pvdz", auxiliary_basis_set="cc-pVTZ-JKFIT")

    rhf = RHF(charge=0, econv=1e-12)(system)
    casci = CI(
        orbitals=[4, 5, 6, 7, 8, 9],
        core_orbitals=[0, 1, 2, 3],
        state=State(nel=14, multiplicity=1, ms=0.0),
        nroot=1,
    )(rhf)
    casci.run()
    assert casci.E[0] == approx(eref_casci)

    avas = AVAS(
        selection_method="separate",
        subspace=["N(2p)"],
        num_active_occ=3,
        num_active_vir=3,
        diagonalize=False,
    )(rhf)
    casci = CI(
        orbitals=[4, 5, 6, 7, 8, 9],
        core_orbitals=[0, 1, 2, 3],
        state=State(nel=14, multiplicity=1, ms=0.0),
        nroot=1,
    )(avas)
    casci.run()
    assert casci.E[0] == approx(eref_casci_avas)

    avas = AVAS(
        selection_method="separate",
        subspace=["N(2p)"],
        num_active_occ=3,
        num_active_vir=3,
        diagonalize=True,
    )(rhf)
    casci = CI(
        orbitals=[4, 5, 6, 7, 8, 9],
        core_orbitals=[0, 1, 2, 3],
        state=State(nel=14, multiplicity=1, ms=0.0),
        nroot=1,
    )(avas)
    casci.run()
    assert casci.E[0] == approx(eref_casci_avas_diagonalize)
