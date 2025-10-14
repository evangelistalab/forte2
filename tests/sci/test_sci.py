from dataclasses import dataclass, field

from forte2 import System, State
from forte2.scf import RHF
from forte2.sci import SelectedCI
from forte2.helpers.comparisons import approx

import pytest


def test_sci1():
    """Test that SelectedCI reproduces the FCI energy on 4 H atoms in a chain with STO-6G basis set."""

    xyz = f"""
    H 0.0 0.0 0.0
    H 0.0 0.0 1.0
    H 0.0 0.0 2.0
    H 0.0 0.0 3.0
    """

    efci = -2.180967812920

    system = System(xyz=xyz, basis_set="sto-6g", auxiliary_basis_set="cc-pVTZ-JKFIT")

    rhf = RHF(charge=0, econv=1e-14)(system)

    sci = SelectedCI(
        states=State(nel=4, multiplicity=1, ms=0.0),
        active_orbitals=list(range(4)),
        selection_algorithm="hbci",
        var_threshold=1e-12,
        pt2_threshold=0.0,
    )(rhf)
    sci.run()

    assert sci.E[0] == approx(efci)


def test_sci2():
    """Test SelectedCI with a single determinant guess."""

    xyz = f"""
    H 0.0 0.0 0.0
    H 0.0 0.0 1.0
    H 0.0 0.0 2.0
    H 0.0 0.0 3.0
    H 0.0 0.0 4.0
    H 0.0 0.0 5.0
    """

    system = System(xyz=xyz, basis_set="cc-pVDZ", auxiliary_basis_set="cc-pVTZ-JKFIT")

    rhf = RHF(charge=0, econv=1e-10)(system)

    sci = SelectedCI(
        states=State(nel=6, multiplicity=1, ms=0.0),
        active_orbitals=list(range(12)),
        selection_algorithm="hbci",
        var_threshold=1e-4,
        pt2_threshold=0.0,
        guess_occ_window=0,
        guess_vir_window=0,
        nroots=1,
    )(rhf)

    sci.run()

    assert sci.E[0] == pytest.approx(-3.321294103198, abs=1e-9)


def test_sci3():
    """Test SelectedCI on multiple states without spin penalty."""
    xyz = f"""
    H 0.0 0.0 0.0
    H 0.0 0.0 1.0
    H 0.0 0.0 2.0
    H 0.0 0.0 3.0
    H 0.0 0.0 4.0
    H 0.0 0.0 5.0
    """

    system = System(xyz=xyz, basis_set="cc-pVDZ", auxiliary_basis_set="cc-pVTZ-JKFIT")

    rhf = RHF(charge=0, econv=1e-10)(system)

    sci = SelectedCI(
        states=State(nel=6, multiplicity=1, ms=0.0),
        active_orbitals=list(range(12)),
        selection_algorithm="hbci",
        var_threshold=1e-5,
        pt2_threshold=0.0,
        guess_occ_window=2,
        guess_vir_window=2,
        nroots=4,
        do_spin_penalty=False,
    )(rhf)

    sci.run()

    assert sci.E[0] == pytest.approx(-3.3213220620, abs=1e-8)
    assert sci.E[3] == pytest.approx(-3.0403077216, abs=1e-8)


def test_sci4():
    """Test SelectedCI on multiple states with spin penalty."""
    xyz = f"""
    H 0.0 0.0 0.0
    H 0.0 0.0 1.0
    H 0.0 0.0 2.0
    H 0.0 0.0 3.0
    H 0.0 0.0 4.0
    H 0.0 0.0 5.0
    """

    system = System(xyz=xyz, basis_set="cc-pVDZ", auxiliary_basis_set="cc-pVTZ-JKFIT")

    rhf = RHF(charge=0, econv=1e-10)(system)

    sci = SelectedCI(
        states=State(nel=6, multiplicity=1, ms=0.0),
        active_orbitals=list(range(12)),
        selection_algorithm="hbci",
        var_threshold=1e-5,
        pt2_threshold=0.0,
        guess_occ_window=2,
        guess_vir_window=2,
        nroots=2,
        do_spin_penalty=True,
    )(rhf)

    sci.run()

    assert sci.E[0] == pytest.approx(-3.3213219202, abs=1e-8)
    assert sci.E[1] == pytest.approx(-3.0403076453, abs=1e-8)


if __name__ == "__main__":
    test_sci3()
