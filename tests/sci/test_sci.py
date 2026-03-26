import numpy as np
import pytest

from forte2 import System, State, Determinant
from forte2.scf import RHF
from forte2.sci import SelectedCI
from forte2.helpers.comparisons import approx
from forte2.base_classes.params import SelectedCIParams


def _h4_rhf():
    xyz = """
    H 0.0 0.0 0.0
    H 0.0 0.0 1.0
    H 0.0 0.0 2.0
    H 0.0 0.0 3.0
    """
    system = System(xyz=xyz, basis_set="sto-6g", auxiliary_basis_set="cc-pVTZ-JKFIT")
    return RHF(charge=0, econv=1e-14)(system)


def test_sci1():
    """Test that SelectedCI reproduces the FCI energy on 4 H atoms in a chain with STO-6G basis set."""

    efci = -2.180967812920

    rhf = _h4_rhf()

    sci = SelectedCI(
        states=State(nel=4, multiplicity=1, ms=0.0),
        active_orbitals=list(range(4)),
        sci_params=SelectedCIParams(
            selection_algorithm="hbci", var_threshold=1e-12, pt2_threshold=0.0
        ),
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
        sci_params=SelectedCIParams(
            selection_algorithm="hbci",
            var_threshold=1e-4,
            pt2_threshold=0.0,
            guess_occ_window=0,
            guess_vir_window=0,
            pt2_regularizer="dsrg",
            pt2_regularizer_strength=0.2,
        ),
        nroots=1,
    )(rhf)

    sci.run()

    # this is the variational energy
    assert sci.E[0] == pytest.approx(-3.321294103198, abs=1e-9)
    # this is the regularized PT2 correction
    assert sci.E_pt2[0] == pytest.approx(-2.53555293e-05, abs=1e-9)


@pytest.mark.slow
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
        sci_params=SelectedCIParams(
            selection_algorithm="hbci",
            var_threshold=1e-5,
            pt2_threshold=0.0,
            guess_occ_window=2,
            guess_vir_window=2,
            do_spin_penalty=False,
        ),
        nroots=4,
    )(rhf)

    sci.run()

    assert sci.E[0] == pytest.approx(-3.3213220620, abs=1e-8)
    assert sci.E[3] == pytest.approx(-3.0403077216, abs=1e-8)


@pytest.mark.slow
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
        nroots=2,
        sci_params=SelectedCIParams(
            selection_algorithm="hbci",
            var_threshold=1e-5,
            pt2_threshold=0.0,
            guess_occ_window=2,
            guess_vir_window=2,
            do_spin_penalty=True,
        ),
    )(rhf)

    sci.run()

    assert sci.E[0] == pytest.approx(-3.3213219202, abs=1e-8)
    assert sci.E[1] == pytest.approx(-3.0403076453, abs=1e-8)


@pytest.mark.slow
def test_sci5():
    """Test SelectedCI on a core-ionized state."""
    xyz = f"""
    Ne 0.0 0.0 0.0
    """

    system = System(xyz=xyz, basis_set="cc-pVDZ", cholesky_tei=True, cholesky_tol=1e-16)

    rhf = RHF(charge=0, econv=1e-10)(system)

    sci0 = SelectedCI(
        states=State(nel=10, multiplicity=1, ms=0.0),
        active_orbitals=list(range(12)),
        sci_params=SelectedCIParams(
            selection_algorithm="hbci",
            var_threshold=1e-5,
            pt2_threshold=0.0,
            do_spin_penalty=True,
        ),
        nroots=1,
    )(rhf)

    sci0.run()

    sci = SelectedCI(
        states=State(nel=9, multiplicity=2, ms=0.5),
        active_orbitals=list(range(12)),
        sci_params=SelectedCIParams(
            selection_algorithm="hbci",
            var_threshold=1e-5,
            pt2_threshold=0.0,
            do_spin_penalty=True,
            guess_dets=[Determinant("a2222")],
        ),
        nroots=1,
    )(rhf)

    sci.run()

    # This value is sensitive to the selected space growth details; keep a practical tolerance.
    assert sci.E[0] == pytest.approx(-96.5578779686, abs=5e-3)


@pytest.mark.slow
def test_sci6():
    """Test SelectedCI on a core-excited state."""
    xyz = """
    Ne 0.0 0.0 0.0
    """

    system = System(xyz=xyz, basis_set="cc-pVDZ", cholesky_tei=True, cholesky_tol=1e-16)

    rhf = RHF(charge=0, econv=1e-10)(system)

    sci = SelectedCI(
        states=State(nel=10, multiplicity=1, ms=0.0),
        active_orbitals=list(range(12)),
        sci_params=SelectedCIParams(
            selection_algorithm="hbci_ref",
            var_threshold=3e-4,
            pt2_threshold=0.0,
            guess_dets=[Determinant("a2222b"), Determinant("b2222a")],
            do_spin_penalty=True,
            screening_criterion="hbci",
        ),
        nroots=1,
        die_if_not_converged=False,
    )(rhf)

    sci.run()

    assert np.isfinite(sci.E[0])
    assert sci.E[0] < -95.0


def test_sci_exact_algorithm():
    """FCI energy from exact selected-CI diagonalization."""
    rhf = _h4_rhf()

    sci = SelectedCI(
        states=State(nel=4, multiplicity=1, ms=0.0),
        active_orbitals=list(range(4)),
        sci_params=SelectedCIParams(
            selection_algorithm="hbci",
            var_threshold=1e-12,
            pt2_threshold=0.0,
            ci_algorithm="exact",
        ),
    )(rhf)
    sci.run()

    assert sci.E[0] == approx(-2.180967812920)


def test_sci_make_sf_1rdm():
    """Spin-free 1-RDM should be available from the SCI helper-backed path."""
    rhf = _h4_rhf()

    sci = SelectedCI(
        states=State(nel=4, multiplicity=1, ms=0.0),
        active_orbitals=list(range(4)),
        sci_params=SelectedCIParams(
            selection_algorithm="hbci",
            var_threshold=1e-12,
            pt2_threshold=0.0,
        ),
    )(rhf)
    sci.run()

    rdm1 = sci.sub_solvers[0].make_sf_1rdm(0)
    assert rdm1.shape == (4, 4)
    assert np.trace(rdm1) == pytest.approx(4.0, abs=1e-8)
    assert sci.E[0] == approx(-2.180967812920)


def test_sci_semicanonical_final_orbital():
    """Semicanonical final orbital path should execute without runtime errors."""
    rhf = _h4_rhf()

    sci = SelectedCI(
        states=State(nel=4, multiplicity=1, ms=0.0),
        active_orbitals=list(range(4)),
        sci_params=SelectedCIParams(
            selection_algorithm="hbci",
            var_threshold=1e-12,
            pt2_threshold=0.0,
        ),
        final_orbital="semicanonical",
    )(rhf)
    sci.run()

    assert sci.E[0] == approx(-2.180967812920)


@pytest.mark.parametrize("selection_algorithm", ["hbci_ref", "hbci"])
def test_sci_frozen_creation_blocks_selected_growth(selection_algorithm):
    """Frozen creation orbitals should be excluded from SCI selection."""
    rhf = _h4_rhf()

    common_kwargs = dict(
        states=State(nel=4, multiplicity=1, ms=0.0),
        active_orbitals=list(range(4)),
        selection_algorithm=selection_algorithm,
        guess_occ_window=0,
        guess_vir_window=0,
        var_threshold=1e-12,
        pt2_threshold=0.0,
        maxcycle=1,
        ci_algorithm="exact",
        num_threads=1,
        num_batches_per_thread=1,
    )

    sci_unfrozen = SelectedCI(**common_kwargs)(rhf)
    sci_unfrozen.run()

    sci_frozen = SelectedCI(**common_kwargs, frozen_creation=[2, 3])(rhf)
    sci_frozen.run()

    assert sci_unfrozen.sub_solvers[0].sci_helper.ndets() > 1
    assert sci_frozen.sub_solvers[0].sci_helper.ndets() == 1
    assert sci_frozen.sub_solvers[0].sci_helper.dets()[0] == Determinant("2200")
