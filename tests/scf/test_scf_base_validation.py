import pytest

from forte2 import System
from forte2.scf import RHF, UHF, ROHF, CUHF, GHF


def _atom():
    # Small open-shell system usable by UHF/CUHF/ROHF/GHF.
    return System(
        xyz="Li 0 0 0",
        basis_set="sto-3g",
        auxiliary_basis_set="def2-universal-JKFIT",
    )


def test_level_shift_scalar_promoted_for_tuple_methods():
    # Regression: a scalar level_shift was only promoted to a 2-tuple for
    # method == "UHF" and only when it was a float. CUHF reuses UHF's
    # tuple-based _apply_level_shift, and an int was never promoted, so both
    # crashed with "'float'/'int' object is not iterable" during run().
    sys = _atom()

    uhf = UHF(charge=0, ms=0.5, level_shift=1)(sys)
    assert uhf.level_shift == (1.0, 1.0)

    sys2 = _atom()
    cuhf = CUHF(charge=0, ms=0.5, level_shift=0.5)(sys2)
    assert cuhf.level_shift == (0.5, 0.5)


def test_level_shift_negative_tuple_element_rejected():
    # Regression: the non-negativity check only handled scalars, so a negative
    # element inside a UHF tuple slipped through and was applied to the alpha
    # channel.
    sys = _atom()
    with pytest.raises(ValueError):
        UHF(charge=0, ms=0.5, level_shift=(-1.0, 0.5))(sys)


def test_level_shift_negative_scalar_rejected():
    sys = _atom()
    with pytest.raises(ValueError):
        RHF(charge=0, level_shift=-0.5)(sys)


def test_tuple_level_shift_rejected_for_non_uhf():
    sys = _atom()
    with pytest.raises(ValueError):
        RHF(charge=0, level_shift=(0.5, 0.5))(sys)
    sys2 = _atom()
    with pytest.raises(ValueError):
        GHF(charge=0, level_shift=(0.5, 0.5))(sys2)


def test_tuple_level_shift_wrong_length_rejected():
    sys = _atom()
    with pytest.raises(ValueError):
        UHF(charge=0, ms=0.5, level_shift=(0.5, 0.5, 0.5))(sys)
