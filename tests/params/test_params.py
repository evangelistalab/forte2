from typing import get_args
import itertools

import pytest

from forte2.base_classes.params import X2CParams
from forte2.data.atom_data import LIGHT_SPEED


def test_x2c_params():
    fields = ("x2c_type", "x2c_model", "snso_type")
    choices = [get_args(X2CParams.__annotations__[name]) for name in fields]
    snso = ("boettger", "dc", "dcb", "row-dependent")
    # x2c_model=None means "1e" once x2c_type is set
    allowed = [
        (None, None, None),
        ("sf", None, None),
        ("sf", "1e", None),
        ("sf", "sap", None),
        ("so", None, None),
        ("so", "1e", None),
        ("so", "sap", None),
        *[("so", model, s) for model in (None, "1e") for s in snso],
    ]
    for combo in itertools.product(*choices):
        is_allowed = X2CParams.is_valid_input(*combo)
        assert is_allowed == (combo in allowed)


def test_x2c_params_defaults():
    assert X2CParams().x2c_type is None and X2CParams().x2c_model is None
    assert X2CParams(x2c_type="so").x2c_model == "1e"


def test_x2c_params_light_speed():
    assert X2CParams(x2c_type="so").light_speed == LIGHT_SPEED
    assert X2CParams(x2c_type="sf", light_speed=50.0).light_speed == 50.0
    for bad in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="light_speed must be positive"):
            X2CParams(x2c_type="so", light_speed=bad)
    with pytest.raises(ValueError, match="light_speed only affects X2C"):
        X2CParams(x2c_type=None, x2c_model=None, light_speed=50.0)
