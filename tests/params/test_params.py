from typing import get_args
import itertools

from forte2.base_classes.params import X2CParams


def test_x2c_params():
    fields = ("x2c_type", "x2c_model", "snso_type")
    choices = [get_args(X2CParams.__annotations__[name]) for name in fields]
    allowed = [
        (None, None, None),
        ("sf", "1e", None),
        ("sf", "sap", None),
        ("so", "1e", None),
        ("so", "1e", "boettger"),
        ("so", "1e", "dc"),
        ("so", "1e", "dcb"),
        ("so", "1e", "row-dependent"),
        ("so", "sap", None),
    ]
    for combo in itertools.product(*choices):
        is_allowed = X2CParams.is_valid_input(*combo)
        assert is_allowed == (combo in allowed)
