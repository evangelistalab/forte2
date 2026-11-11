"""Reject unsupported penalty gradients without running molecular calculations."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from forte2.mcopt import mc_optimizer_grad


def _mc(*, penalty=0.0, executed=False, applied_penalty=None):
    mc = SimpleNamespace(
        lambda_penalty=penalty,
        executed=executed,
        ci_solver=SimpleNamespace(sa_info=SimpleNamespace(nroots_sum=1)),
        active_frozen_orbitals=None,
        freeze_inter_gas_rots=False,
        final_orbitals="original",
        system=SimpleNamespace(two_component=False),
        run=Mock(side_effect=AssertionError("No molecular calculation is expected.")),
    )
    if applied_penalty is not None:
        mc.orb_opt = SimpleNamespace(lambda_penalty=applied_penalty)
    return mc


@pytest.mark.parametrize("executed", [False, True])
def test_nonzero_penalty_rejected_before_nuclear_gradient(executed):
    mc = _mc(penalty=0.5, executed=executed)

    with pytest.raises(NotImplementedError, match="projector-penalized"):
        mc_optimizer_grad._compute_casscf_gradient(mc)

    mc.run.assert_not_called()


def test_changing_option_does_not_release_converged_penalty_state():
    mc = _mc(penalty=0.0, executed=True, applied_penalty=0.5)

    with pytest.raises(NotImplementedError, match="after a penalized run"):
        mc_optimizer_grad._compute_casscf_gradient(mc)

    mc.run.assert_not_called()


@pytest.mark.parametrize(
    "executed, applied_penalty", [(False, None), (True, 0.0), (False, 0.5)]
)
def test_zero_penalty_request_preserves_existing_validation(
    monkeypatch, executed, applied_penalty
):
    mc = _mc(executed=executed, applied_penalty=applied_penalty)
    validate_system = Mock()
    monkeypatch.setattr(
        mc_optimizer_grad, "validate_df_gradient_system", validate_system
    )

    mc_optimizer_grad._validate_casscf_gradient_request(mc)

    validate_system.assert_called_once_with(mc.system, "CASSCF/GASSCF")
    mc.run.assert_not_called()
