import pytest

from experiments.lih_ccpvdz_dsrg_sweep import (
    build_lih_problem,
    solve_sr_bare_problem,
    solve_sr_normal,
)


@pytest.mark.slow
def test_lih_ccpvdz_srdsrg2_bare_and_normal_truncation():
    """Keep the historical bare- versus normal-rank definitions reproducible."""
    problem = build_lih_problem(1.60)

    normal = solve_sr_normal(problem, 2, max_iter=0, max_commutators=1)
    bare = solve_sr_bare_problem(problem, 2, max_iter=0, max_commutators=1)

    assert normal["n_amplitudes"] == bare["n_amplitudes"] == 1496
    assert normal["energy"] == pytest.approx(-8.02904202865641, abs=5.0e-9)
    assert bare["energy"] == pytest.approx(-8.029042028669403, abs=5.0e-9)
    assert normal["history"][0]["rms_update"] == pytest.approx(
        0.03488296046300551, abs=5.0e-9
    )
    assert bare["history"][0]["rms_update"] == pytest.approx(
        0.03742494038701854, abs=5.0e-9
    )
    assert bare["history"][0]["n_terms"] == 77954
    assert (
        abs(bare["history"][0]["rms_update"] - normal["history"][0]["rms_update"])
        > 1.0e-3
    )
