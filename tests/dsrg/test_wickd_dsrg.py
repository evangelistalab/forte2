import contextlib
import importlib.util
import io

import pytest

import forte2
from forte2.helpers import logger

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("wickd") is None,
    reason="wickd is not installed",
)


@pytest.fixture(scope="module")
def h2_sto3g_data():
    logger.set_verbosity_level(0)
    with contextlib.redirect_stdout(io.StringIO()):
        system = forte2.System(
            xyz="H 0.0 0.0 0.0\nH 0.0 0.0 0.74",
            basis_set="sto-3g",
            minao_basis_set=None,
            cholesky_tei=True,
            cholesky_tol=1.0e-12,
        )
        rhf = forte2.RHF(charge=0, e_tol=1.0e-10, d_tol=1.0e-8)(system)
        rhf.run()
    return forte2.wickd_dsrg_data_from_rhf(rhf)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("rank", "solver", "expected"),
    [
        (2, forte2.solve_wickd_dsrg2, -1.137831710258920),
        (3, forte2.solve_wickd_dsrg3, -1.137283834651968),
        (4, forte2.solve_wickd_dsrg4, -1.137283834651968),
    ],
)
def test_wickd_dsrg_h2_sto3g_regression(h2_sto3g_data, rank, solver, expected):
    result = solver(
        h2_sto3g_data,
        flow_param=5.0,
        e_tol=1.0e-10,
        r_tol=1.0e-5,
        maxiter=40,
    )

    assert result.converged
    assert result.max_rank == rank
    assert result.energy == pytest.approx(expected, abs=5.0e-10)
    assert result.equations.rank == rank
