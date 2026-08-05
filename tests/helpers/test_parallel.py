import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_THREAD_LIMIT_ENV = ("OMP_NUM_THREADS", "OMP_THREAD_LIMIT", "SLURM_CPUS_PER_TASK")


@pytest.fixture(scope="module")
def parallel_test_driver(tmp_path_factory):
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    if not compiler or shutil.which(compiler[0]) is None:
        pytest.skip(
            "a C++ compiler is required to test the header-only parallel helpers"
        )

    executable = tmp_path_factory.mktemp("parallel") / "parallel_test_driver"
    source = Path(__file__).with_name("parallel_test_driver.cc")
    command = compiler + ["-std=c++20", "-pthread"]
    if sys.platform == "darwin":
        command.append("-fblocks")
    command += ["-I", str(_ROOT), str(source), "-o", str(executable)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    return executable


def _run_driver(executable, limits=None, *args):
    env = os.environ.copy()
    for variable in _THREAD_LIMIT_ENV:
        env.pop(variable, None)
    env.update(limits or {})
    result = subprocess.run(
        [str(executable), *args], env=env, check=False, capture_output=True, text=True
    )
    if result.returncode == 77:
        pytest.skip("process affinity controls are unavailable")
    result.check_returncode()
    return tuple(int(value) for value in result.stdout.split())


def test_parallel_thread_count_honors_environment_limits(parallel_test_driver):
    baseline, baseline_chunks, baseline_visited = _run_driver(
        parallel_test_driver, {"OMP_NUM_THREADS": "8"}
    )
    assert baseline >= 1
    assert baseline_chunks == baseline
    assert baseline_visited == 3 * baseline

    cases = [
        ({"OMP_NUM_THREADS": "1"}, 1),
        ({"OMP_NUM_THREADS": "1,4"}, 1),
        ({"OMP_NUM_THREADS": "8", "OMP_THREAD_LIMIT": "2"}, min(baseline, 2)),
        ({"OMP_NUM_THREADS": "8", "SLURM_CPUS_PER_TASK": "2"}, min(baseline, 2)),
    ]
    for limits, expected in cases:
        num_threads, chunks, visited = _run_driver(parallel_test_driver, limits)
        assert num_threads == expected
        assert chunks == expected
        assert visited == 3 * expected


def test_worker_threads_are_joined_during_exception_unwinding(parallel_test_driver):
    assert _run_driver(parallel_test_driver, None, "--join-on-exception") == (1,)


@pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="sched affinity is Linux-only"
)
def test_parallel_thread_count_honors_process_affinity(parallel_test_driver):
    num_threads, chunks, visited = _run_driver(parallel_test_driver, None, "--pin-one")
    assert num_threads == 1
    assert chunks == 1
    assert visited == 3
