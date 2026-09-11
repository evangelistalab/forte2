import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import pytest

from forte2.lib import cpp_helpers

_ROOT = Path(__file__).resolve().parents[2]

_THREAD_ENV = (
    "FORTE_NUM_THREADS_OVERRIDE",
    "OMP_NUM_THREADS",
    "OMP_THREAD_LIMIT",
    "SLURM_CPUS_PER_TASK",
)

_PARALLEL_FOR_VARIANTS = (
    "parallel_for",
    "parallel_for_chunked",
    "parallel_for_chunked_thread",
    "parallel_for_chunked_async",
    "parallel_for_interleaved_thread",
    "parallel_for_interleaved_async",
    "parallel_for_dynamic_thread",
    "parallel_for_dynamic_async",
)

# Counts spanning the interesting regimes: empty, single, below SERIAL_THRESHOLD*threads (serial
# fallback), an uneven size that does not divide evenly into blocks, and larger parallel sizes.
_COVERAGE_COUNTS = (0, 1, 2, 7, 101, 1000)


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


def _run_driver(executable, *args):
    """Run one driver mode. The driver self-asserts and communicates via exit code alone:
    0 = passed, 77 = skip (unsupported environment), anything else = failed. On failure the
    driver's stderr is surfaced in the assertion message."""
    env = os.environ.copy()
    for variable in _THREAD_ENV:
        env.pop(variable, None)
    result = subprocess.run(
        [str(executable), *args], env=env, check=False, capture_output=True, text=True
    )
    if result.returncode == 77:
        pytest.skip("driver reported an unsupported environment (exit code 77)")
    assert result.returncode == 0, result.stderr or result.stdout


def test_get_num_threads(monkeypatch):
    def _reset():
        for var in _THREAD_ENV:
            monkeypatch.delenv(var, raising=False)

    _reset()
    # baseline min(hardware, affinity)
    phys_thrds = cpp_helpers.get_num_threads()

    cases = [
        # override bypasses the min-chain, so it wins regardless of hardware
        ([("FORTE_NUM_THREADS_OVERRIDE", " 12345678\t\n\v")], 12345678),
        # a small valid override wins even below the hardware count
        ([("FORTE_NUM_THREADS_OVERRIDE", "2")], 2),
        # malformed / out-of-range overrides fall through to the min-chain
        ([("FORTE_NUM_THREADS_OVERRIDE", "9" * 20)], phys_thrds),  # overflows size_t
        ([("FORTE_NUM_THREADS_OVERRIDE", "1.23")], phys_thrds),
        ([("FORTE_NUM_THREADS_OVERRIDE", "12-3")], phys_thrds),
        ([("FORTE_NUM_THREADS_OVERRIDE", "-1")], phys_thrds),
        ([("FORTE_NUM_THREADS_OVERRIDE", "0")], phys_thrds),
        ([("FORTE_NUM_THREADS_OVERRIDE", "")], phys_thrds),
        # the override does not accept the comma-list syntax
        ([("FORTE_NUM_THREADS_OVERRIDE", "12,6")], phys_thrds),
        ([("FORTE_NUM_THREADS_OVERRIDE", "12"), ("SLURM_CPUS_PER_TASK", "6")], 12),
        # no override: the smallest valid limit wins, bounded by the hardware count
        (
            [
                ("OMP_NUM_THREADS", "12"),
                ("SLURM_CPUS_PER_TASK", "6"),
                ("OMP_THREAD_LIMIT", "3"),
            ],
            min(phys_thrds, 3),
        ),
        # an invalid limit is ignored; the next-smallest valid one applies
        (
            [
                ("OMP_NUM_THREADS", "12"),
                ("SLURM_CPUS_PER_TASK", "4"),
                ("OMP_THREAD_LIMIT", "-5"),
            ],
            min(phys_thrds, 4),
        ),
        # only the first element of an OMP_NUM_THREADS list is used
        ([("OMP_NUM_THREADS", "2,1 , 2")], min(phys_thrds, 2)),
    ]

    for envvars, expected in cases:
        _reset()
        for var, val in envvars:
            monkeypatch.setenv(var, val)
        assert cpp_helpers.get_num_threads() == expected


@pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="sched affinity is Linux-only"
)
def test_get_num_threads_with_affinity_mask(parallel_test_driver):
    # The driver pins itself to a single CPU, then asserts get_num_threads() reports 1.
    _run_driver(parallel_test_driver, "--test-process-affinity")


@pytest.mark.parametrize("variant", _PARALLEL_FOR_VARIANTS)
@pytest.mark.parametrize("count", _COVERAGE_COUNTS)
def test_parallel_for_variants(parallel_test_driver, variant, count):
    # The driver runs the variant and asserts every index in [0, count) is visited exactly once,
    # failing (non-zero exit) otherwise. A returncode of 77 (skip) is handled inside _run_driver.
    _run_driver(
        parallel_test_driver, "--test-parallel-for-variant", variant, str(count)
    )


def test_thread_joiner(parallel_test_driver):
    _run_driver(parallel_test_driver, "--test-thread-joiner")
