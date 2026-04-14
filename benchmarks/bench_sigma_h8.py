"""
Benchmark for sigma-vector methods on H8/cc-pVDZ.

Measures the wall time of each H2xx and H2xx_claude method at various CI space sizes,
averaged over many applications to simulate a Davidson diagonalization workload.

Usage:
    python benchmarks/bench_sigma_h8.py [--nthreads N] [--ntrial N]
"""

import argparse
import time

import numpy as np

import forte2
from forte2 import Determinant, System
from forte2.jkbuilder.mointegrals import RestrictedMOIntegrals
from forte2.scf import RHF


# ---------------------------------------------------------------------------
# Geometry & integrals
# ---------------------------------------------------------------------------


def build_h8_integrals(r_hh: float = 1.0):
    """Build RHF and MO integrals for H8 in a linear chain with cc-pVDZ."""
    coords = "\n".join(f"H 0.0 0.0 {i * r_hh:.6f}" for i in range(8))
    system = System(
        xyz=coords, basis_set="cc-pVDZ", auxiliary_basis_set="cc-pVTZ-JKFIT"
    )
    rhf = RHF(charge=0, econv=1e-12)(system)
    rhf.run()
    norb_total = rhf.C[0].shape[1]
    orbitals = list(range(norb_total))
    ints = RestrictedMOIntegrals(
        system=system, C=rhf.C[0], orbitals=orbitals, core_orbitals=[]
    )
    return ints.E, ints.H, ints.V


# ---------------------------------------------------------------------------
# CI space builder
# ---------------------------------------------------------------------------


def grow_ci_space(E, H, V, hf_string: str, var_threshold: float, nthreads: int):
    """Grow a selected CI space to convergence at the given variational threshold."""
    norb = H.shape[0]
    dets = [Determinant(hf_string)]
    c = np.array([[1.0]])
    helper = forte2.SelectedCIHelper(norb, dets, c, E, H, V)
    helper.set_num_threads(nthreads)
    helper.set_screening_criterion("hbci")

    for _ in range(20):
        prev_ndets = helper.ndets()
        helper.select_hbci(var_threshold, 0.0)
        nd = helper.ndets()
        c0 = np.ones((nd, 1)) / nd**0.5
        helper.set_c(c0)
        if nd == prev_ndets:
            break

    return helper


# ---------------------------------------------------------------------------
# Timing utilities
# ---------------------------------------------------------------------------


def bench_method(
    helper, method_name: str, basis: np.ndarray, nwarm: int = 3, ntrial: int = 100
) -> float:
    """Return mean wall-time in ms for one sigma-vector application."""
    ndets = len(helper.dets())
    sigma = np.zeros(ndets)
    fn = getattr(helper, method_name)

    # Warm-up (also populates the coupling-list cache on first call)
    for _ in range(nwarm):
        sigma[:] = 0.0
        fn(basis, sigma)

    t0 = time.perf_counter()
    for _ in range(ntrial):
        sigma[:] = 0.0
        fn(basis, sigma)
    return (time.perf_counter() - t0) / ntrial * 1e3


def verify_pair(helper, ref_name: str, new_name: str, basis: np.ndarray) -> float:
    """Return max absolute error between two sigma methods."""
    ndets = len(helper.dets())
    s_ref = np.zeros(ndets)
    s_new = np.zeros(ndets)
    getattr(helper, ref_name)(basis, s_ref)
    getattr(helper, new_name)(basis, s_new)
    return float(np.max(np.abs(s_ref - s_new)))


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------


def run_benchmark(nthreads: int, ntrial: int):
    print("=" * 70)
    print(
        f"  H8/cc-pVDZ sigma-vector benchmark  |  threads={nthreads}  ntrial={ntrial}"
    )
    print("=" * 70)

    # Build integrals once
    print("\nBuilding H8/cc-pVDZ integrals (RHF) …", flush=True)
    E, H, V = build_h8_integrals()
    norb = H.shape[0]
    na = nb = 4
    print(f"  norb={norb}, na={na}, nb={nb}")

    # HF determinant string: 4 doubly-occupied orbitals, padded to 64
    hf_string = "2" * na + "0" * (64 - na)

    # Variational thresholds to study different CI space sizes
    thresholds = [5e-2, 1e-2, 5e-3, 1e-3]

    print(
        f"\n{'Threshold':>12}  {'ndets':>8}  "
        f"{'H2aa':>8}  {'H2aa_claude':>11}  {'spd':>5}  "
        f"{'H2bb':>8}  {'H2bb_claude':>11}  {'spd':>5}  "
        f"{'H2ab':>8}  {'H2ab_claude':>11}  {'spd':>5}"
    )
    print("-" * 100)

    for threshold in thresholds:
        helper = grow_ci_space(E, H, V, hf_string, threshold, nthreads)
        helper.set_num_threads(nthreads)

        dets = helper.dets()
        ndets = len(dets)
        c_arr = np.ones((ndets, 1)) / ndets**0.5
        helper.set_c(c_arr)

        basis = np.random.default_rng(0).standard_normal(ndets)

        # Correctness
        err_aa = verify_pair(helper, "H2aa", "H2aa_claude", basis)
        err_bb = verify_pair(helper, "H2bb", "H2bb_claude", basis)
        err_ab = verify_pair(helper, "H2ab", "H2ab_claude", basis)
        assert err_aa < 1e-10, f"H2aa_claude wrong: max_err={err_aa:.2e}"
        assert err_bb < 1e-10, f"H2bb_claude wrong: max_err={err_bb:.2e}"
        assert err_ab < 1e-10, f"H2ab_claude wrong: max_err={err_ab:.2e}"

        t_aa = bench_method(helper, "H2aa", basis, ntrial=ntrial)
        t_aa_claude = bench_method(helper, "H2aa_claude", basis, ntrial=ntrial)
        t_bb = bench_method(helper, "H2bb", basis, ntrial=ntrial)
        t_bb_claude = bench_method(helper, "H2bb_claude", basis, ntrial=ntrial)
        t_ab = bench_method(helper, "H2ab", basis, ntrial=ntrial)
        t_ab_claude = bench_method(helper, "H2ab_claude", basis, ntrial=ntrial)

        print(
            f"{threshold:>12.0e}  {ndets:>8d}  "
            f"{t_aa:>7.2f}ms  {t_aa_claude:>10.2f}ms  {t_aa/t_aa_claude:>4.1f}x  "
            f"{t_bb:>7.2f}ms  {t_bb_claude:>10.2f}ms  {t_bb/t_bb_claude:>4.1f}x  "
            f"{t_ab:>7.2f}ms  {t_ab_claude:>10.2f}ms  {t_ab/t_ab_claude:>4.1f}x"
        )

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="H8/cc-pVDZ sigma-vector benchmark")
    parser.add_argument("--nthreads", type=int, default=1, help="number of threads")
    parser.add_argument("--ntrial", type=int, default=100, help="trials per timing")
    args = parser.parse_args()

    run_benchmark(nthreads=args.nthreads, ntrial=args.ntrial)
