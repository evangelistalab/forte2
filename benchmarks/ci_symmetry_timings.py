#!/usr/bin/env python3
"""Time CI sigma builds and RDM contractions in C1 and C2v.

The real workloads use eight electrons in ten orbitals and ten electrons in
twelve orbitals, each with two GAS spaces. The complex workloads use six
electrons in twelve spinors and eight electrons in sixteen spinors. C2v labels
are interleaved and use target irrep 3. All workloads use seeded,
symmetry-conserving Hamiltonians and distinct left and right CI vectors.
Pass --quick to run only the smaller workload for each scalar type.

Run from the repository root with a rebuilt extension, for example::

    env FORTE_NUM_THREADS_OVERRIDE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1 \
        python benchmarks/ci_symmetry_timings.py
"""

from argparse import ArgumentParser
from functools import partial
from statistics import median
from time import perf_counter
from timeit import Timer

import numpy as np

from forte2.lib.ci_helpers import CISigmaBuilder, CIStrings, RelCISigmaBuilder
from forte2.lib.cpp_helpers import get_num_threads


def make_integrals(rng, irreps, nirrep, spinor):
    """Generate Hermitian H and density-fitted V in physicist's ordering."""
    norb = len(irreps)
    pair_irreps = irreps[:, None] ^ irreps[None, :]
    auxiliary_irreps = np.arange(2 * norb) % nirrep
    h = rng.standard_normal((norb, norb))
    factors = rng.standard_normal((len(auxiliary_irreps), norb, norb))
    if spinor:
        h = h + 1j * rng.standard_normal(h.shape)
        factors = factors + 1j * rng.standard_normal(factors.shape)
    h = (h + h.T.conj()) * (pair_irreps == 0)
    factors = factors + factors.swapaxes(1, 2).conj()
    factors *= auxiliary_irreps[:, None, None] == pair_irreps
    v = np.einsum("Apr,Aqs->pqrs", factors, factors, optimize=True)
    return h, np.ascontiguousarray(v)


def median_call_seconds(operation, repeats):
    timer = Timer(operation)
    warmup = timer.timeit(number=1)
    # Batch short calls to reduce timer noise while keeping the suite quick.
    calls = max(1, min(100, int(0.02 / max(warmup, 1e-9))))
    return median(timer.repeat(repeat=repeats, number=calls)) / calls


def time_case(point_group, nirrep, norb, na, spinor, repeats):
    rng = np.random.default_rng(17)
    irreps = np.arange(norb) % nirrep
    symmetry = nirrep - 1
    h, v = make_integrals(rng, irreps, nirrep, spinor)
    energy = 0.37

    if spinor:
        lists = CIStrings(na, 0, nirrep, symmetry, [irreps.tolist()], [], [])
        builders = {"HZ": RelCISigmaBuilder(lists, energy, h, v, 0, "hz")}
        rdm_names = ("so_2rdm", "so_3rdm")
    else:
        gas = [space.tolist() for space in np.split(irreps, 2)]
        lists = CIStrings(na, na, nirrep, symmetry, gas, [na], [na])
        builders = {
            "HZ": CISigmaBuilder(lists, energy, h, v, 0, "hz"),
            "KH": CISigmaBuilder(lists, energy, h, v, 0, "kh"),
        }
        rdm_names = ("ab_2rdm", "aab_3rdm", "abb_3rdm")

    left, right = rng.standard_normal((2, lists.ndet)).astype(h.dtype)
    if spinor:
        left += 1j * rng.standard_normal(lists.ndet)
        right += 1j * rng.standard_normal(lists.ndet)
    left /= np.linalg.norm(left)
    right /= np.linalg.norm(right)
    sigma = np.empty_like(right)

    if not spinor:
        reference = np.empty_like(right)
        builders["HZ"].Hamiltonian(right, reference)
        builders["KH"].Hamiltonian(right, sigma)
        np.testing.assert_allclose(sigma, reference, rtol=1e-11, atol=1e-11)

    scalar = "complex" if spinor else "real"

    def report(kernel, operation):
        milliseconds = 1000 * median_call_seconds(operation, repeats)
        print(
            f"{point_group:<8} {scalar:<7} {norb:>4} {lists.ndet:>7} "
            f"{kernel:<22} {milliseconds:>10.4f}"
        )

    for algorithm, builder in builders.items():
        report(
            f"{algorithm} integral update",
            partial(builder.set_Hamiltonian, energy, h, v),
        )
        report(f"{algorithm} sigma", partial(builder.Hamiltonian, right, sigma))
        report(
            f"{algorithm} sigma 1e", partial(builder.sigma_one_electron, right, sigma)
        )
        report(
            f"{algorithm} sigma 2e", partial(builder.sigma_two_electron, right, sigma)
        )

    for name in rdm_names:
        report(name, partial(getattr(builders["HZ"], name), left, right))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5, help="Samples per kernel")
    parser.add_argument(
        "--quick", action="store_true", help="Run only the smaller cases"
    )
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")

    start = perf_counter()
    print(f"Median of {args.repeats} samples with {get_num_threads()} Forte thread(s)")
    print(
        f"{'symmetry':<8} {'scalar':<7} {'norb':>4} {'ndet':>7} {'kernel':<22} {'median_ms':>10}"
    )
    cases = [(10, 4, False), (12, 6, True)]
    if not args.quick:
        cases.extend([(12, 5, False), (16, 8, True)])
    for point_group, nirrep in (("C1", 1), ("C2v", 4)):
        for norb, na, spinor in cases:
            time_case(point_group, nirrep, norb, na, spinor, args.repeats)
    print(f"Total benchmark time {perf_counter() - start:.2f} s")


if __name__ == "__main__":
    main()
