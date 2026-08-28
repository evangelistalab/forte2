import itertools

import numpy as np
import pytest

from forte2 import System, State
from forte2.scf import RHF
from forte2.sci import SelectedCI
from forte2.lib.det import Determinant, SlaterRules
from forte2.base_classes.params import SelectedCIParams

# The thresholds are chosen so that each term of the telescoping sum carries real weight on this
# system: the deterministic term is about -1.28e-3, the gap it leaves to EPS2_PSEUDOSTOCH about -2.1e-4,
# and the gap from there to EPS2 about -1.6e-7.
EPS2 = 1e-7
EPS2_PSEUDOSTOCH = 3e-6
EPS2_DETERM = 1e-4
NUM_BATCHES = 16


def _converged_helper(nroots=1):
    """Converge a selected CI and hand back its helper, with the variational space untouched.

    The semistochastic algorithm replaces the final selection cycle, so the helper it leaves behind
    still describes the space the last diagonalization saw. That is what both the deterministic and
    the semistochastic correction have to be evaluated on for their comparison to mean anything.

    The variational threshold is deliberately loose. A selected CI that converges onto the whole
    determinant space has no correction at all to compare.
    """
    xyz = """
    H 0.0 0.0 0.0
    H 0.0 0.0 1.2
    H 0.0 0.0 2.4
    H 0.0 0.0 3.6
    H 0.0 0.0 4.8
    H 0.0 0.0 6.0
    """
    system = System(xyz=xyz, basis_set="6-31g", auxiliary_basis_set="cc-pVTZ-JKFIT")
    rhf = RHF(charge=0, e_tol=1e-12)(system)
    sci = SelectedCI(
        states=State(nel=6, multiplicity=1, ms=0.0),
        active_orbitals=list(range(12)),
        nroots=nroots,
        sci_params=SelectedCIParams(
            var_threshold=1e-3,
            pt2_threshold=EPS2,
            maxcycle=6,
            # the run's own correction is not what is being tested, so make it cheap
            pt2_algorithm="semistochastic",
            pt2_threshold_pseudostoch=EPS2_DETERM,
            pt2_threshold_determ=EPS2_DETERM,
            pt2_num_samples=0,
        ),
    )(rhf)
    sci.run()
    return sci.sub_solvers[0].sci_helper


def _determ(helper, eps2):
    helper.compute_pt2_determ(eps2=eps2, num_batches=NUM_BATCHES)
    return np.array(helper.ept2())


def _semistoch(helper, **kwargs):
    args = dict(
        eps2=EPS2,
        eps2_pseudostoch=EPS2_PSEUDOSTOCH,
        eps2_determ=EPS2_DETERM,
        num_batches=NUM_BATCHES,
        min_batches_pseudostoch=NUM_BATCHES,
        target_error=0.0,
        num_batches_stoch=1,
        batches_per_sample=1,
        num_samples=10,
        sample_size=50,
        seed=0,
    )
    args.update(kwargs)
    helper.compute_pt2_semistoch(**args)
    return np.array(helper.ept2())


def test_determ_pt2_matches_a_brute_force_sum():
    """The connection generator must reach every connection exactly once, with the right sign.

    Everything else in this file compares one path through `generate_connections` against another,
    so a connection generated twice, or missed, or given the wrong sign would cancel out of those
    comparisons. This builds the Epstein-Nesbet sum from Slater rules over the whole determinant
    space instead, which shares no code with the kernel:

        dE2 = sum_a (sum_i <a|H|i> c_i)^2 / (E_0 - <a|H|a>)   over a outside the variational space

    That the couplings are summed over all parents before being squared is Holmes et al. (JCTC 12,
    3674, 2016), Eq. 3, and is the invariant the `sci-var-fix` branch restored. Squaring each
    parent's contribution separately would change the answer here by the cross terms.
    """
    norb = 6
    nel = 6
    xyz = "\n".join(f"H 0.0 0.0 {1.4 * i}" for i in range(nel))
    system = System(xyz=xyz, basis_set="sto-6g", auxiliary_basis_set="cc-pVTZ-JKFIT")
    rhf = RHF(charge=0, e_tol=1e-12)(system)
    sci = SelectedCI(
        states=State(nel=nel, multiplicity=1, ms=0.0),
        active_orbitals=list(range(norb)),
        nroots=1,
        sci_params=SelectedCIParams(
            var_threshold=5e-2,
            pt2_threshold=EPS2,
            maxcycle=4,
            pt2_algorithm="semistochastic",
            pt2_threshold_pseudostoch=EPS2_DETERM,
            pt2_threshold_determ=EPS2_DETERM,
            pt2_num_samples=0,
        ),
    )(rhf)
    sci.run()

    solver = sci.sub_solvers[0]
    helper = solver.sci_helper
    variational = list(helper.dets())
    coeffs = np.asarray(solver.evecs)[:, 0]
    assert 1 < len(variational) < 400, "the variational space must be a proper subset"

    slater = SlaterRules(norb, solver.ints.E, solver.ints.H, solver.ints.V)
    e0 = helper.energies()[0]

    # every determinant of the full space, as an occupation string over the active orbitals
    def occupations(nalpha):
        for alpha in itertools.combinations(range(norb), nalpha):
            for beta in itertools.combinations(range(norb), nel // 2):
                label = []
                for p in range(norb):
                    a, b = p in alpha, p in beta
                    label.append("2" if a and b else "a" if a else "b" if b else "0")
                yield Determinant("".join(label))

    inside = set(variational)
    expected = 0.0
    for det in occupations(nel // 2):
        if det in inside:
            continue
        v = sum(
            slater.slater_rules(det, parent) * c
            for parent, c in zip(variational, coeffs)
        )
        if v != 0.0:
            expected += v * v / (e0 - slater.energy(det))

    helper.compute_pt2_determ(eps2=0.0, num_batches=NUM_BATCHES)
    assert expected != 0.0
    assert helper.ept2()[0] == pytest.approx(expected, rel=1e-10)


def test_semistoch_collapse():
    """With all three thresholds equal, the two estimated terms must vanish identically.

    Both channels of a term are filled in a single pass, keyed on the same determinant and gated by
    the same comparison against the same threshold, so equal thresholds make them bit-identical and
    their difference exactly zero. That holds for any seed and any sample, which is what makes this
    a sharp test of the indexing, signs, weights and channel pairing all at once.
    """
    helper = _converged_helper()
    reference = _determ(helper, EPS2)
    assert abs(reference[0]) > 1e-8, "the correction is too small to test against"

    for seed in (0, 1, 12345):
        for num_samples, sample_size in ((2, 2), (10, 50)):
            got = _semistoch(
                helper,
                eps2=EPS2,
                eps2_pseudostoch=EPS2,
                eps2_determ=EPS2,
                seed=seed,
                num_samples=num_samples,
                sample_size=sample_size,
            )
            assert np.array_equal(helper.ept2_pseudostoch(), np.zeros(1))
            assert np.array_equal(helper.ept2_stoch(), np.zeros(1))
            assert np.array_equal(helper.ept2_stddev(), np.zeros(1))
            # the two vanishing terms leave the total exactly equal to the deterministic one, which
            # in turn matches the reference only up to how the threads happened to sum the batches
            assert np.array_equal(got, helper.ept2_determ())
            assert got == pytest.approx(reference, abs=1e-14)


def test_semistoch_depends_only_on_the_target_threshold():
    """The answer must not depend on where the work is divided between the three terms.

    Li et al. (JCP 149, 214110, 2018), Sec. IV: the correction depends only on eps2, while
    eps2_determ and eps2_pseudostoch affect only the efficiency. Every other test here collapses at least one
    term, so this is the only one where all three carry weight at once and the telescoping has to
    close. A term computed over the wrong threshold gap, or double counted between two terms, moves
    the answer by far more than the sampling error.
    """
    helper = _converged_helper()
    reference = _determ(helper, EPS2)

    splits = [
        (EPS2_DETERM, EPS2_PSEUDOSTOCH),
        (1e-5, 1e-6),
        (3e-4, 1e-6),
    ]
    for eps2_determ, eps2_pseudostoch in splits:
        got = _semistoch(
            helper,
            eps2_pseudostoch=eps2_pseudostoch,
            eps2_determ=eps2_determ,
            num_samples=200,
            sample_size=100,
        )
        terms = np.array(
            [
                helper.ept2_determ()[0],
                helper.ept2_pseudostoch()[0],
                helper.ept2_stoch()[0],
            ]
        )
        assert np.all(
            terms != 0.0
        ), f"a term vanished for {eps2_determ=}, {eps2_pseudostoch=}"
        sigma = np.array(helper.ept2_stddev())[0]
        assert (
            abs(got[0] - reference[0]) < 4.0 * sigma
        ), f"{eps2_determ=}, {eps2_pseudostoch=}: {got[0]:.8e} vs {reference[0]:.8e}, sigma {sigma:.2e}"


def test_semistoch_pseudostoch_stops_early():
    """The pseudo-stochastic term must extrapolate honestly when it stops before the last batch.

    Every other test drives that term to completion with target_error=0, which makes its
    extrapolation exact and its error zero. Here the stopping rule actually fires, so both the
    extrapolation from a subset of batches and its finite-population error bar are under test.
    """
    helper = _converged_helper()
    reference = _determ(helper, EPS2_PSEUDOSTOCH)

    got = _semistoch(
        helper,
        eps2=EPS2_PSEUDOSTOCH,
        eps2_pseudostoch=EPS2_PSEUDOSTOCH,
        num_samples=0,
        min_batches_pseudostoch=2,
        target_error=1e-4,
    )
    evaluated = helper.num_pseudostoch_batches()
    assert 2 <= evaluated < NUM_BATCHES, "the stopping rule did not fire"

    sigma = np.array(helper.ept2_pseudostoch_stddev())[0]
    assert sigma > 0.0, "an extrapolated term cannot have zero error"
    assert abs(got[0] - reference[0]) < 4.0 * sigma


def test_semistoch_step_isolation():
    """Each estimated term alone must reproduce the deterministic answer.

    Collapsing one term at a time leaves the other carrying the whole gap between two thresholds,
    so an error in either shows up undiluted.
    """
    helper = _converged_helper()
    reference_pseudostoch = _determ(helper, EPS2_PSEUDOSTOCH)
    reference = _determ(helper, EPS2)

    # The pseudo-stochastic term alone, evaluated over every batch and so exact. Equal tight and
    # loose thresholds in the stochastic term make it vanish without any sampling.
    got = _semistoch(
        helper, eps2=EPS2_PSEUDOSTOCH, eps2_pseudostoch=EPS2_PSEUDOSTOCH, num_samples=0
    )
    assert np.array_equal(helper.ept2_stoch(), np.zeros(1))
    assert got == pytest.approx(reference_pseudostoch, abs=1e-14)

    # The stochastic term alone, which now covers the gap from EPS2_DETERM down to EPS2.
    got = _semistoch(
        helper,
        eps2_pseudostoch=EPS2_DETERM,
        num_samples=200,
        sample_size=100,
    )
    assert np.array_equal(helper.ept2_pseudostoch(), np.zeros(1))
    sigma = np.array(helper.ept2_stddev())[0]
    assert sigma > 0.0
    assert abs(got[0] - reference[0]) < 4.0 * sigma


def test_semistoch_error_bar_and_batch_scaling():
    """The reported error must describe the actual spread, and batching must not bias the mean.

    The error bar is the deliverable of the stochastic term, so it is checked against the empirical
    spread over independent seeds rather than merely for being small. The same comparison run with
    the connected space split into batches also checks the extrapolation by the sampled fraction,
    which is unbiased but not exact for any single sample.
    """
    helper = _converged_helper()
    reference = _determ(helper, EPS2)

    def sample_means(**kwargs):
        means, sigmas = [], []
        for seed in range(20):
            got = _semistoch(
                helper,
                eps2_pseudostoch=EPS2_DETERM,
                num_samples=20,
                sample_size=100,
                seed=seed,
                **kwargs,
            )
            means.append(got[0])
            sigmas.append(np.array(helper.ept2_stddev())[0])
        return np.array(means), np.array(sigmas)

    means, sigmas = sample_means()
    empirical = means.std(ddof=1)
    reported = sigmas.mean()
    assert (
        0.5 < empirical / reported < 2.0
    ), f"reported error {reported:.3e} does not describe the spread {empirical:.3e}"
    assert abs(means.mean() - reference[0]) < 3.0 * empirical / np.sqrt(len(means))

    # One batch of four per sample, scaled by four. Unbiased, but noisier, so it is only the mean
    # that has to agree.
    batched, batched_sigmas = sample_means(num_batches_stoch=4, batches_per_sample=1)
    spread = np.sqrt(
        empirical**2 / len(means) + batched.std(ddof=1) ** 2 / len(batched)
    )
    assert abs(batched.mean() - means.mean()) < 3.0 * spread
    assert batched_sigmas.mean() > reported


def test_semistoch_rejects_inconsistent_input():
    """The thresholds and the variational correction are silent failures if left unchecked."""
    with pytest.raises(ValueError, match="pt2_threshold_pseudostoch"):
        SelectedCIParams(
            pt2_algorithm="semistochastic",
            pt2_threshold=1e-6,
            pt2_threshold_pseudostoch=1e-8,
            pt2_threshold_determ=1e-4,
        )
    with pytest.raises(ValueError, match="not linear in the square"):
        SelectedCIParams(
            pt2_algorithm="semistochastic", energy_correction="variational"
        )
    with pytest.raises(ValueError, match="at least 2 to"):
        SelectedCIParams(pt2_algorithm="semistochastic", pt2_num_samples=1)
    with pytest.raises(ValueError, match="pt2_min_batches_pseudostoch"):
        SelectedCIParams(pt2_algorithm="semistochastic", pt2_min_batches_pseudostoch=1)
    with pytest.raises(ValueError, match="pt2_batches_per_sample"):
        SelectedCIParams(
            pt2_algorithm="semistochastic",
            pt2_num_batches_stoch=2,
            pt2_batches_per_sample=3,
        )
