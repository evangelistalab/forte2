#include <algorithm>
#include <atomic>
#include <cmath>
#include <future>
#include <numeric>

#include "helpers/logger.h"
#include "helpers/random.hpp"
#include "helpers/timer.hpp"

#include "sci_helper.h"
#include "sci_helper_contributions.hpp"

namespace forte2 {

namespace {

/// @brief The sample mean and standard deviation of a set of per-root observations
/// @param values The observations, laid out as values[k * nroots + r]
/// @param count The number of observations
///
/// The standard deviation is zero for a single observation, which cannot support an estimate.
void mean_and_stddev(std::span<const double> values, size_t count, size_t nroots,
                     std::span<double> mean, std::span<double> stddev) {
    std::fill(mean.begin(), mean.end(), 0.0);
    std::fill(stddev.begin(), stddev.end(), 0.0);
    if (count == 0)
        return;
    for (size_t k{0}; k < count; ++k)
        for (size_t r{0}; r < nroots; ++r)
            mean[r] += values[k * nroots + r];
    for (size_t r{0}; r < nroots; ++r)
        mean[r] /= static_cast<double>(count);
    if (count < 2)
        return;
    for (size_t k{0}; k < count; ++k) {
        for (size_t r{0}; r < nroots; ++r) {
            const double d = values[k * nroots + r] - mean[r];
            stddev[r] += d * d;
        }
    }
    for (size_t r{0}; r < nroots; ++r)
        stddev[r] = std::sqrt(stddev[r] / static_cast<double>(count - 1));
}

} // namespace

double SelectedCIHelper::pt2_denominator(double delta) const {
    if (pt2_regularizer_ == PT2Regularizer::Shift)
        return 1.0 / (delta + pt2_regularizer_strength_);
    if (pt2_regularizer_ == PT2Regularizer::DSRG)
        return regularized_denominator(delta, pt2_regularizer_strength_);
    return 1.0 / delta;
}

void SelectedCIHelper::pt2_difference_batch(Pt2Scratch& s, double eps_tight, double eps_loose,
                                            size_t num_batches, size_t batch_id,
                                            const DetSet& existing_dets, std::span<double> out,
                                            size_t& num_dets) const {
    auto& map = s.map;
    auto& slots = s.slots;
    map.clear();
    slots.clear();
    const size_t stride = 2 * nroots_;

    generate_contributions(
        s.conn, eps_tight, num_batches, batch_id, std::span<const uint64_t>{},
        [&](const Determinant& det, const double* c_parent, size_t /*det_index*/, double coupling,
            double criterion) {
            if (criterion <= eps_tight)
                return;
            if (existing_dets.count(det))
                return;
            auto [it, emplaced] = map.try_emplace(det, slots.size());
            if (emplaced)
                slots.resize(slots.size() + stride, 0.0);
            const size_t idx = it->second;
            // repeating the tight channel's own test at the looser threshold is what makes the
            // two channels agree bit for bit when the thresholds are equal
            const bool loose = criterion > eps_loose;
            for (size_t r{0}; r < nroots_; ++r) {
                const double v = coupling * c_parent[r];
                slots[idx + r] += v;
                if (loose)
                    slots[idx + nroots_ + r] += v;
            }
        });

    for (const auto& [det, idx] : map) {
        const double energy = slater_rules_.energy(det);
        for (size_t r{0}; r < nroots_; ++r) {
            const double delta = root_energies_[r] - energy;
            out[r] += compute_delta_ept2(delta, slots[idx + r]) -
                      compute_delta_ept2(delta, slots[idx + nroots_ + r]);
        }
    }
    num_dets += map.size();
}

void SelectedCIHelper::pt2_stoch_batch(Pt2Scratch& s, double eps_tight, double eps_loose,
                                       size_t num_batches, size_t batch_id,
                                       const DetSet& existing_dets,
                                       std::span<const uint64_t> parent_mask,
                                       const PT2WeightMap& weights, std::span<double> out) const {
    auto& map = s.map;
    auto& slots = s.slots;
    map.clear();
    slots.clear();
    const size_t stride = 4 * nroots_;

    generate_contributions(
        s.conn, eps_tight, num_batches, batch_id, parent_mask,
        [&](const Determinant& det, const double* c_parent, size_t det_index, double coupling,
            double criterion) {
            if (criterion <= eps_tight)
                return;
            if (existing_dets.count(det))
                return;
            // the mask admits only sampled parents, so the lookup always succeeds
            const auto& [xw, yw] = weights.find(det_index)->second;
            auto [it, emplaced] = map.try_emplace(det, slots.size());
            if (emplaced)
                slots.resize(slots.size() + stride, 0.0);
            const size_t idx = it->second;
            const bool loose = criterion > eps_loose;
            for (size_t r{0}; r < nroots_; ++r) {
                const double v = coupling * c_parent[r];
                const double vv = v * v;
                slots[idx + r] += xw * v;
                slots[idx + nroots_ + r] += yw * vv;
                if (loose) {
                    slots[idx + 2 * nroots_ + r] += xw * v;
                    slots[idx + 3 * nroots_ + r] += yw * vv;
                }
            }
        });

    for (const auto& [det, idx] : map) {
        const double energy = slater_rules_.energy(det);
        for (size_t r{0}; r < nroots_; ++r) {
            const double xt = slots[idx + r];
            const double yt = slots[idx + nroots_ + r];
            const double xl = slots[idx + 2 * nroots_ + r];
            const double yl = slots[idx + 3 * nroots_ + r];
            // the two channels are paired term by term so that identical channels cancel exactly;
            // grouping this as (xt * xt + yt) - (xl * xl + yl) would not, because the intermediate
            // sums round
            out[r] +=
                ((xt * xt - xl * xl) + (yt - yl)) * pt2_denominator(root_energies_[r] - energy);
        }
    }
}

void SelectedCIHelper::compute_pt2_semistoch(double eps2, double eps2_pseudostoch,
                                             double eps2_determ, size_t num_batches,
                                             size_t min_batches_pseudostoch, double target_error,
                                             size_t num_batches_stoch, size_t batches_per_sample,
                                             size_t num_samples, size_t sample_size,
                                             uint64_t seed) {
    if (num_batches == 0 or num_batches_stoch == 0)
        throw std::invalid_argument("compute_pt2_semistoch requires positive batch counts");
    if (batches_per_sample == 0 or batches_per_sample > num_batches_stoch)
        throw std::invalid_argument(
            "compute_pt2_semistoch requires 0 < batches_per_sample <= num_batches_stoch");
    if (num_samples == 1)
        throw std::invalid_argument("compute_pt2_semistoch needs at least two samples to "
                                    "estimate an error, or none to skip the stochastic step");
    if (num_samples > 0 and sample_size < 2)
        throw std::invalid_argument("compute_pt2_semistoch requires sample_size >= 2");
    if (eps2 > eps2_pseudostoch or eps2_pseudostoch > eps2_determ)
        throw std::invalid_argument(
            "compute_pt2_semistoch requires eps2 <= eps2_pseudostoch <= eps2_determ");
    if (num_samples > 0 and energy_correction_ == EnergyCorrection::Variational)
        throw std::invalid_argument("The variational energy correction is not linear in the square "
                                    "of the coupling, so the stochastic step would be biased");

    local_timer pt2_timer;

    compute_det_energies();
    prepare_strings();
    update_hbci_ints();

    // Built once from the whole variational space and shared by every step and sample. Building it
    // from a sample instead would admit determinants that are variational but unsampled, whose
    // small energy denominators would introduce a large sample-size-dependent bias.
    const DetSet existing_dets(dets_.begin(), dets_.end());

    const size_t ndets = dets_.size();
    const auto num_threads = get_num_threads();

    ept2_determ_.assign(nroots_, 0.0);
    ept2_pseudostoch_.assign(nroots_, 0.0);
    ept2_stoch_.assign(nroots_, 0.0);
    ept2_pseudostoch_stddev_.assign(nroots_, 0.0);
    ept2_stoch_stddev_.assign(nroots_, 0.0);
    num_pt2_dets_ = 0;

    // == Step 1: the correction down to eps2_determ, evaluated exactly ==
    {
        std::atomic<size_t> next_batch(0);
        std::vector<std::vector<double>> local(num_threads, std::vector<double>(nroots_, 0.0));
        std::vector<size_t> local_dets(num_threads, 0);
        std::vector<std::future<void>> workers;
        for (size_t t{0}; t < num_threads; ++t) {
            workers.push_back(std::async(std::launch::async, [&, t] {
                Pt2Scratch s;
                while (true) {
                    const size_t batch_id = next_batch.fetch_add(1);
                    if (batch_id >= num_batches)
                        break;
                    pt2_determ_batch(s, eps2_determ, num_batches, batch_id, existing_dets, local[t],
                                     local_dets[t]);
                }
            }));
        }
        for (auto& w : workers)
            w.get();
        for (size_t t{0}; t < num_threads; ++t) {
            for (size_t r{0}; r < nroots_; ++r)
                ept2_determ_[r] += local[t][r];
            num_pt2_dets_ += local_dets[t];
        }
    }

    // == Step 2: the gain from eps2_determ down to eps2_pseudostoch, from a subset of batches ==
    //
    // The sampling unit is the batch rather than the connected determinant: contributions within a
    // batch share parents and are correlated, and the per-determinant contribution is heavy-tailed,
    // so a per-batch sum is much better behaved than an individual term.
    {
        std::vector<double> batch_sums;
        batch_sums.reserve(num_batches * nroots_);
        std::vector<double> mean(nroots_), sd(nroots_);
        size_t evaluated = 0;
        // A single batch has no spread to measure, and stopping on the zero it would report would
        // present a one-batch extrapolation as if it were exact. Li et al. can stop after one batch
        // because they take their statistics over individual perturbative determinants, of which
        // one batch holds many; the per-batch sums used here need at least two.
        const size_t min_evaluated = std::max<size_t>(min_batches_pseudostoch, 2);
        // Batches are evaluated a round at a time and the rule can only be checked between rounds,
        // so the check points are fixed here rather than left to fall wherever the rounds land.
        // Rounds are still at most one batch per thread, but they are cut short to stop exactly on
        // a check point, which keeps the number of batches evaluated -- and therefore the energy --
        // the same on every machine. Letting the rounds run free would tie the check points to the
        // thread count, which makes min_batches_pseudostoch inert below it and stops the rule
        // firing at all once the thread count reaches the batch count.
        size_t next_check = min_evaluated;

        while (evaluated < num_batches) {
            size_t round = std::min(num_threads, num_batches - evaluated);
            if (evaluated < next_check)
                round = std::min(round, next_check - evaluated);
            std::vector<std::vector<double>> local(round, std::vector<double>(nroots_, 0.0));
            std::vector<size_t> local_dets(round, 0);
            std::vector<std::future<void>> workers;
            for (size_t t{0}; t < round; ++t) {
                workers.push_back(std::async(std::launch::async, [&, t] {
                    Pt2Scratch s;
                    pt2_difference_batch(s, eps2_pseudostoch, eps2_determ, num_batches,
                                         evaluated + t, existing_dets, local[t], local_dets[t]);
                }));
            }
            for (auto& w : workers)
                w.get();
            for (auto& s : local)
                batch_sums.insert(batch_sums.end(), s.begin(), s.end());
            evaluated += round;

            if (evaluated == next_check and evaluated < num_batches) {
                mean_and_stddev(batch_sums, evaluated, nroots_, mean, sd);
                const double fraction = static_cast<double>(evaluated) /
                                        static_cast<double>(num_batches);
                double worst = 0.0;
                for (size_t r{0}; r < nroots_; ++r) {
                    // the finite-population correction takes the error to zero as the subset
                    // approaches the whole set, which it must since that limit is exact
                    const double err = static_cast<double>(num_batches) * sd[r] /
                                       std::sqrt(static_cast<double>(evaluated)) *
                                       std::sqrt(1.0 - fraction);
                    worst = std::max(worst, std::abs(err));
                }
                // step 3 spends the rest of the error budget, so stop well inside it
                if (worst < 0.4 * target_error)
                    break;
                // the error falls as the square root of the batches evaluated, so doubling keeps
                // the number of checks logarithmic without overshooting the target by much
                next_check = std::min(2 * next_check, num_batches);
            }
        }

        mean_and_stddev(batch_sums, evaluated, nroots_, mean, sd);
        const double fraction = static_cast<double>(evaluated) / static_cast<double>(num_batches);
        for (size_t r{0}; r < nroots_; ++r) {
            ept2_pseudostoch_[r] = static_cast<double>(num_batches) * mean[r];
            ept2_pseudostoch_stddev_[r] = static_cast<double>(num_batches) * sd[r] /
                                   std::sqrt(static_cast<double>(evaluated)) *
                                   std::sqrt(1.0 - fraction);
        }
        num_pseudostoch_batches_ = evaluated;
    }

    // == Step 3: the gain from eps2_pseudostoch down to eps2, from samples of the
    // variational space ==
    if (num_samples > 0) {
        // sampling proportional to the norm of a determinant's coefficients across roots lets one
        // sample serve every root at once; the estimator is unbiased for any positive probability
        std::vector<double> sampling_weights(ndets, 0.0);
        for (size_t i{0}; i < ndets; ++i) {
            double norm2 = 0.0;
            for (size_t r{0}; r < nroots_; ++r) {
                const double c = c_[i * nroots_ + r];
                norm2 += c * c;
            }
            sampling_weights[i] = std::sqrt(norm2);
        }
        const AliasTable alias{std::span<const double>(sampling_weights)};

        const size_t mask_words = (ndets + 63) / 64;
        const double norm = static_cast<double>(num_batches_stoch) /
                            (static_cast<double>(batches_per_sample) *
                             static_cast<double>(sample_size) *
                             static_cast<double>(sample_size - 1));

        std::vector<double> estimates(num_samples * nroots_, 0.0);
        std::atomic<size_t> next_sample(0);
        std::vector<std::future<void>> workers;
        for (size_t t{0}; t < num_threads; ++t) {
            workers.push_back(std::async(std::launch::async, [&] {
                std::vector<uint64_t> mask(mask_words);
                PT2WeightMap weights;
                Pt2Scratch scratch;
                // the pool a partial Fisher-Yates shuffle draws the batch subset from
                std::vector<size_t> batch_pool(num_batches_stoch);

                while (true) {
                    const size_t s = next_sample.fetch_add(1);
                    if (s >= num_samples)
                        break;
                    // deriving each stream from the sample index rather than the thread keeps the
                    // result independent of how the samples happen to be scheduled
                    Rng rng(seed ^ splitmix64(s + 1));

                    std::fill(mask.begin(), mask.end(), 0ULL);
                    weights.clear();
                    for (size_t d{0}; d < sample_size; ++d) {
                        const size_t i = alias.sample(rng);
                        weights[i].first += 1.0;
                        mask[i >> 6] |= 1ULL << (i & 63);
                    }
                    for (auto& entry : weights) {
                        const double wp = entry.second.first / alias.probability(entry.first);
                        entry.second.first = wp;
                        entry.second.second =
                            wp * static_cast<double>(sample_size - 1) - wp * wp;
                    }

                    // Restoring the identity permutation first is what keeps the subset a function
                    // of the sample index alone. Carrying the shuffled pool over to the next
                    // sample would make the draw depend on which samples this thread happened to
                    // handle before it.
                    std::iota(batch_pool.begin(), batch_pool.end(), 0);
                    for (size_t j{0}; j < batches_per_sample; ++j)
                        std::swap(batch_pool[j], batch_pool[j + rng.below(num_batches_stoch - j)]);

                    std::span<double> out(estimates.data() + s * nroots_, nroots_);
                    for (size_t j{0}; j < batches_per_sample; ++j) {
                        pt2_stoch_batch(scratch, eps2, eps2_pseudostoch, num_batches_stoch,
                                        batch_pool[j], existing_dets,
                                        std::span<const uint64_t>(mask), weights, out);
                    }
                    for (size_t r{0}; r < nroots_; ++r)
                        out[r] *= norm;
                }
            }));
        }
        for (auto& w : workers)
            w.get();

        mean_and_stddev(estimates, num_samples, nroots_, ept2_stoch_, ept2_stoch_stddev_);
        for (size_t r{0}; r < nroots_; ++r)
            ept2_stoch_stddev_[r] /= std::sqrt(static_cast<double>(num_samples));
    }

    ept2_.assign(nroots_, 0.0);
    ept2_stddev_.assign(nroots_, 0.0);
    for (size_t r{0}; r < nroots_; ++r) {
        ept2_[r] = ept2_determ_[r] + ept2_pseudostoch_[r] + ept2_stoch_[r];
        // the two estimated terms use independent randomness, so their variances add
        ept2_stddev_[r] = std::sqrt(ept2_pseudostoch_stddev_[r] * ept2_pseudostoch_stddev_[r] +
                                    ept2_stoch_stddev_[r] * ept2_stoch_stddev_[r]);
    }

    pt2_time_ = pt2_timer.elapsed_seconds();
    LOG(log_level_) << "Semistochastic PT2 with eps2 = " << eps2 << ": " << num_pt2_dets_
                    << " determinants deterministically, " << num_pseudostoch_batches_ << " of "
                    << num_batches << " batches pseudo-stochastically, " << num_samples
                    << " samples of " << sample_size << " determinants, in " << pt2_time_ << " s";
}

} // namespace forte2
