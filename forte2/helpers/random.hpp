#pragma once

#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

namespace forte2 {

/// @brief The splitmix64 finalizer/mixer: spreads the bits of a 64-bit integer
/// @param x The input value to mix
/// @return The mixed value
inline uint64_t splitmix64(uint64_t x) {
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    x = x ^ (x >> 31);
    return x;
}

/// @brief Advance a splitmix64 state and return the next value
/// @param state The generator state, advanced in place
///
/// Used to seed Rng and to derive independent streams from a single user-supplied seed.
inline uint64_t splitmix64_next(uint64_t& state) {
    return splitmix64(state += 0x9e3779b97f4a7c15ULL);
}

/// @brief A xoshiro256++ pseudorandom generator
///
/// The standard library distribution classes are deliberately avoided here: the C++ standard
/// specifies their interface but not their algorithm, so libstdc++ and libc++ produce different
/// sequences from the same seed and any seeded test would be platform dependent.
class Rng {
  public:
    /// @brief Construct a generator from a seed
    explicit Rng(uint64_t seed) {
        uint64_t state = seed;
        for (auto& s : s_)
            s = splitmix64_next(state);
    }

    /// @return The next 64-bit value
    uint64_t next() {
        const uint64_t result = rotl(s_[0] + s_[3], 23) + s_[0];
        const uint64_t t = s_[1] << 17;
        s_[2] ^= s_[0];
        s_[3] ^= s_[1];
        s_[1] ^= s_[2];
        s_[0] ^= s_[3];
        s_[2] ^= t;
        s_[3] = rotl(s_[3], 45);
        return result;
    }

    /// @return A double uniformly distributed in [0, 1)
    double uniform() { return static_cast<double>(next() >> 11) * 0x1.0p-53; }

    /// @return An integer uniformly distributed in [0, n), without modulo bias
    size_t below(size_t n) {
        if (n == 0)
            throw std::invalid_argument("Rng::below requires n > 0");
        const uint64_t range = static_cast<uint64_t>(n);
        // (-range) % range is 2^64 mod range, the size of the biased tail to reject
        const uint64_t reject_below = (0ULL - range) % range;
        uint64_t x;
        do {
            x = next();
        } while (x < reject_below);
        return static_cast<size_t>(x % range);
    }

  private:
    static uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

    uint64_t s_[4];
};

/// @brief A Walker alias table for sampling a discrete distribution in constant time
///
/// Construction is linear in the number of outcomes and sampling is O(1), which is what makes it
/// practical to draw repeatedly from a variational space with millions of determinants.
class AliasTable {
  public:
    /// @brief Build the table from unnormalized, non-negative weights
    /// @param weights The weights, at least one of which must be positive
    ///
    /// Zero weights are allowed and are never sampled.
    explicit AliasTable(std::span<const double> weights)
        : prob_(weights.size()), alias_(weights.size()), p_(weights.size()) {
        const size_t n = weights.size();
        if (n == 0)
            throw std::invalid_argument("AliasTable requires at least one weight");

        double total = 0.0;
        for (const double w : weights) {
            if (w < 0.0)
                throw std::invalid_argument("AliasTable requires non-negative weights");
            total += w;
        }
        if (total <= 0.0)
            throw std::invalid_argument("AliasTable requires at least one positive weight");

        // scaled[i] is n * p_i, so the average is exactly one
        std::vector<double> scaled(n);
        for (size_t i{0}; i < n; ++i) {
            p_[i] = weights[i] / total;
            scaled[i] = p_[i] * static_cast<double>(n);
        }

        std::vector<size_t> small, large;
        small.reserve(n);
        large.reserve(n);
        for (size_t i{0}; i < n; ++i) {
            (scaled[i] < 1.0 ? small : large).push_back(i);
        }

        while (!small.empty() && !large.empty()) {
            const size_t s = small.back();
            small.pop_back();
            const size_t l = large.back();
            large.pop_back();
            prob_[s] = scaled[s];
            alias_[s] = l;
            // move the mass that s did not use back onto l
            scaled[l] -= 1.0 - scaled[s];
            (scaled[l] < 1.0 ? small : large).push_back(l);
        }
        // whatever remains is within rounding of a full bucket
        for (const size_t i : large)
            prob_[i] = 1.0;
        for (const size_t i : small)
            prob_[i] = 1.0;
    }

    /// @return An index drawn with probability proportional to its weight
    size_t sample(Rng& rng) const {
        const size_t i = rng.below(prob_.size());
        // prob_[i] is zero for a zero-weight outcome, so it is never returned here
        return rng.uniform() < prob_[i] ? i : alias_[i];
    }

    /// @return The normalized probability of outcome i
    double probability(size_t i) const { return p_[i]; }

    /// @return The number of outcomes
    size_t size() const { return prob_.size(); }

  private:
    /// @brief Probability of keeping the bucket index rather than following the alias
    std::vector<double> prob_;
    /// @brief The outcome to fall back to for each bucket
    std::vector<size_t> alias_;
    /// @brief Normalized probabilities, needed by the importance-sampling estimator
    std::vector<double> p_;
};

} // namespace forte2
