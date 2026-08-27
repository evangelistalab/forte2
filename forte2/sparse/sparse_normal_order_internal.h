#pragma once

#include <memory>

#include "sparse/sparse_normal_order.h"

namespace forte2 {

/// Reusable generalized-normal-ordering state for streaming bare operator terms.
class GeneralizedNormalOrderComputer {
  public:
    GeneralizedNormalOrderComputer(const SparseState& vacuum, std::size_t norb, int max_cumulant,
                                   double screen_thresh, int max_rank);
    ~GeneralizedNormalOrderComputer();

    GeneralizedNormalOrderComputer(const GeneralizedNormalOrderComputer&) = delete;
    GeneralizedNormalOrderComputer& operator=(const GeneralizedNormalOrderComputer&) = delete;
    GeneralizedNormalOrderComputer(GeneralizedNormalOrderComputer&&) noexcept;
    GeneralizedNormalOrderComputer& operator=(GeneralizedNormalOrderComputer&&) noexcept;

    /// Add the generalized-normal-order expansion of one bare term to result.
    void add_term(const SQOperatorString& term, sparse_scalar_t coefficient,
                  GeneralizedNormalOrderedSparseOperator& result);

    /// Return whether a bare term can produce any retained generalized-normal-ordered term.
    bool could_contribute(const SQOperatorString& term) const;

    /// Apply final rank and coefficient screening to an accumulated result.
    GeneralizedNormalOrderedSparseOperator
    clean(const GeneralizedNormalOrderedSparseOperator& result) const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace forte2
