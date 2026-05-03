#pragma once

#include "ci/determinant.h"
#include "sparse/sparse_operator.h"

namespace forte2 {

/// @brief Computes sparse operator products with cheap many-body rank screening.
///
/// This class is intended for algorithms that immediately discard products that cannot contribute
/// to a determinant-reference normal-ordered operator through a fixed many-body rank.  It does not
/// normal order the result itself; it only skips SQOperatorString pairs for which even the maximum
/// possible same-mode contractions cannot reduce the product to the requested rank.
class RankScreenedProductComputer {
  public:
    explicit RankScreenedProductComputer(int max_rank, double screen_thresh = 1.0e-12);

    /// @return True if the pair can possibly contribute to the retained normal rank.
    bool could_contribute(const SQOperatorString& lhs, const SQOperatorString& rhs) const;

    /// @return The commutator with impossible high-rank term pairs skipped.
    SparseOperator commutator(const SparseOperator& lhs, const SparseOperator& rhs) const;

  private:
    int max_rank_;
    double screen_thresh_;
};

/// @return The commutator with impossible high-rank term pairs skipped.
SparseOperator rank_screened_commutator(const SparseOperator& lhs, const SparseOperator& rhs,
                                        int max_rank, double screen_thresh = 1.0e-12);

} // namespace forte2
