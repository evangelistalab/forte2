#pragma once

#include "sparse/sparse_normal_order.h"

namespace forte2 {

/// Compute commutators while fusing bare multiplication, generalized normal ordering, and rank
/// truncation. The full intermediate bare commutator is never materialized.
class GeneralizedNormalOrderedProductComputer {
  public:
    GeneralizedNormalOrderedProductComputer(int max_rank, double screen_thresh = 1.0e-12);

    GeneralizedNormalOrderedSparseOperator
    commutator(const GeneralizedNormalOrderedSparseOperator& lhs,
               const GeneralizedNormalOrderedSparseOperator& rhs) const;

  private:
    int max_rank_;
    double screen_thresh_;
};

GeneralizedNormalOrderedSparseOperator
generalized_normal_ordered_commutator(const GeneralizedNormalOrderedSparseOperator& lhs,
                                      const GeneralizedNormalOrderedSparseOperator& rhs,
                                      int max_rank, double screen_thresh = 1.0e-12);

} // namespace forte2
