#pragma once

#include <optional>

#include "sparse/cumulant_reference.h"
#include "sparse/sparse_normal_order.h"

namespace forte2 {

/// Compute commutators while fusing bare multiplication, generalized normal ordering, and rank
/// truncation. The full intermediate bare commutator is never materialized.
class GeneralizedNormalOrderedProductComputer {
  public:
    GeneralizedNormalOrderedProductComputer(int max_rank, double screen_thresh = 1.0e-12);
    GeneralizedNormalOrderedProductComputer(const CumulantReference& reference, int max_rank,
                                            double screen_thresh = 1.0e-12);

    /// @return Whether higher moments are reconstructed from a truncated cumulant hierarchy.
    bool uses_cumulant_truncation() const;

    GeneralizedNormalOrderedSparseOperator
    commutator(const GeneralizedNormalOrderedSparseOperator& lhs,
               const GeneralizedNormalOrderedSparseOperator& rhs) const;

  private:
    int max_rank_;
    double screen_thresh_;
    std::optional<CumulantReference> reference_;
};

GeneralizedNormalOrderedSparseOperator
generalized_normal_ordered_commutator(const GeneralizedNormalOrderedSparseOperator& lhs,
                                      const GeneralizedNormalOrderedSparseOperator& rhs,
                                      int max_rank, double screen_thresh = 1.0e-12);

GeneralizedNormalOrderedSparseOperator cumulant_truncated_generalized_normal_ordered_commutator(
    const GeneralizedNormalOrderedSparseOperator& lhs,
    const GeneralizedNormalOrderedSparseOperator& rhs, int max_rank,
    double screen_thresh = 1.0e-12);

} // namespace forte2
