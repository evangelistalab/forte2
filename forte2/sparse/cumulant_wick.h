#pragma once

#include "sparse/cumulant_reference.h"
#include "sparse/sparse_normal_order.h"

namespace forte2 {

/// @brief Direct generalized Wick products using explicit spin-orbital density cumulants.
///
/// This engine is independent of GeneralizedNormalOrderedProductComputer. It does not convert its
/// operands to bare SparseOperator objects and supports input terms and density cumulants through
/// rank four.
class CumulantWickEngine {
  public:
    CumulantWickEngine(const CumulantReference& reference, int max_rank,
                       double screen_thresh = 1.0e-12);

    /// @return The explicit cumulant reference used by this engine.
    const CumulantReference& reference() const;

    /// @return The maximum retained many-body rank, or -1 for no output-rank truncation.
    int max_rank() const;

    /// @return The numerical screening threshold.
    double screen_thresh() const;

    /// Compute the generalized-normal-ordered product lhs * rhs, treating unavailable cumulants as
    /// zero.
    GeneralizedNormalOrderedSparseOperator
    product(const GeneralizedNormalOrderedSparseOperator& lhs,
            const GeneralizedNormalOrderedSparseOperator& rhs) const;

    /// Compute the generalized-normal-ordered commutator [lhs, rhs].
    GeneralizedNormalOrderedSparseOperator
    commutator(const GeneralizedNormalOrderedSparseOperator& lhs,
               const GeneralizedNormalOrderedSparseOperator& rhs) const;

  private:
    void validate_operand(const GeneralizedNormalOrderedSparseOperator& op) const;
    void add_product(const GeneralizedNormalOrderedSparseOperator& lhs,
                     const GeneralizedNormalOrderedSparseOperator& rhs, sparse_scalar_t factor,
                     GeneralizedNormalOrderedSparseOperator& result) const;

    CumulantReference reference_;
    int max_rank_ = -1;
    double screen_thresh_ = 1.0e-12;
};

} // namespace forte2
