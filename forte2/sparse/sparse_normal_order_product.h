#pragma once

#include "sparse/sparse_normal_order.h"

namespace forte2 {

/// @brief Computes products directly in determinant-normal-ordered quasiparticle form.
class NormalOrderedProductComputer {
  public:
    explicit NormalOrderedProductComputer(int max_rank, double screen_thresh = 1.0e-12);

    /// @return True if this pair can possibly contribute through the retained rank.
    bool could_contribute(const NormalOrderedString& lhs, const NormalOrderedString& rhs) const;

    /// @return The normal-ordered commutator truncated to max_rank.
    NormalOrderedSparseOperator commutator(const NormalOrderedSparseOperator& lhs,
                                           const NormalOrderedSparseOperator& rhs) const;

  private:
    bool could_product_contribute(const NormalOrderedString& lhs,
                                  const NormalOrderedString& rhs) const;

    int max_rank_;
    double screen_thresh_;
};

/// @return The adjoint of a normal-ordered sparse operator.
NormalOrderedSparseOperator adjoint(const NormalOrderedSparseOperator& op,
                                    double screen_thresh = 1.0e-12);

/// @return The normal-ordered commutator truncated to max_rank.
NormalOrderedSparseOperator normal_ordered_commutator(const NormalOrderedSparseOperator& lhs,
                                                      const NormalOrderedSparseOperator& rhs,
                                                      int max_rank, double screen_thresh = 1.0e-12);

} // namespace forte2
