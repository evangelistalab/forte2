#include <algorithm>
#include <cmath>
#include <functional>
#include <stdexcept>

#include "sparse/sparse_operator_product.h"

namespace forte2 {

RankScreenedProductComputer::RankScreenedProductComputer(int max_rank, double screen_thresh)
    : max_rank_(max_rank), screen_thresh_(screen_thresh) {
    if (max_rank < 0) {
        throw std::invalid_argument("RankScreenedProductComputer: max_rank must be non-negative");
    }
    if (screen_thresh < 0.0) {
        throw std::invalid_argument(
            "RankScreenedProductComputer: screen_thresh must be non-negative");
    }
}

bool RankScreenedProductComputer::could_contribute(const SQOperatorString& lhs,
                                                   const SQOperatorString& rhs) const {
    const int total_count = lhs.count() + rhs.count();
    if (total_count <= 2 * max_rank_) {
        return true;
    }

    const auto cre_union = lhs.cre() | rhs.cre();
    const auto ann_union = lhs.ann() | rhs.ann();
    const int max_same_mode_contractions = cre_union.intersection_count(ann_union);
    const int lower_bound_count = total_count - 2 * max_same_mode_contractions;
    return lower_bound_count <= 2 * max_rank_;
}

SparseOperator RankScreenedProductComputer::commutator(const SparseOperator& lhs,
                                                       const SparseOperator& rhs) const {
    SQOperatorProductComputer computer;
    SparseOperator result;
    result.reserve(std::min(lhs.size() * rhs.size(), std::size_t{250000}));
    const std::function<void(const SQOperatorString&, const sparse_scalar_t)> add_to_result =
        [&result](const SQOperatorString& sqop, const sparse_scalar_t c) { result[sqop] += c; };

    for (const auto& [lhs_op, lhs_c] : lhs.elements()) {
        for (const auto& [rhs_op, rhs_c] : rhs.elements()) {
            const sparse_scalar_t factor = lhs_c * rhs_c;
            if (std::abs(factor) < screen_thresh_) {
                continue;
            }
            if (do_ops_commute(lhs_op, rhs_op)) {
                continue;
            }
            if (not could_contribute(lhs_op, rhs_op)) {
                continue;
            }
            computer.commutator(lhs_op, rhs_op, factor, add_to_result);
        }
    }
    return result;
}

SparseOperator rank_screened_commutator(const SparseOperator& lhs, const SparseOperator& rhs,
                                        int max_rank, double screen_thresh) {
    return RankScreenedProductComputer(max_rank, screen_thresh).commutator(lhs, rhs);
}

} // namespace forte2
