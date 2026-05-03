#include "sparse/sparse_normal_order_product.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <stdexcept>

#include "sparse/sq_operator_string.h"

namespace forte2 {

namespace {

SQOperatorString as_quasiparticle_string(const NormalOrderedString& term) {
    return SQOperatorString(term.cre(), term.ann());
}

NormalOrderedString as_normal_ordered_string(const SQOperatorString& term) {
    return NormalOrderedString(term.cre(), term.ann());
}

NormalOrderedSparseOperator clean_normal_ordered_operator(const NormalOrderedSparseOperator& op,
                                                          int max_rank, double screen_thresh) {
    NormalOrderedSparseOperator cleaned(op.reference());
    cleaned.reserve(op.size());
    for (const auto& [term, coefficient] : op.elements()) {
        if ((max_rank < 0 or term.count() <= 2 * max_rank) and
            std::abs(coefficient) > screen_thresh) {
            cleaned.add(term, coefficient);
        }
    }
    return cleaned;
}

} // namespace

NormalOrderedProductComputer::NormalOrderedProductComputer(int max_rank, double screen_thresh)
    : max_rank_(max_rank), screen_thresh_(screen_thresh) {
    if (max_rank < 0) {
        throw std::invalid_argument("NormalOrderedProductComputer: max_rank must be non-negative");
    }
    if (screen_thresh < 0.0) {
        throw std::invalid_argument("NormalOrderedProductComputer: screen_thresh must be non-negative");
    }
}

bool NormalOrderedProductComputer::could_contribute(const NormalOrderedString& lhs,
                                                    const NormalOrderedString& rhs) const {
    const int total_count = lhs.count() + rhs.count();
    if (total_count <= 2 * max_rank_) {
        return true;
    }

    return could_product_contribute(lhs, rhs) or could_product_contribute(rhs, lhs);
}

bool NormalOrderedProductComputer::could_product_contribute(const NormalOrderedString& lhs,
                                                            const NormalOrderedString& rhs) const {
    const int total_count = lhs.count() + rhs.count();
    if (total_count <= 2 * max_rank_) {
        return true;
    }

    const int max_contractions = lhs.ann().fast_a_and_b_count(rhs.cre());
    const int lower_bound_count = total_count - 2 * max_contractions;
    return lower_bound_count <= 2 * max_rank_;
}

NormalOrderedSparseOperator
NormalOrderedProductComputer::commutator(const NormalOrderedSparseOperator& lhs,
                                         const NormalOrderedSparseOperator& rhs) const {
    if (lhs.reference() != rhs.reference()) {
        throw std::invalid_argument(
            "NormalOrderedProductComputer::commutator: references must match");
    }

    SQOperatorProductComputer computer;
    NormalOrderedSparseOperator result(lhs.reference());
    result.reserve(std::min(lhs.size() * rhs.size(), std::size_t{250000}));

    const std::function<void(const SQOperatorString&, const sparse_scalar_t)> add_to_result =
        [this, &result](const SQOperatorString& term, const sparse_scalar_t coefficient) {
            if (term.count() <= 2 * max_rank_ and std::abs(coefficient) > screen_thresh_) {
                result.add(as_normal_ordered_string(term), coefficient);
            }
        };

    for (const auto& [lhs_term, lhs_coefficient] : lhs.elements()) {
        const auto lhs_qp = as_quasiparticle_string(lhs_term);
        for (const auto& [rhs_term, rhs_coefficient] : rhs.elements()) {
            const sparse_scalar_t factor = lhs_coefficient * rhs_coefficient;
            if (std::abs(factor) < screen_thresh_) {
                continue;
            }
            const bool lhs_rhs_contributes = could_product_contribute(lhs_term, rhs_term);
            const bool rhs_lhs_contributes = could_product_contribute(rhs_term, lhs_term);
            if (not lhs_rhs_contributes and not rhs_lhs_contributes) {
                continue;
            }
            const auto rhs_qp = as_quasiparticle_string(rhs_term);
            if (do_ops_commute(lhs_qp, rhs_qp)) {
                continue;
            }
            if (lhs_rhs_contributes) {
                computer.product(lhs_qp, rhs_qp, factor, add_to_result);
            }
            if (rhs_lhs_contributes) {
                computer.product(rhs_qp, lhs_qp, -factor, add_to_result);
            }
        }
    }

    return clean_normal_ordered_operator(result, max_rank_, screen_thresh_);
}

NormalOrderedSparseOperator adjoint(const NormalOrderedSparseOperator& op, double screen_thresh) {
    if (screen_thresh < 0.0) {
        throw std::invalid_argument("adjoint: screen_thresh must be non-negative");
    }

    NormalOrderedSparseOperator result(op.reference());
    result.reserve(op.size());
    for (const auto& [term, coefficient] : op.elements()) {
        if (std::abs(coefficient) > screen_thresh) {
            result.add(NormalOrderedString(term.ann(), term.cre()), std::conj(coefficient));
        }
    }
    return result;
}

NormalOrderedSparseOperator normal_ordered_commutator(const NormalOrderedSparseOperator& lhs,
                                                      const NormalOrderedSparseOperator& rhs,
                                                      int max_rank, double screen_thresh) {
    return NormalOrderedProductComputer(max_rank, screen_thresh).commutator(lhs, rhs);
}

} // namespace forte2
