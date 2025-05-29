#include <algorithm>
#include <numeric>
#include <cmath>

#include "ci/sparse_exp.h"

namespace forte2 {

size_t num_attempts_ = 0;
size_t num_success_ = 0;

SparseExp::SparseExp(int maxk, double screen_thresh) : maxk_(maxk), screen_thresh_(screen_thresh) {}

SparseState SparseExp::apply_op(const SparseOperator& sop, const SparseState& state,
                                double scaling_factor) {
    return apply_exp_operator(OperatorType::Excitation, sop, state, scaling_factor);
}

SparseState SparseExp::apply_op(const SparseOperatorList& sop_list, const SparseState& state,
                                double scaling_factor) {
    SparseOperator sop = sop_list.to_operator();
    return apply_op(sop, state, scaling_factor);
}

SparseState SparseExp::apply_antiherm(const SparseOperator& sop, const SparseState& state,
                                      double scaling_factor) {
    return apply_exp_operator(OperatorType::Antihermitian, sop, state, scaling_factor);
}

SparseState SparseExp::apply_antiherm(const SparseOperatorList& sop_list, const SparseState& state,
                                      double scaling_factor) {
    SparseOperator sop = sop_list.to_operator();
    return apply_antiherm(sop, state, scaling_factor);
}

SparseState SparseExp::apply_exp_operator(OperatorType op_type, const SparseOperator& sop,
                                          const SparseState& state, double scaling_factor) {
    SparseState exp_state(state);
    SparseState old_terms(state);
    SparseState new_terms;

    for (int k = 1; k <= maxk_; k++) {
        old_terms *= scaling_factor / static_cast<double>(k);
        if (op_type == OperatorType::Excitation) {
            new_terms = apply_operator_lin(sop, old_terms, screen_thresh_);
        } else if (op_type == OperatorType::Antihermitian) {
            new_terms = apply_operator_antiherm(sop, old_terms, screen_thresh_);
        }
        double norm = 0.0;
        double inf_norm = 0.0;
        exp_state += new_terms;
        for (const auto& [det, c] : new_terms) {
            norm += std::pow(std::abs(c), 2.0);
            inf_norm = std::max(inf_norm, std::abs(c));
        }
        norm = std::sqrt(norm);
        if (inf_norm < screen_thresh_) {
            break;
        }
        old_terms = new_terms;
    }
    return exp_state;
}

SparseFactExp::SparseFactExp(double screen_thresh) : screen_thresh_(screen_thresh) {}

SparseState SparseFactExp::apply_op(const SparseOperatorList& sop, const SparseState& state,
                                    bool inverse) {
    // initialize a state object
    SparseState result(state);

    // temporary space to store new elements. This avoids reallocation
    Buffer<std::pair<Determinant, sparse_scalar_t>> new_terms;

    Determinant new_det;
    Determinant sign_mask;
    Determinant idx;
    for (size_t m = 0, nterms = sop.size(); m < nterms; m++) {
        size_t n = inverse ? nterms - m - 1 : m;
        const auto& [sqop, coefficient] = sop(n);
        if (not sqop.is_nilpotent()) {
            std::string msg =
                "compute_on_the_fly_excitation is implemented only for nilpotent operators."
                "Operator " +
                sqop.str() + " is not nilpotent";
            throw std::runtime_error(msg);
        }
        compute_sign_mask(sqop.cre(), sqop.ann(), sign_mask, idx);
        const auto t = (inverse ? -1.0 : 1.0) * coefficient;
        const auto screen_thresh_div_t = screen_thresh_ / std::abs(t);
        // loop over all determinants
        for (const auto& [det, c] : result) {
            // do not apply this operator to this determinant if we expect the new determinant
            // to have an amplitude less than screen_thresh
            // test if we can apply this operator to this determinant
            if ((std::abs(c) > screen_thresh_div_t) and
                det.faster_can_apply_operator(sqop.cre(), sqop.ann())) {
                const auto sign =
                    faster_apply_operator_to_det(det, new_det, sqop.cre(), sqop.ann(), sign_mask);
                new_terms.push_back(std::make_pair(new_det, c * t * sign));
            }
        }
        for (const auto& [det, c] : new_terms) {
            result[det] += c;
        }

        // reset the buffer
        new_terms.reset();
    }
    return result;
}

SparseState SparseFactExp::apply_antiherm(const SparseOperatorList& sop, const SparseState& state,
                                          bool inverse) {

    // initialize a state object
    SparseState result(state);
    Buffer<std::pair<Determinant, sparse_scalar_t>> new_terms;

    Determinant new_det;
    Determinant sign_mask;
    Determinant idx;
    for (size_t m = 0, nterms = sop.size(); m < nterms; m++) {
        size_t n = inverse ? nterms - m - 1 : m;

        const auto& [sqop, coefficient] = sop(n);
        if (not sqop.is_nilpotent()) {
            std::string msg =
                "compute_on_the_fly_antihermitian is implemented only for nilpotent operators."
                "Operator " +
                sqop.str() + " is not nilpotent";
            throw std::runtime_error(msg);
        }
        compute_sign_mask(sqop.cre(), sqop.ann(), sign_mask, idx);
        const auto t = (inverse ? -1.0 : 1.0) * coefficient;
        const auto screen_thresh_div_t = screen_thresh_ / std::abs(t);
        // loop over all determinants
        for (const auto& [det, c] : result) {
            // do not apply this operator to this determinant if we expect the new determinant
            // to have an amplitude less than screen_thresh
            // (here we use the approximation sin(x) ~ x, for x small)
            if (std::abs(c) > screen_thresh_div_t) {
                if (det.faster_can_apply_operator(sqop.cre(), sqop.ann())) {
                    const auto theta = t * faster_apply_operator_to_det(det, new_det, sqop.cre(),
                                                                        sqop.ann(), sign_mask);
                    new_terms.emplace_back(det, c * (std::cos(theta) - 1.0));
                    new_terms.emplace_back(new_det, c * std::sin(theta));
                } else if (det.faster_can_apply_operator(sqop.ann(), sqop.cre())) {
                    const auto theta = -t * faster_apply_operator_to_det(det, new_det, sqop.ann(),
                                                                         sqop.cre(), sign_mask);
                    new_terms.emplace_back(det, c * (std::cos(theta) - 1.0));
                    new_terms.emplace_back(new_det, c * std::sin(theta));
                }
            }
        }
        for (const auto& [det, c] : new_terms) {
            result[det] += c;
        }

        // reset the buffer
        new_terms.reset();
    }
    return result;
}

} // namespace forte2
