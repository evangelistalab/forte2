#include <algorithm>
#include <numeric>
#include <cmath>
#include <thread>
#include <future>
#include <chrono>

#include "ci/sparse_exp.h"
#include "helpers/logger.h"
#include "helpers/timer.hpp"

namespace forte2 {

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
                                    bool inverse, bool reverse) {
    // initialize a state object
    SparseState result(state);

    // temporary space to store new elements. This avoids reallocation
    Buffer<std::pair<Determinant, sparse_scalar_t>> new_terms;

    Determinant new_det;
    Determinant sign_mask;
    Determinant idx;
    for (size_t m = 0, nterms = sop.size(); m < nterms; m++) {
        size_t n = (inverse ^ reverse) ? nterms - m - 1 : m;
        const auto& [sqop, coefficient] = sop(n);
        bool is_idempotent = !sqop.is_nilpotent();

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
                if (is_idempotent) {
                    new_terms.emplace_back(det, c * (std::exp(t) - 1.0));
                } else {
                    const auto sign = faster_apply_operator_to_det(det, new_det, sqop.cre(),
                                                                   sqop.ann(), sign_mask);
                    new_terms.emplace_back(new_det, c * t * sign);
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

SparseState SparseFactExp::apply_antiherm_serial(const SparseOperatorList& sop,
                                                 const SparseState& state, bool inverse,
                                                 bool reverse) {
    // initialize a state object
    SparseState result(state);
    Buffer<std::pair<Determinant, sparse_scalar_t>> new_terms;

    Determinant new_det;
    Determinant sign_mask;
    Determinant idx;
    for (size_t m = 0, nterms = sop.size(); m < nterms; m++) {
        size_t n = (inverse ^ reverse) ? nterms - m - 1 : m;

        const auto& [sqop, coefficient] = sop(n);
        bool is_idempotent = !sqop.is_nilpotent();

        compute_sign_mask(sqop.cre(), sqop.ann(), sign_mask, idx);
        const auto t = (inverse ? -1.0 : 1.0) * coefficient;
        const auto screen_thresh_div_t = screen_thresh_ / std::abs(t);
        // loop over all determinants
        for (const auto& [det, c] : result) {
            // do not apply this operator to this determinant if we expect the new determinant
            // to have an amplitude less than screen_thresh
            // (here we use the approximation sin(x) ~ x, for x small)
            if (std::abs(c) > screen_thresh_div_t) {
                if (is_idempotent and det.faster_can_apply_operator(sqop.cre(), sqop.ann())) {
                    new_terms.emplace_back(det, c * (std::polar(1.0, 2.0 * std::imag(t)) - 1.0));
                } else {
                    if (det.faster_can_apply_operator(sqop.cre(), sqop.ann())) {
                        const auto sign = faster_apply_operator_to_det(det, new_det, sqop.cre(),
                                                                       sqop.ann(), sign_mask);
                        new_terms.emplace_back(det, c * (std::cos(std::abs(t)) - 1.0));
                        new_terms.emplace_back(new_det, sign * c * std::polar(1.0, std::arg(t)) *
                                                            std::sin(std::abs(t)));
                    } else if (det.faster_can_apply_operator(sqop.ann(), sqop.cre())) {
                        const auto sign = faster_apply_operator_to_det(det, new_det, sqop.ann(),
                                                                       sqop.cre(), sign_mask);
                        new_terms.emplace_back(det, c * (std::cos(std::abs(t)) - 1.0));
                        new_terms.emplace_back(new_det, -sign * c * std::polar(1.0, -std::arg(t)) *
                                                            std::sin(std::abs(t)));
                    }
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

SparseState SparseFactExp::apply_antiherm(const SparseOperatorList& sop, const SparseState& state,
                                          bool inverse, bool reverse) {
    // initialize a state object
    SparseState result(state);

    Determinant sign_mask;
    Determinant idx;
    size_t num_threads = std::max<size_t>(1, std::thread::hardware_concurrency());
    std::vector<Buffer<std::pair<Determinant, sparse_scalar_t>>> buffers(num_threads);
    double prepping_time = 0.0;
    double parallel_time = 0.0;
    double serial_time = 0.0;
    size_t num_dets = 0;

    for (size_t m = 0, nterms = sop.size(); m < nterms; m++) {
        size_t n = (inverse ^ reverse) ? nterms - m - 1 : m;

        const auto& [sqop, coefficient] = sop(n);
        bool is_idempotent = !sqop.is_nilpotent();

        compute_sign_mask(sqop.cre(), sqop.ann(), sign_mask, idx);
        const auto t = (inverse ? -1.0 : 1.0) * coefficient;
        const auto screen_thresh_div_t = screen_thresh_ / std::abs(t);

        local_timer prep_timer;

        auto result_views = split_sparse_state_buckets(result, num_threads);
        if (result_views.size() != 1) {
            num_dets += result.size();
        }
        prepping_time += prep_timer.elapsed_seconds();

        std::vector<std::future<void>> futures;
        local_timer parallel_timer;

        for (size_t i = 0; i < result_views.size(); ++i) {
            auto& view = result_views[i];
            auto& buffer = buffers[i];
            futures.emplace_back(
                std::async(std::launch::async, [this, &result, &sqop, &view, &buffer, t, &sign_mask,
                                                screen_thresh_div_t, is_idempotent]() {
                    apply_antiherm_kernel(result, sqop, view, buffer, t, sign_mask,
                                          screen_thresh_div_t, is_idempotent);
                }));
        }

        for (auto& future : futures) {
            future.get(); // wait for all threads to finish
        }
        parallel_time += parallel_timer.elapsed_seconds();
        local_timer serial_timer;
        for (size_t i = 0; i < result_views.size(); ++i) {
            auto& buffer = buffers[i];
            for (const auto& [det, c] : buffer) {
                result[det] += c;
            }
        }
        serial_time += serial_timer.elapsed_seconds();
    }
    // Print timing information
    LOG_INFO1 << "Prepping time: " << prepping_time << " seconds\n";
    LOG_INFO1 << "Parallel time: " << parallel_time << " seconds\n";
    LOG_INFO1 << "Serial time: " << serial_time << " seconds\n";
    LOG_INFO1 << "Number of determinants processed for async: " << num_dets << "\n";

    return result;
}

void SparseFactExp::apply_antiherm_kernel(
    const SparseState& result, const SQOperatorString& sqop,
    const std::pair<size_t, size_t>& bucket_range,
    Buffer<std::pair<Determinant, sparse_scalar_t>>& new_terms, const sparse_scalar_t t,
    const Determinant& sign_mask, double screen_thresh_div_t, bool is_idempotent) {
    Determinant new_det;
    new_terms.reset();

    for (size_t i = bucket_range.first; i < bucket_range.second; ++i) {
        for (auto it = result.begin(i); it != result.end(i); ++it) {
            const auto& [det, c] = *it;

            // do not apply this operator to this determinant if we expect the new determinant
            // to have an amplitude less than screen_thresh
            // (here we use the approximation sin(x) ~ x, for x small)
            if (std::abs(c) > screen_thresh_div_t) {
                if (is_idempotent and det.faster_can_apply_operator(sqop.cre(), sqop.ann())) {
                    new_terms.emplace_back(det, c * (std::polar(1.0, 2.0 * std::imag(t)) - 1.0));
                } else {
                    if (det.faster_can_apply_operator(sqop.cre(), sqop.ann())) {
                        const auto sign = faster_apply_operator_to_det(det, new_det, sqop.cre(),
                                                                       sqop.ann(), sign_mask);
                        new_terms.emplace_back(det, c * (std::cos(std::abs(t)) - 1.0));
                        new_terms.emplace_back(new_det, sign * c * std::polar(1.0, std::arg(t)) *
                                                            std::sin(std::abs(t)));
                    } else if (det.faster_can_apply_operator(sqop.ann(), sqop.cre())) {
                        const auto sign = faster_apply_operator_to_det(det, new_det, sqop.ann(),
                                                                       sqop.cre(), sign_mask);
                        new_terms.emplace_back(det, c * (std::cos(std::abs(t)) - 1.0));
                        new_terms.emplace_back(new_det, -sign * c * std::polar(1.0, -std::arg(t)) *
                                                            std::sin(std::abs(t)));
                    }
                }
            }
        }
    }
}

std::pair<SparseState, SparseState>
SparseFactExp::apply_antiherm_deriv(const SQOperatorString& sqop, const sparse_scalar_t t,
                                    const SparseState& state) {

    // initialize a state object
    SparseState result_x;
    SparseState result_y;

    Determinant new_det;
    Determinant sign_mask;
    Determinant idx;
    if (not sqop.is_nilpotent()) {
        std::string msg = "apply_antiherm_deriv is implemented only for nilpotent operators."
                          "Operator " +
                          sqop.str() + " is not nilpotent";
        throw std::runtime_error(msg);
    }

    compute_sign_mask(sqop.cre(), sqop.ann(), sign_mask, idx);

    const auto tabs = std::abs(t);
    const auto sint = std::sin(tabs);
    const auto cost = std::cos(tabs);
    const auto sinct = sinc_taylor(tabs);

    const auto phi = std::arg(t);
    const auto sinphi = std::sin(phi);
    const auto cosphi = std::cos(phi);

    const sparse_scalar_t c1 = std::pow(cosphi, 2) * cost + std::pow(sinphi, 2) * sinct;
    const sparse_scalar_t c2 = cosphi * sinphi * cost - cosphi * sinphi * sinct;
    const sparse_scalar_t c3 = -cosphi * sint;
    const sparse_scalar_t c4 = std::pow(sinphi, 2) * cost + std::pow(cosphi, 2) * sinct;
    const sparse_scalar_t c5 = -sinphi * sint;
    const sparse_scalar_t uimag = std::complex<double>(0.0, 1.0);

    for (const auto& [det, c] : state) {
        if (det.faster_can_apply_operator(sqop.cre(),
                                          sqop.ann())) { // case where sqop can be applied to det
            const auto sign =
                faster_apply_operator_to_det(det, new_det, sqop.cre(), sqop.ann(), sign_mask);
            result_x[det] += c * c3;
            result_x[new_det] += c * sign * (c1 + uimag * c2);
            result_y[det] += c * c5;
            result_y[new_det] += c * sign * (c2 + uimag * c4);
        } else if (det.faster_can_apply_operator(
                       sqop.ann(), sqop.cre())) { // case where sqop^+ can be applied to det
            const auto sign =
                faster_apply_operator_to_det(det, new_det, sqop.ann(), sqop.cre(), sign_mask);
            result_x[det] += c * c3;
            result_x[new_det] += c * sign * (-c1 + uimag * c2);
            result_y[det] += c * c5;
            result_y[new_det] += c * sign * (-c2 + uimag * c4);
        }
    }

    return std::make_pair(result_x, result_y);
}

} // namespace forte2
