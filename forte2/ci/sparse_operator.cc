#include <algorithm>
#include <cmath>
#include <complex>
#include <format>
#include <numeric>

#include "helpers/combinatorial.h"
#include "helpers/string_algorithms.h"

#include "sparse_operator.h"

namespace forte2 {

std::string format_term_in_sum(sparse_scalar_t coefficient, const std::string& term) {
    // if (term == "[ ]") {
    //     if (coefficient == 0.0) {
    //         return "";
    //     } else {
    //         return (coefficient > 0.0 ? "+ " : "") + to_string_with_precision(coefficient, 12);
    //     }
    // }
    // if (coefficient == 0.0) {
    //     return "";
    // } else if (coefficient == 1.0) {
    //     return "+ " + term;
    // } else if (coefficient == -1.0) {
    //     return "- " + term;
    // } else if (coefficient == static_cast<int>(coefficient)) {
    //     return to_string_with_precision(coefficient, 12) + " * " + term;
    // } else {
    //     std::string s = to_string_with_precision(coefficient, 12);
    //     s.erase(s.find_last_not_of('0') + 1, std::string::npos);
    //     return s + " * " + term;
    // }
    // return "";
    // if constexpr (std::is_same_v<sparse_scalar_t, std::complex<double>>) {
    // }
    // if constexpr (std::is_same_v<sparse_scalar_t, double>) {
    //     return std::format("{} * {}", coefficient, term);
    // }
    return std::format("({} + {}i) * {}", std::real(coefficient), std::imag(coefficient), term);
}

// void SparseOperator::add_term(const SQOperator& sqop) { op_list_.push_back(sqop); }

std::vector<std::string> SparseOperator::str() const {
    std::vector<std::string> v;
    for (const auto& [sqop, c] : this->elements()) {
        if (std::abs(c) < 1.0e-12)
            continue;
        v.push_back(format_term_in_sum(c, sqop.str()));
    }
    // sort v to guarantee a consistent order
    std::sort(v.begin(), v.end());
    return v;
}

std::string SparseOperator::latex() const {
    std::vector<std::string> v;
    for (const auto& [sqop, c] : this->elements()) {
        const std::string s = to_string_latex(c) + "\\;" + sqop.latex();
        v.push_back(s);
    }
    // sort v to guarantee a consistent order
    std::sort(v.begin(), v.end());
    return join(v, " + ");
}

void SparseOperator::add_term_from_str(const std::string& s, sparse_scalar_t coefficient,
                                       bool allow_reordering) {
    auto [sqop, phase] = make_sq_operator_string(s, allow_reordering);
    add(sqop, phase * coefficient);
}

SparseOperator SparseOperatorList::to_operator() const {
    SparseOperator op;
    for (const auto& [sqop, c] : elements()) {
        op.add(sqop, c);
    }
    return op;
}

SparseOperator operator*(const SparseOperator& lhs, const SparseOperator& rhs) {
    SparseOperator result;
    for (const auto& [sqop_lhs, c_lhs] : lhs.elements()) {
        for (const auto& [sqop_rhs, c_rhs] : rhs.elements()) {
            const auto prod = sqop_lhs * sqop_rhs;
            for (const auto& [sqop, c] : prod) {
                if (c * c_lhs * c_rhs != 0.0) {
                    result[sqop] += c * c_lhs * c_rhs;
                }
            }
        }
    }
    return result;
}

SparseOperator product(const SparseOperator& lhs, const SparseOperator& rhs) {
    SQOperatorProductComputer computer;
    SparseOperator C;
    for (const auto& [lhs_op, lhs_c] : lhs.elements()) {
        for (const auto& [rhs_op, rhs_c] : rhs.elements()) {
            computer.product(
                lhs_op, rhs_op, lhs_c * rhs_c,
                [&C](const SQOperatorString& sqop, const sparse_scalar_t c) { C.add(sqop, c); });
        }
    }
    return C;
}

SparseOperator commutator(const SparseOperator& lhs, const SparseOperator& rhs) {
    // place the elements in a map to avoid duplicates and to simplify the addition
    SQOperatorProductComputer computer;
    SparseOperator C;
    for (const auto& [lhs_op, lhs_c] : lhs.elements()) {
        for (const auto& [rhs_op, rhs_c] : rhs.elements()) {
            computer.commutator(
                lhs_op, rhs_op, lhs_c * rhs_c,
                [&C](const SQOperatorString& sqop, const sparse_scalar_t c) { C[sqop] += c; });
        }
    }
    return C;
}

void SparseOperatorList::add_term_from_str(std::string str, sparse_scalar_t coefficient,
                                           bool allow_reordering) {
    auto [sqop, phase] = make_sq_operator_string(str, allow_reordering);
    add(sqop, phase * coefficient);
}

std::vector<std::string> SparseOperatorList::str() const {
    std::vector<std::string> v;
    for (const auto& [sqop, c] : this->elements()) {
        if (std::abs(c) < 1.0e-12)
            continue;
        v.push_back(format_term_in_sum(c, sqop.str()));
    }
    return v;
}

void SparseOperatorList::add_term(const std::vector<std::tuple<bool, bool, int>>& op_list,
                                  double coefficient, bool allow_reordering) {
    auto [sqop, sign] = make_sq_operator_string_from_list(op_list, allow_reordering);
    add(sqop, sign * coefficient);
}

SparseOperator sparse_operator_hamiltonian(double scalar, np_matrix& oei_a, np_matrix& oei_b,
                                           np_tensor4& tei_aa, np_tensor4& tei_ab,
                                           np_tensor4& tei_bb, double screen_thresh) {
    SparseOperator H;
    auto oei_a_view = oei_a.view();
    auto oei_b_view = oei_b.view();
    auto tei_aa_view = tei_aa.view();
    auto tei_ab_view = tei_ab.view();
    auto tei_bb_view = tei_bb.view();
    size_t nmo = oei_a.shape(0);

    H.add_term_from_str("[]", scalar);

    for (size_t p = 0; p < nmo; p++) {
        for (size_t q = 0; q < nmo; q++) {
            // std::abs used to handle complex numbers
            if (std::abs(oei_a_view(p, q)) > screen_thresh) {
                H.add(SQOperatorString({p}, {}, {q}, {}), oei_a_view(p, q));
            }
            if (std::abs(oei_b_view(p, q)) > screen_thresh) {
                H.add(SQOperatorString({}, {p}, {}, {q}), oei_b_view(p, q));
            }
        }
    }
    for (size_t p = 0; p < nmo; p++) {
        for (size_t q = p + 1; q < nmo; q++) {
            for (size_t r = 0; r < nmo; r++) {
                for (size_t s = r + 1; s < nmo; s++) {
                    if (std::abs(tei_aa_view(p, q, r, s)) > screen_thresh) {
                        H.add(SQOperatorString({p, q}, {}, {s, r}, {}), tei_aa_view(p, q, r, s));
                    }
                    if (std::abs(tei_bb_view(p, q, r, s)) > screen_thresh) {
                        H.add(SQOperatorString({}, {p, q}, {}, {s, r}), tei_bb_view(p, q, r, s));
                    }
                }
            }
        }
    }
    for (size_t p = 0; p < nmo; p++) {
        for (size_t q = 0; q < nmo; q++) {
            for (size_t r = 0; r < nmo; r++) {
                for (size_t s = 0; s < nmo; s++) {
                    if (std::abs(tei_ab_view(p, q, r, s)) > screen_thresh) {
                        H.add(SQOperatorString({p}, {q}, {r}, {s}), tei_ab_view(p, q, r, s));
                    }
                }
            }
        }
    }
    return H;
}

} // namespace forte2
