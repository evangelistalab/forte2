#include "sparse/sparse_normal_order.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <format>
#include <stdexcept>
#include <tuple>

#include "helpers/string_algorithms.h"

#include "sparse/sparse_operator.h"
#include "sparse/sparse_state.h"

namespace forte2 {

namespace {

struct NormalOp {
    bool creation;
    bool alpha;
    int orbital;
};

std::vector<NormalOp> to_normal_ops(const op_tuple_t& ops) {
    std::vector<NormalOp> result;
    result.reserve(ops.size());
    for (const auto& [creation, alpha, orbital] : ops) {
        result.push_back({creation, alpha, orbital});
    }
    return result;
}

std::vector<NormalOp> physical_ops(const NormalOrderedString& str, const Determinant& reference) {
    return to_normal_ops(str.op_tuple(reference));
}

std::string op_string(const NormalOp& op) {
    return std::to_string(op.orbital) + (op.alpha ? "a" : "b") + (op.creation ? "+" : "-");
}

std::string op_latex(const NormalOp& op) {
    return "\\hat{a}_{" + std::to_string(op.orbital) + (op.alpha ? " \\alpha}" : " \\beta}") +
           (op.creation ? "^\\dagger" : "");
}

bool occupied(const Determinant& reference, const NormalOp& op) {
    return op.alpha ? reference.na(op.orbital) : reference.nb(op.orbital);
}

bool is_normal_creator(const Determinant& reference, const NormalOp& op) {
    return occupied(reference, op) ? not op.creation : op.creation;
}

size_t spin_orbital_index(const NormalOp& op) {
    return op.alpha ? static_cast<size_t>(op.orbital)
                    : Determinant::beta_bit_offset + static_cast<size_t>(op.orbital);
}

bool normal_less(const Determinant& reference, const NormalOp& lhs, const NormalOp& rhs) {
    const bool lhs_creator = is_normal_creator(reference, lhs);
    const bool rhs_creator = is_normal_creator(reference, rhs);
    if (lhs_creator != rhs_creator) {
        return lhs_creator;
    }

    const auto lhs_index = spin_orbital_index(lhs);
    const auto rhs_index = spin_orbital_index(rhs);
    if (lhs_index != rhs_index) {
        return lhs_creator ? lhs_index < rhs_index : lhs_index > rhs_index;
    }
    return std::tie(lhs.creation, lhs.alpha, lhs.orbital) <
           std::tie(rhs.creation, rhs.alpha, rhs.orbital);
}

bool same_spin_orbital(const NormalOp& lhs, const NormalOp& rhs) {
    return lhs.alpha == rhs.alpha and lhs.orbital == rhs.orbital;
}

bool adjoint_pair(const NormalOp& lhs, const NormalOp& rhs) {
    return same_spin_orbital(lhs, rhs) and lhs.creation != rhs.creation;
}

double contraction_value(const Determinant& reference, const NormalOp& lhs, const NormalOp& rhs) {
    if (not adjoint_pair(lhs, rhs)) {
        return 0.0;
    }
    const bool occ = occupied(reference, lhs);
    if (lhs.creation and not rhs.creation) {
        return occ ? 1.0 : 0.0;
    }
    return occ ? 0.0 : 1.0;
}

double apply_normal_op_to_det(Determinant& det, const NormalOp& op) {
    if (op.creation) {
        return op.alpha ? det.create_a(op.orbital) : det.create_b(op.orbital);
    }
    return op.alpha ? det.destroy_a(op.orbital) : det.destroy_b(op.orbital);
}

SQOperatorString one_op_sparse_string(const NormalOp& op) {
    auto cre = Determinant::zero();
    auto ann = Determinant::zero();
    if (op.creation) {
        if (op.alpha) {
            cre.set_na(op.orbital, true);
        } else {
            cre.set_nb(op.orbital, true);
        }
    } else {
        if (op.alpha) {
            ann.set_na(op.orbital, true);
        } else {
            ann.set_nb(op.orbital, true);
        }
    }
    return SQOperatorString(cre, ann);
}

std::string format_normal_term(sparse_scalar_t coefficient, const std::string& term) {
    return std::format("({} + {}i) * {}", std::real(coefficient), std::imag(coefficient), term);
}

bool make_normal_ordered_string(const Determinant& reference, const std::vector<NormalOp>& ops,
                                NormalOrderedString& result) {
    auto cre = Determinant::zero();
    auto ann = Determinant::zero();
    for (const auto& op : ops) {
        Determinant& target = is_normal_creator(reference, op) ? cre : ann;
        const auto bit = spin_orbital_index(op);
        if (target.get_bit(bit)) {
            return false;
        }
        target.set_bit(bit, true);
    }
    result = NormalOrderedString(cre, ann);
    return true;
}

void normal_order_term(const Determinant& reference, const std::vector<NormalOp>& ops,
                       sparse_scalar_t coefficient, NormalOrderedSparseOperator& result,
                       double screen_thresh) {
    if (std::abs(coefficient) <= screen_thresh) {
        return;
    }

    for (size_t i = 0; i + 1 < ops.size(); ++i) {
        if (normal_less(reference, ops[i + 1], ops[i])) {
            auto swapped = ops;
            std::swap(swapped[i], swapped[i + 1]);
            normal_order_term(reference, swapped, -coefficient, result, screen_thresh);

            const double contraction = contraction_value(reference, ops[i], ops[i + 1]);
            if (contraction != 0.0) {
                auto contracted = ops;
                contracted.erase(contracted.begin() + static_cast<std::ptrdiff_t>(i),
                                 contracted.begin() + static_cast<std::ptrdiff_t>(i + 2));
                normal_order_term(reference, contracted, contraction * coefficient, result,
                                  screen_thresh);
            }
            return;
        }
    }

    NormalOrderedString term;
    if (make_normal_ordered_string(reference, ops, term)) {
        result.add(term, coefficient);
    }
}

} // namespace

NormalOrderedString::NormalOrderedString() = default;

NormalOrderedString::NormalOrderedString(const Determinant& cre, const Determinant& ann)
    : cre_(cre), ann_(ann) {
    Determinant temp = Determinant::zero();
    compute_sign_mask(cre_, ann_, sign_mask_, temp);
}

const Determinant& NormalOrderedString::cre() const { return cre_; }

const Determinant& NormalOrderedString::ann() const { return ann_; }

const Determinant& NormalOrderedString::sign_mask() const { return sign_mask_; }

bool NormalOrderedString::is_identity() const {
    return cre_.count_all() == 0 and ann_.count_all() == 0;
}

int NormalOrderedString::count() const { return cre_.count_all() + ann_.count_all(); }

int NormalOrderedString::many_body_rank() const { return (count() + 1) / 2; }

op_tuple_t NormalOrderedString::op_tuple(const Determinant& reference) const {
    op_tuple_t terms;
    auto acre = cre_.get_alfa_occ(cre_.norb());
    auto bcre = cre_.get_beta_occ(cre_.norb());
    auto aann = ann_.get_alfa_occ(ann_.norb());
    auto bann = ann_.get_beta_occ(ann_.norb());
    std::reverse(aann.begin(), aann.end());
    std::reverse(bann.begin(), bann.end());

    auto append = [&terms, &reference](bool normal_creation, bool alpha, int orbital) {
        const bool occ = alpha ? reference.na(orbital) : reference.nb(orbital);
        const bool physical_creation = normal_creation ? not occ : occ;
        terms.emplace_back(physical_creation, alpha, orbital);
    };

    for (auto p : acre) {
        append(true, true, p);
    }
    for (auto p : bcre) {
        append(true, false, p);
    }
    for (auto p : bann) {
        append(false, false, p);
    }
    for (auto p : aann) {
        append(false, true, p);
    }
    return terms;
}

std::string NormalOrderedString::str() const { return str(Determinant::zero()); }

std::string NormalOrderedString::str(const Determinant& reference) const {
    std::vector<std::string> terms;
    for (const auto& op : physical_ops(*this, reference)) {
        terms.push_back(op_string(op));
    }
    return "{" + join(terms, " ") + "}";
}

std::string NormalOrderedString::latex() const { return latex(Determinant::zero()); }

std::string NormalOrderedString::latex(const Determinant& reference) const {
    std::string s = "\\left\\{";
    for (const auto& op : physical_ops(*this, reference)) {
        s += op_latex(op);
    }
    s += "\\right\\}";
    return s;
}

bool NormalOrderedString::operator==(const NormalOrderedString& other) const {
    return cre_ == other.cre_ and ann_ == other.ann_;
}

bool NormalOrderedString::operator<(const NormalOrderedString& other) const {
    if (cre_ != other.cre_) {
        return cre_ < other.cre_;
    }
    return ann_ < other.ann_;
}

std::size_t NormalOrderedString::Hash::operator()(const NormalOrderedString& str) const {
    const auto h1 = std::hash<Determinant>()(str.cre());
    const auto h2 = std::hash<Determinant>()(str.ann());
    return hash_combine(h1, h2);
}

NormalOrderedSparseOperator::NormalOrderedSparseOperator() = default;

NormalOrderedSparseOperator::NormalOrderedSparseOperator(const Determinant& reference)
    : reference_(reference) {}

NormalOrderedSparseOperator::NormalOrderedSparseOperator(const Determinant& reference,
                                                         const NormalOrderedString& str,
                                                         sparse_scalar_t coefficient)
    : base_t(str, coefficient), reference_(reference) {}

const Determinant& NormalOrderedSparseOperator::reference() const { return reference_; }

sparse_scalar_t NormalOrderedSparseOperator::coefficient(const NormalOrderedString& str) const {
    return (*this)[str];
}

std::vector<std::string> NormalOrderedSparseOperator::str() const {
    std::vector<std::string> result;
    result.reserve(this->size());
    for (const auto& [term, coefficient] : this->elements()) {
        if (std::abs(coefficient) < 1.0e-12) {
            continue;
        }
        result.push_back(format_normal_term(coefficient, term.str(reference_)));
    }
    std::sort(result.begin(), result.end());
    return result;
}

std::string NormalOrderedSparseOperator::latex() const {
    std::vector<std::string> result;
    result.reserve(this->size());
    for (const auto& [term, coefficient] : this->elements()) {
        if (std::abs(coefficient) < 1.0e-12) {
            continue;
        }
        result.push_back(to_string_latex(coefficient) + "\\;" + term.latex(reference_));
    }
    std::sort(result.begin(), result.end());
    return join(result, " + ");
}

NormalOrderedSparseOperator NormalOrderedSparseOperator::truncate(int max_rank,
                                                                  double screen_thresh) const {
    if (max_rank < 0) {
        throw std::invalid_argument(
            "NormalOrderedSparseOperator::truncate: max_rank must be non-negative");
    }
    if (screen_thresh < 0.0) {
        throw std::invalid_argument(
            "NormalOrderedSparseOperator::truncate: screen_thresh must be non-negative");
    }

    NormalOrderedSparseOperator result(reference_);
    for (const auto& [term, coefficient] : this->elements()) {
        if ((term.count() <= 2 * max_rank) and (std::abs(coefficient) > screen_thresh)) {
            result.add(term, coefficient);
        }
    }
    return result;
}

SparseOperator NormalOrderedSparseOperator::to_sparse_operator(double screen_thresh) const {
    if (screen_thresh < 0.0) {
        throw std::invalid_argument(
            "NormalOrderedSparseOperator::to_sparse_operator: screen_thresh must be non-negative");
    }

    auto zero = Determinant::zero();
    const auto identity = SQOperatorString(zero, zero);
    SparseOperator result;

    for (const auto& [term, coefficient] : this->elements()) {
        if (std::abs(coefficient) <= screen_thresh) {
            continue;
        }

        SparseOperator expanded(identity, sparse_scalar_t(1.0));
        for (const auto& op : physical_ops(term, reference_)) {
            SparseOperator op_as_sparse(one_op_sparse_string(op), sparse_scalar_t(1.0));
            expanded = expanded * op_as_sparse;
        }
        expanded *= coefficient;
        result += expanded;
    }

    SparseOperator cleaned;
    for (const auto& [sqop, coefficient] : result.elements()) {
        if (std::abs(coefficient) > screen_thresh) {
            cleaned.add(sqop, coefficient);
        }
    }
    return cleaned;
}

SparseState NormalOrderedSparseOperator::apply_to_state(const SparseState& state,
                                                        double screen_thresh) const {
    if (screen_thresh < 0.0) {
        throw std::invalid_argument(
            "NormalOrderedSparseOperator::apply_to_state: screen_thresh must be non-negative");
    }

    SparseState result;
    for (const auto& [term, coefficient] : this->elements()) {
        if (std::abs(coefficient) <= screen_thresh) {
            continue;
        }
        const auto ops = physical_ops(term, reference_);
        for (const auto& [det, state_coefficient] : state.elements()) {
            if (std::abs(coefficient * state_coefficient) <= screen_thresh) {
                continue;
            }
            auto new_det = det;
            double sign = 1.0;
            for (auto it = ops.rbegin(); it != ops.rend(); ++it) {
                sign *= apply_normal_op_to_det(new_det, *it);
                if (sign == 0.0) {
                    break;
                }
            }
            if (sign != 0.0) {
                result[new_det] += sign * coefficient * state_coefficient;
            }
        }
    }
    return result;
}

bool NormalOrderedSparseOperator::operator==(const NormalOrderedSparseOperator& other) const {
    if (reference_ != other.reference_) {
        return false;
    }
    return base_t::operator==(other);
}

NormalOrderedSparseOperator normal_order(const SparseOperator& op, const Determinant& reference,
                                         double screen_thresh, int max_rank) {
    if (screen_thresh < 0.0) {
        throw std::invalid_argument("normal_order: screen_thresh must be non-negative");
    }
    if (max_rank < -1) {
        throw std::invalid_argument("normal_order: max_rank must be non-negative or -1");
    }

    NormalOrderedSparseOperator result(reference);
    for (const auto& [sqop, coefficient] : op.elements()) {
        normal_order_term(reference, to_normal_ops(sqop.op_tuple()), coefficient, result,
                          screen_thresh);
    }

    NormalOrderedSparseOperator cleaned(reference);
    for (const auto& [term, coefficient] : result.elements()) {
        if ((max_rank < 0 or term.count() <= 2 * max_rank) and
            (std::abs(coefficient) > screen_thresh)) {
            cleaned.add(term, coefficient);
        }
    }
    return cleaned;
}

} // namespace forte2
