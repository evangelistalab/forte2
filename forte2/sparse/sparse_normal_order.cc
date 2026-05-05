#include "sparse/sparse_normal_order.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <format>
#include <stdexcept>
#include <tuple>
#include <unordered_map>

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

using NormalOrderExpansion = std::vector<std::pair<NormalOrderedString, sparse_scalar_t>>;
using SparseExpansion = std::vector<std::pair<SQOperatorString, sparse_scalar_t>>;

struct NormalOrderCacheKey {
    Determinant reference;
    SQOperatorString sqop;
    int max_rank;

    bool operator==(const NormalOrderCacheKey& other) const {
        return reference == other.reference and sqop == other.sqop and max_rank == other.max_rank;
    }
};

struct NormalOrderCacheKeyHash {
    std::size_t operator()(const NormalOrderCacheKey& key) const {
        std::size_t h = std::hash<Determinant>()(key.reference);
        h = hash_combine(h, SQOperatorString::Hash{}(key.sqop));
        return hash_combine(h, std::hash<int>()(key.max_rank));
    }
};

struct SparseExpansionCacheKey {
    Determinant reference;
    NormalOrderedString term;

    bool operator==(const SparseExpansionCacheKey& other) const {
        return reference == other.reference and term == other.term;
    }
};

struct SparseExpansionCacheKeyHash {
    std::size_t operator()(const SparseExpansionCacheKey& key) const {
        std::size_t h = std::hash<Determinant>()(key.reference);
        return hash_combine(h, NormalOrderedString::Hash{}(key.term));
    }
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
                    : Determinant::beta_storage_offset + static_cast<size_t>(op.orbital);
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
        return op.alpha ? det.create_alpha(op.orbital) : det.create_beta(op.orbital);
    }
    return op.alpha ? det.destroy_alpha(op.orbital) : det.destroy_beta(op.orbital);
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

SQOperatorString sparse_string_from_ops(const std::vector<NormalOp>& ops) {
    auto cre = Determinant::zero();
    auto ann = Determinant::zero();
    for (const auto& op : ops) {
        auto& target = op.creation ? cre : ann;
        const auto bit = spin_orbital_index(op);
        if (target.get_bit(bit)) {
            throw std::invalid_argument(
                "sparse_string_from_ops: duplicate second-quantized operator");
        }
        target.set_bit(bit, true);
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
                       double screen_thresh, int max_rank) {
    if (std::abs(coefficient) <= screen_thresh) {
        return;
    }
    const int max_count = max_rank < 0 ? -1 : 2 * max_rank;

    for (size_t i = 0; i + 1 < ops.size(); ++i) {
        if (normal_less(reference, ops[i + 1], ops[i])) {
            auto swapped = ops;
            std::swap(swapped[i], swapped[i + 1]);
            normal_order_term(reference, swapped, -coefficient, result, screen_thresh, max_rank);

            const double contraction = contraction_value(reference, ops[i], ops[i + 1]);
            if (contraction != 0.0) {
                auto contracted = ops;
                contracted.erase(contracted.begin() + static_cast<std::ptrdiff_t>(i),
                                 contracted.begin() + static_cast<std::ptrdiff_t>(i + 2));
                normal_order_term(reference, contracted, contraction * coefficient, result,
                                  screen_thresh, max_rank);
            }
            return;
        }
    }

    if (max_count >= 0 and static_cast<int>(ops.size()) > max_count) {
        return;
    }
    NormalOrderedString term;
    if (make_normal_ordered_string(reference, ops, term)) {
        result.add(term, coefficient);
    }
}

const NormalOrderExpansion& normal_order_expansion(const Determinant& reference,
                                                   const SQOperatorString& sqop, int max_rank) {
    thread_local std::unordered_map<NormalOrderCacheKey, NormalOrderExpansion,
                                    NormalOrderCacheKeyHash>
        cache;
    constexpr size_t max_cache_size = 500000;
    if (cache.size() > max_cache_size) {
        cache.clear();
    }

    NormalOrderCacheKey key{reference, sqop, max_rank};
    if (auto it = cache.find(key); it != cache.end()) {
        return it->second;
    }

    auto [it, inserted] = cache.emplace(std::move(key), NormalOrderExpansion{});
    NormalOrderedSparseOperator ordered(reference);
    normal_order_term(reference, to_normal_ops(sqop.op_tuple()), sparse_scalar_t(1.0), ordered, 0.0,
                      max_rank);

    auto& expansion = it->second;
    expansion.reserve(ordered.size());
    for (const auto& [term, coefficient] : ordered.elements()) {
        if (max_rank < 0 or term.count() <= 2 * max_rank) {
            expansion.emplace_back(term, coefficient);
        }
    }
    return expansion;
}

const SparseExpansion& sparse_expansion(const Determinant& reference,
                                        const NormalOrderedString& term) {
    thread_local std::unordered_map<SparseExpansionCacheKey, SparseExpansion,
                                    SparseExpansionCacheKeyHash>
        cache;
    constexpr size_t max_cache_size = 500000;
    if (cache.size() > max_cache_size) {
        cache.clear();
    }

    SparseExpansionCacheKey key{reference, term};
    if (auto it = cache.find(key); it != cache.end()) {
        return it->second;
    }

    auto [it, inserted] = cache.emplace(std::move(key), SparseExpansion{});

    auto zero = Determinant::zero();
    const auto identity = SQOperatorString(zero, zero);
    SparseOperator expanded(identity, sparse_scalar_t(1.0));
    for (const auto& op : physical_ops(term, reference)) {
        SparseOperator op_as_sparse(one_op_sparse_string(op), sparse_scalar_t(1.0));
        expanded = expanded * op_as_sparse;
    }

    auto& expansion = it->second;
    expansion.reserve(expanded.size());
    for (const auto& [sqop, coefficient] : expanded.elements()) {
        expansion.emplace_back(sqop, coefficient);
    }
    return expansion;
}

int subset_to_front_phase(const std::vector<size_t>& selected, size_t nops) {
    std::vector<bool> is_selected(nops, false);
    for (const auto pos : selected) {
        is_selected[pos] = true;
    }

    int inversions = 0;
    for (size_t i = 0; i < nops; ++i) {
        if (is_selected[i]) {
            continue;
        }
        for (size_t j = i + 1; j < nops; ++j) {
            if (is_selected[j]) {
                ++inversions;
            }
        }
    }
    return inversions % 2 == 0 ? 1 : -1;
}

int contracted_body_rank(const std::vector<NormalOp>& ops, const std::vector<size_t>& selected) {
    int ncre = 0;
    int nann = 0;
    for (const auto pos : selected) {
        if (ops[pos].creation) {
            ++ncre;
        } else {
            ++nann;
        }
    }
    return ncre == nann ? ncre : -1;
}

std::vector<NormalOp> select_ops(const std::vector<NormalOp>& ops,
                                 const std::vector<size_t>& selected) {
    std::vector<NormalOp> result;
    result.reserve(selected.size());
    for (const auto pos : selected) {
        result.push_back(ops[pos]);
    }
    return result;
}

std::vector<NormalOp> remove_selected_ops(const std::vector<NormalOp>& ops,
                                          const std::vector<size_t>& selected) {
    std::vector<bool> is_selected(ops.size(), false);
    for (const auto pos : selected) {
        is_selected[pos] = true;
    }

    std::vector<NormalOp> result;
    result.reserve(ops.size() - selected.size());
    for (size_t i = 0; i < ops.size(); ++i) {
        if (not is_selected[i]) {
            result.push_back(ops[i]);
        }
    }
    return result;
}

SparseState apply_ops_to_state(const std::vector<NormalOp>& ops, const SparseState& state,
                               double screen_thresh) {
    SparseState result;
    for (const auto& [det, coefficient] : state.elements()) {
        if (std::abs(coefficient) <= screen_thresh) {
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
            result[new_det] += sign * coefficient;
        }
    }
    return result;
}

sparse_scalar_t vacuum_norm(const SparseState& vacuum) {
    sparse_scalar_t norm = 0.0;
    for (const auto& [det, coefficient] : vacuum.elements()) {
        norm += std::conj(coefficient) * coefficient;
    }
    return norm;
}

bool sparse_state_equal(const SparseState& lhs, const SparseState& rhs) {
    if (lhs.size() != rhs.size()) {
        return false;
    }
    for (const auto& [det, coefficient] : lhs.elements()) {
        const auto it = rhs.elements().find(det);
        if (it == rhs.elements().end() or it->second != coefficient) {
            return false;
        }
    }
    return true;
}

sparse_scalar_t vacuum_expectation(const SparseState& vacuum, const std::vector<NormalOp>& ops,
                                   double screen_thresh) {
    const auto norm = vacuum_norm(vacuum);
    if (std::abs(norm) <= screen_thresh) {
        throw std::invalid_argument("generalized normal ordering requires a nonzero vacuum");
    }

    const auto ket = apply_ops_to_state(ops, vacuum, screen_thresh);
    sparse_scalar_t expectation = 0.0;
    for (const auto& [det, coefficient] : ket.elements()) {
        const auto it = vacuum.elements().find(det);
        if (it != vacuum.elements().end()) {
            expectation += std::conj(it->second) * coefficient;
        }
    }
    return expectation / norm;
}

template <typename Func>
void enumerate_generalized_contractions(const std::vector<NormalOp>& ops, int max_cumulant,
                                        Func&& func) {
    const auto nops = ops.size();
    const int max_selected_count = max_cumulant < 0 ? static_cast<int>(nops) : 2 * max_cumulant;
    std::vector<size_t> selected;

    auto visit = [&](auto&& self, size_t pos) -> void {
        if (static_cast<int>(selected.size()) > max_selected_count) {
            return;
        }
        if (pos == nops) {
            if (selected.empty()) {
                return;
            }
            const int rank = contracted_body_rank(ops, selected);
            if (rank < 0 or (max_cumulant >= 0 and rank > max_cumulant)) {
                return;
            }
            func(selected);
            return;
        }

        self(self, pos + 1);
        selected.push_back(pos);
        self(self, pos + 1);
        selected.pop_back();
    };
    visit(visit, 0);
}

void sparse_expansion_of_generalized_normal_term(const SparseState& vacuum, int max_cumulant,
                                                 const SQOperatorString& term,
                                                 SparseOperator& result,
                                                 sparse_scalar_t coefficient,
                                                 double screen_thresh) {
    if (std::abs(coefficient) <= screen_thresh) {
        return;
    }

    const auto ops = to_normal_ops(term.op_tuple());
    result.add(term, coefficient);

    enumerate_generalized_contractions(ops, max_cumulant, [&](const std::vector<size_t>& selected) {
        const int phase = subset_to_front_phase(selected, ops.size());
        const auto contracted = select_ops(ops, selected);
        const auto contraction = vacuum_expectation(vacuum, contracted, screen_thresh);
        if (std::abs(contraction) <= screen_thresh) {
            return;
        }

        const auto remainder = sparse_string_from_ops(remove_selected_ops(ops, selected));
        sparse_expansion_of_generalized_normal_term(
            vacuum, max_cumulant, remainder, result,
            -coefficient * static_cast<double>(phase) * contraction, screen_thresh);
    });
}

} // namespace

NormalOrderedString::NormalOrderedString() = default;

NormalOrderedString::NormalOrderedString(const Determinant& cre, const Determinant& ann)
    : cre_(cre), ann_(ann) {}

const Determinant& NormalOrderedString::cre() const { return cre_; }

const Determinant& NormalOrderedString::ann() const { return ann_; }

const Determinant& NormalOrderedString::sign_mask() const {
    if (not sign_mask_valid_) {
        compute_sign_mask(cre_, ann_, sign_mask_);
        sign_mask_valid_ = true;
    }
    return sign_mask_;
}

bool NormalOrderedString::is_identity() const {
    return cre_.count_all() == 0 and ann_.count_all() == 0;
}

int NormalOrderedString::count() const { return cre_.count_all() + ann_.count_all(); }

int NormalOrderedString::many_body_rank() const { return (count() + 1) / 2; }

op_tuple_t NormalOrderedString::op_tuple(const Determinant& reference) const {
    op_tuple_t terms;
    auto acre = cre_.get_alpha_occ();
    auto bcre = cre_.get_beta_occ();
    auto aann = ann_.get_alpha_occ();
    auto bann = ann_.get_beta_occ();
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

    SparseOperator result;

    for (const auto& [term, coefficient] : this->elements()) {
        if (std::abs(coefficient) <= screen_thresh) {
            continue;
        }
        for (const auto& [sqop, factor] : sparse_expansion(reference_, term)) {
            result.add(sqop, coefficient * factor);
        }
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
        if (std::abs(coefficient) <= screen_thresh) {
            continue;
        }
        for (const auto& [term, factor] : normal_order_expansion(reference, sqop, max_rank)) {
            result.add(term, coefficient * factor);
        }
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

GeneralizedNormalOrderedSparseOperator::GeneralizedNormalOrderedSparseOperator() = default;

GeneralizedNormalOrderedSparseOperator::GeneralizedNormalOrderedSparseOperator(
    const SparseState& vacuum, std::size_t norb, int max_cumulant)
    : vacuum_(vacuum), norb_(norb), max_cumulant_(max_cumulant) {
    if (max_cumulant < -1) {
        throw std::invalid_argument(
            "GeneralizedNormalOrderedSparseOperator: max_cumulant must be non-negative or -1");
    }
}

GeneralizedNormalOrderedSparseOperator::GeneralizedNormalOrderedSparseOperator(
    const SparseState& vacuum, std::size_t norb, int max_cumulant, const SQOperatorString& str,
    sparse_scalar_t coefficient)
    : base_t(str, coefficient), vacuum_(vacuum), norb_(norb), max_cumulant_(max_cumulant) {
    if (max_cumulant < -1) {
        throw std::invalid_argument(
            "GeneralizedNormalOrderedSparseOperator: max_cumulant must be non-negative or -1");
    }
}

const SparseState& GeneralizedNormalOrderedSparseOperator::vacuum() const { return vacuum_; }

std::size_t GeneralizedNormalOrderedSparseOperator::norb() const { return norb_; }

int GeneralizedNormalOrderedSparseOperator::max_cumulant() const { return max_cumulant_; }

sparse_scalar_t
GeneralizedNormalOrderedSparseOperator::coefficient(const SQOperatorString& str) const {
    return (*this)[str];
}

std::vector<std::string> GeneralizedNormalOrderedSparseOperator::str() const {
    std::vector<std::string> result;
    result.reserve(this->size());
    for (const auto& [term, coefficient] : this->elements()) {
        if (std::abs(coefficient) < 1.0e-12) {
            continue;
        }
        result.push_back(format_normal_term(coefficient, term.str()));
    }
    std::sort(result.begin(), result.end());
    return result;
}

std::string GeneralizedNormalOrderedSparseOperator::latex() const {
    std::vector<std::string> result;
    result.reserve(this->size());
    for (const auto& [term, coefficient] : this->elements()) {
        if (std::abs(coefficient) < 1.0e-12) {
            continue;
        }
        result.push_back(to_string_latex(coefficient) + "\\;" + term.latex());
    }
    std::sort(result.begin(), result.end());
    return join(result, " + ");
}

GeneralizedNormalOrderedSparseOperator
GeneralizedNormalOrderedSparseOperator::truncate(int max_rank, double screen_thresh) const {
    if (max_rank < 0) {
        throw std::invalid_argument(
            "GeneralizedNormalOrderedSparseOperator::truncate: max_rank must be non-negative");
    }
    if (screen_thresh < 0.0) {
        throw std::invalid_argument(
            "GeneralizedNormalOrderedSparseOperator::truncate: screen_thresh must be non-negative");
    }

    GeneralizedNormalOrderedSparseOperator result(vacuum_, norb_, max_cumulant_);
    for (const auto& [term, coefficient] : this->elements()) {
        if ((term.count() <= 2 * max_rank) and (std::abs(coefficient) > screen_thresh)) {
            result.add(term, coefficient);
        }
    }
    return result;
}

SparseOperator
GeneralizedNormalOrderedSparseOperator::to_sparse_operator(double screen_thresh) const {
    if (screen_thresh < 0.0) {
        throw std::invalid_argument(
            "GeneralizedNormalOrderedSparseOperator::to_sparse_operator: screen_thresh must be "
            "non-negative");
    }

    SparseOperator result;
    for (const auto& [term, coefficient] : this->elements()) {
        if (std::abs(coefficient) <= screen_thresh) {
            continue;
        }
        sparse_expansion_of_generalized_normal_term(vacuum_, max_cumulant_, term, result,
                                                    coefficient, screen_thresh);
    }

    SparseOperator cleaned;
    for (const auto& [sqop, coefficient] : result.elements()) {
        if (std::abs(coefficient) > screen_thresh) {
            cleaned.add(sqop, coefficient);
        }
    }
    return cleaned;
}

SparseState GeneralizedNormalOrderedSparseOperator::apply_to_state(const SparseState& state,
                                                                   double screen_thresh) const {
    return apply_operator_lin(to_sparse_operator(screen_thresh), state, screen_thresh);
}

bool GeneralizedNormalOrderedSparseOperator::operator==(
    const GeneralizedNormalOrderedSparseOperator& other) const {
    if (norb_ != other.norb_ or max_cumulant_ != other.max_cumulant_ or
        not sparse_state_equal(vacuum_, other.vacuum_)) {
        return false;
    }
    return base_t::operator==(other);
}

GeneralizedNormalOrderedSparseOperator
generalized_normal_order(const SparseOperator& op, const SparseState& vacuum, std::size_t norb,
                         int max_cumulant, double screen_thresh, int max_rank) {
    if (screen_thresh < 0.0) {
        throw std::invalid_argument("generalized_normal_order: screen_thresh must be non-negative");
    }
    if (max_cumulant < -1) {
        throw std::invalid_argument(
            "generalized_normal_order: max_cumulant must be non-negative or -1");
    }
    if (max_rank < -1) {
        throw std::invalid_argument(
            "generalized_normal_order: max_rank must be non-negative or -1");
    }

    if (std::abs(vacuum_norm(vacuum)) <= screen_thresh) {
        throw std::invalid_argument("generalized_normal_order: vacuum must be nonzero");
    }

    GeneralizedNormalOrderedSparseOperator result(vacuum, norb, max_cumulant);
    for (const auto& [sqop, coefficient] : op.elements()) {
        if (std::abs(coefficient) <= screen_thresh) {
            continue;
        }

        const auto ops = to_normal_ops(sqop.op_tuple());
        result.add(sqop, coefficient);
        enumerate_generalized_contractions(
            ops, max_cumulant, [&](const std::vector<size_t>& selected) {
                const auto remainder_ops = remove_selected_ops(ops, selected);
                if (max_rank >= 0 and static_cast<int>(remainder_ops.size()) > 2 * max_rank) {
                    return;
                }

                const int phase = subset_to_front_phase(selected, ops.size());
                const auto contracted = select_ops(ops, selected);
                const auto contraction = vacuum_expectation(vacuum, contracted, screen_thresh);
                if (std::abs(contraction) <= screen_thresh) {
                    return;
                }
                result.add(sparse_string_from_ops(remainder_ops),
                           coefficient * static_cast<double>(phase) * contraction);
            });
    }

    GeneralizedNormalOrderedSparseOperator cleaned(vacuum, norb, max_cumulant);
    for (const auto& [term, coefficient] : result.elements()) {
        if ((max_rank < 0 or term.count() <= 2 * max_rank) and
            (std::abs(coefficient) > screen_thresh)) {
            cleaned.add(term, coefficient);
        }
    }
    return cleaned;
}

} // namespace forte2
