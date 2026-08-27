#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include "sparse/cumulant_wick.h"

namespace forte2 {
namespace {

struct WickLeg {
    std::size_t position;
    std::size_t mode;
    bool creation;
    bool lhs;
};

struct ElementaryContraction {
    std::uint16_t selected = 0;
    std::vector<std::size_t> canonical_positions;
    sparse_scalar_t value = 0.0;
    bool eta = false;
};

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

int permutation_phase(const std::vector<std::size_t>& order) {
    int phase = 1;
    for (std::size_t i = 0; i < order.size(); ++i) {
        for (std::size_t j = i + 1; j < order.size(); ++j) {
            if (order[i] > order[j]) {
                phase = -phase;
            }
        }
    }
    return phase;
}

template <typename Func>
void enumerate_combinations(const std::vector<std::size_t>& positions, std::size_t count,
                            Func&& func) {
    std::vector<std::size_t> selected;
    selected.reserve(count);
    auto visit = [&](auto&& self, std::size_t begin) -> void {
        if (selected.size() == count) {
            func(selected);
            return;
        }
        const auto needed = count - selected.size();
        for (std::size_t i = begin; i + needed <= positions.size(); ++i) {
            selected.push_back(positions[i]);
            self(self, i + 1);
            selected.pop_back();
        }
    };
    visit(visit, 0);
}

std::vector<WickLeg> product_legs(const SQOperatorString& lhs, const SQOperatorString& rhs) {
    std::vector<WickLeg> legs;
    legs.reserve(static_cast<std::size_t>(lhs.count() + rhs.count()));
    auto append = [&](const SQOperatorString& term, bool is_lhs) {
        for (const auto& [creation, alpha, orbital] : term.op_tuple()) {
            const auto mode =
                alpha ? static_cast<std::size_t>(orbital)
                      : Determinant::beta_storage_offset + static_cast<std::size_t>(orbital);
            legs.push_back({legs.size(), mode, creation, is_lhs});
        }
    };
    append(lhs, true);
    append(rhs, false);
    return legs;
}

bool unique_modes(const std::vector<WickLeg>& legs, const std::vector<std::size_t>& positions) {
    auto modes = Determinant::zero();
    for (const auto position : positions) {
        const auto mode = legs[position].mode;
        if (modes.get_bit(mode)) {
            return false;
        }
        modes.set_bit(mode, true);
    }
    return true;
}

ElementaryContraction make_elementary_contraction(const CumulantReference& reference,
                                                  const std::vector<WickLeg>& legs,
                                                  const std::vector<std::size_t>& creators,
                                                  const std::vector<std::size_t>& annihilators) {
    ElementaryContraction contraction;
    if (not unique_modes(legs, creators) or not unique_modes(legs, annihilators)) {
        return contraction;
    }

    auto cre = Determinant::zero();
    auto ann = Determinant::zero();
    bool touches_lhs = false;
    bool touches_rhs = false;
    for (const auto position : creators) {
        cre.set_bit(legs[position].mode, true);
        contraction.selected |= static_cast<std::uint16_t>(1U << position);
        touches_lhs |= legs[position].lhs;
        touches_rhs |= not legs[position].lhs;
    }
    for (const auto position : annihilators) {
        ann.set_bit(legs[position].mode, true);
        contraction.selected |= static_cast<std::uint16_t>(1U << position);
        touches_lhs |= legs[position].lhs;
        touches_rhs |= not legs[position].lhs;
    }
    if (not touches_lhs or not touches_rhs) {
        contraction.selected = 0;
        return contraction;
    }

    contraction.canonical_positions = creators;
    std::sort(contraction.canonical_positions.begin(), contraction.canonical_positions.end(),
              [&](std::size_t lhs, std::size_t rhs) { return legs[lhs].mode < legs[rhs].mode; });
    auto canonical_ann = annihilators;
    std::sort(canonical_ann.begin(), canonical_ann.end(),
              [&](std::size_t lhs, std::size_t rhs) { return legs[lhs].mode > legs[rhs].mode; });
    contraction.canonical_positions.insert(contraction.canonical_positions.end(),
                                           canonical_ann.begin(), canonical_ann.end());

    if (cre.count_all() == 1) {
        contraction.eta = creators[0] > annihilators[0];
        const auto gamma = reference.cumulant(cre, ann);
        contraction.value =
            contraction.eta ? sparse_scalar_t{cre == ann ? 1.0 : 0.0} - gamma : gamma;
    } else {
        contraction.value = reference.cumulant(cre, ann);
    }
    return contraction;
}

std::vector<ElementaryContraction> elementary_contractions(const CumulantReference& reference,
                                                           const std::vector<WickLeg>& legs,
                                                           double /* screen_thresh */) {
    std::vector<std::size_t> creators;
    std::vector<std::size_t> annihilators;
    for (const auto& leg : legs) {
        (leg.creation ? creators : annihilators).push_back(leg.position);
    }

    std::vector<ElementaryContraction> contractions;
    const auto max_rank = std::min<std::size_t>(static_cast<std::size_t>(reference.max_cumulant()),
                                                std::min(creators.size(), annihilators.size()));
    for (std::size_t rank = 1; rank <= max_rank; ++rank) {
        enumerate_combinations(creators, rank, [&](const auto& selected_cre) {
            enumerate_combinations(annihilators, rank, [&](const auto& selected_ann) {
                auto contraction =
                    make_elementary_contraction(reference, legs, selected_cre, selected_ann);
                if (contraction.selected != 0 and contraction.value != sparse_scalar_t{0.0}) {
                    contractions.push_back(std::move(contraction));
                }
            });
        });
    }
    return contractions;
}

bool append_remainder(const std::vector<WickLeg>& legs, std::uint16_t selected,
                      std::vector<std::size_t>& order, SQOperatorString& remainder) {
    auto cre = Determinant::zero();
    auto ann = Determinant::zero();
    std::vector<std::size_t> creators;
    std::vector<std::size_t> annihilators;
    for (const auto& leg : legs) {
        if ((selected & static_cast<std::uint16_t>(1U << leg.position)) != 0) {
            continue;
        }
        auto& bits = leg.creation ? cre : ann;
        if (bits.get_bit(leg.mode)) {
            return false;
        }
        bits.set_bit(leg.mode, true);
        (leg.creation ? creators : annihilators).push_back(leg.position);
    }
    std::sort(creators.begin(), creators.end(),
              [&](std::size_t lhs, std::size_t rhs) { return legs[lhs].mode < legs[rhs].mode; });
    std::sort(annihilators.begin(), annihilators.end(),
              [&](std::size_t lhs, std::size_t rhs) { return legs[lhs].mode > legs[rhs].mode; });
    order.insert(order.end(), creators.begin(), creators.end());
    order.insert(order.end(), annihilators.begin(), annihilators.end());
    remainder = SQOperatorString(cre, ann);
    return true;
}

void add_contraction_term(const std::vector<WickLeg>& legs,
                          const std::vector<ElementaryContraction>& contractions,
                          const std::vector<std::size_t>& selected_contractions,
                          std::uint16_t selected_legs, sparse_scalar_t contraction_value,
                          sparse_scalar_t coefficient, int max_rank, double screen_thresh,
                          GeneralizedNormalOrderedSparseOperator& result) {
    const auto remaining_count =
        legs.size() - static_cast<std::size_t>(std::popcount(selected_legs));
    if (max_rank >= 0 and remaining_count > static_cast<std::size_t>(2 * max_rank)) {
        return;
    }

    std::vector<std::size_t> order;
    order.reserve(legs.size());
    int eta_phase = 1;
    for (const auto index : selected_contractions) {
        const auto& contraction = contractions[index];
        order.insert(order.end(), contraction.canonical_positions.begin(),
                     contraction.canonical_positions.end());
        if (contraction.eta) {
            eta_phase = -eta_phase;
        }
    }

    SQOperatorString remainder;
    if (not append_remainder(legs, selected_legs, order, remainder)) {
        return;
    }
    const auto value =
        coefficient * contraction_value * static_cast<double>(eta_phase * permutation_phase(order));
    if (std::abs(value) > screen_thresh) {
        result.add(remainder, value);
    }
}

void add_term_product(const CumulantReference& reference, const SQOperatorString& lhs,
                      const SQOperatorString& rhs, sparse_scalar_t coefficient, int max_rank,
                      double screen_thresh, GeneralizedNormalOrderedSparseOperator& result) {
    const auto legs = product_legs(lhs, rhs);
    if (legs.size() > 16) {
        throw std::invalid_argument(
            "CumulantWickEngine: rank-two input terms may contain at most eight legs");
    }
    const auto contractions = elementary_contractions(reference, legs, screen_thresh);

    std::vector<std::size_t> selected_contractions;
    auto visit = [&](auto&& self, std::size_t begin, std::uint16_t selected_legs,
                     sparse_scalar_t contraction_value) -> void {
        add_contraction_term(legs, contractions, selected_contractions, selected_legs,
                             contraction_value, coefficient, max_rank, screen_thresh, result);
        for (std::size_t index = begin; index < contractions.size(); ++index) {
            const auto& contraction = contractions[index];
            if ((selected_legs & contraction.selected) != 0) {
                continue;
            }
            selected_contractions.push_back(index);
            self(self, index + 1, selected_legs | contraction.selected,
                 contraction_value * contraction.value);
            selected_contractions.pop_back();
        }
    };
    visit(visit, 0, 0, sparse_scalar_t{1.0});
}

GeneralizedNormalOrderedSparseOperator
clean_result(const GeneralizedNormalOrderedSparseOperator& result, double screen_thresh) {
    GeneralizedNormalOrderedSparseOperator cleaned(result.vacuum(), result.norb(),
                                                   result.max_cumulant());
    cleaned.reserve(result.size());
    for (const auto& [term, coefficient] : result.elements()) {
        if (std::abs(coefficient) > screen_thresh) {
            cleaned.add(term, coefficient);
        }
    }
    return cleaned;
}

} // namespace

CumulantWickEngine::CumulantWickEngine(const CumulantReference& reference, int max_rank,
                                       double screen_thresh)
    : reference_(reference), max_rank_(max_rank), screen_thresh_(screen_thresh) {
    if (max_rank_ < -1) {
        throw std::invalid_argument("CumulantWickEngine: max_rank must be non-negative or -1");
    }
    if (screen_thresh_ < 0.0) {
        throw std::invalid_argument("CumulantWickEngine: screen_thresh must be non-negative");
    }
    if (reference_.max_cumulant() > 3) {
        throw std::invalid_argument(
            "CumulantWickEngine: the current implementation supports cumulants through rank three");
    }
}

const CumulantReference& CumulantWickEngine::reference() const { return reference_; }

int CumulantWickEngine::max_rank() const { return max_rank_; }

double CumulantWickEngine::screen_thresh() const { return screen_thresh_; }

void CumulantWickEngine::validate_operand(const GeneralizedNormalOrderedSparseOperator& op) const {
    if (op.norb() != reference_.norb()) {
        throw std::invalid_argument("CumulantWickEngine: operand and reference norb must match");
    }
    if (op.max_cumulant() != reference_.max_cumulant()) {
        throw std::invalid_argument(
            "CumulantWickEngine: operand and reference max_cumulant values must match");
    }
    if (not sparse_state_equal(op.vacuum(), reference_.vacuum())) {
        throw std::invalid_argument("CumulantWickEngine: operand and reference vacua must match");
    }
    for (const auto& [term, coefficient] : op.elements()) {
        (void)coefficient;
        if (term.count() > 4) {
            throw std::invalid_argument(
                "CumulantWickEngine: the current implementation supports rank-two input terms");
        }
    }
}

void CumulantWickEngine::add_product(const GeneralizedNormalOrderedSparseOperator& lhs,
                                     const GeneralizedNormalOrderedSparseOperator& rhs,
                                     sparse_scalar_t factor,
                                     GeneralizedNormalOrderedSparseOperator& result) const {
    for (const auto& [lhs_term, lhs_coefficient] : lhs.elements()) {
        for (const auto& [rhs_term, rhs_coefficient] : rhs.elements()) {
            const auto coefficient = factor * lhs_coefficient * rhs_coefficient;
            if (std::abs(coefficient) <= screen_thresh_) {
                continue;
            }
            add_term_product(reference_, lhs_term, rhs_term, coefficient, max_rank_, screen_thresh_,
                             result);
        }
    }
}

GeneralizedNormalOrderedSparseOperator
CumulantWickEngine::product(const GeneralizedNormalOrderedSparseOperator& lhs,
                            const GeneralizedNormalOrderedSparseOperator& rhs) const {
    validate_operand(lhs);
    validate_operand(rhs);
    GeneralizedNormalOrderedSparseOperator result(reference_.vacuum(), reference_.norb(),
                                                  reference_.max_cumulant());
    add_product(lhs, rhs, sparse_scalar_t{1.0}, result);
    return clean_result(result, screen_thresh_);
}

GeneralizedNormalOrderedSparseOperator
CumulantWickEngine::commutator(const GeneralizedNormalOrderedSparseOperator& lhs,
                               const GeneralizedNormalOrderedSparseOperator& rhs) const {
    validate_operand(lhs);
    validate_operand(rhs);
    GeneralizedNormalOrderedSparseOperator result(reference_.vacuum(), reference_.norb(),
                                                  reference_.max_cumulant());
    add_product(lhs, rhs, sparse_scalar_t{1.0}, result);
    add_product(rhs, lhs, sparse_scalar_t{-1.0}, result);
    return clean_result(result, screen_thresh_);
}

} // namespace forte2
