#include "sparse/sparse_normal_order_product.h"

#include <algorithm>
#include <bit>
#include <bitset>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>

namespace forte2 {

namespace {

struct NormalTermData {
    const NormalOrderedString* term;
    sparse_scalar_t coefficient;
    int count;
    int cre_count;
    int ann_count;
};

struct PairContribution {
    bool lhs_rhs;
    bool rhs_lhs;
    bool commute;
};

bool above_threshold(const sparse_scalar_t coefficient, double screen_thresh_squared) {
    return std::norm(coefficient) > screen_thresh_squared;
}

std::vector<NormalTermData> collect_terms(const NormalOrderedSparseOperator& op,
                                          double screen_thresh) {
    const double screen_thresh_squared = screen_thresh * screen_thresh;
    std::vector<NormalTermData> terms;
    terms.reserve(op.size());
    for (const auto& [term, coefficient] : op.elements()) {
        if (above_threshold(coefficient, screen_thresh_squared)) {
            const int cre_count = static_cast<int>(term.cre().count_all());
            const int ann_count = static_cast<int>(term.ann().count_all());
            terms.push_back({&term, coefficient, cre_count + ann_count, cre_count, ann_count});
        }
    }
    return terms;
}

NormalOrderedSparseOperator clean_normal_ordered_operator(const NormalOrderedSparseOperator& op,
                                                          int max_rank, double screen_thresh) {
    const double screen_thresh_squared = screen_thresh * screen_thresh;
    NormalOrderedSparseOperator cleaned(op.reference());
    cleaned.reserve(op.size());
    for (const auto& [term, coefficient] : op.elements()) {
        if ((max_rank < 0 or term.count() <= 2 * max_rank) and
            above_threshold(coefficient, screen_thresh_squared)) {
            cleaned.add(term, coefficient);
        }
    }
    return cleaned;
}

PairContribution analyze_pair(const NormalTermData& lhs, const NormalTermData& rhs,
                              int max_rank) {
    const int total_count = lhs.count + rhs.count;
    const int max_count = 2 * max_rank;
    if (total_count > max_count) {
        const int required_contractions = (total_count - max_count + 1) / 2;
        bool lhs_rhs = false;
        if (lhs.ann_count >= required_contractions and rhs.cre_count >= required_contractions) {
            lhs_rhs = lhs.term->ann().fast_a_and_b_count(rhs.term->cre()) >=
                      required_contractions;
        }
        bool rhs_lhs = false;
        if (rhs.ann_count >= required_contractions and lhs.cre_count >= required_contractions) {
            rhs_lhs = rhs.term->ann().fast_a_and_b_count(lhs.term->cre()) >=
                      required_contractions;
        }
        return {lhs_rhs, rhs_lhs, false};
    }

    if ((lhs.count * rhs.count) % 2 != 0) {
        return {true, true, false};
    }

    const bool common_l_cre_r_cre =
        lhs.cre_count > 0 and rhs.cre_count > 0 and
        not lhs.term->cre().fast_a_and_b_eq_zero(rhs.term->cre());
    const bool common_l_cre_r_ann =
        lhs.cre_count > 0 and rhs.ann_count > 0 and
        not lhs.term->cre().fast_a_and_b_eq_zero(rhs.term->ann());
    const bool common_l_ann_r_ann =
        lhs.ann_count > 0 and rhs.ann_count > 0 and
        not lhs.term->ann().fast_a_and_b_eq_zero(rhs.term->ann());
    const bool common_l_ann_r_cre =
        lhs.ann_count > 0 and rhs.cre_count > 0 and
        not lhs.term->ann().fast_a_and_b_eq_zero(rhs.term->cre());
    const bool commute =
        not common_l_cre_r_cre and not common_l_ann_r_ann and not common_l_ann_r_cre and
        not common_l_cre_r_ann;
    return {true, true, commute};
}

class TruncatedNormalProductComputer {
  public:
    explicit TruncatedNormalProductComputer(int max_rank) : max_rank_(max_rank) {}

    void product(const NormalOrderedString& lhs, const NormalOrderedString& rhs,
                 sparse_scalar_t factor,
                 const std::function<void(const NormalOrderedString&, sparse_scalar_t)>& func) {
        ucon_rhs_cre_ = rhs.cre() - lhs.ann();
        if (not lhs.cre().fast_a_and_b_eq_zero(ucon_rhs_cre_)) {
            return;
        }
        con_rhs_cre_ = rhs.cre() - ucon_rhs_cre_;
        ucon_rhs_ann_ = rhs.ann() - con_rhs_cre_;
        if (not lhs.ann().fast_a_and_b_eq_zero(ucon_rhs_ann_)) {
            return;
        }

        phase_ = factor;
        rhs_cre_ = rhs.cre();
        rhs_ann_ = rhs.ann();
        lhs_cre_ = lhs.cre();
        lhs_ann_ = lhs.ann();

        if (const auto ucon_rhs_cre_count = ucon_rhs_cre_.count_all(); ucon_rhs_cre_count > 0) {
            phase_ *= ((lhs_ann_.count_all() * ucon_rhs_cre_count) % 2) == 0 ? 1.0 : -1.0;
            for (size_t i = ucon_rhs_cre_.fast_find_and_clear_first_one(0); i != ~0ULL;
                 i = ucon_rhs_cre_.fast_find_and_clear_first_one(i)) {
                rhs_cre_.set_bit(i, false);
                phase_ *= rhs_cre_.slater_sign(i);
                lhs_cre_.set_bit(i, true);
                phase_ *= lhs_cre_.slater_sign_reverse(i);
            }
        }

        if (const auto ucon_rhs_ann_count = ucon_rhs_ann_.count_all(); ucon_rhs_ann_count > 0) {
            phase_ *= ((rhs_cre_.count_all() * ucon_rhs_ann_count) % 2) == 0 ? 1.0 : -1.0;
            for (size_t i = ucon_rhs_ann_.fast_find_and_clear_first_one(0); i != ~0ULL;
                 i = ucon_rhs_ann_.fast_find_and_clear_first_one(i)) {
                rhs_ann_.set_bit(i, false);
                phase_ *= rhs_ann_.slater_sign_reverse(i);
                lhs_ann_.set_bit(i, true);
                phase_ *= lhs_ann_.slater_sign(i);
            }
        }

        auto rhs_comm_trivial = rhs_cre_ & rhs_ann_ & lhs_ann_;
        if (rhs_comm_trivial.count_all() != 0) {
            rhs_cre_ -= rhs_comm_trivial;
            rhs_ann_ -= rhs_comm_trivial;
            ucon_rhs_cre_ = rhs_cre_;
            for (size_t i = rhs_comm_trivial.fast_find_and_clear_first_one(0); i != ~0ULL;
                 i = rhs_comm_trivial.fast_find_and_clear_first_one(i)) {
                phase_ *= rhs_cre_.slater_sign_reverse(i) * rhs_ann_.slater_sign_reverse(i);
            }
        }

        auto lhs_comm_trivial = lhs_cre_ & lhs_ann_ & rhs_cre_;
        if (lhs_comm_trivial.count_all() != 0) {
            rhs_cre_ -= lhs_comm_trivial;
            lhs_ann_ -= lhs_comm_trivial;
            for (size_t i = lhs_comm_trivial.fast_find_and_clear_first_one(0); i != ~0ULL;
                 i = lhs_comm_trivial.fast_find_and_clear_first_one(i)) {
                phase_ *= lhs_ann_.slater_sign(i) * rhs_cre_.slater_sign(i);
            }
        }

        const auto ncontr = rhs_cre_.count_all();
        if (ncontr == 0) {
            if (lhs_cre_.count_all() + lhs_ann_.count_all() <= 2 * max_rank_) {
                func(NormalOrderedString(lhs_cre_, lhs_ann_), phase_);
            }
            return;
        }
        if (ncontr > max_contracted_ops_) {
            throw std::runtime_error(
                "TruncatedNormalProductComputer: too many simultaneous contractions");
        }

        ucon_rhs_cre_ = rhs_cre_;
        for (size_t i = ucon_rhs_cre_.fast_find_and_clear_first_one(0); i != ~0ULL;
             i = ucon_rhs_cre_.fast_find_and_clear_first_one(i)) {
            phase_ *= lhs_ann_.slater_sign(i) * rhs_cre_.slater_sign(i);
        }

        size_t nbits = ncontr;
        auto contracted_rhs_cre = rhs_cre_;
        contracted_rhs_cre.find_set_bits(set_bits_, nbits);
        rhs_cre_ = lhs_ann_ - contracted_rhs_cre;
        const auto identity_lhs_ann = rhs_cre_;

        for (size_t i = 0; i < ncontr; i++) {
            sign_[i] = (rhs_cre_.slater_sign_reverse(set_bits_[i]) *
                            lhs_cre_.slater_sign_reverse(set_bits_[i]) >
                        0.0)
                           ? 0
                           : 1;
        }

        const int min_count =
            static_cast<int>(lhs_cre_.count_all() + lhs_ann_.count_all() - ncontr);
        const int max_swapped = (2 * max_rank_ - min_count) / 2;
        if (max_swapped < 0) {
            return;
        }
        if (ncontr >= std::numeric_limits<unsigned long long>::digits) {
            throw std::runtime_error("TruncatedNormalProductComputer: contraction mask overflow");
        }

        auto emit_mask = [this, &identity_lhs_ann, &func](unsigned long long mask) {
            auto new_lhs_cre = lhs_cre_;
            auto new_lhs_ann = identity_lhs_ann;
            double contraction_phase = 1.0;
            for (auto bits = mask; bits != 0; bits &= bits - 1) {
                const auto j = std::countr_zero(bits);
                contraction_phase *= sign_[j] ? 1.0 : -1.0;
                new_lhs_cre.set_bit(set_bits_[j], true);
                new_lhs_ann.set_bit(set_bits_[j], true);
            }
            func(NormalOrderedString(new_lhs_cre, new_lhs_ann), phase_ * contraction_phase);
        };

        const auto next_combination = [](unsigned long long mask) {
            const unsigned long long smallest = mask & -mask;
            const unsigned long long ripple = mask + smallest;
            return (((ripple ^ mask) >> 2) / smallest) | ripple;
        };

        const unsigned long long limit = 1ULL << ncontr;
        const auto max_bits = std::min<int>(max_swapped, static_cast<int>(ncontr));
        for (int nbits = 0; nbits <= max_bits; ++nbits) {
            if (nbits == 0) {
                emit_mask(0);
                continue;
            }
            unsigned long long mask = (1ULL << nbits) - 1ULL;
            while (mask < limit) {
                emit_mask(mask);
                mask = next_combination(mask);
            }
        }
    }

  private:
    constexpr static size_t max_contracted_ops_ = 32;

    int max_rank_;
    Determinant lhs_cre_;
    Determinant lhs_ann_;
    Determinant rhs_cre_;
    Determinant rhs_ann_;
    Determinant ucon_rhs_cre_;
    Determinant con_rhs_cre_;
    Determinant ucon_rhs_ann_;
    sparse_scalar_t phase_;
    std::vector<size_t> set_bits_ = std::vector<size_t>(max_contracted_ops_, 0);
    std::bitset<max_contracted_ops_> sign_;
};

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

    TruncatedNormalProductComputer computer(max_rank_);
    const double screen_thresh_squared = screen_thresh_ * screen_thresh_;
    const auto lhs_terms = collect_terms(lhs, screen_thresh_);
    const auto rhs_terms = collect_terms(rhs, screen_thresh_);
    NormalOrderedSparseOperator result(lhs.reference());
    result.reserve(std::min(lhs_terms.size() * rhs_terms.size(), std::size_t{250000}));

    const std::function<void(const NormalOrderedString&, const sparse_scalar_t)> add_to_result =
        [this, &result,
         screen_thresh_squared](const NormalOrderedString& term,
                                const sparse_scalar_t coefficient) {
            if (above_threshold(coefficient, screen_thresh_squared)) {
                result.add(term, coefficient);
            }
        };

    for (const auto& lhs_term : lhs_terms) {
        for (const auto& rhs_term : rhs_terms) {
            const sparse_scalar_t factor = lhs_term.coefficient * rhs_term.coefficient;
            if (not above_threshold(factor, screen_thresh_squared)) {
                continue;
            }
            const auto contribution = analyze_pair(lhs_term, rhs_term, max_rank_);
            if (not contribution.lhs_rhs and not contribution.rhs_lhs) {
                continue;
            }
            if (contribution.commute) {
                continue;
            }
            if (contribution.lhs_rhs) {
                computer.product(*lhs_term.term, *rhs_term.term, factor, add_to_result);
            }
            if (contribution.rhs_lhs) {
                computer.product(*rhs_term.term, *lhs_term.term, -factor, add_to_result);
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
