#include <cmath>
#include <complex>
#include <stdexcept>
#include <string>

#include "determinant/determinant_helpers.h"
#include "sparse/cumulant_reference.h"

namespace forte2 {

CumulantReference::CumulantReference(const SparseState& vacuum, std::size_t norb, int max_cumulant,
                                     double screen_thresh)
    : vacuum_(vacuum), norb_(norb), max_cumulant_(max_cumulant), screen_thresh_(screen_thresh) {
    if (norb_ == 0 or norb_ > Determinant::norb()) {
        throw std::invalid_argument("CumulantReference: norb must be between 1 and " +
                                    std::to_string(Determinant::norb()));
    }
    if (max_cumulant_ < 1 or max_cumulant_ > 2) {
        throw std::invalid_argument(
            "CumulantReference: the current implementation supports max_cumulant 1 or 2");
    }
    if (screen_thresh_ < 0.0) {
        throw std::invalid_argument("CumulantReference: screen_thresh must be non-negative");
    }

    for (const auto& [det, coefficient] : vacuum_.elements()) {
        (void)det;
        norm_ += std::conj(coefficient) * coefficient;
    }
    if (std::abs(norm_) <= screen_thresh_) {
        throw std::invalid_argument("CumulantReference: vacuum must have nonzero norm");
    }

    cumulants_.resize(static_cast<std::size_t>(max_cumulant_ + 1));
    build_orbital_spaces();
    build_one_body_density();
    if (max_cumulant_ >= 2) {
        build_two_body_cumulant();
    }
}

const SparseState& CumulantReference::vacuum() const { return vacuum_; }

std::size_t CumulantReference::norb() const { return norb_; }

int CumulantReference::max_cumulant() const { return max_cumulant_; }

double CumulantReference::screen_thresh() const { return screen_thresh_; }

const Determinant& CumulantReference::core_modes() const { return core_modes_; }

const Determinant& CumulantReference::active_modes() const { return active_modes_; }

const Determinant& CumulantReference::virtual_modes() const { return virtual_modes_; }

std::size_t CumulantReference::determinant_mode(std::size_t orbital, bool alpha) const {
    if (orbital >= norb_) {
        throw std::out_of_range("CumulantReference: orbital index is outside the reference space");
    }
    return alpha ? orbital : Determinant::beta_storage_offset + orbital;
}

std::size_t CumulantReference::compact_mode(std::size_t mode) const {
    if (mode < norb_) {
        return mode;
    }
    if (mode >= Determinant::beta_storage_offset and
        mode < Determinant::beta_storage_offset + norb_) {
        return norb_ + mode - Determinant::beta_storage_offset;
    }
    throw std::out_of_range("CumulantReference: spin-orbital index is outside the reference space");
}

void CumulantReference::validate_indices(const Determinant& cre, const Determinant& ann) const {
    if (cre.count_all() != ann.count_all()) {
        throw std::invalid_argument("CumulantReference: creator and annihilator ranks must match");
    }
    if (not all_modes_.is_superset_of(cre) or not all_modes_.is_superset_of(ann)) {
        throw std::out_of_range(
            "CumulantReference: cumulant indices are outside the reference space");
    }
}

sparse_scalar_t CumulantReference::gamma(std::size_t p, bool p_alpha, std::size_t q,
                                         bool q_alpha) const {
    return gamma_mode(determinant_mode(p, p_alpha), determinant_mode(q, q_alpha));
}

sparse_scalar_t CumulantReference::eta(std::size_t p, bool p_alpha, std::size_t q,
                                       bool q_alpha) const {
    const auto delta = (p == q and p_alpha == q_alpha) ? 1.0 : 0.0;
    return sparse_scalar_t{delta} - gamma(p, p_alpha, q, q_alpha);
}

sparse_scalar_t CumulantReference::rdm(const Determinant& cre, const Determinant& ann) const {
    validate_indices(cre, ann);
    if (cre.count_all() == 0) {
        return sparse_scalar_t{1.0};
    }
    return expectation(SQOperatorString(cre, ann));
}

sparse_scalar_t CumulantReference::cumulant(const Determinant& cre, const Determinant& ann) const {
    validate_indices(cre, ann);
    const auto rank = static_cast<int>(cre.count_all());
    if (rank == 0) {
        return sparse_scalar_t{1.0};
    }
    if (rank > max_cumulant_) {
        throw std::out_of_range("CumulantReference: requested cumulant rank is unavailable");
    }
    if (rank == 1) {
        std::vector<std::size_t> cre_modes(1);
        std::vector<std::size_t> ann_modes(1);
        std::size_t ncre = 0;
        std::size_t nann = 0;
        cre.find_set_bits(cre_modes, ncre);
        ann.find_set_bits(ann_modes, nann);
        return gamma_mode(cre_modes[0], ann_modes[0]);
    }
    if (not active_modes_.is_superset_of(cre) or not active_modes_.is_superset_of(ann)) {
        return sparse_scalar_t{0.0};
    }
    const auto& values = cumulants_[static_cast<std::size_t>(rank)];
    const auto it = values.find(SQOperatorString(cre, ann));
    return it == values.end() ? sparse_scalar_t{0.0} : it->second;
}

std::size_t CumulantReference::cumulant_size(int rank) const {
    if (rank < 1 or rank > max_cumulant_) {
        throw std::out_of_range("CumulantReference: requested cumulant rank is unavailable");
    }
    if (rank == 1) {
        std::size_t count = 0;
        for (const auto& value : gamma_) {
            count += std::abs(value) > screen_thresh_;
        }
        return count;
    }
    return cumulants_[static_cast<std::size_t>(rank)].size();
}

sparse_scalar_t CumulantReference::expectation(const SQOperatorString& term) const {
    sparse_scalar_t value = 0.0;
    Determinant new_det;
    for (const auto& [det, coefficient] : vacuum_.elements()) {
        if (not det.can_apply_operator(term.cre(), term.ann())) {
            continue;
        }
        const auto phase =
            apply_operator_to_det_unchecked(det, new_det, term.cre(), term.ann(), term.sign_mask());
        const auto bra = vacuum_.elements().find(new_det);
        if (bra != vacuum_.elements().end()) {
            value += std::conj(bra->second) * phase * coefficient;
        }
    }
    return value / norm_;
}

sparse_scalar_t CumulantReference::gamma_mode(std::size_t p, std::size_t q) const {
    const auto cp = compact_mode(p);
    const auto cq = compact_mode(q);
    return gamma_[cp * (2 * norb_) + cq];
}

void CumulantReference::build_orbital_spaces() {
    for (std::size_t p = 0; p < norb_; ++p) {
        all_modes_.set_na(p, true);
        all_modes_.set_nb(p, true);
    }

    auto occupied_support = Determinant::zero();
    bool first = true;
    for (const auto& [det, coefficient] : vacuum_.elements()) {
        if (std::abs(coefficient) <= screen_thresh_) {
            continue;
        }
        if (not all_modes_.is_superset_of(det)) {
            throw std::invalid_argument(
                "CumulantReference: vacuum contains occupations outside norb");
        }
        occupied_support |= det;
        if (first) {
            core_modes_ = det;
            first = false;
        } else {
            core_modes_ &= det;
        }
    }
    if (first) {
        throw std::invalid_argument("CumulantReference: vacuum has no significant determinants");
    }

    active_modes_ = occupied_support - core_modes_;
    virtual_modes_ = all_modes_ - occupied_support;
}

void CumulantReference::build_one_body_density() {
    const auto nso = 2 * norb_;
    gamma_.assign(nso * nso, sparse_scalar_t{0.0});

    core_modes_.for_each_occ([&](std::size_t p) {
        gamma_[compact_mode(p) * nso + compact_mode(p)] = sparse_scalar_t{1.0};
    });

    std::vector<std::size_t> active(active_modes_.count_all());
    std::size_t nactive = 0;
    active_modes_.find_set_bits(active, nactive);
    for (const auto p : active) {
        for (const auto q : active) {
            auto cre = Determinant::zero();
            auto ann = Determinant::zero();
            cre.set_bit(p, true);
            ann.set_bit(q, true);
            const auto value = expectation(SQOperatorString(cre, ann));
            if (std::abs(value) > screen_thresh_) {
                gamma_[compact_mode(p) * nso + compact_mode(q)] = value;
            }
        }
    }
}

void CumulantReference::build_two_body_cumulant() {
    std::vector<std::size_t> active(active_modes_.count_all());
    std::size_t nactive = 0;
    active_modes_.find_set_bits(active, nactive);
    auto& lambda2 = cumulants_[2];

    for (std::size_t pi = 0; pi < active.size(); ++pi) {
        for (std::size_t qi = pi + 1; qi < active.size(); ++qi) {
            auto cre = Determinant::zero();
            cre.set_bit(active[pi], true);
            cre.set_bit(active[qi], true);
            for (std::size_t ri = 0; ri < active.size(); ++ri) {
                for (std::size_t si = ri + 1; si < active.size(); ++si) {
                    auto ann = Determinant::zero();
                    ann.set_bit(active[ri], true);
                    ann.set_bit(active[si], true);
                    const auto moment = expectation(SQOperatorString(cre, ann));
                    const auto disconnected =
                        gamma_mode(active[pi], active[ri]) * gamma_mode(active[qi], active[si]) -
                        gamma_mode(active[pi], active[si]) * gamma_mode(active[qi], active[ri]);
                    const auto value = moment - disconnected;
                    if (std::abs(value) > screen_thresh_) {
                        lambda2.emplace(SQOperatorString(cre, ann), value);
                    }
                }
            }
        }
    }
}

} // namespace forte2
