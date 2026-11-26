#include "ci_strings.h"
#include "ci/flat/ci_strings.h"

#include <algorithm>
#include <stdexcept>

namespace forte2 {

CIStrings::CIStrings(size_t na, size_t nb, int nirrep, int symmetry,
                     const std::vector<std::vector<int>>& orbital_symmetry,
                     const std::vector<int>& gas_min, const std::vector<int>& gas_max)
    : na_(na), nb_(nb), symmetry_(symmetry), flat_orbital_symmetry_(orbital_symmetry),
      flat_gas_min_(gas_min), flat_gas_max_(gas_max) {
    if (gas_min.size() > orbital_symmetry.size() || gas_max.size() > orbital_symmetry.size())
        throw std::invalid_argument(
            "CIStrings: GAS occupation bounds must refer to existing GAS slots.");

    auto basis = std::make_shared<GASStringBasis>(nirrep, orbital_symmetry);
    if (symmetry < 0 || symmetry >= basis->nirrep())
        throw std::invalid_argument("Target symmetry is outside the irrep range.");
    basis_ = basis;

    auto [alpha, beta, pairs] =
        get_ci_occupation_patterns(na, nb, gas_min, gas_max, basis->sizes());
    occupation_pairs_ = std::move(pairs);

    const auto setup_spin = [&](auto& addresses, std::vector<occupation_t> occupations) {
        auto excited = generate_1h1p_occupations(occupations, gas_size());
        for (size_t holes = 0; holes < 4; ++holes) {
            auto next =
                holes < 3 ? generate_1h_occupations(occupations) : std::vector<occupation_t>{};
            basis->prepare(occupations);
            addresses[holes].emplace(basis, std::move(occupations));
            occupations = std::move(next);
        }
        basis->prepare(excited);
        addresses.back().emplace(basis, std::move(excited));
    };
    setup_spin(alpha_strings_, std::move(alpha));
    setup_spin(beta_strings_, std::move(beta));
    basis->prepare_substitutions();

    for (const auto& [alpha_occ, beta_occ] : occupation_pairs_) {
        for (int alpha_irrep = 0; alpha_irrep < nirrep; ++alpha_irrep) {
            const int alpha_class = static_cast<int>(alpha_occ) * nirrep + alpha_irrep;
            const int beta_class = static_cast<int>(beta_occ) * nirrep + (symmetry ^ alpha_irrep);
            const auto alpha_size = strings(Spin::Alpha).size(alpha_class);
            const auto beta_size = strings(Spin::Beta).size(beta_class);
            if (alpha_size == 0 || beta_size == 0)
                continue;

            const auto next = checked_ci_sum(ndet_, checked_ci_product(alpha_size, beta_size));
            block_indices_.emplace(std::pair{alpha_class, beta_class}, blocks_.size());
            blocks_.push_back({alpha_class, beta_class, alpha_size, beta_size, ndet_});
            ndet_ = next;
        }
    }
}

const ci_flat::CIStrings& CIStrings::flat_strings() const {
    std::lock_guard<std::mutex> lock(flat_strings_mutex_);
    if (!flat_strings_) {
        flat_strings_ = std::make_shared<ci_flat::CIStrings>(
            na_, nb_, nirrep(), symmetry_, flat_orbital_symmetry_, flat_gas_min_, flat_gas_max_);
    }
    return *flat_strings_;
}

std::vector<Determinant> CIStrings::make_determinants() const {
    std::vector<Determinant> dets;
    dets.reserve(ndet_);
    std::vector<LocalStringAddress> work(ngas_spaces());
    for (const auto& block : blocks_) {
        // Compose each beta string once per determinant block.
        std::vector<String> beta;
        beta.reserve(block.beta_size);
        for (size_t b = 0; b < block.beta_size; ++b)
            beta.push_back(strings(Spin::Beta).string_at(block.beta_class, b, work));
        for (size_t a = 0; a < block.alpha_size; ++a) {
            const auto alpha = strings(Spin::Alpha).string_at(block.alpha_class, a, work);
            for (const auto& b : beta)
                dets.emplace_back(alpha, b);
        }
    }
    return dets;
}

size_t CIStrings::determinant_address(const Determinant& determinant) const {
    std::vector<LocalStringAddress> work(ngas_spaces());
    occupation_t occupation(ngas_spaces());
    const auto alpha = strings(Spin::Alpha).find(determinant.a_string(), work, occupation);
    const auto beta = strings(Spin::Beta).find(determinant.b_string(), work, occupation);
    if (!alpha || !beta)
        throw std::out_of_range("Determinant is outside the CI space.");
    const auto it = block_indices_.find({alpha->string_class, beta->string_class});
    if (it == block_indices_.end())
        throw std::out_of_range("Determinant is outside the CI space.");

    const auto& block = blocks_[it->second];
    return block.offset + alpha->rank * block.beta_size + beta->rank;
}

Determinant CIStrings::determinant(size_t address) const {
    if (address >= ndet_)
        throw std::out_of_range("Determinant address is outside the CI space.");
    const auto& block =
        *std::prev(std::ranges::upper_bound(blocks_, address, {}, &DeterminantBlock::offset));
    const size_t local = address - block.offset;
    std::vector<LocalStringAddress> work(ngas_spaces());
    const auto alpha =
        strings(Spin::Alpha).string_at(block.alpha_class, local / block.beta_size, work);
    const auto beta =
        strings(Spin::Beta).string_at(block.beta_class, local % block.beta_size, work);
    return Determinant(alpha, beta);
}

} // namespace forte2
