#pragma once

#include <array>
#include <map>
#include <optional>
#include <span>
#include <vector>

#include "ci/ci_string_address.h"
#include "helpers/spin.h"

namespace forte2 {

/// A row-major alpha-by-beta coefficient block of a single allowed GAS occupation pair.
/// Both spin classes carry a total irrep. The block belongs to their XOR irrep.
struct DeterminantBlock {
    int alpha_class;
    int beta_class;
    size_t alpha_size;
    size_t beta_size;
    size_t offset;

    size_t size() const { return alpha_size * beta_size; }
    int string_class(Spin spin) const { return is_alpha(spin) ? alpha_class : beta_class; }
    int spectator_class(Spin spin) const { return is_alpha(spin) ? beta_class : alpha_class; }
    size_t spectator_size(Spin spin) const { return is_alpha(spin) ? beta_size : alpha_size; }
};

/// The symmetry-resolved GAS determinant space and its intermediate spin-string products.
/// Spin string spaces share the local GAS basis. Only the allowed alpha/beta products get
/// coefficient blocks. Full determinant strings can be composed at the API boundary.
class CIStrings {
  public:
    CIStrings(size_t na, size_t nb, int nirrep, int symmetry,
              const std::vector<std::vector<int>>& orbital_symmetry,
              const std::vector<int>& gas_min, const std::vector<int>& gas_max);

    size_t na() const { return na_; }
    size_t nb() const { return nb_; }
    size_t nas() const { return strings(Spin::Alpha).total_size(); }
    size_t nbs() const { return strings(Spin::Beta).total_size(); }
    size_t ndet() const { return ndet_; }
    int symmetry() const { return symmetry_; }
    int nirrep() const { return basis_->nirrep(); }
    size_t ngas_spaces() const { return basis_->ngas(); }
    size_t norb() const { return basis_->norb(); }
    const auto& gas_size() const { return basis_->sizes(); }
    const auto& orbital_irreps() const { return basis_->irreps(); }

    const StringAddress& strings(Spin spin, size_t holes = 0) const {
        return *(is_alpha(spin) ? alpha_strings_ : beta_strings_)[holes];
    }
    const StringAddress& excited_strings(Spin spin) const {
        return *(is_alpha(spin) ? alpha_strings_ : beta_strings_).back();
    }
    const auto& blocks() const { return blocks_; }

    const auto& gas_alpha_occupations() const { return strings(Spin::Alpha).occupations(); }
    const auto& gas_beta_occupations() const { return strings(Spin::Beta).occupations(); }
    const auto& gas_occupations() const { return occupation_pairs_; }

    size_t determinant_address(const Determinant& determinant) const;
    Determinant determinant(size_t address) const;
    std::vector<Determinant> make_determinants() const;

  private:
    size_t na_;
    size_t nb_;
    int symmetry_;
    std::shared_ptr<const GASStringBasis> basis_;
    /// Variational, one-, two-, and three-hole, and one-hole/one-particle string spaces.
    std::array<std::optional<StringAddress>, 5> alpha_strings_, beta_strings_;
    std::vector<std::pair<size_t, size_t>> occupation_pairs_;
    std::vector<DeterminantBlock> blocks_;
    std::map<std::pair<int, int>, size_t> block_indices_;
    size_t ndet_ = 0;
};

/// View a determinant block with the spectator index contiguous. Alpha already uses this
/// order. Beta uses the supplied scratch space for the transpose.
template <typename Scalar>
std::span<Scalar> gather_block(std::span<Scalar> source, std::span<Scalar> scratch, Spin spin,
                               const DeterminantBlock& block) {
    const auto coefficients = source.subspan(block.offset, block.size());
    if (is_alpha(spin))
        return coefficients;
    for (size_t a = 0; a < block.alpha_size; ++a)
        for (size_t b = 0; b < block.beta_size; ++b)
            scratch[b * block.alpha_size + a] = coefficients[a * block.beta_size + b];
    return scratch.first(block.size());
}

/// Add a block from active-spin-major order to the full coefficient vector.
template <typename Scalar>
void scatter_block(std::span<Scalar> source, std::span<Scalar> destination, Spin spin,
                   const DeterminantBlock& block) {
    const auto coefficients = destination.subspan(block.offset, block.size());
    if (is_alpha(spin)) {
        for (size_t i = 0; i < block.size(); ++i)
            coefficients[i] += source[i];
    } else {
        for (size_t a = 0; a < block.alpha_size; ++a)
            for (size_t b = 0; b < block.beta_size; ++b)
                coefficients[a * block.beta_size + b] += source[b * block.alpha_size + a];
    }
}

} // namespace forte2
