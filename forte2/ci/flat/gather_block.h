#pragma once

#include <span>
#include <vector>

#include "ci/flat/ci_strings.h"
#include "helpers/spin.h"

namespace forte2::ci_flat {

/// @brief View a flat coefficient block with the spectator index contiguous.
template <typename Scalar>
std::span<Scalar> gather_block(std::span<Scalar> source, std::vector<Scalar>& scratch, Spin spin,
                               const CIStrings& lists, int alpha_class, int beta_class) {
    const auto block = lists.string_class()->block_index(alpha_class, beta_class);
    const auto alpha_size = lists.alpha_address()->strpcls(alpha_class);
    const auto beta_size = lists.beta_address()->strpcls(beta_class);
    const auto coefficients = source.subspan(lists.block_offset(block), alpha_size * beta_size);
    if (is_alpha(spin))
        return coefficients;
    for (size_t a = 0; a < alpha_size; ++a)
        for (size_t b = 0; b < beta_size; ++b)
            scratch[b * alpha_size + a] = coefficients[a * beta_size + b];
    return std::span(scratch).first(coefficients.size());
}

} // namespace forte2::ci_flat
