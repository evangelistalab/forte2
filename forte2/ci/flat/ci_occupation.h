#pragma once

#include <array>
#include <cstddef>
#include <tuple>
#include <utility>
#include <vector>

namespace forte2::ci_flat {

const size_t max_active_spaces = 6;
using occupation_t = std::array<int, max_active_spaces>;

std::tuple<size_t, std::vector<occupation_t>, std::vector<occupation_t>,
           std::vector<std::pair<size_t, size_t>>>
get_ci_occupation_patterns(size_t na, size_t nb, const std::vector<int>& min_occ,
                           const std::vector<int>& max_occ, const std::vector<int>& size);

/// @brief Generates 1h occupations from the given gas occupations. This function is used when
/// making lists of coupling coefficients for 1h, 2h, and 3h strings.
std::vector<std::array<int, 6>>
generate_1h_occupations(const std::vector<std::array<int, 6>>& gas_occupations);

/// @brief Generates 1h1p occupations from the given gas occupations. This function is used when
/// making lists of coupling coefficients for 1h1p strings and it is needed because in GAS it is
/// possible to have intermediate states that fall outside of the GAS occupation constraints.
/// @param gas_occupations The GAS occupations of the strings
/// @param gas_size The number of orbitals in each GAS space
/// @return The given occupations and every occupation reached by moving one electron into any
/// GAS space that has orbitals
std::vector<std::array<int, 6>>
generate_1h1p_occupations(const std::vector<std::array<int, 6>>& gas_occupations,
                          const std::vector<int>& gas_size);

} // namespace forte2::ci_flat