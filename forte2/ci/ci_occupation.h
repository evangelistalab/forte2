#pragma once

#include <tuple>
#include <array>
#include <vector>
#include <string>

namespace forte2 {

const size_t max_active_spaces = 6;
using occupation_t = std::array<int, max_active_spaces>;

std::tuple<size_t, std::vector<std::array<int, 6>>, std::vector<std::array<int, 6>>,
           std::vector<std::pair<size_t, size_t>>>
get_gas_occupation(size_t na, size_t nb, const std::vector<int>& gas_min,
                   const std::vector<int>& gas_max, const std::vector<int>& gas_size);

std::tuple<size_t, std::vector<occupation_t>, std::vector<occupation_t>,
           std::vector<std::pair<size_t, size_t>>>
get_ci_occupation_patterns(size_t na, size_t nb, const std::vector<int>& min_occ,
                           const std::vector<int>& max_occ, const std::vector<int>& size);

std::tuple<std::vector<occupation_t>, std::vector<occupation_t>,
           std::vector<std::pair<size_t, size_t>>>
generate_gas_occupations(int na, int nb, const occupation_t& gas_min_el,
                         const occupation_t& gas_max_el, const occupation_t& gas_size);

std::vector<std::array<int, 6>>
generate_1h_occupations(const std::vector<std::array<int, 6>>& gas_occupations);

std::string occupation_table(size_t num_spaces,
                             const std::vector<std::array<int, 6>>& alfa_occupation,
                             const std::vector<std::array<int, 6>>& beta_occupation,
                             const std::vector<std::pair<size_t, size_t>>& occupation_pairs);

} // namespace forte2