#pragma once

#include <cstddef>
#include <span>
#include <utility>
#include <vector>

namespace forte2 {

/// Occupation of each GAS slot in input order, including empty slots.
using occupation_t = std::vector<int>;

/// @brief The alpha and beta GAS occupations allowed by the electron counts and GAS bounds.
/// @details Each entry in pairs indexes one alpha occupation and one beta occupation whose
/// combined electron counts satisfy the GAS bounds.
struct GASOccupations {
    std::vector<occupation_t> alpha;
    std::vector<occupation_t> beta;
    std::vector<std::pair<size_t, size_t>> pairs;
};

GASOccupations get_ci_occupation_patterns(size_t na, size_t nb, std::span<const int> min_occ,
                                          std::span<const int> max_occ,
                                          std::span<const int> gas_size);

/// Distinct occupations reached by removing one electron.
std::vector<occupation_t> generate_1h_occupations(std::span<const occupation_t> occupations);

/// Original occupations and distinct occupations reached by moving one electron.
std::vector<occupation_t> generate_1h1p_occupations(std::span<const occupation_t> occupations,
                                                    std::span<const int> gas_size);

} // namespace forte2
