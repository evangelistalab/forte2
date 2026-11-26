#include <algorithm>
#include <iterator>
#include <utility>

#include "ci_occupation.h"

namespace forte2::ci_flat {

void add_occupation_pair(const occupation_t& alpha_occ, const occupation_t& beta_occ,
                         std::vector<occupation_t>& alpha_occupations,
                         std::vector<occupation_t>& beta_occupations,
                         std::vector<std::pair<size_t, size_t>>& occupation_pairs) {
    size_t alpha_index;
    if (auto alpha_it = std::find(alpha_occupations.begin(), alpha_occupations.end(), alpha_occ);
        alpha_it == alpha_occupations.end()) {
        alpha_occupations.push_back(alpha_occ);
        alpha_index = alpha_occupations.size() - 1;
    } else {
        alpha_index = std::distance(alpha_occupations.begin(), alpha_it);
    }
    size_t beta_index;
    if (auto beta_it = std::find(beta_occupations.begin(), beta_occupations.end(), beta_occ);
        beta_it == beta_occupations.end()) {
        beta_occupations.push_back(beta_occ);
        beta_index = beta_occupations.size() - 1;
    } else {
        beta_index = std::distance(beta_occupations.begin(), beta_it);
    }
    occupation_pairs.push_back(std::make_pair(alpha_index, beta_index));
}

void recursive_ormas_generation(int na, int nb, const occupation_t& min_occ,
                                const occupation_t& max_occ, const occupation_t& space_size,
                                size_t num_spaces, size_t space_count, occupation_t alpha_occ,
                                occupation_t beta_occ, std::vector<occupation_t>& alpha_occupations,
                                std::vector<occupation_t>& beta_occupations,
                                std::vector<std::pair<size_t, size_t>>& occupation_pairs) {
    // if we reach the last space
    if (space_count == num_spaces - 1) {
        int occ_max = std::max(na, nb);
        int occ_min = std::min(na, nb);
        int occ_total = na + nb;
        // ensure that the number of electrons in the last space is within the bounds
        if (occ_total <= max_occ[space_count] and occ_total >= min_occ[space_count] and
            occ_max <= space_size[space_count] and occ_min >= 0) {
            // add the last space to the occupation
            alpha_occ[space_count] = na;
            beta_occ[space_count] = nb;
            add_occupation_pair(alpha_occ, beta_occ, alpha_occupations, beta_occupations,
                                occupation_pairs);
        }
    } else {
        // the minimum number of alpha electrons (na_min) is the total number of electrons minus
        // the maximum number that can filled in the beta levels of this space. To ensure that the
        // number is positive, we take the maximum of this number and zero.
        int na_min = std::max(min_occ[space_count] - space_size[space_count], 0);
        // the maximum number of alpha electrons (na_max) is the total number of electrons, or
        // at most the size of this space
        int na_max = std::min(max_occ[space_count], space_size[space_count]);
        for (int space_na = na_min; space_na <= na_max; space_na++) {
            // the minimum number of beta electrons (nb_min) is the total number of electrons
            // minus those allocated to the alpha levels of this space. To ensure that the number is
            // positive, we take the maximum of this number and zero.
            int nb_min = std::max(min_occ[space_count] - space_na, 0);
            // the maximum number of beta electrons (nb_max) is the total number of electrons
            // minus the number of alpha electrons, or at most the size of this space
            int nb_max = std::min(max_occ[space_count] - space_na, space_size[space_count]);
            for (int space_nb = nb_min; space_nb <= nb_max; space_nb++) {
                int occ_total = space_na + space_nb;
                // this next check might be useless
                if (occ_total <= max_occ[space_count] and occ_total >= min_occ[space_count]) {
                    // add the current space to the occupation
                    alpha_occ[space_count] = space_na;
                    beta_occ[space_count] = space_nb;
                    recursive_ormas_generation(na - space_na, nb - space_nb, min_occ, max_occ,
                                               space_size, num_spaces, space_count + 1, alpha_occ,
                                               beta_occ, alpha_occupations, beta_occupations,
                                               occupation_pairs);
                }
            }
        }
    }
}

std::tuple<size_t, std::vector<occupation_t>, std::vector<occupation_t>,
           std::vector<std::pair<size_t, size_t>>>
generate_ormas_occupations(size_t na, size_t nb, const occupation_t& min_occ,
                           const occupation_t& max_occ, const occupation_t& space_size,
                           size_t num_spaces) {
    occupation_t alpha_occ{};
    occupation_t beta_occ{};
    std::vector<occupation_t> alpha_occupations;
    std::vector<occupation_t> beta_occupations;
    std::vector<std::pair<size_t, size_t>> occupation_pairs;
    recursive_ormas_generation(na, nb, min_occ, max_occ, space_size, num_spaces, 0, alpha_occ,
                               beta_occ, alpha_occupations, beta_occupations, occupation_pairs);
    return std::make_tuple(num_spaces, std::move(alpha_occupations), std::move(beta_occupations),
                           std::move(occupation_pairs));
}

std::tuple<size_t, std::vector<occupation_t>, std::vector<occupation_t>,
           std::vector<std::pair<size_t, size_t>>>
get_ci_occupation_patterns(size_t na, size_t nb, const std::vector<int>& occ_min,
                           const std::vector<int>& occ_max, const std::vector<int>& gas_size) {
    // The vectors of maximum number of electrons, minimum number of electrons,
    // and the number of orbitals
    occupation_t max_occ{};
    occupation_t min_occ{};
    occupation_t space_size{};

    // Keep empty GAS slots so the bounds and occupation arrays refer to the same spaces.
    const size_t num_spaces = gas_size.size();
    for (size_t n = 0; n < num_spaces; n++) {
        space_size[n] = gas_size[n];
        const int capacity = std::min(2 * static_cast<size_t>(gas_size[n]), na + nb);
        max_occ[n] = n < occ_max.size() ? std::min(occ_max[n], capacity) : capacity;
        min_occ[n] = n < occ_min.size() ? occ_min[n] : 0;
    }
    return generate_ormas_occupations(na, nb, min_occ, max_occ, space_size, num_spaces);
}

std::vector<std::array<int, 6>>
generate_1h_occupations(const std::vector<std::array<int, 6>>& gas_occupations) {
    std::vector<std::array<int, 6>> one_hole_occ;
    // Loop over all the GAS alpha/beta occupations
    for (const auto& gas_occupation : gas_occupations) {
        for (size_t n = 0; n < 6; n++) {
            // Check if we can remove an electron from the GAS
            if (gas_occupation[n] >= 1) {
                // If so, remove it and store the new occupation
                std::array<int, 6> new_occ = gas_occupation;
                new_occ[n] -= 1;
                // Check if the new occupation is already in the list
                auto it = std::find(one_hole_occ.begin(), one_hole_occ.end(), new_occ);
                if (it == one_hole_occ.end()) {
                    one_hole_occ.push_back(new_occ);
                }
            }
        }
    }
    return one_hole_occ;
}

std::vector<std::array<int, 6>>
generate_1h1p_occupations(const std::vector<std::array<int, 6>>& gas_occupations,
                          const std::vector<int>& gas_size) {
    const size_t num_gas_spaces = std::min(gas_size.size(), max_active_spaces);
    std::vector<std::array<int, 6>> one_hole_one_particle_occ;
    auto add_unique = [&](const std::array<int, 6>& occ) {
        if (std::find(one_hole_one_particle_occ.begin(), one_hole_one_particle_occ.end(), occ) ==
            one_hole_one_particle_occ.end()) {
            one_hole_one_particle_occ.push_back(occ);
        }
    };
    for (const auto& gas_occupation : gas_occupations) {
        for (size_t n = 0; n < num_gas_spaces; n++) {
            if (gas_occupation[n] >= 1) {
                // Generate all valid GAS occupations resulting from
                // moving an electron this GAS to all other GASes
                for (size_t s = 0; s < num_gas_spaces; s++) {
                    std::array<int, 6> new_occ = gas_occupation;
                    new_occ[n] -= 1;
                    new_occ[s] += 1;
                    if (new_occ[s] <= gas_size[s])
                        add_unique(new_occ);
                }
            }
        }
    }
    // Edge case: when all GASes are empty, we still need a single allowed
    // GAS occupation list, this is so that an empty spectator string is
    // generated for the Knowles-Handy algorithm
    for (const auto& gas_occupation : gas_occupations) {
        add_unique(gas_occupation);
    }
    return one_hole_one_particle_occ;
}

} // namespace forte2::ci_flat
