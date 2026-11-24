#include "ci_occupation.h"

#include <algorithm>
#include <map>
#include <set>

namespace forte2 {
namespace {

size_t occupation_index(const occupation_t& occupation, std::vector<occupation_t>& occupations,
                        std::map<occupation_t, size_t>& indices) {
    const auto [it, inserted] = indices.try_emplace(occupation, occupations.size());
    if (inserted)
        occupations.push_back(occupation);
    return it->second;
}

} // namespace

GASOccupations get_ci_occupation_patterns(size_t na, size_t nb, std::span<const int> min_occ,
                                          std::span<const int> max_occ,
                                          std::span<const int> gas_size) {
    const size_t ngas = gas_size.size();
    // Remaining capacity per spin and remaining electron bounds at each GAS slot.
    std::vector<int> capacity(ngas + 1), minimum(ngas + 1), maximum(ngas + 1);
    occupation_t lower(ngas), upper(ngas);
    for (size_t g = ngas; g-- > 0;) {
        lower[g] = g < min_occ.size() ? std::max(min_occ[g], 0) : 0;
        upper[g] = g < max_occ.size() ? std::min(max_occ[g], 2 * gas_size[g]) : 2 * gas_size[g];
        if (lower[g] > upper[g])
            return {};
        capacity[g] = capacity[g + 1] + gas_size[g];
        minimum[g] = minimum[g + 1] + lower[g];
        maximum[g] = maximum[g + 1] + upper[g];
    }

    GASOccupations result;
    if (na > static_cast<size_t>(capacity.front()) || nb > static_cast<size_t>(capacity.front()))
        return result;

    occupation_t alpha(ngas), beta(ngas);
    std::map<occupation_t, size_t> alpha_indices, beta_indices;

    const auto enumerate = [&](this const auto& self, size_t g, int a, int b) -> void {
        // Empty slots already have zero occupations and require no recursion.
        while (g < ngas && gas_size[g] == 0)
            ++g;
        if (a > capacity[g] || b > capacity[g] || a + b < minimum[g] || a + b > maximum[g])
            return;
        if (g == ngas) {
            const auto ai = occupation_index(alpha, result.alpha, alpha_indices);
            const auto bi = occupation_index(beta, result.beta, beta_indices);
            result.pairs.emplace_back(ai, bi);
            return;
        }

        const int max_a = std::min({a, upper[g], gas_size[g]});
        for (int ag = std::max({0, lower[g] - gas_size[g], a - capacity[g + 1]}); ag <= max_a;
             ++ag) {
            const int min_b = std::max({0, lower[g] - ag, b - capacity[g + 1]});
            const int max_b = std::min({b, upper[g] - ag, gas_size[g]});
            alpha[g] = ag;
            for (int bg = min_b; bg <= max_b; ++bg) {
                beta[g] = bg;
                self(g + 1, a - ag, b - bg);
            }
        }
    };
    enumerate(0, static_cast<int>(na), static_cast<int>(nb));
    return result;
}

std::vector<occupation_t> generate_1h_occupations(std::span<const occupation_t> occupations) {
    std::vector<occupation_t> result;
    std::set<occupation_t> seen;
    for (const auto& occupation : occupations) {
        auto candidate = occupation;
        for (size_t g = 0; g < occupation.size(); ++g) {
            if (occupation[g] == 0)
                continue;
            --candidate[g];
            if (seen.insert(candidate).second)
                result.push_back(candidate);
            ++candidate[g];
        }
    }
    return result;
}

std::vector<occupation_t> generate_1h1p_occupations(std::span<const occupation_t> occupations,
                                                    std::span<const int> gas_size) {
    std::vector<occupation_t> result;
    std::set<occupation_t> seen;
    for (const auto& occupation : occupations) {
        auto candidate = occupation;
        for (size_t from = 0; from < occupation.size(); ++from) {
            if (occupation[from] == 0)
                continue;
            --candidate[from];
            for (size_t to = 0; to < occupation.size(); ++to) {
                if (candidate[to] == gas_size[to])
                    continue;
                ++candidate[to];
                if (seen.insert(candidate).second)
                    result.push_back(candidate);
                --candidate[to];
            }
            ++candidate[from];
        }
        // A zero-electron string is still a spectator.
        if (seen.insert(occupation).second)
            result.push_back(occupation);
    }
    return result;
}

} // namespace forte2
