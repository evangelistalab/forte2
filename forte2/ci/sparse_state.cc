#include <algorithm>
#include <cmath>
#include <numeric>
#include <algorithm>

#include "helpers/string_algorithms.h"
#include "ci/determinant.hpp"
#include "ci/sparse_state.h"

namespace forte2 {

std::string SparseState::str(int n) const {
    n = std::max(n, static_cast<int>(Determinant::norb()));
    std::string s;
    for (const auto& [det, c] : elements()) {
        if (std::abs(c) > 1.0e-8) {
            s += forte2::str(det, n) + " * " + to_string_with_precision(c, 8) + "\n";
        }
    }
    return s;
}

std::vector<SparseStateView> split_sparse_state(const SparseState& state, size_t n) {
    std::vector<SparseStateView> views;
    if (n == 1 || state.size() <= n) {
        views.emplace_back(state.begin(), state.end());
        return views;
    }
    views.reserve(n);

    auto it = state.begin();
    size_t chunk_size = state.size() / n;
    size_t remainder = state.size() % n;

    for (size_t i = 0; i < n; ++i) {
        size_t current_chunk_size = chunk_size + (i < remainder ? 1 : 0);
        auto end_it = it;
        // this would be extremely inefficient for std::unordered_map
        // but SparseState uses ankerl::unordered_dense::map which 
        // uses a flat layout internally, so iterators are very efficient
        std::advance(end_it, current_chunk_size);
        views.emplace_back(it, end_it);
        it = end_it;
    }
    return views;
}

} // namespace forte2
