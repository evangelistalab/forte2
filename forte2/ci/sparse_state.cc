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
    if (n == 1 || state.size() <= 100000) {
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
        std::advance(end_it, current_chunk_size);
        views.emplace_back(it, end_it);
        it = end_it;
    }
    return views;
}

std::vector<std::pair<size_t, size_t>> split_sparse_state_buckets(const SparseState& state,
                                                                  size_t n) {
    std::vector<std::pair<size_t, size_t>> buckets;
    size_t num_buckets = state.bucket_count();

    if (n == 1 || state.size() <= n) {
        buckets.emplace_back(0, num_buckets);
        return buckets;
    }
    buckets.reserve(num_buckets);
    size_t chunk_size = num_buckets / n;
    size_t remainder = num_buckets % n;
    size_t start = 0;
    for (size_t i = 0; i < n; ++i) {
        size_t current_chunk_size = chunk_size + (i < remainder ? 1 : 0);
        size_t end = start + current_chunk_size;
        if (end > num_buckets) {
            end = num_buckets;
        }
        buckets.emplace_back(start, end);
        start = end;
    }
    return buckets;
}

} // namespace forte2
