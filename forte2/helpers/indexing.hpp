#pragma once

#include <cmath>
#include <cstddef>
#include <utility>

namespace forte2 {
/// @brief Compute the index of the pair (i, j) where max(i,j) >= min(i,j).
/// This is used to map pairs of indices to a single index in a 1D array.
template <typename T> std::size_t pair_index_gen(T i, T j, T dim) {
    return static_cast<std::size_t>(i) * static_cast<std::size_t>(dim) +
           static_cast<std::size_t>(j);
}

/// @brief Compute the index of the pair (i, j) where max(i,j) >= min(i,j).
/// This is used to map pairs of indices to a single index in a 1D array.
template <typename T> std::size_t pair_index_geq(T i, T j) {
    std::size_t a = i, b = j;
    if (a < b)
        std::swap(a, b);
    return (a * (a + 1)) / 2 + b;
}

/// @brief Compute the index of the pair (i, j) where max(i,j) > min(i,j).
/// This is used to map pairs of distinct indices to a single index in a 1D array.
template <typename T> std::size_t pair_index_gt(T i, T j) {
    std::size_t a = i, b = j;
    if (a < b)
        std::swap(a, b);
    return (a * (a - 1)) / 2 + b;
}

/// @brief From the index of the pair (i, j) where max(i,j) > min(i,j) compute the indices i and j
/// (zero based).
template <typename T> std::pair<T, T> inv_pair_index_gt(T n) {
    // solve for i = floor((1 + sqrt(1+8n)) / 2)
    double d = static_cast<double>(n);
    T i = static_cast<T>(std::floor((1.0 + std::sqrt(1.0 + 8.0 * d)) / 2.0));
    T j = n - (i * (i - 1)) / 2;
    return {i, j};
}

/// @brief Compute the index of the triplet (i, j, k) where i > j > k.
/// This is used to map pairs of indices to a single index in a 1D array.
template <typename T> std::size_t triplet_index_gt(T i, T j, T k) {
    std::size_t a = i, b = j, c = k;
    if (a < b)
        std::swap(a, b);
    if (a < c)
        std::swap(a, c);
    if (b < c)
        std::swap(b, c);
    return (a * (a - 1) * (a - 2)) / 6 + (b * (b - 1)) / 2 + c;
}

/// @brief Compute the index of the triplet (i, j, k) where i > j and k is unrestricted.
/// This is used to map triplets of indices to a single index in a 1D array.
/// @param i The first index.
/// @param j The second index.
/// @param k The third index.
/// @param dimk The dimension of the third index.
/// @return The index of the triplet (i, j, k).
template <typename T> std::size_t triplet_index_aab(T i, T j, T k, T dimk) {
    return pair_index_gt(i, j) * static_cast<std::size_t>(dimk) + static_cast<std::size_t>(k);
}

/// @brief Compute the index of the triplet (i, j, k) where j > k and i is unrestricted.
/// This is used to map triplets of indices to a single index in a 1D array.
/// @param i The first index.
/// @param j The second index.
/// @param k The third index.
/// @param dimjk The dimension of the composite index (j, k).
/// @return The index of the triplet (i, j, k).
template <typename T> std::size_t triplet_index_abb(T i, T j, T k, T dimjk) {
    return static_cast<std::size_t>(i) * static_cast<std::size_t>(dimjk) + pair_index_gt(j, k);
}

} // namespace forte2