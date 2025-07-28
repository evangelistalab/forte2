#pragma once

#include <cmath>
#include <utility>

namespace forte2 {
/// @brief Compute the index of the pair (i, j) where max(i,j) >= min(i,j).
/// This is used to map pairs of indices to a single index in a 1D array.
template <typename T> T pair_index_geq(T i, T j) {
    if (i < j)
        std::swap(i, j);
    return (i * (i + 1)) / 2 + j;
}

/// @brief Compute the index of the pair (i, j) where max(i,j) > min(i,j).
/// This is used to map pairs of distinct indices to a single index in a 1D array.
template <typename T> T pair_index_gt(T i, T j) {
    if (i < j)
        std::swap(i, j);
    return (i * (i - 1)) / 2 + j;
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
template <typename T> T triplet_index_gt(T i, T j, T k) {
    if (i < j)
        std::swap(i, j);
    if (i < k)
        std::swap(i, k);
    if (j < k)
        std::swap(j, k);
    return (i * (i - 1) * (i - 2)) / 6 + (j * (j - 1)) / 2 + k;
}

/// @brief Compute the index of the triplet (i, j, k) where i > j and k is unrestricted.
/// This is used to map triplets of indices to a single index in a 1D array.
/// @param i The first index.
/// @param j The second index.
/// @param k The third index.
/// @param dimk The dimension of the third index.
/// @return The index of the triplet (i, j, k).
template <typename T> T triplet_index_aab(T i, T j, T k, T dimk) {
    return pair_index_gt(i, j) * dimk + k;
}

/// @brief Compute the index of the triplet (i, j, k) where j > k and i is unrestricted.
/// This is used to map triplets of indices to a single index in a 1D array.
/// @param i The first index.
/// @param j The second index.
/// @param k The third index.
/// @param dimjk The dimension of the composite index (j, k).
/// @return The index of the triplet (i, j, k).
template <typename T> T triplet_index_abb(T i, T j, T k, T dimjk) {
    return i * dimjk + pair_index_gt(j, k);
}

} // namespace forte2