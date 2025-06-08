

/// @brief Compute the index of the pair (i, j) where max(i,j) >= min(i,j).
/// This is used to map pairs of indices to a single index in a 1D array.
template <typename T> T pair_index_geq(T i, T j) {
    if (i < j)
        std::swap(i, j);
    return (i * (i + 1)) / 2 + j;
}

/// @brief Compute the index of the pair (i, j) where max(i,j) > min(i,j).
template <typename T> T pair_index_gt(T i, T j) {
    if (i < j)
        std::swap(i, j);
    return (i * (i - 1)) / 2 + j;
}