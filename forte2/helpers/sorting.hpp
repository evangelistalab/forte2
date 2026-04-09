#pragma once

#include <vector>
#include <algorithm>
#include <numeric>

namespace forte2 {

/// @brief Generate the permutation that sorts the input vector according to the given comparison
/// function
/// @tparam T Type of the elements in the vector
/// @tparam Compare The comparison function type
/// @param vec The input vector to be sorted
/// @param compare The comparison operator
/// @return A vector of indices representing the permutation that sorts the input vector
template <typename T, typename Compare>
std::vector<std::size_t> sort_permutation(const std::vector<T>& vec, Compare&& compare) {
    std::vector<std::size_t> p(vec.size());
    std::iota(p.begin(), p.end(), 0);
    std::sort(p.begin(), p.end(),
              [&](std::size_t i, std::size_t j) { return compare(vec[i], vec[j]); });
    return p;
}

/// @brief Apply a permutation to a vector in place
/// @tparam T The type of the elements in the vector
/// @param vec The vector to be permuted
/// @param p The permutation vector
template <typename T>
void apply_permutation_in_place(std::vector<T>& vec, const std::vector<std::size_t>& p) {
    std::vector<bool> done(vec.size());
    for (std::size_t i = 0; i < vec.size(); ++i) {
        if (done[i]) {
            continue;
        }
        done[i] = true;
        std::size_t prev_j = i;
        std::size_t j = p[i];
        while (i != j) {
            std::swap(vec[prev_j], vec[j]);
            done[j] = true;
            prev_j = j;
            j = p[j];
        }
    }
}

/// @brief Append unique elements from a sorted vector to another vector in place
/// @tparam T The type of the elements in the vectors
/// @tparam Less The comparison function type
/// @param a The vector we are appending to
/// @param a_sorted The sorted version of vector a
/// @param b_sorted_uniq The vector from which we are appending unique elements
/// @param less The comparison operator less(x, y) returns true if x < y
template <class T, class Less = std::less<T>>
void append_unique_from_sorted_inplace(std::vector<T>& a, const std::vector<T>& a_sorted,
                                       const std::vector<T>& b_sorted_uniq, Less less = Less{}) {
    std::size_t i = 0, j = 0;
    const std::size_t n = a_sorted.size(), m = b_sorted_uniq.size();
    while (i < n && j < m) {
        const T& x = a_sorted[i];
        const T& y = b_sorted_uniq[j];
        if (less(x, y)) {
            ++i;
        } // y is greater, skip x
        else if (less(y, x)) {
            a.push_back(y);
            ++j;
        } // y can be appended
        else {
            ++i;
            ++j;
        } // y is already in a, skip it
    }
    // add the remaining y's as they are greater than all x's
    for (; j < m; ++j)
        a.push_back(b_sorted_uniq[j]);
}

} // namespace forte2