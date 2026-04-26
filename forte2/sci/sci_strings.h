#pragma once

#include <vector>

#include "helpers/unordered_dense.h"
#include "ci/determinant.h"

namespace forte2 {

class SelectedCIStrings {
  public:
    // == Class Constructors ==
    /// @brief Construct from a list of determinants
    SelectedCIStrings(size_t norb, std::vector<Determinant>& sorted_dets);

    /// @brief Default constructor
    SelectedCIStrings() = default;

    /// @return The number of orbitals
    size_t norb() const { return norb_; }

    /// @return The number of determinants
    size_t ndets() const { return ndets_; }

    /// @return The sorted determinants
    const std::vector<Determinant>& sorted_dets() const { return sorted_dets_; }

    /// @return The i-th sorted first string
    const String& sorted_first_string(size_t i) const { return sorted_first_string_[i]; }

    /// @return The i-th sorted second string
    const String& sorted_second_string(size_t i) const { return sorted_second_string_[i]; }

    /// @return The permutation that sorts the determinants (perm[i] gives the index in the original
    /// det ordering)
    const std::vector<size_t>& det_permutation() const { return det_permutation_; }

    /// @return The range of indices of the determinants that go with the i-th unique first string
    /// as (start, end) indices in sorted_dets_
    const std::pair<size_t, size_t> range(size_t i) const { return first_string_range_[i]; }

    /// @return The index sorted second string that corresponds to the i-th determinant in
    /// sorted_dets_
    const size_t sorted_dets_second_string(size_t i) const { return sorted_dets_second_string_[i]; }

    /// @return The number of unique first strings
    size_t first_string_size() const { return sorted_first_string_.size(); }

    /// @return For a given first string, this maps the second string index to the determinant index
    /// in the original ordering
    const std::vector<ankerl::unordered_dense::map<size_t, size_t, std::hash<size_t>>>&
    second_string_to_sorted_det_index() const {
        return second_string_to_sorted_det_index_;
    }

    /// @return For a given first string, this maps the second string index to the determinant index
    /// in the original ordering
    const std::vector<ankerl::unordered_dense::map<size_t, size_t, std::hash<size_t>>>&
    second_string_to_det_index() const {
        return second_string_to_det_index_;
    }

    /// @return The one-hole strings
    const std::vector<String>& one_hole_first_strings() const { return one_hole_first_strings_; }

    /// @return The substitution list of one-hole strings
    const std::vector<std::vector<std::tuple<size_t, size_t, double>>>&
    one_hole_first_string_list() const {
        return one_hole_first_string_list_;
    }

    /// @return The inverse substitution list of one-hole strings
    const std::vector<std::vector<std::tuple<size_t, size_t, double>>>&
    one_hole_first_string_list_inv() const {
        return one_hole_first_string_list_inv_;
    }

    /// @return The substitution list of one-hole strings sorted by the index of the orbital that
    /// was removed
    /// Stores orbital -> tuples of (hole_idx, I, sign) where I is the index of the string and
    /// hole_idx is the index of the removed orbital
    const std::vector<std::vector<std::tuple<size_t, size_t, double>>>&
    one_hole_first_string_list_by_orbital() const {
        return one_hole_first_string_list_by_orbital_;
    }

    /// @brief Map from one-hole string to its index
    const ankerl::unordered_dense::map<String, size_t, String::Hash>&
    one_hole_first_strings_index() const {
        return one_hole_first_strings_index_;
    }

    /// @return The one-hole strings for the second string
    const std::vector<String>& one_hole_second_strings() const { return one_hole_second_strings_; }

    /// @return The substitution list of one-hole strings for the second string
    const std::vector<std::vector<std::tuple<size_t, size_t, double>>>&
    one_hole_second_string_list() const {
        return one_hole_second_string_list_;
    }

    /// @return The inverse substitution list of one-hole strings for the second string
    const std::vector<std::vector<std::tuple<size_t, size_t, double>>>&
    one_hole_second_string_list_inv() const {
        return one_hole_second_string_list_inv_;
    }

    /// @return The substitution list of one-hole strings for the second string sorted by the
    /// index of the orbital that was removed
    /// Stores orbital -> tuples of (hole_idx, I, sign) where I is the index of the string and
    /// hole_idx is the index of the removed orbital
    const std::vector<std::vector<std::tuple<size_t, size_t, double>>>&
    one_hole_second_string_list_by_orbital() const {
        return one_hole_second_string_list_by_orbital_;
    }

    /// @return The two-hole strings
    const std::vector<String>& two_hole_strings() const { return two_hole_strings_; }

    /// @return The substitution list of two-hole strings
    const std::vector<std::vector<std::tuple<size_t, size_t, size_t, double>>>&
    two_hole_string_list() const {
        return two_hole_string_list_;
    }

    /// @return The inverse substitution list of two-hole strings
    const std::vector<std::vector<std::tuple<size_t, size_t, size_t, double>>>&
    two_hole_string_list_inv() const {
        return two_hole_string_list_inv_;
    }

  private:
    // == Class Private Functions ==
    void initialize_sorted_strings(std::vector<Determinant>& dets);
    void build_second_string_to_det_index();
    void build_one_hole_strings_and_lists(
        const std::vector<String>& sorted_strings, std::vector<String>& one_hole_strings,
        std::vector<std::vector<std::tuple<size_t, size_t, double>>>& list,
        std::vector<std::vector<std::tuple<size_t, size_t, double>>>& inverse_list,
        std::vector<std::vector<std::tuple<size_t, size_t, double>>>& list_by_orbital,
        ankerl::unordered_dense::map<String, size_t, String::Hash>& index_map);
    void build_two_hole_strings();

    // == Class Protected Variables ==

    /// @brief Number of orbitals
    size_t norb_ = 0;
    /// @brief Number of determinants
    size_t ndets_ = 0;
    /// @brief The sorted determinants
    std::vector<Determinant> sorted_dets_;
    /// @brief The permutation that sorts the determinants
    /// det_permutation_[i] gives the index in the original det ordering
    std::vector<size_t> det_permutation_;
    /// @brief The range of each unique first string. first_string_range_[i] = (start, end)
    /// where start and end are indices in sorted_dets_ and sorted_dets_second_string_.
    std::vector<std::pair<size_t, size_t>> first_string_range_;
    /// @brief The unique first strings
    std::vector<String> sorted_first_string_;
    /// @brief The unique second strings
    std::vector<String> sorted_second_string_;
    /// @brief The unique addresses of the second strings corresponding to each determinant in
    /// sorted_dets_
    std::vector<size_t> sorted_dets_second_string_;
    /// @brief Map from first string to its index
    ankerl::unordered_dense::map<String, size_t, String::Hash> first_string_index_;
    /// @brief Map from second string to its index
    ankerl::unordered_dense::map<String, size_t, String::Hash> second_string_index_;
    /// @brief For each unique first string, map from second string index to determinant index
    std::vector<ankerl::unordered_dense::map<size_t, size_t, std::hash<size_t>>>
        second_string_to_sorted_det_index_;
    /// @brief For each unique first string, map from second string index to determinant index in
    /// the original ordering
    std::vector<ankerl::unordered_dense::map<size_t, size_t, std::hash<size_t>>>
        second_string_to_det_index_;

    /// @brief Precomputed list of one-hole strings
    std::vector<String> one_hole_first_strings_;
    /// @brief Map from one-hole string to its index
    ankerl::unordered_dense::map<String, size_t, String::Hash> one_hole_first_strings_index_;
    /// @brief Precomputed list of one-particle strings with sign for each orbital
    /// Stores I -> tuples of (orbital, K, sign) where K is the index of the one-hole string
    std::vector<std::vector<std::tuple<size_t, size_t, double>>> one_hole_first_string_list_;
    /// @brief Precomputed list of one-particle strings with sign for each orbital (inverse)
    /// Stores K -> tuples of (orbital, I, sign) where I is the index of the string
    std::vector<std::vector<std::tuple<size_t, size_t, double>>> one_hole_first_string_list_inv_;
    /// @brief Precomputed list of one-hole strings sorted by the index of the orbital that was
    /// removed Stores orbital -> tuples of (hole_idx, I, sign) where I is the index of the string
    /// and hole_idx is the index of the removed orbital
    std::vector<std::vector<std::tuple<size_t, size_t, double>>>
        one_hole_first_string_list_by_orbital_;

    /// @brief Precomputed list of one-hole strings
    std::vector<String> one_hole_second_strings_;
    /// @brief Map from one-hole string to its index
    ankerl::unordered_dense::map<String, size_t, String::Hash> one_hole_second_strings_index_;
    /// @brief Precomputed list of one-particle strings with sign for each orbital
    /// Stores I -> tuples of (orbital, K, sign) where K is the index of the one-hole string
    std::vector<std::vector<std::tuple<size_t, size_t, double>>> one_hole_second_string_list_;
    /// @brief Precomputed list of one-particle strings with sign for each orbital (inverse)
    /// Stores K -> tuples of (orbital, I, sign) where I is the index of the string
    std::vector<std::vector<std::tuple<size_t, size_t, double>>> one_hole_second_string_list_inv_;
    /// @brief Precomputed list of one-hole strings sorted by the index of the orbital that was
    /// removed Stores orbital -> tuples of (hole_idx, I, sign) where I is the index of the string
    /// and hole_idx is the index of the removed orbital
    std::vector<std::vector<std::tuple<size_t, size_t, double>>>
        one_hole_second_string_list_by_orbital_;

    /// @brief Two-hole strings
    std::vector<String> two_hole_strings_;
    /// @brief Map from two-hole string to its index
    ankerl::unordered_dense::map<String, size_t, String::Hash> two_hole_strings_index_;
    /// @brief Precomputed list of two-hole strings with sign for each pair of orbitals
    /// Stores I -> tuples of (orbital1, orbital2, K, sign) where K is the index of the two-hole
    /// string
    std::vector<std::vector<std::tuple<size_t, size_t, size_t, double>>> two_hole_string_list_;
    /// @brief Precomputed list of two-hole strings with sign for each pair of orbitals (inverse)
    /// Stores K -> tuples of (orbital1, orbital2, I, sign) where I is the index of the string
    std::vector<std::vector<std::tuple<size_t, size_t, size_t, double>>> two_hole_string_list_inv_;
};

} // namespace forte2
