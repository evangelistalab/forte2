#include "helpers/logger.h"
#include "helpers/timer.hpp"
#include "helpers/unordered_dense.h"
#include "helpers/sorting.hpp"
#include "helpers/ndarray.h"
#include "helpers/np_vector_functions.h"

#include "determinant_helpers.h"
#include "sci_helper.h"

namespace forte2 {

SelectedCIStrings::SelectedCIStrings(size_t norb, std::vector<Determinant>& dets) : norb_(norb) {
    initialize_sorted_strings(dets);
    build_second_string_to_det_index();

    build_one_hole_strings_and_lists(sorted_first_string_, one_hole_first_strings_,
                                     one_hole_first_string_list_, one_hole_first_string_list_inv_,
                                     one_hole_first_strings_index_);
    build_one_hole_strings_and_lists(sorted_second_string_, one_hole_second_strings_,
                                     one_hole_second_string_list_, one_hole_second_string_list_inv_,
                                     one_hole_second_strings_index_);
    // build_one_hole_first_strings();
    // build_one_hole_second_strings();
    build_two_hole_strings();
}

void SelectedCIStrings::initialize_sorted_strings(std::vector<Determinant>& dets) {
    det_permutation_ = sort_permutation(dets, Determinant::reverse_less_than);
    apply_permutation_in_place(dets, det_permutation_);

    ndets_ = dets.size();

    size_t i = 0;
    String first_string = dets[0].a_string();
    String second_string = dets[0].b_string();
    String old_first_string = first_string;

    first_string_range_.push_back({i, i + 1});
    sorted_first_string_.push_back(first_string);
    sorted_second_string_.push_back(second_string);
    first_string_index_[first_string] = 0;
    second_string_index_[second_string] = 0;
    sorted_dets_second_string_.push_back(second_string_index_[second_string]);

    for (size_t j{1}; j < ndets_; j++) {
        first_string = dets[j].a_string();
        second_string = dets[j].b_string();
        // check if the second string is new, and if so, add it and assign it an index
        if (second_string_index_.find(second_string) == second_string_index_.end()) {
            second_string_index_[second_string] = second_string_index_.size();
            sorted_second_string_.push_back(second_string);
        }
        sorted_dets_second_string_.push_back(second_string_index_[second_string]);
        // if the first string changed, store the range
        if (first_string != old_first_string) {
            first_string_range_[i].second = j; // end of the range
            // start a new range
            i++;
            first_string_range_.push_back(std::make_pair(j, j + 1));
            sorted_first_string_.push_back(first_string);
            first_string_index_[first_string] = i;
            old_first_string = first_string;
        }
    }
    // set the end of the last range
    first_string_range_[i].second = ndets_;
}

void SelectedCIStrings::build_second_string_to_det_index() {
    second_string_to_det_index_.reserve(first_string_range_.size());
    for (const auto [start, end] : first_string_range_) {
        ankerl::unordered_dense::map<size_t, size_t, std::hash<size_t>> map;
        for (size_t idx{start}; idx < end; ++idx) {
            map[sorted_dets_second_string_[idx]] = det_permutation_[idx];
        }
        second_string_to_det_index_.push_back(std::move(map));
    }

    second_string_to_sorted_det_index_.reserve(first_string_range_.size());
    for (const auto [start, end] : first_string_range_) {
        ankerl::unordered_dense::map<size_t, size_t, std::hash<size_t>> map;
        for (size_t idx{start}; idx < end; ++idx) {
            map[sorted_dets_second_string_[idx]] = idx;
        }
        second_string_to_sorted_det_index_.push_back(std::move(map));
    }
}

void SelectedCIStrings::build_one_hole_strings_and_lists(
    const std::vector<String>& sorted_strings, std::vector<String>& one_hole_strings,
    std::vector<std::vector<std::tuple<size_t, size_t, double>>>& list,
    std::vector<std::vector<std::tuple<size_t, size_t, double>>>& inverse_list,
    ankerl::unordered_dense::map<String, size_t, String::Hash>& index_map) {
    list.reserve(sorted_strings.size());
    std::vector<size_t> occ(norb_, 0); // at most norb occupied orbitals
    for (size_t i = 0, imax{sorted_strings.size()}; i < imax; ++i) {
        const auto& str = sorted_strings[i];
        // Find the occupied orbitals in the first string
        size_t n = 0;
        str.find_set_bits(occ, n);

        std::vector<std::tuple<size_t, size_t, double>> list_entry;
        list_entry.reserve(n);

        // for each occupied orbital, create the one-hole string and store it
        for (size_t p = 0; p < n; ++p) {
            const size_t orb = occ[p];
            String one_hole = str;
            one_hole.set_bit(orb, false);
            // insert one-hole string into map if not already present
            auto [it, inserted] = index_map.try_emplace(one_hole, index_map.size());
            // if inserted, also add to the list of one-hole strings
            if (inserted)
                one_hole_strings.push_back(one_hole);
            list_entry.emplace_back(orb, it->second, str.slater_sign(orb));
        }
        list.emplace_back(std::move(list_entry));
    }

    // create the inverse mapping from one-hole strings to full strings
    inverse_list.resize(index_map.size());
    for (size_t i = 0, imax{sorted_first_string_.size()}; i < imax; ++i) {
        for (const auto& [orb, hole_idx, sign] : list[i]) {
            inverse_list[hole_idx].emplace_back(orb, i, sign);
        }
    }
}

void SelectedCIStrings::build_two_hole_strings() {
    two_hole_string_list_.reserve(sorted_first_string_.size());
    std::vector<size_t> occ(norb_, 0); // at most norb occupied orbitals
    for (size_t i = 0, imax{sorted_first_string_.size()}; i < imax; ++i) {
        const auto& first_str = sorted_first_string_[i];
        // Find the occupied orbitals in the first string
        size_t n = 0;
        first_str.find_set_bits(occ, n);

        std::vector<std::tuple<size_t, size_t, size_t, double>> two_hole_string_list_entry;
        two_hole_string_list_entry.reserve(n * (n - 1) / 2);

        // For each pair of occupied orbitals, create the two-hole string and store it (p < q)
        for (size_t p = 0; p < n; ++p) {
            const size_t orb_p = occ[p];
            for (size_t q = p + 1; q < n; ++q) {
                const size_t orb_q = occ[q];
                String two_hole = first_str;
                double sign = 1.0;
                two_hole.set_bit(orb_p, false);
                sign *= two_hole.slater_sign(orb_p);
                two_hole.set_bit(orb_q, false);
                sign *= two_hole.slater_sign(orb_q);
                // insert two-hole string into map if not already present
                auto [it, inserted] =
                    two_hole_strings_index_.try_emplace(two_hole, two_hole_strings_index_.size());
                // if inserted, also add to the list of two-hole strings
                if (inserted)
                    two_hole_strings_.push_back(two_hole);
                two_hole_string_list_entry.emplace_back(orb_p, orb_q, it->second, sign);
            }
        }
        two_hole_string_list_.emplace_back(std::move(two_hole_string_list_entry));
    }

    // Create the inverse mapping from two-hole strings to full strings
    two_hole_string_list_inv_.resize(two_hole_strings_index_.size());
    for (size_t i = 0, imax{sorted_first_string_.size()}; i < imax; ++i) {
        for (const auto& [orb_p, orb_q, hole_idx, sign] : two_hole_string_list_[i]) {
            two_hole_string_list_inv_[hole_idx].emplace_back(orb_p, orb_q, i, sign);
        }
    }
}

} // namespace forte2
