#include <stdexcept>

#include "helpers/ndarray.h"
#include "helpers/np_matrix_functions.h"
#include "helpers/spin.h"

#include "sci_helper.h"

namespace forte2 {

double SelectedCIHelper::find_matching_dets_1trdm(size_t left_root, size_t right_root,
                                                  const SelectedCIStrings& left_list,
                                                  const SelectedCIStrings& right_list,
                                                  const std::vector<double>& left_c,
                                                  const std::vector<double>& right_c,
                                                  size_t left_nroots, size_t right_nroots, size_t i,
                                                  size_t j, double sign) const {
    double result = 0.0;

    // Find the range of determinants with the current alpha string
    const auto& [istart, iend] = left_list.range(i);
    const auto& [jstart, jend] = right_list.range(j);
    const auto& right_det_permutation = right_list.det_permutation();
    const auto& left_det_permutation = left_list.det_permutation();

    // Here we choose to loop over the smaller range and look up the determinants in the larger
    // range by using the hash map. The map keys are list-local second-string indices, so we
    // map from the right <-> left actual second-string index.
    if (iend - istart >= jend - jstart) {
        const auto& i_map = left_list.second_string_to_det_index()[i];
        for (size_t jj{jstart}; jj < jend; ++jj) {
            const auto right_idx_j = right_list.sorted_dets_second_string(jj);
            // find the corresponding second string in the left list and get its index
            const auto& second_string_j = right_list.sorted_second_string(right_idx_j);
            const auto left_idx_j = left_list.find_second_string_index(second_string_j);
            if (left_idx_j.has_value()) {
                if (const auto it = i_map.find(*left_idx_j); it != i_map.end()) {
                    result += sign * left_c[left_nroots * it->second + left_root] *
                              right_c[right_nroots * right_det_permutation[jj] + right_root];
                }
            }
        }
    } else {
        const auto& j_map = right_list.second_string_to_det_index()[j];
        for (size_t ii{istart}; ii < iend; ++ii) {
            const auto left_idx_i = left_list.sorted_dets_second_string(ii);
            // find the corresponding second string in the right list and get its index
            const auto& second_string_i = left_list.sorted_second_string(left_idx_i);
            const auto right_idx_i = right_list.find_second_string_index(second_string_i);
            if (right_idx_i.has_value()) {
                if (const auto it = j_map.find(*right_idx_i); it != j_map.end()) {
                    result += sign * left_c[left_nroots * left_det_permutation[ii] + left_root] *
                              right_c[right_nroots * it->second + right_root];
                }
            }
        }
    }

    return result;
}

np_matrix SelectedCIHelper::compute_s_1trdm(const SelectedCIHelper& right_helper, size_t left_root,
                                            size_t right_root, Spin spin) const {
    const auto& left_helper = *this;

    if (left_helper.norb_ != right_helper.norb_) {
        throw std::runtime_error("SCI transition RDMs: The number of MOs must be the same in the "
                                 "two wave functions.");
    }
    if (left_helper.na_ != right_helper.na_ || left_helper.nb_ != right_helper.nb_) {
        throw std::runtime_error("SCI transition RDMs: The number of alpha and beta electrons must "
                                 "be the same in the two wave functions.");
    }
    if (left_root >= left_helper.nroots_ || right_root >= right_helper.nroots_) {
        throw std::runtime_error("SCI transition RDMs: Root index out of range.");
    }

    const auto& left_c = left_helper.c_;
    const auto& right_c = right_helper.c_;
    const auto left_nroots = left_helper.nroots_;
    const auto right_nroots = right_helper.nroots_;

    auto rdm = make_zeros<nb::numpy, double, 2>({norb_, norb_});
    double* rdm_data = rdm.data();

    // pick the appropriate string lists based on the spin
    const auto& left_list = is_alpha(spin) ? left_helper.ab_list_ : left_helper.ba_list_;
    const auto& right_list = is_alpha(spin) ? right_helper.ab_list_ : right_helper.ba_list_;

    const auto right_first_string_size = right_list.first_string_size();
    const auto& right_one_hole_first_strings = right_list.one_hole_first_strings();
    const auto& left_one_hole_first_strings_index = left_list.one_hole_first_strings_index();

    // Loop over all unique strings of the right state
    for (size_t j{0}; j < right_first_string_size; ++j) {
        const auto& sublist_right = right_list.one_hole_first_string_list()[j];
        // loop over all single excitations in the right string. a_p |j> -> +/-|k>
        for (const auto& [p, right_hole_idx, sign_p] : sublist_right) {
            // get the one-hole alpha string from the right solver
            const auto& K = right_one_hole_first_strings[right_hole_idx];
            // find the index of the one-hole alpha string K in the left solver
            if (const auto it = left_one_hole_first_strings_index.find(K);
                it != left_one_hole_first_strings_index.end()) {
                // if found, get the corresponding inv_sublist on the left
                const auto& inv_sublist = left_list.one_hole_first_string_list_inv()[it->second];
                for (const auto& [q, i, sign_q] : inv_sublist) {
                    const double sign = sign_p * sign_q;
                    rdm_data[q * norb_ + p] += find_matching_dets_1trdm(
                        left_root, right_root, left_list, right_list, left_c, right_c, left_nroots,
                        right_nroots, i, j, sign);
                }
            }
        }
    }

    return rdm;
}

np_matrix SelectedCIHelper::compute_a_1trdm(const SelectedCIHelper& right_helper, size_t left_root,
                                            size_t right_root) const {
    return compute_s_1trdm(right_helper, left_root, right_root, Spin::Alpha);
}

np_matrix SelectedCIHelper::compute_b_1trdm(const SelectedCIHelper& right_helper, size_t left_root,
                                            size_t right_root) const {
    return compute_s_1trdm(right_helper, left_root, right_root, Spin::Beta);
}

np_matrix SelectedCIHelper::compute_sf_1trdm(const SelectedCIHelper& right_helper, size_t left_root,
                                             size_t right_root) const {
    auto sf_1trdm = make_zeros<nb::numpy, double, 2>({norb_, norb_});
    if (norb_ > 0) {
        auto a_1trdm = compute_a_1trdm(right_helper, left_root, right_root);
        auto b_1trdm = compute_b_1trdm(right_helper, left_root, right_root);
        matrix::daxpy<double>(1.0, a_1trdm, sf_1trdm);
        matrix::daxpy<double>(1.0, b_1trdm, sf_1trdm);
    }
    return sf_1trdm;
}

} // namespace forte2
