#include "helpers/ndarray.h"
#include "helpers/np_matrix_functions.h"

#include "sci_helper.h"

namespace forte2 {

double SelectedCIHelper::find_matching_dets_1rdm(size_t left_root, size_t right_root,
                                                 const SelectedCIStrings& list, size_t i, size_t j,
                                                 double sign) const {
    double result = 0.0;

    // Find the range of determinants with the current alpha string
    const auto& [istart, iend] = list.range(i);
    const auto& [jstart, jend] = list.range(j);
    const auto& det_permutation = list.det_permutation();

    // Here we choose to loop over the smaller range and look up the determinants in the larger
    // range by using the hash map
    if (iend - istart >= jend - jstart) {
        const auto& i_map = list.second_string_to_det_index()[i];
        for (size_t jj{jstart}; jj < jend; ++jj) {
            const auto idx_j = list.sorted_dets_second_string(jj);
            if (const auto it = i_map.find(idx_j); it != i_map.end()) {
                result += sign * c_[nroots_ * it->second + left_root] *
                          c_[nroots_ * det_permutation[jj] + right_root];
            }
        }
    } else {
        const auto& j_map = list.second_string_to_det_index()[j];
        for (size_t ii{istart}; ii < iend; ++ii) {
            const auto idx_i = list.sorted_dets_second_string(ii);
            if (const auto it = j_map.find(idx_i); it != j_map.end()) {
                result += sign * c_[nroots_ * det_permutation[ii] + left_root] *
                          c_[nroots_ * it->second + right_root];
            }
        }
    }

    return result;
}

np_matrix SelectedCIHelper::compute_a_1rdm(size_t left_root, size_t right_root) const {
    auto rdm = make_zeros<nb::numpy, double, 2>({norb_, norb_});
    double* rdm_data = rdm.data();

    const auto first_string_size = ab_list_.first_string_size();
    // Loop over all unique alpha strings
    for (size_t i{0}; i < first_string_size; ++i) {
        const auto& sublist = ab_list_.one_hole_first_string_list()[i];
        for (const auto& [p, hole_idx, sign_p] : sublist) {
            const auto& inv_sublist = ab_list_.one_hole_first_string_list_inv()[hole_idx];
            for (const auto& [q, j, sign_q] : inv_sublist) {
                const double sign = sign_p * sign_q;
                rdm_data[p * norb_ + q] +=
                    find_matching_dets_1rdm(left_root, right_root, ab_list_, i, j, sign);
            }
        }
    }

    return rdm;
}

np_matrix SelectedCIHelper::compute_b_1rdm(size_t left_root, size_t right_root) const {
    auto rdm = make_zeros<nb::numpy, double, 2>({norb_, norb_});
    double* rdm_data = rdm.data();

    const auto first_string_size = ba_list_.first_string_size();
    // Loop over all unique beta strings
    for (size_t i{0}; i < first_string_size; ++i) {
        const auto& sublist = ba_list_.one_hole_first_string_list()[i];
        for (const auto& [p, hole_idx, sign_p] : sublist) {
            const auto& inv_sublist = ba_list_.one_hole_first_string_list_inv()[hole_idx];
            for (const auto& [q, j, sign_q] : inv_sublist) {
                const double sign = sign_p * sign_q;
                rdm_data[p * norb_ + q] +=
                    find_matching_dets_1rdm(left_root, right_root, ba_list_, i, j, sign);
            }
        }
    }

    return rdm;
}

np_matrix SelectedCIHelper::compute_sf_1rdm(size_t left_root, size_t right_root) const {
    auto sf_1rdm = make_zeros<nb::numpy, double, 2>({norb_, norb_});
    if (norb_ > 0) {
        auto a_1rdm = compute_a_1rdm(left_root, right_root);
        auto b_1rdm = compute_b_1rdm(left_root, right_root);
        matrix::daxpy<double>(1.0, a_1rdm, sf_1rdm);
        matrix::daxpy<double>(1.0, b_1rdm, sf_1rdm);
    }
    return sf_1rdm;
}

} // namespace forte2
