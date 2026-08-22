#include "helpers/indexing.hpp"
#include "helpers/ndarray.h"

#include "sci_helper.h"

namespace forte2 {

np_matrix SelectedCIHelper::compute_aa_2rdm(size_t left_root, size_t right_root) const {
    // calculate the number of pairs of orbitals p > q
    const size_t npairs = (norb_ * (norb_ - 1)) / 2;
    auto rdm = make_zeros<nb::numpy, double, 2>({npairs, npairs});
    double* rdm_data = rdm.data();

    const auto first_string_size = ab_list_.first_string_size();
    // Loop over all unique alpha strings
    for (size_t i{0}; i < first_string_size; ++i) {
        const auto& sublist = ab_list_.two_hole_string_list()[i];
        for (const auto& [p, q, hole_idx, sign_pq] : sublist) { // (p < q)
            const size_t pq = pair_index_gt(p, q);
            const auto& inv_sublist = ab_list_.two_hole_string_list_inv()[hole_idx];
            for (const auto& [r, s, j, sign_rs] : inv_sublist) { // (r < s)
                const size_t rs = pair_index_gt(r, s);
                const double sign = sign_pq * sign_rs;
                rdm_data[pq * npairs + rs] +=
                    find_matching_dets_1rdm(left_root, right_root, ab_list_, i, j, sign);
            }
        }
    }
    return rdm;
}

np_matrix SelectedCIHelper::compute_bb_2rdm(size_t left_root, size_t right_root) const {
    // calculate the number of pairs of orbitals p > q
    const size_t npairs = (norb_ * (norb_ - 1)) / 2;
    auto rdm = make_zeros<nb::numpy, double, 2>({npairs, npairs});
    double* rdm_data = rdm.data();

    const auto first_string_size = ba_list_.first_string_size();
    // Loop over all unique alpha strings
    for (size_t i{0}; i < first_string_size; ++i) {
        const auto& sublist = ba_list_.two_hole_string_list()[i];
        for (const auto& [p, q, hole_idx, sign_pq] : sublist) { // (p < q)
            const size_t pq = pair_index_gt(p, q);
            const auto& inv_sublist = ba_list_.two_hole_string_list_inv()[hole_idx];
            for (const auto& [r, s, j, sign_rs] : inv_sublist) { // (r < s)
                const size_t rs = pair_index_gt(r, s);
                const double sign = sign_pq * sign_rs;
                rdm_data[pq * npairs + rs] +=
                    find_matching_dets_1rdm(left_root, right_root, ba_list_, i, j, sign);
            }
        }
    }
    return rdm;
}

np_tensor4 SelectedCIHelper::compute_ab_2rdm(size_t left_root, size_t right_root) const {
    auto rdm = make_zeros<nb::numpy, double, 4>({norb_, norb_, norb_, norb_});
    double* rdm_data = rdm.data();

    const auto first_string_size = ab_list_.first_string_size();
    const auto& det_permutation = ab_list_.det_permutation();
    // Loop over all unique alpha strings
    for (size_t i{0}; i < first_string_size; ++i) {
        const auto& i_map = ab_list_.second_string_to_det_index()[i];
        // Loop over all single excitations in the alpha string.
        // a+_q a_p |i_a> -> +/-|j_a>
        const auto& sublist_a = ab_list_.one_hole_first_string_list()[i];
        for (const auto& [p, hole_idx, sign_p] : sublist_a) {
            const auto& inv_sublist_a = ab_list_.one_hole_first_string_list_inv()[hole_idx];
            for (const auto& [q, j, sign_q] : inv_sublist_a) {
                const auto& [jstart, jend] = ab_list_.range(j);
                // Loop over all the beta strings with the same alpha string
                for (size_t jj{jstart}; jj < jend; ++jj) {
                    const auto idx_j = ab_list_.sorted_dets_second_string(jj);
                    // Loop over single excitations in the beta string.
                    // a+_s a_r |j_b> -> +/-|k_b>
                    const auto& sublist_b = ab_list_.one_hole_second_string_list()[idx_j];
                    for (const auto& [r, hole_idx_b, sign_r] : sublist_b) {
                        const auto& inv_sublist_b =
                            ab_list_.one_hole_second_string_list_inv()[hole_idx_b];
                        for (const auto& [s, k, sign_s] : inv_sublist_b) {
                            const double sign = sign_p * sign_q * sign_r * sign_s;
                            // Here we get (leaving note since this is a bit subtle):
                            //     <j_a k_b|a+_r a_s a+_q a_p|i_a i_b>
                            //   = <j_a k_b|a+_q a+_r a_s a_p|i_a i_b> = gamma2(q_a,r_b,p_a,s_b)
                            // Check if the determinant with the new beta string exists
                            if (const auto it = i_map.find(k); it != i_map.end()) {
                                rdm_data[q * norb3_ + r * norb2_ + p * norb_ + s] +=
                                    sign * c_[nroots_ * it->second + left_root] *
                                    c_[nroots_ * det_permutation[jj] + right_root];
                            }
                        }
                    }
                }
            }
        }
    }
    return rdm;
}

} // namespace forte2
