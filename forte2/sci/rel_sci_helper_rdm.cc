#include <complex>

#include "helpers/indexing.hpp"
#include "helpers/ndarray.h"
#include "helpers/np_matrix_functions.h"

#include "rel_sci_helper.h"

namespace forte2 {

// Each determinant is its own first string, so string i is determinant perm[i]. The bra is
// conjugated: gamma1[p][q] = <L| a^+_p a_q |R>, gamma2[p][q][r][s] = <L| a^+_p a^+_q a_s a_r |R>.

np_matrix_complex RelSelectedCIHelper::compute_so_1rdm(size_t left_root, size_t right_root) const {
    auto rdm = make_zeros<nb::numpy, std::complex<double>, 2>({norb_, norb_});

    if (nel_ < 1 || norb_ < 1)
        return rdm;

    std::complex<double>* rdm_data = rdm.data();

    const auto first_string_size = ab_list_.first_string_size();
    const auto& perm = ab_list_.det_permutation();

    for (size_t i{0}; i < first_string_size; ++i) {
        const std::complex<double> c_bra = std::conj(c_[nroots_ * perm[i] + left_root]);
        const auto& sublist = ab_list_.one_hole_first_string_list()[i];
        for (const auto& [p, hole_idx, sign_p] : sublist) {
            const auto& inv_sublist = ab_list_.one_hole_first_string_list_inv()[hole_idx];
            for (const auto& [q, j, sign_q] : inv_sublist) {
                const double sign = sign_p * sign_q;
                rdm_data[p * norb_ + q] += sign * c_bra * c_[nroots_ * perm[j] + right_root];
            }
        }
    }

    return rdm;
}

np_tensor4_complex RelSelectedCIHelper::compute_so_2rdm(size_t left_root, size_t right_root) const {
    if (norb_ < 2 || nel_ < 2)
        return make_zeros<nb::numpy, std::complex<double>, 4>({norb_, norb_, norb_, norb_});

    // Accumulate in the packed (p > q, r > s) representation, then expand to the full
    // antisymmetric tensor (matching RelCISigmaBuilder::compute_so_2rdm).
    const size_t npairs = (norb_ * (norb_ - 1)) / 2;
    auto rdm = make_zeros<nb::numpy, std::complex<double>, 2>({npairs, npairs});
    std::complex<double>* rdm_data = rdm.data();

    const auto first_string_size = ab_list_.first_string_size();
    const auto& perm = ab_list_.det_permutation();

    for (size_t i{0}; i < first_string_size; ++i) {
        const std::complex<double> c_bra = std::conj(c_[nroots_ * perm[i] + left_root]);
        const auto& sublist = ab_list_.two_hole_string_list()[i];
        for (const auto& [p, q, hole_idx, sign_pq] : sublist) { // (p < q)
            const size_t pq = pair_index_gt(p, q);
            const auto& inv_sublist = ab_list_.two_hole_string_list_inv()[hole_idx];
            for (const auto& [r, s, j, sign_rs] : inv_sublist) { // (r < s)
                const size_t rs = pair_index_gt(r, s);
                const double sign = sign_pq * sign_rs;
                rdm_data[pq * npairs + rs] += sign * c_bra * c_[nroots_ * perm[j] + right_root];
            }
        }
    }

    return matrix::packed_tensor4_to_tensor4<std::complex<double>>(rdm);
}

} // namespace forte2
