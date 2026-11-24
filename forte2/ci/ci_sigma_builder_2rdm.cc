#include "helpers/timer.hpp"
#include "helpers/np_matrix_functions.h"
#include "helpers/np_vector_functions.h"
#include "helpers/indexing.hpp"
#include "helpers/blas.h"

#include "ci_sigma_builder.h"

#include "ci/flat/gather_block.h"

namespace forte2 {

np_matrix CISigmaBuilder::compute_ss_2rdm(np_vector C_left, np_vector C_right, Spin spin) const {
    const auto& flat_lists = lists_.flat_strings();
    local_timer timer;

    const auto na = flat_lists.na();
    const auto nb = flat_lists.nb();
    const size_t norb = flat_lists.norb();

    // if there are less than two orbitals, return an empty matrix
    if (norb < 2) {
        return make_zeros<nb::numpy, double, 2>({0, 0});
    }

    // calculate the number of pairs of orbitals p > q
    const size_t npairs = (norb * (norb - 1)) / 2;
    auto rdm = make_zeros<nb::numpy, double, 2>({npairs, npairs});

    // skip building the RDM if there are not enough electrons
    if ((is_alpha(spin) and (na < 2)) or (is_beta(spin) and (nb < 2)))
        return rdm;

    auto Cl_span = vector::as_span<double>(C_left);
    auto Cr_span = vector::as_span<double>(C_right);

    auto rdm_data = rdm.data();

    const auto& alpha_address = flat_lists.alpha_address();
    const auto& beta_address = flat_lists.beta_address();

    int num_2h_classes = is_alpha(spin) ? flat_lists.alpha_address_2h()->nclasses()
                                        : flat_lists.beta_address_2h()->nclasses();

    for (int class_K = 0; class_K < num_2h_classes; ++class_K) {
        size_t maxK = is_alpha(spin) ? flat_lists.alpha_address_2h()->strpcls(class_K)
                                     : flat_lists.beta_address_2h()->strpcls(class_K);

        // loop over blocks of matrix C
        for (const auto& [nI, class_Ia, class_Ib] : flat_lists.determinant_classes()) {
            if (flat_lists.block_size(nI) == 0)
                continue;

            auto tl = gather_block(Cl_span, TL, spin, flat_lists, class_Ia, class_Ib);

            for (const auto& [nJ, class_Ja, class_Jb] : flat_lists.determinant_classes()) {
                // The string class on which we don't act must be the same for I and J
                if ((is_alpha(spin) and (class_Ib != class_Jb)) or
                    (is_beta(spin) and (class_Ia != class_Ja)))
                    continue;
                if (flat_lists.block_size(nJ) == 0)
                    continue;

                const size_t maxL = is_alpha(spin) ? beta_address->strpcls(class_Ib)
                                                   : alpha_address->strpcls(class_Ia);
                if (maxL > 0) {
                    // Get a pointer to the correct block of matrix C
                    auto tr = gather_block(Cr_span, TR, spin, flat_lists, class_Ja, class_Jb);
                    for (size_t K{0}; K < maxK; ++K) {
                        auto& Kllist = is_alpha(spin)
                                           ? flat_lists.get_alpha_2h_list(class_K, K, class_Ia)
                                           : flat_lists.get_beta_2h_list(class_K, K, class_Ib);
                        auto& Krlist = is_alpha(spin)
                                           ? flat_lists.get_alpha_2h_list(class_K, K, class_Ja)
                                           : flat_lists.get_beta_2h_list(class_K, K, class_Jb);
                        for (const auto& [sign_K, p, q, I] : Kllist) {
                            const size_t pq_index = pair_index_gt(p, q);
                            for (const auto& [sign_L, r, s, J] : Krlist) {
                                const size_t rs_index = pair_index_gt(r, s);
                                const double rdm_element =
                                    dot(maxL, tl.data() + I * maxL, 1, tr.data() + J * maxL, 1);
                                rdm_data[pq_index * npairs + rs_index] +=
                                    sign_K * sign_L * rdm_element;
                            }
                        }
                    }
                }
            }
        }
    }
    rdm2_aa_timer_ += timer.elapsed_seconds();
    return rdm;
}

np_matrix CISigmaBuilder::compute_aa_2rdm(np_vector C_left, np_vector C_right) const {
    return compute_ss_2rdm(C_left, C_right, Spin::Alpha);
}

np_matrix CISigmaBuilder::compute_bb_2rdm(np_vector C_left, np_vector C_right) const {
    return compute_ss_2rdm(C_left, C_right, Spin::Beta);
}

np_tensor4 CISigmaBuilder::compute_sf_2rdm(np_vector C_left, np_vector C_right) const {
    size_t norb = lists_.norb();
    auto rdm_sf = make_zeros<nb::numpy, double, 4>({norb, norb, norb, norb});

    if (norb < 1) {
        return rdm_sf; // No 2-RDM for less than 1 orbitals
    }

    auto rdm_sf_v = rdm_sf.view();
    // Mixed-spin contribution (1 orbital or more)
    {
        auto rdm_ab = compute_ab_2rdm(C_left, C_right);
        auto rdm_ab_v = rdm_ab.view();
        for (size_t p{0}; p < norb; ++p) {
            for (size_t q{0}; q < norb; ++q) {
                for (size_t r{0}; r < norb; ++r) {
                    for (size_t s{0}; s < norb; ++s) {
                        rdm_sf_v(p, q, r, s) += rdm_ab_v(p, q, r, s) + rdm_ab_v(q, p, s, r);
                    }
                }
            }
        }
    }

    if (norb < 2) {
        return rdm_sf; // No same-spin contributions to the 2-RDM for less than 2 orbitals
    }

    // To reduce the  memory footprint, we compute the aa and bb contributions in a packed
    // format and one at a time.
    for (auto spin : {Spin::Alpha, Spin::Beta}) {
        auto rdm_ss = compute_ss_2rdm(C_left, C_right, spin);
        auto rdm_ss_v = rdm_ss.view();
        for (size_t p{1}, pq{0}; p < norb; ++p) {
            for (size_t q{0}; q < p; ++q, ++pq) { // p > q
                for (size_t r{1}, rs{0}; r < norb; ++r) {
                    for (size_t s{0}; s < r; ++s, ++rs) { // r > s
                        auto element = rdm_ss_v(pq, rs);
                        rdm_sf_v(p, q, r, s) += element;
                        rdm_sf_v(q, p, r, s) -= element;
                        rdm_sf_v(p, q, s, r) -= element;
                        rdm_sf_v(q, p, s, r) += element;
                    }
                }
            }
        }
    }

    return rdm_sf;
}

} // namespace forte2
