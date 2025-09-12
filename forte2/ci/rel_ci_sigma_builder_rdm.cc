#include <iostream>

#include "helpers/timer.hpp"
#include "helpers/np_matrix_functions.h"
#include "helpers/np_vector_functions.h"
#include "helpers/indexing.hpp"
#include "helpers/blas.h"

#include "rel_ci_sigma_builder.h"
#include "ci_sigma_builder.h"

namespace forte2 {

np_matrix_complex RelCISigmaBuilder::compute_1rdm(np_vector_complex C_left,
                                                  np_vector_complex C_right) const {
    const auto na = lists_.na();
    const auto norb = lists_.norb();
    auto rdm = make_zeros<nb::numpy, std::complex<double>, 2>({norb, norb});

    // skip building the RDM if there are no electrons or there are zero orbitals
    if (na < 1 || norb < 1)
        return rdm;

    const auto& alfa_address = lists_.alfa_address();
    const auto& beta_address = lists_.beta_address();
    auto rdm_view = rdm.view();
    auto Cl_span = vector_complex::as_span(C_left);
    auto Cr_span = vector_complex::as_span(C_right);

    // placeholder spin
    Spin spin = Spin::Alpha;

    // loop over blocks of the right state
    for (const auto& [nI, class_Ia, class_Ib] : lists_.determinant_classes()) {
        if (lists_.block_size(nI) == 0)
            continue;

        auto tr = gather_block(Cr_span, TR, spin, lists_, class_Ia, class_Ib);

        // loop over blocks of the left state
        for (const auto& [nJ, class_Ja, class_Jb] : lists_.determinant_classes()) {
            // The string class on which we don't act must be the same for I and J
            if ((is_alpha(spin) and (class_Ib != class_Jb)) or
                (is_beta(spin) and (class_Ia != class_Ja)))
                continue;
            if (lists_.block_size(nJ) == 0)
                continue;

            auto tl = gather_block(Cl_span, TL, spin, lists_, class_Ja, class_Jb);

            const size_t maxL =
                is_alpha(spin) ? beta_address->strpcls(class_Ib) : alfa_address->strpcls(class_Ia);

            const auto& pq_vo_list = is_alpha(spin) ? lists_.get_alfa_vo_list(class_Ia, class_Ja)
                                                    : lists_.get_beta_vo_list(class_Ib, class_Jb);

            for (const auto& [pq, vo_list] : pq_vo_list) {
                const auto& [p, q] = pq;
                std::complex<double> rdm_element = 0.0;
                for (const auto& [sign, I, J] : vo_list) {
                    // Compute the RDM element contribution
                    for (size_t idx{0}; idx != maxL; ++idx) {
                        rdm_element += sign * tr[I * maxL + idx] * std::conj(tl[J * maxL + idx]);
                    }
                }
                rdm_view(p, q) += rdm_element;
            }
        }
    }
    return rdm;
}

np_tensor4_complex RelCISigmaBuilder::compute_2rdm(np_vector_complex C_left,
                                                   np_vector_complex C_right) const {
    Spin spin = Spin::Alpha; // placeholder spin

    const auto na = lists_.na();
    const size_t norb = lists_.norb();

    // if there are less than two orbitals or two electrons, return an empty matrix
    if (norb < 2 || na < 2) {
        return make_zeros<nb::numpy, std::complex<double>, 4>({0, 0, 0, 0});
    }

    // calculate the number of pairs of orbitals p > q
    const size_t npairs = (norb * (norb - 1)) / 2;
    auto rdm = make_zeros<nb::numpy, std::complex<double>, 2>({npairs, npairs});

    auto Cl_span = vector_complex::as_span(C_left);
    auto Cr_span = vector_complex::as_span(C_right);

    auto rdm_data = rdm.data();

    const auto& alfa_address = lists_.alfa_address();
    const auto& beta_address = lists_.beta_address();

    int num_2h_classes = is_alpha(spin) ? lists_.alfa_address_2h()->nclasses()
                                        : lists_.beta_address_2h()->nclasses();

    for (int class_K = 0; class_K < num_2h_classes; ++class_K) {
        size_t maxK = is_alpha(spin) ? lists_.alfa_address_2h()->strpcls(class_K)
                                     : lists_.beta_address_2h()->strpcls(class_K);

        // loop over blocks of matrix C
        for (const auto& [nI, class_Ia, class_Ib] : lists_.determinant_classes()) {

            if (lists_.block_size(nI) == 0)
                continue;

            auto tr = gather_block(Cr_span, TR, spin, lists_, class_Ia, class_Ib);

            for (const auto& [nJ, class_Ja, class_Jb] : lists_.determinant_classes()) {
                // The string class on which we don't act must be the same for I and J
                if ((is_alpha(spin) and (class_Ib != class_Jb)) or
                    (is_beta(spin) and (class_Ia != class_Ja)))
                    continue;
                if (lists_.block_size(nJ) == 0)
                    continue;

                const size_t maxL = is_alpha(spin) ? beta_address->strpcls(class_Ib)
                                                   : alfa_address->strpcls(class_Ia);
                if (maxL > 0) {
                    // Get a pointer to the correct block of matrix C
                    auto tl = gather_block(Cl_span, TL, spin, lists_, class_Ja, class_Jb);
                    for (size_t K{0}; K < maxK; ++K) {
                        auto& Krlist = is_alpha(spin)
                                           ? lists_.get_alfa_2h_list(class_K, K, class_Ia)
                                           : lists_.get_beta_2h_list(class_K, K, class_Ib);
                        auto& Kllist = is_alpha(spin)
                                           ? lists_.get_alfa_2h_list(class_K, K, class_Ja)
                                           : lists_.get_beta_2h_list(class_K, K, class_Jb);
                        for (const auto& [sign_K, p, q, I] : Krlist) {
                            const size_t pq_index = pair_index_gt(p, q);
                            for (const auto& [sign_L, r, s, J] : Kllist) {
                                const size_t rs_index = pair_index_gt(r, s);
                                const std::complex<double> rdm_element =
                                    dot(maxL, tr.data() + I * maxL, 1, tl.data() + J * maxL, 1);
                                rdm_data[pq_index * npairs + rs_index] +=
                                    static_cast<std::complex<double>>(sign_K * sign_L) *
                                    rdm_element;
                            }
                        }
                    }
                }
            }
        }
    }

    return matrix_complex::packed_tensor4_to_tensor4(rdm);
}

np_tensor4_complex RelCISigmaBuilder::compute_2cumulant(np_vector_complex C_left,
                                                        np_vector_complex C_right) const {
    // Compute the 1-RDM
    auto G1 = compute_1rdm(C_left, C_right);
    // Compute the 2-RDM (this will hold the cumulant)
    auto L2 = compute_2rdm(C_left, C_right);

    // Evaluate L2[p,q,r,s] = G2[p,q,r,s] - G1[p,r] * G1[q,s] + 0.5 * G1[p,s] * G1[q,r]
    auto G1_v = G1.view();
    auto L2_v = L2.view();

    const auto norb = lists_.norb();
    for (size_t p{0}; p < norb; ++p) {
        for (size_t q{0}; q < norb; ++q) {
            for (size_t r{0}; r < norb; ++r) {
                for (size_t s{0}; s < norb; ++s) {
                    L2_v(p, q, r, s) += -G1_v(p, r) * G1_v(q, s) + G1_v(p, s) * G1_v(q, r);
                }
            }
        }
    }
    return L2;
}

} // namespace forte2
