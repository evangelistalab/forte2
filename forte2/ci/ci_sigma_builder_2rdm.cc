#include "helpers/timer.hpp"
#include "helpers/np_matrix_functions.h"
#include "helpers/np_vector_functions.h"
#include "helpers/indexing.hpp"
#include "helpers/blas.h"

#include "ci_sigma_builder.h"

namespace forte2 {

np_matrix CISigmaBuilder::compute_ss_2rdm(np_vector C_left, np_vector C_right, Spin spin) const {
    local_timer timer;

    const auto na = lists_.na();
    const auto nb = lists_.nb();
    const size_t norb = lists_.norb();

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
                                const double rdm_element =
                                    dot(maxL, tr.data() + I * maxL, 1, tl.data() + J * maxL, 1);
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

np_tensor4 CISigmaBuilder::compute_ab_2rdm(np_vector C_left, np_vector C_right) const {
    local_timer timer;

    const auto na = lists_.na();
    const auto nb = lists_.nb();
    const auto norb = lists_.norb();
    const auto norb2 = norb * norb;
    const auto norb3 = norb2 * norb;

    auto rdm = make_zeros<nb::numpy, double, 4>({norb, norb, norb, norb});

    // skip building the RDM if there are no electrons or there are zero orbitals
    if ((na < 1) or (nb < 1) or (norb < 1)) {
        return rdm;
    }

    // Create a lambda to compute the index in the 4D tensor
    auto index = [norb, norb2, norb3](size_t p, size_t q, size_t r, size_t s) {
        return p * norb3 + q * norb2 + r * norb + s;
    };

    auto rdm_data = rdm.data();

    auto Cl_span = vector::as_span<double>(C_left);
    auto Cr_span = vector::as_span<double>(C_right);

    const int num_1h_class_Ka = lists_.alfa_address_1h()->nclasses();
    const int num_1h_class_Kb = lists_.beta_address_1h()->nclasses();

    for (int class_Ka = 0; class_Ka < num_1h_class_Ka; ++class_Ka) {
        const auto maxKa = lists_.alfa_address_1h()->strpcls(class_Ka);
        for (int class_Kb = 0; class_Kb < num_1h_class_Kb; ++class_Kb) {
            const auto maxKb = lists_.beta_address_1h()->strpcls(class_Kb);
            // loop over blocks of matrix C
            for (const auto& [nI, class_Ia, class_Ib] : lists_.determinant_classes()) {
                if (lists_.block_size(nI) == 0)
                    continue;

                const auto maxIb = lists_.beta_address()->strpcls(class_Ib);
                const auto Cr_offset = lists_.block_offset(nI);

                for (const auto& [nJ, class_Ja, class_Jb] : lists_.determinant_classes()) {
                    if (lists_.block_size(nJ) == 0)
                        continue;

                    const auto maxJb = lists_.beta_address()->strpcls(class_Jb);
                    const auto Cl_offset = lists_.block_offset(nJ);

                    for (size_t Ka = 0; Ka < maxKa; ++Ka) {
                        auto& Ka_right_list = lists_.get_alfa_1h_list(class_Ka, Ka, class_Ia);
                        auto& Ka_left_list = lists_.get_alfa_1h_list(class_Ka, Ka, class_Ja);
                        for (size_t Kb = 0; Kb < maxKb; ++Kb) {
                            auto& Kb_right_list = lists_.get_beta_1h_list(class_Kb, Kb, class_Ib);
                            auto& Kb_left_list = lists_.get_beta_1h_list(class_Kb, Kb, class_Jb);
                            for (const auto& [sign_u, u, Ja] : Ka_left_list) {
                                for (const auto& [sign_v, v, Jb] : Kb_left_list) {
                                    const auto ClJ =
                                        sign_u * sign_v * Cl_span[Cl_offset + Ja * maxJb + Jb];
                                    for (const auto& [sign_x, x, Ia] : Ka_right_list) {
                                        const auto Cr_Ia_offset = Cr_offset + Ia * maxIb;
                                        for (const auto& [sign_y, y, Ib] : Kb_right_list) {
                                            rdm_data[index(u, v, x, y)] +=
                                                sign_x * sign_y * ClJ * Cr_span[Cr_Ia_offset + Ib];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    rdm2_ab_timer_ += timer.elapsed_seconds();
    return rdm;
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

// np_tensor4 CISigmaBuilder::compute_aa_2rdm_full(np_vector C_left, np_vector C_right,
//                                                 bool alfa) const {
//     auto rdm = compute_aa_2rdm(C_left, C_right, alfa);
//     const auto norb = lists_.norb();
//     auto rdm_full = make_zeros<nb::numpy, double, 4>({norb, norb, norb, norb});
//     auto rdm_v = rdm.view();
//     auto rdm_full_v = rdm_full.view();

//     for (size_t p{1}; p < norb; ++p) {
//         for (size_t q{0}; q < p; ++q) { // p > q
//             auto pq = pair_index_gt(p, q);
//             for (size_t r{1}; r < norb; ++r) {
//                 for (size_t s{0}; s < r; ++s) { // r > s
//                     auto rs = pair_index_gt(r, s);
//                     auto element = rdm_v(pq, rs);
//                     rdm_full_v(p, q, r, s) = element;
//                     rdm_full_v(q, p, r, s) = -element;
//                     rdm_full_v(p, q, s, r) = -element;
//                     rdm_full_v(q, p, s, r) = element;
//                 }
//             }
//         }
//     }
//     return rdm_full;
// }

np_tensor4 CISigmaBuilder::compute_sf_2cumulant(np_vector C_left, np_vector C_right) const {
    // Compute the spin-free 1-RDM
    auto G1 = compute_sf_1rdm(C_left, C_right);
    // Compute the spin-free 2-RDM (this will hold the cumulant)
    auto L2 = compute_sf_2rdm(C_left, C_right);

    // Evaluate L2[p,q,r,s] = G2[p,q,r,s] - G1[p,r] * G1[q,s] + 0.5 * G1[p,s] * G1[q,r]
    auto G1_v = G1.view();
    auto L2_v = L2.view();

    const auto norb = lists_.norb();
    for (size_t p{0}; p < norb; ++p) {
        for (size_t q{0}; q < norb; ++q) {
            for (size_t r{0}; r < norb; ++r) {
                for (size_t s{0}; s < norb; ++s) {
                    L2_v(p, q, r, s) += -G1_v(p, r) * G1_v(q, s) + 0.5 * G1_v(p, s) * G1_v(q, r);
                }
            }
        }
    }
    return L2;
}

} // namespace forte2
