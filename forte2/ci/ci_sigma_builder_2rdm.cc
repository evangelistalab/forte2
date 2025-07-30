#include "helpers/timer.hpp"
#include "helpers/np_matrix_functions.h"
#include "helpers/np_vector_functions.h"
#include "helpers/indexing.hpp"
#include "helpers/blas.h"

#include "ci_sigma_builder.h"

namespace forte2 {

np_matrix CISigmaBuilder::compute_ss_2rdm(np_vector C_left, np_vector C_right, bool alfa) const {
    local_timer timer;

    const size_t norb = lists_.norb();
    const size_t npairs = (norb * (norb - 1)) / 2;

    auto rdm = make_zeros<nb::numpy, double, 2>({npairs, npairs});

    const auto na = lists_.na();
    const auto nb = lists_.nb();
    if ((alfa and (na < 2)) or ((!alfa) and (nb < 2)))
        return rdm;

    auto Cl_span = vector::as_span(C_left);
    auto Cr_span = vector::as_span(C_right);

    auto rdm_data = rdm.data();

    const auto& alfa_address = lists_.alfa_address();
    const auto& beta_address = lists_.beta_address();

    int num_2h_classes =
        alfa ? lists_.alfa_address_2h()->nclasses() : lists_.beta_address_2h()->nclasses();

    for (int class_K = 0; class_K < num_2h_classes; ++class_K) {
        size_t maxK = alfa ? lists_.alfa_address_2h()->strpcls(class_K)
                           : lists_.beta_address_2h()->strpcls(class_K);

        // loop over blocks of matrix C
        for (const auto& [nI, class_Ia, class_Ib] : lists_.determinant_classes()) {
            if (lists_.block_size(nI) == 0)
                continue;

            auto tr = gather_block(Cr_span, TR, alfa, lists_, class_Ia, class_Ib);

            for (const auto& [nJ, class_Ja, class_Jb] : lists_.determinant_classes()) {
                // The string class on which we don't act must be the same for I and J
                if ((alfa and (class_Ib != class_Jb)) or (not alfa and (class_Ia != class_Ja)))
                    continue;
                if (lists_.block_size(nJ) == 0)
                    continue;

                const size_t maxL =
                    alfa ? beta_address->strpcls(class_Ib) : alfa_address->strpcls(class_Ia);
                if (maxL > 0) {
                    // Get a pointer to the correct block of matrix C
                    auto tl = gather_block(Cl_span, TL, alfa, lists_, class_Ja, class_Jb);
                    for (size_t K{0}; K < maxK; ++K) {
                        auto& Krlist = alfa ? lists_.get_alfa_2h_list(class_K, K, class_Ia)
                                            : lists_.get_beta_2h_list(class_K, K, class_Ib);
                        auto& Kllist = alfa ? lists_.get_alfa_2h_list(class_K, K, class_Ja)
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
    return compute_ss_2rdm(C_left, C_right, true);
}

np_matrix CISigmaBuilder::compute_bb_2rdm(np_vector C_left, np_vector C_right) const {
    return compute_ss_2rdm(C_left, C_right, false);
}

np_tensor4 CISigmaBuilder::compute_ab_2rdm(np_vector C_left, np_vector C_right) const {
    local_timer timer;
    size_t norb = lists_.norb();
    auto rdm = make_zeros<nb::numpy, double, 4>({norb, norb, norb, norb});

    auto norb2 = norb * norb;
    auto norb3 = norb2 * norb;
    auto index = [norb, norb2, norb3](size_t p, size_t q, size_t r, size_t s) {
        return p * norb3 + q * norb2 + r * norb + s;
    };

    const auto na = lists_.na();
    const auto nb = lists_.nb();
    if ((na < 1) or (nb < 1))
        return rdm;

    auto rdm_data = rdm.data();

    auto Cl_span = vector::as_span(C_left);
    auto Cr_span = vector::as_span(C_right);

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
    auto rdm_sf_v = rdm_sf.view();

    if (norb < 1) {
        return rdm_sf; // No 2-RDM for less than 1 orbitals
    }

    // Mixed-spin contribution
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
    for (auto spin : {true, false}) {
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

// /**
//  * Compute the aa/bb two-particle density matrix for a given wave function
//  * @param alfa flag for alfa or beta component, true = aa, false = bb
//  */
// np_matrix CISigmaBuilder::compute_aa_2rdm(np_vector C_left, np_vector C_right,
//                                                      bool alfa) {
//     const size_t norb = lists_.norb();
//     const size_t npairs = (norb * (norb - 1)) / 2;
//     const auto& alfa_address = lists_.alfa_address();
//     const auto& beta_address = C_left.beta_address();

//     auto rdm = make_zeros<nb::numpy, double, 2>({npairs, npairs});

//     const auto na = lists_.na();
//     const auto nb = lists_.nb();
//     if ((alfa and (na < 2)) or ((!alfa) and (nb < 2)))
//         return rdm;

//     auto rdm_view = rdm.view();

//     for (const auto& [nI, class_Ia, class_Ib] : lists_.determinant_classes()) {
//         if (lists_.block_size(nI) == 0)
//             continue;

//         gather_block(C_right, CR, alfa, lists_, class_Ia, class_Ib);

//         for (const auto& [nJ, class_Ja, class_Jb] : lists_.determinant_classes()) {
//             // The string class on which we don't act must be the same for I and J
//             if ((alfa and (class_Ib != class_Jb)) or (not alfa and (class_Ia != class_Ja)))
//                 continue;
//             if (lists_.block_size(nJ) == 0)
//                 continue;

//             gather_block(C_left, CL, alfa, lists_, class_Ja, class_Jb);

//             // get the size of the string of spin opposite to the one we are acting on
//             size_t maxL = alfa ? beta_address->strpcls(class_Ib) :
//             alfa_address->strpcls(class_Ia);

//             if ((class_Ia == class_Ja) and (class_Ib == class_Jb)) {
//                 // OO terms
//                 // Loop over (p>q) == (p>q)
//                 const auto& pq_oo_list =
//                     alfa ? lists_.get_alfa_oo_list(class_Ia) : lists_.get_beta_oo_list(class_Ib);
//                 for (const auto& [pq, oo_list] : pq_oo_list) {
//                     const auto& [p, q] = pq;
//                     double rdm_element = 0.0;
//                     for (const auto& I : oo_list) {
//                         // rdm_element += psi::C_DDOT(maxL, Cl[I], 1, Cr[I], 1);
//                         rdm_element += matrix::dot_rows(CL, I, CR, I);
//                     }
//                     size_t pq_index = p * (p + 1) / 2 + q;
//                     rdm_view(pq_index, pq_index) += rdm_element;
//                 }
//             }

//             // VVOO terms
//             const auto& pqrs_vvoo_list = alfa ? lists_.get_alfa_vvoo_list(class_Ia, class_Ja)
//                                               : lists_.get_beta_vvoo_list(class_Ib, class_Jb);
//             for (const auto& [pqrs, vvoo_list] : pqrs_vvoo_list) {
//                 const auto& [p, q, r, s] = pqrs;

//                 double rdm_element = 0.0;
//                 for (const auto& [sign, I, J] : vvoo_list) {
//                     rdm_element += sign * matrix::dot_rows(CL, J, CR, I);
//                 }
//                 size_t pq_index = p * (p + 1) / 2 + q;
//                 size_t rs_index = r * (r + 1) / 2 + s;
//                 rdm_view(pq_index, rs_index) += rdm_element;
//             }
//         }
//     }

//     return rdm;
// }

// ambit::Tensor CIVector::compute_ab_2rdm(CIVector& C_left, CIVector& C_right) {
//     size_t ncmo = C_left.ncmo_;
//     const auto& alfa_address = C_left.alfa_address_;
//     const auto& beta_address = C_left.beta_address_;
//     const auto& lists = C_left.lists_;

//     auto rdm = ambit::Tensor::build(ambit::CoreTensor, "2RDM_AB", {ncmo, ncmo, ncmo, ncmo});

//     auto na = alfa_address->nones();
//     auto nb = beta_address->nones();
//     if ((na < 1) or (nb < 1))
//         return rdm;

//     auto& rdm_data = rdm.data();

//     const auto& mo_sym = lists->string_class()->mo_sym();
//     // Loop over blocks of matrix C
//     for (const auto& [nI, class_Ia, class_Ib] : lists->determinant_classes()) {
//         if (lists->block_size(nI) == 0)
//             continue;

//         auto h_Ib = lists->string_class()->beta_string_classes()[class_Ib].second;
//         const auto Cr = C_right.C_[nI]->pointer();

//         for (const auto& [nJ, class_Ja, class_Jb] : lists->determinant_classes()) {
//             if (lists->block_size(nJ) == 0)
//                 continue;

//             auto h_Jb = lists->string_class()->beta_string_classes()[class_Jb].second;
//             const auto Cl = C_left.C_[nJ]->pointer();

//             const auto& pq_vo_alfa = lists->get_alfa_vo_list(class_Ia, class_Ja);
//             const auto& rs_vo_beta = lists->get_beta_vo_list(class_Ib, class_Jb);

//             for (const auto& [rs, vo_beta_list] : rs_vo_beta) {
//                 const size_t beta_list_size = vo_beta_list.size();
//                 if (beta_list_size == 0)
//                     continue;

//                 const auto& [r, s] = rs;
//                 const auto rs_sym = mo_sym[r] ^ mo_sym[s];

//                 // Make sure that the symmetry of the J beta string is the same as the symmetry
//                 // of the I beta string times the symmetry of the rs product
//                 if (h_Jb != (h_Ib ^ rs_sym))
//                     continue;

//                 for (const auto& [pq, vo_alfa_list] : pq_vo_alfa) {
//                     const auto& [p, q] = pq;
//                     const auto pq_sym = mo_sym[p] ^ mo_sym[q];
//                     // ensure that the product pqrs is totally symmetric
//                     if (pq_sym != rs_sym)
//                         continue;

//                     double rdm_element = 0.0;
//                     for (const auto& [sign_a, Ia, Ja] : vo_alfa_list) {
//                         for (const auto& [sign_b, Ib, Jb] : vo_beta_list) {
//                             rdm_element += Cl[Ja][Jb] * Cr[Ia][Ib] * sign_a * sign_b;
//                         }
//                     }
//                     rdm_data[tei_index(p, r, q, s, ncmo)] += rdm_element;
//                 } // End loop over p,q
//             }
//         }
//     }

//     return rdm;
// }
