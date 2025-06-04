#include "helpers/timer.hpp"
#include "helpers/np_matrix_functions.h"
#include "helpers/np_vector_functions.h"

#include "ci_sigma_builder.h"

namespace forte2 {

np_matrix CISigmaBuilder::compute_2rdm_aa_same_irrep(np_vector C_left, np_vector C_right,
                                                     bool alfa) const {
    const size_t norb = lists_.norb();
    const size_t npairs = (norb * (norb - 1)) / 2;
    const auto& alfa_address = lists_.alfa_address();
    const auto& beta_address = lists_.beta_address();

    auto rdm = make_zeros<nb::numpy, double, 2>({npairs, npairs});

    const auto na = lists_.na();
    const auto nb = lists_.nb();
    if ((alfa and (na < 2)) or ((!alfa) and (nb < 2)))
        return rdm;

    auto Cl_view = C_left.view();
    auto Cr_view = C_right.view();
    // Copy Cr to C and Cl to S
    for (size_t i{0}, imax{C.size()}; i < imax; ++i) {
        C[i] = Cr_view(i);
        S[i] = Cl_view(i);
    }

    auto rdm_view = rdm.view();

    int num_2h_classes =
        alfa ? lists_.alfa_address_2h()->nclasses() : lists_.beta_address_2h()->nclasses();

    for (int class_K = 0; class_K < num_2h_classes; ++class_K) {
        size_t maxK = alfa ? lists_.alfa_address_2h()->strpcls(class_K)
                           : lists_.beta_address_2h()->strpcls(class_K);

        // loop over blocks of matrix C
        for (const auto& [nI, class_Ia, class_Ib] : lists_.determinant_classes()) {
            if (lists_.detpblk(nI) == 0)
                continue;

            gather_block2(C, TR, alfa, lists_, class_Ia, class_Ib);

            for (const auto& [nJ, class_Ja, class_Jb] : lists_.determinant_classes()) {
                // The string class on which we don't act must be the same for I and J
                if ((alfa and (class_Ib != class_Jb)) or (not alfa and (class_Ia != class_Ja)))
                    continue;
                if (lists_.detpblk(nJ) == 0)
                    continue;

                // Get a pointer to the correct block of matrix C
                gather_block2(S, TL, alfa, lists_, class_Ja, class_Jb);

                size_t maxL =
                    alfa ? beta_address->strpcls(class_Ib) : alfa_address->strpcls(class_Ia);
                if (maxL > 0) {
                    for (size_t K = 0; K < maxK; ++K) {
                        auto& Krlist = alfa ? lists_.get_alfa_2h_list(class_K, K, class_Ia)
                                            : lists_.get_beta_2h_list(class_K, K, class_Ib);
                        auto& Kllist = alfa ? lists_.get_alfa_2h_list(class_K, K, class_Ja)
                                            : lists_.get_beta_2h_list(class_K, K, class_Jb);
                        for (const auto& [sign_K, p, q, I] : Krlist) {
                            for (const auto& [sign_L, r, s, J] : Kllist) {
                                const size_t pq_index = p * (p - 1) / 2 + q;
                                const size_t rs_index = r * (r - 1) / 2 + s;
                                double rdm_element = 0.0;
                                for (size_t idx{0}; idx != maxL; ++idx) {
                                    rdm_element += TR[I * maxL + idx] * TL[J * maxL + idx];
                                }
                                rdm_view(pq_index, rs_index) += sign_K * sign_L * rdm_element;
                                // matrix::dot_rows(CL, J, CR, I, maxL);
                            }
                        }
                    }
                }
            }
        }
    }
    return rdm;
}

np_tensor4 CISigmaBuilder::compute_2rdm_ab_same_irrep(np_vector C_left, np_vector C_right) {
    size_t norb = lists_.norb();
    auto rdm = make_zeros<nb::numpy, double, 4>({norb, norb, norb, norb});

    const auto na = lists_.na();
    const auto nb = lists_.nb();
    if ((na < 1) or (nb < 1))
        return rdm;

    auto rdm_view = rdm.view();

    auto Cl_view = C_left.view();
    auto Cr_view = C_right.view();
    // Copy Cr to C and Cl to S
    auto& L = S;
    for (size_t i{0}, imax{C.size()}; i < imax; ++i) {
        C[i] = Cr_view(i);
        L[i] = Cl_view(i);
    }

    const int num_1h_class_Ka = lists_.alfa_address_1h()->nclasses();
    const int num_1h_class_Kb = lists_.beta_address_1h()->nclasses();

    for (int class_Ka = 0; class_Ka < num_1h_class_Ka; ++class_Ka) {
        size_t maxKa = lists_.alfa_address_1h()->strpcls(class_Ka);

        for (int class_Kb = 0; class_Kb < num_1h_class_Kb; ++class_Kb) {
            size_t maxKb = lists_.beta_address_1h()->strpcls(class_Kb);

            // loop over blocks of matrix C
            for (const auto& [nI, class_Ia, class_Ib] : lists_.determinant_classes()) {
                if (lists_.detpblk(nI) == 0)
                    continue;

                const auto maxIb = lists_.beta_address()->strpcls(class_Ib);
                const auto Cr_offset = lists_.block_offset(nI);

                for (const auto& [nJ, class_Ja, class_Jb] : lists_.determinant_classes()) {
                    if (lists_.detpblk(nJ) == 0)
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
                                        sign_u * sign_v * L[Cl_offset + Ja * maxJb + Jb];
                                    for (const auto& [sign_x, x, Ia] : Ka_right_list) {
                                        for (const auto& [sign_y, y, Ib] : Kb_right_list) {
                                            rdm_view(u, v, x, y) += sign_x * sign_y * ClJ *
                                                                    C[Cr_offset + Ia * maxIb + Ib];
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
    return rdm;
}
} // namespace forte2

// /**
//  * Compute the aa/bb two-particle density matrix for a given wave function
//  * @param alfa flag for alfa or beta component, true = aa, false = bb
//  */
// np_matrix CISigmaBuilder::compute_2rdm_aa_same_irrep(np_vector C_left, np_vector C_right,
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
//         if (lists_.detpblk(nI) == 0)
//             continue;

//         gather_block(C_right, CR, alfa, lists_, class_Ia, class_Ib);

//         for (const auto& [nJ, class_Ja, class_Jb] : lists_.determinant_classes()) {
//             // The string class on which we don't act must be the same for I and J
//             if ((alfa and (class_Ib != class_Jb)) or (not alfa and (class_Ia != class_Ja)))
//                 continue;
//             if (lists_.detpblk(nJ) == 0)
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

// ambit::Tensor CIVector::compute_2rdm_ab_same_irrep(CIVector& C_left, CIVector& C_right) {
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
//         if (lists->detpblk(nI) == 0)
//             continue;

//         auto h_Ib = lists->string_class()->beta_string_classes()[class_Ib].second;
//         const auto Cr = C_right.C_[nI]->pointer();

//         for (const auto& [nJ, class_Ja, class_Jb] : lists->determinant_classes()) {
//             if (lists->detpblk(nJ) == 0)
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
