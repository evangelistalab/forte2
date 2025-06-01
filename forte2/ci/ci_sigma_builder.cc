#include "helpers/timer.hpp"
#include "helpers/np_matrix_functions.h"
#include "helpers/np_vector_functions.h"

#include "ci_sigma_builder.h"

namespace forte2 {

np_matrix CISigmaBuilder::CR;
np_matrix CISigmaBuilder::CL;

// double CISigmaBuilder::hdiag_timer = 0.0;
// double CISigmaBuilder::h1_aa_timer = 0.0;
// double CISigmaBuilder::h1_bb_timer = 0.0;
// double CISigmaBuilder::h2_aaaa_timer = 0.0;
// double CISigmaBuilder::h2_aabb_timer = 0.0;
// double CISigmaBuilder::h2_bbbb_timer = 0.0;

void CISigmaBuilder::allocate_temp_space(const CIStrings& lists) {
    // if CR is already allocated (e.g., because we computed several roots) make sure
    // we do not allocate a matrix of smaller size. So let's find out the size of the current CR
    // size_t current_size = CR ? CR->rowdim() : 0;

    // // Find the largest size of the symmetry blocks
    size_t max_size = 0;

    for (int class_Ia = 0; class_Ia < lists.alfa_address()->nclasses(); ++class_Ia) {
        max_size = std::max(max_size, lists.alfa_address()->strpcls(class_Ia));
    }
    for (int class_Ib = 0; class_Ib < lists.beta_address()->nclasses(); ++class_Ib) {
        max_size = std::max(max_size, lists.beta_address()->strpcls(class_Ib));
    }

    // Allocate the temporary arrays CR and CL with the largest block size
    // if (max_size > current_size) {
    CR = make_zeros<nb::numpy, double, 2>({max_size, max_size});
    CL = make_zeros<nb::numpy, double, 2>({max_size, max_size});
}

void CISigmaBuilder::release_temp_space() {
    CR = np_matrix();
    CL = np_matrix();
}

CISigmaBuilder::CISigmaBuilder(const CIStrings& lists, double E, np_matrix& H, np_tensor4& V)
    : lists_(lists), E_(E), H_(H), V_(V), slater_rules_(lists.norb(), E, H, V) {}

np_vector CISigmaBuilder::form_Hdiag_det() {
    CIVector Hdiag(lists_);
    Determinant I;
    Hdiag.for_each_element([&](const size_t& /*n*/, const int& class_Ia, const int& class_Ib,
                               const size_t& Ia, const size_t& Ib, double& c) {
        I.set_str(lists_.alfa_str(class_Ia, Ia), lists_.beta_str(class_Ib, Ib));
        c = slater_rules_.energy(I);
    });

    auto Hdiag_np = make_zeros<nb::numpy, double, 1>({lists_.ndet()});
    Hdiag.copy_to(Hdiag_np);
    return Hdiag_np;
}

void CISigmaBuilder::Hamiltonian(CIVector& basis, CIVector& sigma) const {
    local_timer t;
    sigma.zero();

    H0(basis, sigma);
    H1(basis, sigma, true);
    H1(basis, sigma, false);

    H2_aabb(basis, sigma);
    H2_aaaa(basis, sigma, true);
    H2_aaaa(basis, sigma, false);
    hdiag_timer_ += t.elapsed_seconds();
    build_count_++;
}

void CISigmaBuilder::H0(CIVector& basis, CIVector& sigma) const {
    for (const auto& [n, _1, _2] : lists_.determinant_classes()) {
        matrix::copy(basis.C(n), sigma.C(n));
        matrix::scale(sigma.C(n), E_);
    }
}

void CISigmaBuilder::H1(CIVector& basis, CIVector& sigma, bool alfa) const {
    auto h = H_.view();
    // loop over blocks of matrix C
    for (const auto& [nI, class_Ia, class_Ib] : lists_.determinant_classes()) {
        if (lists_.detpblk(nI) == 0)
            continue;
        auto Cr = basis.gather_C_block(CR, alfa, lists_.alfa_address(), lists_.beta_address(),
                                       class_Ia, class_Ib, false);

        for (const auto& [nJ, class_Ja, class_Jb] : lists_.determinant_classes()) {

            // If we act on the alpha string, the beta string classes of the result must be the same
            if (alfa and (class_Ib != class_Jb))
                continue;
            // If we act on the beta string, the alpha string classes of the result must be the same
            if (not alfa and (class_Ia != class_Ja))
                continue;

            if (lists_.detpblk(nJ) == 0)
                continue;

            auto Cl = sigma.gather_C_block(CL, alfa, lists_.alfa_address(), lists_.beta_address(),
                                           class_Ja, class_Jb, !alfa);

            size_t maxL = alfa ? lists_.beta_address()->strpcls(class_Ib)
                               : lists_.alfa_address()->strpcls(class_Ia);

            const auto& pq_vo_list = alfa ? lists_.get_alfa_vo_list(class_Ia, class_Ja)
                                          : lists_.get_beta_vo_list(class_Ib, class_Jb);

            for (const auto& [pq, vo_list] : pq_vo_list) {
                const auto& [p, q] = pq;
                const double Hpq = alfa ? h(p, q) : h(p, q);
                for (const auto& [sign, I, J] : vo_list) {
                    matrix::daxpy_rows(sign * Hpq, Cr, I, Cl, J);
                }
            }
            sigma.scatter_C_block(Cl, alfa, lists_.alfa_address(), lists_.beta_address(), class_Ja,
                                  class_Jb);
        }
    }
}

void CISigmaBuilder::H2_aaaa(CIVector& basis, CIVector& sigma, bool alfa) const {
    auto v = V_.view();
    for (const auto& [nI, class_Ia, class_Ib] : lists_.determinant_classes()) {
        if (lists_.detpblk(nI) == 0)
            continue;

        const auto Cr = basis.gather_C_block(CR, alfa, lists_.alfa_address(), lists_.beta_address(),
                                             class_Ia, class_Ib, false);

        for (const auto& [nJ, class_Ja, class_Jb] : lists_.determinant_classes()) {
            // The string class on which we don't act must be the same for I and J
            if ((alfa and (class_Ib != class_Jb)) or (not alfa and (class_Ia != class_Ja)))
                continue;
            if (lists_.detpblk(nJ) == 0)
                continue;

            auto Cl = sigma.gather_C_block(CL, alfa, lists_.alfa_address(), lists_.beta_address(),
                                           class_Ja, class_Jb, !alfa);

            // get the size of the string of spin opposite to the one we are acting on
            size_t maxL = alfa ? lists_.beta_address()->strpcls(class_Ib)
                               : lists_.alfa_address()->strpcls(class_Ia);

            if ((class_Ia == class_Ja) and (class_Ib == class_Jb)) {
                // OO terms
                // Loop over (p>q) == (p>q)
                const auto& pq_oo_list =
                    alfa ? lists_.get_alfa_oo_list(class_Ia) : lists_.get_beta_oo_list(class_Ib);
                for (const auto& [pq, oo_list] : pq_oo_list) {
                    const auto& [p, q] = pq;
                    const double integral =
                        alfa ? v(p, q, p, q) - v(p, q, q, p) : v(p, q, p, q) - v(p, q, q, p);

                    for (const auto& I : oo_list) {
                        matrix::daxpy_rows(integral, Cr, I, Cl, I);
                        // C_DAXPY(maxL, integral, Cr[I], 1, Cl[I], 1);
                    }
                }
            }

            // VVOO terms
            const auto& pqrs_vvoo_list = alfa ? lists_.get_alfa_vvoo_list(class_Ia, class_Ja)
                                              : lists_.get_beta_vvoo_list(class_Ib, class_Jb);
            for (const auto& [pqrs, vvoo_list] : pqrs_vvoo_list) {
                const auto& [p, q, r, s] = pqrs;
                const double integral1 =
                    alfa ? v(p, q, r, s) - v(p, q, s, r) : v(p, q, r, s) - v(p, q, s, r);
                for (const auto& [sign, I, J] : vvoo_list) {
                    matrix::daxpy_rows(sign * integral1, Cr, I, Cl, J);
                }
            }
            sigma.scatter_C_block(Cl, alfa, lists_.alfa_address(), lists_.beta_address(), class_Ja,
                                  class_Jb);
        }
    }
}

// no gather / scatter implementation of H2_aabb 2
void CISigmaBuilder::H2_aabb(CIVector& basis, CIVector& sigma) const {
    auto v = V_.view();
    const auto& mo_sym = lists_.string_class()->mo_sym();
    // Loop over blocks of matrix C
    for (const auto& [nI, class_Ia, class_Ib] : lists_.determinant_classes()) {
        if (lists_.detpblk(nI) == 0)
            continue;

        auto h_Ib = lists_.string_class()->beta_string_classes()[class_Ib].second;
        const size_t maxIa = lists_.alfa_address()->strpcls(class_Ia);
        auto C_view = basis.C(nI).view();

        for (const auto& [nJ, class_Ja, class_Jb] : lists_.determinant_classes()) {
            if (lists_.detpblk(nJ) == 0)
                continue;

            auto h_Jb = lists_.string_class()->beta_string_classes()[class_Jb].second;
            auto HC_view = sigma.C(nJ).view();

            const auto& pq_vo_alfa = lists_.get_alfa_vo_list(class_Ia, class_Ja);
            const auto& rs_vo_beta = lists_.get_beta_vo_list(class_Ib, class_Jb);

            for (const auto& [rs, vo_beta_list] : rs_vo_beta) {
                const size_t beta_list_size = vo_beta_list.size();
                if (beta_list_size == 0)
                    continue;

                const auto& [r, s] = rs;
                const auto rs_sym = mo_sym[r] ^ mo_sym[s];

                // Make sure that the symmetry of the J beta string is the same as the symmetry
                // of the I beta string times the symmetry of the rs product
                if (h_Jb != (h_Ib ^ rs_sym))
                    continue;

                for (const auto& [pq, vo_alfa_list] : pq_vo_alfa) {
                    const size_t a_list_size = vo_alfa_list.size();
                    if (a_list_size == 0)
                        continue;

                    const auto& [p, q] = pq;
                    const auto pq_sym = mo_sym[p] ^ mo_sym[q];
                    // ensure that the product pqrs is totally symmetric
                    if (pq_sym != rs_sym)
                        continue;

                    // Grab the integral
                    const double integral = v(p, r, q, s);

                    for (const auto& [sign_a, Ia, Ja] : vo_alfa_list) {
                        for (const auto& [sign_b, Ib, Jb] : vo_beta_list) {
                            HC_view(Ja, Jb) += C_view(Ia, Ib) * integral * sign_a * sign_b;
                        }
                    }
                }
            }
        }
    }
}

// no gather/scatter implementation of H2_aabb
// void CISigmaBuilder::H2_aabb(CIVector& basis, CIVector& sigma) const {
//     auto v = V_.view();
//     const auto& mo_sym = lists_.string_class()->mo_sym();
//     // Loop over blocks of matrix C
//     for (const auto& [nI, class_Ia, class_Ib] : lists_.determinant_classes()) {
//         if (lists_.detpblk(nI) == 0)
//             continue;

//         auto h_Ib = lists_.string_class()->beta_string_classes()[class_Ib].second;
//         const size_t maxIa = lists_.alfa_address()->strpcls(class_Ia);
//         auto C_view = basis.C(nI).view();

//         for (const auto& [nJ, class_Ja, class_Jb] : lists_.determinant_classes()) {
//             if (lists_.detpblk(nJ) == 0)
//                 continue;

//             auto h_Jb = lists_.string_class()->beta_string_classes()[class_Jb].second;
//             const size_t maxJa = lists_.alfa_address()->strpcls(class_Ja);
//             auto HC_view = sigma.C(nJ).view();

//             const auto& pq_vo_alfa = lists_.get_alfa_vo_list(class_Ia, class_Ja);
//             const auto& rs_vo_beta = lists_.get_beta_vo_list(class_Ib, class_Jb);

//             for (const auto& [rs, vo_beta_list] : rs_vo_beta) {
//                 const size_t beta_list_size = vo_beta_list.size();
//                 if (beta_list_size == 0)
//                     continue;

//                 const auto& [r, s] = rs;
//                 const auto rs_sym = mo_sym[r] ^ mo_sym[s];

//                 // Make sure that the symmetry of the J beta string is the same as the symmetry
//                 of
//                 // the I beta string times the symmetry of the rs product
//                 if (h_Jb != (h_Ib ^ rs_sym))
//                     continue;

//                 for (const auto& [pq, vo_alfa_list] : pq_vo_alfa) {
//                     const auto& [p, q] = pq;
//                     const auto pq_sym = mo_sym[p] ^ mo_sym[q];
//                     // ensure that the product pqrs is totally symmetric
//                     if (pq_sym != rs_sym)
//                         continue;

//                     // Grab the integral
//                     const double integral = v(p, r, q, s);

//                     for (const auto& [sign_a, Ia, Ja] : vo_alfa_list) {
//                         const auto factor = integral * sign_a;
//                         for (const auto& [sign_b, Ib, Jb] : vo_beta_list) {
//                             HC_view(Ja, Jb) += C_view(Ia, Ib) * sign_b * factor;
//                         }
//                     }
//                 }
//             }
//         }
//     }
// }

// original implementation of H2_aabb
// void CISigmaBuilder::H2_aabb(CIVector& basis, CIVector& sigma) const {
//     auto v = V_.view();
//     const auto& mo_sym = lists_.string_class()->mo_sym();
//     auto Cr = CR;
//     auto Cl = CL;
//     auto Cr_view = Cr.view();
//     auto Cl_view = Cl.view();
//     // Loop over blocks of matrix C
//     for (const auto& [nI, class_Ia, class_Ib] : lists_.determinant_classes()) {
//         if (lists_.detpblk(nI) == 0)
//             continue;

//         auto h_Ib = lists_.string_class()->beta_string_classes()[class_Ib].second;
//         const size_t maxIa = lists_.alfa_address()->strpcls(class_Ia);
//         const auto C = basis.C(nI);
//         auto C_view = C.view();

//         for (const auto& [nJ, class_Ja, class_Jb] : lists_.determinant_classes()) {
//             if (lists_.detpblk(nJ) == 0)
//                 continue;

//             auto h_Jb = lists_.string_class()->beta_string_classes()[class_Jb].second;
//             const size_t maxJa = lists_.alfa_address()->strpcls(class_Ja);
//             auto HC = sigma.C(nJ);
//             auto HC_view = HC.view();

//             const auto& pq_vo_alfa = lists_.get_alfa_vo_list(class_Ia, class_Ja);
//             const auto& rs_vo_beta = lists_.get_beta_vo_list(class_Ib, class_Jb);

//             for (const auto& [rs, vo_beta_list] : rs_vo_beta) {
//                 const size_t beta_list_size = vo_beta_list.size();
//                 if (beta_list_size == 0)
//                     continue;

//                 const auto& [r, s] = rs;
//                 const auto rs_sym = mo_sym[r] ^ mo_sym[s];

//                 // Make sure that the symmetry of the J beta string is the same as the symmetry
//                 of
//                 // the I beta string times the symmetry of the rs product
//                 if (h_Jb != (h_Ib ^ rs_sym))
//                     continue;

//                 // Zero the block of CL used to store the sigma
//                 // this should be faster than CL->zero(); when beta_list_size is smaller than
//                 // the number of columns of CL
//                 for (size_t Ja{0}; Ja < maxJa; ++Ja) {
//                     // const auto cl = Cl[Ja];
//                     // std::fill(cl, cl + beta_list_size, 0.0);
//                     for (size_t idx{0}; idx < beta_list_size; ++idx)
//                         Cl_view(Ja, idx) = 0.0;
//                 }

//                 // Gather cols of C into CR with the correct sign
//                 for (size_t Ia{0}; Ia < maxIa; ++Ia) {
//                     // const auto c = C[Ia];
//                     // auto cr = Cr[Ia];
//                     for (size_t idx{0}; const auto& [sign, I, _] : vo_beta_list) {
//                         // cr[idx] = c[I] * sign;
//                         Cr_view(Ia, idx) = C_view(Ia, I) * sign;
//                         idx++;
//                     }
//                 }

//                 for (const auto& [pq, vo_alfa_list] : pq_vo_alfa) {
//                     const auto& [p, q] = pq;
//                     const auto pq_sym = mo_sym[p] ^ mo_sym[q];
//                     // ensure that the product pqrs is totally symmetric
//                     if (pq_sym != rs_sym)
//                         continue;

//                     // Grab the integral
//                     const double integral = v(p, r, q, s);

//                     for (const auto& [sign, I, J] : vo_alfa_list) {
//                         const auto factor = integral * sign;
//                         // std::transform(Cr[I], Cr[I] + beta_list_size, Cl[J], Cl[J],
//                         //                [factor](double xi, double yi) { return factor * xi
//                         // +yi;
//                         //                });
//                         // C_DAXPY(beta_list_size, integral * sign, Cr[I], 1, Cl[J], 1);
//                         // the urolled version of the loop above is faster
//                         // const auto& ClJ = Cl[J];
//                         // const auto& CrI = Cr[I];
//                         // const auto factor = integral * sign;
//                         for (size_t idx{0}; idx != beta_list_size; ++idx) {
//                             Cl_view(J, idx) += factor * Cr_view(I, idx);
//                         }
//                     }
//                 } // End loop over p,q

//                 // Scatter cols of CL into HC (the sign was included before in the gathering)
//                 for (size_t Ja = 0; Ja < maxJa; ++Ja) {
//                     // auto hc = HC[Ja];
//                     // const auto cl = Cl[Ja];
//                     for (size_t idx{0}; const auto& [_1, _2, J] : vo_beta_list) {
//                         // hc[J] += cl[idx];
//                         HC_view(Ja, J) += Cl_view(Ja, idx);
//                         idx++;
//                     }
//                 }
//             }
//         }
//     }
// }

void CISigmaBuilder::Hamiltonian2(np_vector basis, np_vector sigma) const {
    local_timer t;
    vector::zero(sigma);

    H0(basis, sigma);
    H1(basis, sigma, true);
    H1(basis, sigma, false);

    H2_aabb(basis, sigma);
    H2_aaaa(basis, sigma, true);
    H2_aaaa(basis, sigma, false);
    hdiag_timer_ += t.elapsed_seconds();
    build_count_++;
}

void CISigmaBuilder::H0(np_vector basis, np_vector sigma) const {
    vector::copy(basis, sigma);
    vector::scale(sigma, E_);
}

void CISigmaBuilder::gather_C_block(np_vector source, np_matrix dest, bool alfa,
                                    const CIStrings& lists, int class_Ia, int class_Ib) {
    const auto block_index = lists.string_class()->block_index(class_Ia, class_Ib);
    const auto offset = lists.block_offset(block_index);
    const auto maxIa = lists.alfa_address()->strpcls(class_Ia);
    const auto maxIb = lists.beta_address()->strpcls(class_Ib);
    auto m = dest.view();
    const auto c = source.view();

    if (alfa) {
        for (size_t Ia{0}; Ia < maxIa; ++Ia)
            for (size_t Ib{0}; Ib < maxIb; ++Ib)
                m(Ia, Ib) = c(offset + Ia * maxIb + Ib);
    } else {
        for (size_t Ia{0}; Ia < maxIa; ++Ia)
            for (size_t Ib{0}; Ib < maxIb; ++Ib)
                m(Ib, Ia) = c(offset + Ia * maxIb + Ib);
    }
}

void CISigmaBuilder::zero_block(np_matrix dest, bool alfa, const CIStrings& lists, int class_Ia,
                                int class_Ib) {
    const auto maxIa = lists.alfa_address()->strpcls(class_Ia);
    const auto maxIb = lists.beta_address()->strpcls(class_Ib);
    auto m = dest.view();
    if (alfa) {
        for (size_t Ia{0}; Ia < maxIa; ++Ia)
            for (size_t Ib{0}; Ib < maxIb; ++Ib)
                m(Ia, Ib) = 0.0;
    } else {
        for (size_t Ib{0}; Ib < maxIb; ++Ib)
            for (size_t Ia{0}; Ia < maxIa; ++Ia)
                m(Ib, Ia) = 0.0;
    }
}

void CISigmaBuilder::scatter_C_block(np_matrix source, np_vector dest, bool alfa,
                                     const CIStrings& lists, int class_Ia, int class_Ib) {
    size_t maxIa = lists.alfa_address()->strpcls(class_Ia);
    size_t maxIb = lists.beta_address()->strpcls(class_Ib);

    auto block_index = lists.string_class()->block_index(class_Ia, class_Ib);
    auto offset = lists.block_offset(block_index);

    auto m = source.view();
    auto c = dest.view();

    if (alfa) {
        // Add m to C
        for (size_t Ia{0}; Ia < maxIa; ++Ia)
            for (size_t Ib{0}; Ib < maxIb; ++Ib)
                c(offset + Ia * maxIb + Ib) += m(Ia, Ib);
    } else {
        // Add m transposed to C
        for (size_t Ia{0}; Ia < maxIa; ++Ia)
            for (size_t Ib{0}; Ib < maxIb; ++Ib)
                c(offset + Ia * maxIb + Ib) += m(Ib, Ia);
    }
}

void CISigmaBuilder::H1(np_vector basis, np_vector sigma, bool alfa) const {
    auto h = H_.view();
    // loop over blocks of matrix C
    for (const auto& [nI, class_Ia, class_Ib] : lists_.determinant_classes()) {
        if (lists_.detpblk(nI) == 0)
            continue;

        gather_C_block(basis, CR, alfa, lists_, class_Ia, class_Ib);

        for (const auto& [nJ, class_Ja, class_Jb] : lists_.determinant_classes()) {
            if (lists_.detpblk(nJ) == 0)
                continue;

            // If we act on the alpha string, the beta string classes must be the same
            if (alfa and (class_Ib != class_Jb))
                continue;
            // If we act on the beta string, the alpha string classes must be the same
            if (not alfa and (class_Ia != class_Ja))
                continue;

            zero_block(CL, alfa, lists_, class_Ja, class_Jb);

            size_t maxL = alfa ? lists_.beta_address()->strpcls(class_Ib)
                               : lists_.alfa_address()->strpcls(class_Ia);

            const auto& pq_vo_list = alfa ? lists_.get_alfa_vo_list(class_Ia, class_Ja)
                                          : lists_.get_beta_vo_list(class_Ib, class_Jb);

            for (const auto& [pq, vo_list] : pq_vo_list) {
                const auto& [p, q] = pq;
                const double Hpq = alfa ? h(p, q) : h(p, q);
                for (const auto& [sign, I, J] : vo_list) {
                    matrix::daxpy_rows(sign * Hpq, CR, I, CL, J);
                }
            }
            scatter_C_block(CL, sigma, alfa, lists_, class_Ja, class_Jb);
        }
    }
}

void CISigmaBuilder::H2_aaaa(np_vector basis, np_vector sigma, bool alfa) const {
    auto v = V_.view();
    for (const auto& [nI, class_Ia, class_Ib] : lists_.determinant_classes()) {
        if (lists_.detpblk(nI) == 0)
            continue;

        gather_C_block(basis, CR, alfa, lists_, class_Ia, class_Ib);

        for (const auto& [nJ, class_Ja, class_Jb] : lists_.determinant_classes()) {
            // The string class on which we don't act must be the same for I and J
            if ((alfa and (class_Ib != class_Jb)) or (not alfa and (class_Ia != class_Ja)))
                continue;
            if (lists_.detpblk(nJ) == 0)
                continue;

            zero_block(CL, alfa, lists_, class_Ja, class_Jb);

            // get the size of the string of spin opposite to the one we are acting on
            size_t maxL = alfa ? lists_.beta_address()->strpcls(class_Ib)
                               : lists_.alfa_address()->strpcls(class_Ia);

            if ((class_Ia == class_Ja) and (class_Ib == class_Jb)) {
                // OO terms
                // Loop over (p>q) == (p>q)
                const auto& pq_oo_list =
                    alfa ? lists_.get_alfa_oo_list(class_Ia) : lists_.get_beta_oo_list(class_Ib);
                for (const auto& [pq, oo_list] : pq_oo_list) {
                    const auto& [p, q] = pq;
                    const double integral =
                        alfa ? v(p, q, p, q) - v(p, q, q, p) : v(p, q, p, q) - v(p, q, q, p);

                    for (const auto& I : oo_list) {
                        matrix::daxpy_rows(integral, CR, I, CL, I);
                    }
                }
            }

            // VVOO terms
            const auto& pqrs_vvoo_list = alfa ? lists_.get_alfa_vvoo_list(class_Ia, class_Ja)
                                              : lists_.get_beta_vvoo_list(class_Ib, class_Jb);
            for (const auto& [pqrs, vvoo_list] : pqrs_vvoo_list) {
                const auto& [p, q, r, s] = pqrs;
                const double integral1 =
                    alfa ? v(p, q, r, s) - v(p, q, s, r) : v(p, q, r, s) - v(p, q, s, r);
                for (const auto& [sign, I, J] : vvoo_list) {
                    matrix::daxpy_rows(sign * integral1, CR, I, CL, J);
                }
            }
            scatter_C_block(CL, sigma, alfa, lists_, class_Ja, class_Jb);
        }
    }
}

// no gather / scatter implementation of H2_aabb 2
void CISigmaBuilder::H2_aabb(np_vector basis, np_vector sigma) const {
    auto v = V_.view();
    const auto& mo_sym = lists_.string_class()->mo_sym();

    auto C_view = basis.view();
    auto HC_view = sigma.view();

    // Loop over blocks of matrix C
    for (const auto& [nI, class_Ia, class_Ib] : lists_.determinant_classes()) {
        if (lists_.detpblk(nI) == 0)
            continue;

        const auto C_offset = lists_.block_offset(nI);
        const size_t maxIb = lists_.beta_address()->strpcls(class_Ib);

        const auto h_Ib = lists_.string_class()->beta_string_classes()[class_Ib].second;

        for (const auto& [nJ, class_Ja, class_Jb] : lists_.determinant_classes()) {
            if (lists_.detpblk(nJ) == 0)
                continue;

            const auto HC_offset = lists_.block_offset(nJ);
            const auto maxJb = lists_.beta_address()->strpcls(class_Jb);

            const auto h_Jb = lists_.string_class()->beta_string_classes()[class_Jb].second;

            const auto& pq_vo_alfa = lists_.get_alfa_vo_list(class_Ia, class_Ja);
            const auto& rs_vo_beta = lists_.get_beta_vo_list(class_Ib, class_Jb);

            for (const auto& [rs, vo_beta_list] : rs_vo_beta) {
                const auto beta_list_size = vo_beta_list.size();
                if (beta_list_size == 0)
                    continue;

                const auto& [r, s] = rs;
                const auto rs_sym = mo_sym[r] ^ mo_sym[s];

                // Make sure that the symmetry of the J beta string is the same as the symmetry
                // of the I beta string times the symmetry of the rs product
                if (h_Jb != (h_Ib ^ rs_sym))
                    continue;

                for (const auto& [pq, vo_alfa_list] : pq_vo_alfa) {
                    const auto a_list_size = vo_alfa_list.size();
                    if (a_list_size == 0)
                        continue;

                    const auto& [p, q] = pq;
                    const auto pq_sym = mo_sym[p] ^ mo_sym[q];
                    // ensure that the product pqrs is totally symmetric
                    if (pq_sym != rs_sym)
                        continue;

                    // Grab the integral
                    const auto integral = v(p, r, q, s);

                    for (const auto& [sign_a, Ia, Ja] : vo_alfa_list) {
                        for (const auto& [sign_b, Ib, Jb] : vo_beta_list) {
                            HC_view(HC_offset + Ja * maxJb + Jb) +=
                                C_view(C_offset + Ia * maxIb + Ib) * integral * sign_a * sign_b;
                        }
                    }
                }
            }
        }
    }
}

} // namespace forte2
