#include "helpers/timer.hpp"
#include "helpers/np_matrix_functions.h"
#include "helpers/np_vector_functions.h"

#include "ci_sigma_builder.h"

namespace forte2 {

CISigmaBuilder::CISigmaBuilder(const CIStrings& lists, double E, np_matrix& H, np_tensor4& V)
    : lists_(lists), E_(E), H_(H), V_(V), slater_rules_(lists.norb(), E, H, V) {
    // Find the largest size of the symmetry blocks
    size_t max_size = 0;
    for (int class_Ia = 0; class_Ia < lists.alfa_address()->nclasses(); ++class_Ia) {
        max_size = std::max(max_size, lists.alfa_address()->strpcls(class_Ia));
    }
    for (int class_Ib = 0; class_Ib < lists.beta_address()->nclasses(); ++class_Ib) {
        max_size = std::max(max_size, lists.beta_address()->strpcls(class_Ib));
    }
    TR.resize(max_size * max_size);
    TL.resize(max_size * max_size);
    C.resize(lists_.ndet());
    S.resize(lists_.ndet());
}

np_vector CISigmaBuilder::form_Hdiag_csf(const std::vector<Determinant>& dets,
                                         const CISpinAdapter& spin_adapter,
                                         bool spin_adapt_full_preconditioner) const {
    auto Hdiag = make_zeros<nb::numpy, double, 1>({spin_adapter.ncsf()});
    auto Hdiag_view = Hdiag.view();
    // Compute the diagonal elements of the Hamiltonian in the CSF basis
    if (spin_adapt_full_preconditioner) {
        for (size_t i{0}, imax{spin_adapter.ncsf()}; i < imax; ++i) {
            double energy = 0.0;
            int I = 0;
            for (const auto& [det_add_I, c_I] : spin_adapter.csf(i)) {
                int J = 0;
                for (const auto& [det_add_J, c_J] : spin_adapter.csf(i)) {
                    if (I == J) {
                        energy += c_I * c_J * slater_rules_.energy(dets[det_add_I]);
                    } else if (I < J) {
                        if (c_I * c_J != 0.0) {
                            energy += 2.0 * c_I * c_J *
                                      slater_rules_.slater_rules(dets[det_add_I], dets[det_add_J]);
                        }
                    }
                    J++;
                }
                I++;
            }
            Hdiag_view(i) = energy;
        }
    } else {
        for (size_t i{0}, imax{spin_adapter.ncsf()}; i < imax; ++i) {
            double energy = 0.0;
            for (const auto& [det_add_I, c_I] : spin_adapter.csf(i)) {
                energy += c_I * c_I * slater_rules_.energy(dets[det_add_I]);
            }
            Hdiag_view(i) = energy;
        }
    }
    return Hdiag;
}

double CISigmaBuilder::slater_rules_csf(const std::vector<Determinant>& dets,
                                        const CISpinAdapter& spin_adapter, size_t I,
                                        size_t J) const {
    double matrix_element = 0.0;
    for (const auto& [det_add_I, c_I] : spin_adapter.csf(I)) {
        for (const auto& [det_add_J, c_J] : spin_adapter.csf(J)) {
            if (det_add_I == det_add_J) {
                matrix_element += c_I * c_J * slater_rules_.energy(dets[det_add_I]);
            } else if (det_add_I < det_add_J) {
                if (c_I * c_J != 0.0) {
                    matrix_element += 2.0 * c_I * c_J *
                                      slater_rules_.slater_rules(dets[det_add_I], dets[det_add_J]);
                }
            }
        }
    }
    return matrix_element;
}

void CISigmaBuilder::Hamiltonian(np_vector basis, np_vector sigma) const {
    local_timer t;
    vector::zero(sigma);

    auto b_view = basis.view();
    for (size_t i{0}, imax{C.size()}; i < imax; ++i) {
        C[i] = b_view(i);
    }

    H0();
    H1(true);
    H1(false);

    H2_aabb();
    H2_aaaa(true);
    H2_aaaa(false);

    auto S_view = sigma.view();
    for (size_t i{0}, imax{S.size()}; i < imax; ++i) {
        S_view(i) += S[i];
    }
    hdiag_timer_ += t.elapsed_seconds();
    build_count_++;
}

void CISigmaBuilder::H0() const {
    for (size_t i{0}, imax{C.size()}; i < imax; ++i) {
        S[i] = E_ * C[i];
    }
}

void gather_block(np_vector source, np_matrix dest, bool alfa, const CIStrings& lists, int class_Ia,
                  int class_Ib) {
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

void zero_block(np_matrix dest, bool alfa, const CIStrings& lists, int class_Ia, int class_Ib) {
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

void scatter_block(np_matrix source, np_vector dest, bool alfa, const CIStrings& lists,
                   int class_Ia, int class_Ib) {
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

void gather_block2(std::vector<double>& source, std::vector<double>& dest, bool alfa,
                   const CIStrings& lists, int class_Ia, int class_Ib) {
    const auto block_index = lists.string_class()->block_index(class_Ia, class_Ib);
    const auto offset = lists.block_offset(block_index);
    const auto maxIa = lists.alfa_address()->strpcls(class_Ia);
    const auto maxIb = lists.beta_address()->strpcls(class_Ib);
    auto m = dest.data();
    const auto c = source.data();

    if (alfa) {
        for (size_t Ia{0}; Ia < maxIa; ++Ia)
            for (size_t Ib{0}; Ib < maxIb; ++Ib)
                m[Ia * maxIb + Ib] = c[offset + Ia * maxIb + Ib];
    } else {
        for (size_t Ia{0}; Ia < maxIa; ++Ia)
            for (size_t Ib{0}; Ib < maxIb; ++Ib)
                m[Ib * maxIa + Ia] = c[offset + Ia * maxIb + Ib];
    }
}

void zero_block2(std::vector<double>& dest, bool alfa, const CIStrings& lists, int class_Ia,
                 int class_Ib) {
    const auto maxIa = lists.alfa_address()->strpcls(class_Ia);
    const auto maxIb = lists.beta_address()->strpcls(class_Ib);
    auto m = dest.data();
    if (alfa) {
        for (size_t Ia{0}; Ia < maxIa; ++Ia)
            for (size_t Ib{0}; Ib < maxIb; ++Ib)
                m[Ia * maxIb + Ib] = 0.0;
    } else {
        for (size_t Ib{0}; Ib < maxIb; ++Ib)
            for (size_t Ia{0}; Ia < maxIa; ++Ia)
                m[Ib * maxIa + Ia] = 0.0;
    }
}

void scatter_block2(std::vector<double>& source, std::vector<double>& dest, bool alfa,
                    const CIStrings& lists, int class_Ia, int class_Ib) {
    size_t maxIa = lists.alfa_address()->strpcls(class_Ia);
    size_t maxIb = lists.beta_address()->strpcls(class_Ib);

    auto block_index = lists.string_class()->block_index(class_Ia, class_Ib);
    auto offset = lists.block_offset(block_index);

    auto m = source.data();
    auto c = dest.data();

    if (alfa) {
        // Add m to C
        for (size_t Ia{0}; Ia < maxIa; ++Ia)
            for (size_t Ib{0}; Ib < maxIb; ++Ib)
                c[offset + Ia * maxIb + Ib] += m[Ia * maxIb + Ib];
    } else {
        // Add m transposed to C
        for (size_t Ia{0}; Ia < maxIa; ++Ia)
            for (size_t Ib{0}; Ib < maxIb; ++Ib)
                c[offset + Ia * maxIb + Ib] += m[Ib * maxIa + Ia];
    }
}

// void CISigmaBuilder::H1(np_vector basis, np_vector sigma, bool alfa) const {
//     auto h = H_.view();
//     // loop over blocks of matrix C
//     for (const auto& [nI, class_Ia, class_Ib] : lists_.determinant_classes()) {
//         if (lists_.detpblk(nI) == 0)
//             continue;

//         gather_block(basis, CR, alfa, lists_, class_Ia, class_Ib);

//         for (const auto& [nJ, class_Ja, class_Jb] : lists_.determinant_classes()) {
//             if (lists_.detpblk(nJ) == 0)
//                 continue;

//             // If we act on the alpha string, the beta string classes must be the same
//             if (alfa and (class_Ib != class_Jb))
//                 continue;
//             // If we act on the beta string, the alpha string classes must be the same
//             if (not alfa and (class_Ia != class_Ja))
//                 continue;

//             zero_block(CL, alfa, lists_, class_Ja, class_Jb);

//             size_t maxL = alfa ? lists_.beta_address()->strpcls(class_Ib)
//                                : lists_.alfa_address()->strpcls(class_Ia);

//             const auto& pq_vo_list = alfa ? lists_.get_alfa_vo_list(class_Ia, class_Ja)
//                                           : lists_.get_beta_vo_list(class_Ib, class_Jb);

//             for (const auto& [pq, vo_list] : pq_vo_list) {
//                 const auto& [p, q] = pq;
//                 const double Hpq = alfa ? h(p, q) : h(p, q);
//                 for (const auto& [sign, I, J] : vo_list) {
//                     matrix::daxpy_rows(sign * Hpq, CR, I, CL, J);
//                 }
//             }
//             scatter_block(CL, sigma, alfa, lists_, class_Ja, class_Jb);
//         }
//     }
// }

void CISigmaBuilder::H1(bool alfa) const {
    auto h = H_.view();
    // loop over blocks of matrix C
    for (const auto& [nI, class_Ia, class_Ib] : lists_.determinant_classes()) {
        if (lists_.detpblk(nI) == 0)
            continue;

        // gather_block(basis, CR, alfa, lists_, class_Ia, class_Ib);

        gather_block2(C, TR, alfa, lists_, class_Ia, class_Ib);

        for (const auto& [nJ, class_Ja, class_Jb] : lists_.determinant_classes()) {
            if (lists_.detpblk(nJ) == 0)
                continue;

            // If we act on the alpha string, the beta string classes must be the same
            if (alfa and (class_Ib != class_Jb))
                continue;
            // If we act on the beta string, the alpha string classes must be the same
            if (not alfa and (class_Ia != class_Ja))
                continue;

            // zero_block(CL, alfa, lists_, class_Ja, class_Jb);
            zero_block2(TL, alfa, lists_, class_Ja, class_Jb);

            size_t maxL = alfa ? lists_.beta_address()->strpcls(class_Ib)
                               : lists_.alfa_address()->strpcls(class_Ia);

            const auto& pq_vo_list = alfa ? lists_.get_alfa_vo_list(class_Ia, class_Ja)
                                          : lists_.get_beta_vo_list(class_Ib, class_Jb);

            for (const auto& [pq, vo_list] : pq_vo_list) {
                const auto& [p, q] = pq;
                const double Hpq = alfa ? h(p, q) : h(p, q);
                for (const auto& [sign, I, J] : vo_list) {
                    for (size_t idx{0}; idx != maxL; ++idx) {
                        TL[J * maxL + idx] += sign * Hpq * TR[I * maxL + idx];
                    }
                }
            }
            scatter_block2(TL, S, alfa, lists_, class_Ja, class_Jb);
        }
    }
}

// void CISigmaBuilder::H2_aaaa(np_vector basis, np_vector sigma, bool alfa) const {
//     auto v = V_.view();
//     for (const auto& [nI, class_Ia, class_Ib] : lists_.determinant_classes()) {
//         if (lists_.detpblk(nI) == 0)
//             continue;

//         gather_block(basis, CR, alfa, lists_, class_Ia, class_Ib);

//         for (const auto& [nJ, class_Ja, class_Jb] : lists_.determinant_classes()) {
//             // The string class on which we don't act must be the same for I and J
//             if ((alfa and (class_Ib != class_Jb)) or (not alfa and (class_Ia != class_Ja)))
//                 continue;
//             if (lists_.detpblk(nJ) == 0)
//                 continue;

//             zero_block(CL, alfa, lists_, class_Ja, class_Jb);

//             // get the size of the string of spin opposite to the one we are acting on
//             size_t maxL = alfa ? lists_.beta_address()->strpcls(class_Ib)
//                                : lists_.alfa_address()->strpcls(class_Ia);

//             if ((class_Ia == class_Ja) and (class_Ib == class_Jb)) {
//                 // OO terms
//                 // Loop over (p>q) == (p>q)
//                 const auto& pq_oo_list =
//                     alfa ? lists_.get_alfa_oo_list(class_Ia) : lists_.get_beta_oo_list(class_Ib);
//                 for (const auto& [pq, oo_list] : pq_oo_list) {
//                     const auto& [p, q] = pq;
//                     const double integral =
//                         alfa ? v(p, q, p, q) - v(p, q, q, p) : v(p, q, p, q) - v(p, q, q, p);

//                     for (const auto& I : oo_list) {
//                         matrix::daxpy_rows(integral, CR, I, CL, I);
//                     }
//                 }
//             }

//             // VVOO terms
//             const auto& pqrs_vvoo_list = alfa ? lists_.get_alfa_vvoo_list(class_Ia, class_Ja)
//                                               : lists_.get_beta_vvoo_list(class_Ib, class_Jb);
//             for (const auto& [pqrs, vvoo_list] : pqrs_vvoo_list) {
//                 const auto& [p, q, r, s] = pqrs;
//                 const double integral1 =
//                     alfa ? v(p, q, r, s) - v(p, q, s, r) : v(p, q, r, s) - v(p, q, s, r);
//                 for (const auto& [sign, I, J] : vvoo_list) {
//                     matrix::daxpy_rows(sign * integral1, CR, I, CL, J);
//                 }
//             }
//             scatter_block(CL, sigma, alfa, lists_, class_Ja, class_Jb);
//         }
//     }
// }

void CISigmaBuilder::H2_aaaa(bool alfa) const {
    auto v = V_.view();

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

            zero_block2(TL, alfa, lists_, class_Ja, class_Jb);

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
                        for (size_t idx{0}; idx != maxL; ++idx) {
                            TL[I * maxL + idx] += integral * TR[I * maxL + idx];
                        }
                        // matrix::daxpy_rows(integral, CR, I, CL, I);
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
                    // matrix::daxpy_rows(sign * integral1, CR, I, CL, J);
                    for (size_t idx{0}; idx != maxL; ++idx) {
                        TL[J * maxL + idx] += sign * integral1 * TR[I * maxL + idx];
                    }
                }
            }
            scatter_block2(TL, S, alfa, lists_, class_Ja, class_Jb);
        }
    }
}

void CISigmaBuilder::H2_aabb() const {
    auto v = V_.view();
    const auto& mo_sym = lists_.string_class()->mo_sym();
    // Loop over blocks of matrix C
    for (const auto& [nI, class_Ia, class_Ib] : lists_.determinant_classes()) {
        if (lists_.detpblk(nI) == 0)
            continue;

        auto h_Ib = lists_.string_class()->beta_string_classes()[class_Ib].second;
        const auto C_offset = lists_.block_offset(nI);
        const size_t maxIb = lists_.beta_address()->strpcls(class_Ib);
        const size_t maxIa = lists_.alfa_address()->strpcls(class_Ia);

        // Loop over blocks of matrix HC
        for (const auto& [nJ, class_Ja, class_Jb] : lists_.determinant_classes()) {
            if (lists_.detpblk(nJ) == 0)
                continue;

            auto h_Jb = lists_.string_class()->beta_string_classes()[class_Jb].second;
            const auto HC_offset = lists_.block_offset(nJ);
            const auto maxJa = lists_.alfa_address()->strpcls(class_Ja);
            const auto maxJb = lists_.beta_address()->strpcls(class_Jb);

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

                // Zero the block of CL used to store the sigma
                // this should be faster than CL->zero(); when beta_list_size is smaller than
                // the number of columns of CL
                for (size_t Ja{0}; Ja < maxJa; ++Ja) {
                    for (size_t idx{0}; idx < beta_list_size; ++idx)
                        TL[Ja * beta_list_size + idx] = 0.0;
                }

                // Gather cols of C into CR with the correct sign
                for (size_t Ia{0}; Ia < maxIa; ++Ia) {
                    for (size_t idx{0}; const auto& [sign, Ib, _] : vo_beta_list) {
                        TR[Ia * beta_list_size + idx] = C[C_offset + Ia * maxIb + Ib] * sign;
                        idx++;
                    }
                }

                for (const auto& [pq, vo_alfa_list] : pq_vo_alfa) {
                    const auto& [p, q] = pq;
                    const auto pq_sym = mo_sym[p] ^ mo_sym[q];
                    // ensure that the product pqrs is totally symmetric
                    if (pq_sym != rs_sym)
                        continue;

                    // Grab the integral
                    const auto integral = v(p, r, q, s);

                    for (const auto& [sign_a, Ia, Ja] : vo_alfa_list) {
                        const auto factor = integral * sign_a;
                        for (size_t idx{0}; idx != beta_list_size; ++idx) {
                            TL[Ja * beta_list_size + idx] += factor * TR[Ia * beta_list_size + idx];
                        }
                    }
                } // End loop over p,q

                // Scatter cols of CL into HC (the sign was included before in the gathering)
                for (size_t Ja{0}; Ja < maxJa; ++Ja) {
                    for (size_t idx{0}; const auto& [_1, _2, Jb] : vo_beta_list) {
                        S[HC_offset + Ja * maxJb + Jb] += TL[Ja * beta_list_size + idx];
                        idx++;
                    }
                }
            }
        }
    }
}

} // namespace forte2
