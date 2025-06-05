#include <vector>
#include <future>
#include <iostream>
#include <cassert>
#include <algorithm>
#include <thread>

#include "helpers/timer.hpp"
#include "helpers/np_vector_functions.h"
#include "helpers/blas.h"

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

    set_H(H);
    set_V(V);
}

void CISigmaBuilder::set_H(np_matrix H) {
    if (H.ndim() != 2) {
        throw std::runtime_error("H must be a 2D matrix.");
    }
    if (H.shape(0) != lists_.norb() || H.shape(1) != lists_.norb()) {
        throw std::runtime_error("H shape does not match the number of orbitals.");
    }
    H_ = H;
    const size_t norb = lists_.norb();
    h_pq.resize(norb * norb);
    auto h = H.view();
    for (size_t p = 0; p < norb; ++p) {
        for (size_t q = 0; q < norb; ++q) {
            h_pq[p * norb + q] = h(p, q);
        }
    }
}

void CISigmaBuilder::set_V(np_tensor4 V) {
    if (V.ndim() != 4) {
        throw std::runtime_error("V must be a 4D tensor.");
    }
    if (V.shape(0) != lists_.norb() || V.shape(1) != lists_.norb() || V.shape(2) != lists_.norb() ||
        V.shape(3) != lists_.norb()) {
        throw std::runtime_error("V shape does not match the number of orbitals.");
    }
    V_ = V;

    const size_t norb = lists_.norb();
    const size_t norb2 = norb * norb;
    const size_t npairs = (norb * (norb - 1)) / 2; // Number of pairs (p, r) with p > r
    v_pr_qs.resize(norb2 * norb2);
    v_pr_qs_a.resize(npairs * npairs);
    auto v = V.view();
    for (size_t p = 0; p < norb; ++p) {
        for (size_t r = 0; r < norb; ++r) {
            const auto pr_index = p * norb + r;
            for (size_t q = 0; q < norb; ++q) {
                for (size_t s = 0; s < norb; ++s) {
                    const auto qs_index = q * norb + s;
                    v_pr_qs[pr_index * norb2 + qs_index] = v(p, r, q, s);
                }
            }
        }
    }
    for (int p = 1; p < norb; ++p) {
        for (int r = 0; r < p; ++r) {
            const auto pr_index = (p * (p - 1)) / 2 + r;
            for (int q = 1; q < norb; ++q) {
                for (int s = 0; s < q; ++s) {
                    const auto qs_index = (q * (q - 1)) / 2 + s;
                    v_pr_qs_a[pr_index * npairs + qs_index] = v(p, r, q, s) - v(p, r, s, q);
                }
            }
        }
    }
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
    auto b_span = vector::as_span(basis);
    auto s_span = vector::as_span(sigma);

    H0(b_span, s_span);
    H1_aa_gemm(b_span, s_span, true);
    H1_aa_gemm(b_span, s_span, false);

    local_timer h_aabb_timer;
    H2_aabb_gemm(b_span, s_span);
    haabb_timer_ += h_aabb_timer.elapsed_seconds();

    local_timer h_aaaa_timer;
    H2_aaaa_gemm(b_span, s_span, true);
    haaaa_timer_ += h_aaaa_timer.elapsed_seconds();

    local_timer h_bbbb_timer;
    H2_aaaa_gemm(b_span, s_span, false);
    hbbbb_timer_ += h_bbbb_timer.elapsed_seconds();

    hdiag_timer_ += t.elapsed_seconds();
    build_count_++;
}

void CISigmaBuilder::H0(std::span<double> basis, std::span<double> sigma) const {
    for (size_t i{0}, imax{basis.size()}; i < imax; ++i) {
        sigma[i] = E_ * basis[i];
    }
}

std::span<double> gather_block(std::span<double> source, std::span<double> dest, bool alfa,
                               const CIStrings& lists, int class_Ia, int class_Ib) {
    const auto block_index = lists.string_class()->block_index(class_Ia, class_Ib);
    const auto offset = lists.block_offset(block_index);
    const auto maxIa = lists.alfa_address()->strpcls(class_Ia);
    const auto maxIb = lists.beta_address()->strpcls(class_Ib);

    if (alfa) {
        std::span<double> dest_span(source.data() + offset, maxIa * maxIb);
        return dest_span;
    }
    for (size_t Ia{0}; Ia < maxIa; ++Ia)
        for (size_t Ib{0}; Ib < maxIb; ++Ib)
            dest[Ib * maxIa + Ia] = source[offset + Ia * maxIb + Ib];
    return dest;
}

void zero_block(std::span<double> dest, bool alfa, const CIStrings& lists, int class_Ia,
                int class_Ib) {
    const auto maxIa = lists.alfa_address()->strpcls(class_Ia);
    const auto maxIb = lists.beta_address()->strpcls(class_Ib);

    if (alfa) {
        for (size_t Ia{0}; Ia < maxIa; ++Ia)
            for (size_t Ib{0}; Ib < maxIb; ++Ib)
                dest[Ia * maxIb + Ib] = 0.0;
    } else {
        for (size_t Ib{0}; Ib < maxIb; ++Ib)
            for (size_t Ia{0}; Ia < maxIa; ++Ia)
                dest[Ib * maxIa + Ia] = 0.0;
    }
}

void scatter_block(std::span<double> source, std::span<double> dest, bool alfa,
                   const CIStrings& lists, int class_Ia, int class_Ib) {
    size_t maxIa = lists.alfa_address()->strpcls(class_Ia);
    size_t maxIb = lists.beta_address()->strpcls(class_Ib);

    auto block_index = lists.string_class()->block_index(class_Ia, class_Ib);
    auto offset = lists.block_offset(block_index);

    if (alfa) {
        // Add m to C
        for (size_t Ia{0}; Ia < maxIa; ++Ia)
            for (size_t Ib{0}; Ib < maxIb; ++Ib)
                dest[offset + Ia * maxIb + Ib] += source[Ia * maxIb + Ib];
    } else {
        // Add m transposed to C
        for (size_t Ia{0}; Ia < maxIa; ++Ia)
            for (size_t Ib{0}; Ib < maxIb; ++Ib)
                dest[offset + Ia * maxIb + Ib] += source[Ib * maxIa + Ia];
    }
}

void CISigmaBuilder::H1(std::span<double> basis, std::span<double> sigma, bool alfa) const {
    auto h = H_.view();

    // loop over blocks of matrix C
    for (const auto& [nI, class_Ia, class_Ib] : lists_.determinant_classes()) {
        if (lists_.detpblk(nI) == 0)
            continue;

        auto tr = gather_block(basis, TR, alfa, lists_, class_Ia, class_Ib);

        for (const auto& [nJ, class_Ja, class_Jb] : lists_.determinant_classes()) {
            if (lists_.detpblk(nJ) == 0)
                continue;

            // If we act on the alpha string, the beta string classes must be the same
            if (alfa and (class_Ib != class_Jb))
                continue;
            // If we act on the beta string, the alpha string classes must be the same
            if (not alfa and (class_Ia != class_Ja))
                continue;

            zero_block(TL, alfa, lists_, class_Ja, class_Jb);

            size_t maxL = alfa ? lists_.beta_address()->strpcls(class_Ib)
                               : lists_.alfa_address()->strpcls(class_Ia);

            const auto& pq_vo_list = alfa ? lists_.get_alfa_vo_list(class_Ia, class_Ja)
                                          : lists_.get_beta_vo_list(class_Ib, class_Jb);

            for (const auto& [pq, vo_list] : pq_vo_list) {
                const auto& [p, q] = pq;
                const double Hpq = alfa ? h(p, q) : h(p, q);
                for (const auto& [sign, I, J] : vo_list) {
                    for (size_t idx{0}; idx != maxL; ++idx) {
                        TL[J * maxL + idx] += sign * Hpq * tr[I * maxL + idx];
                    }
                }
            }
            scatter_block(TL, sigma, alfa, lists_, class_Ja, class_Jb);
        }
    }
}

void CISigmaBuilder::H2_aaaa(std::span<double> basis, std::span<double> sigma, bool alfa) const {
    auto v = V_.view();

    // loop over blocks of matrix C
    for (const auto& [nI, class_Ia, class_Ib] : lists_.determinant_classes()) {
        if (lists_.detpblk(nI) == 0)
            continue;

        auto tr = gather_block(basis, TR, alfa, lists_, class_Ia, class_Ib);

        for (const auto& [nJ, class_Ja, class_Jb] : lists_.determinant_classes()) {
            // The string class on which we don't act must be the same for I and J
            if ((alfa and (class_Ib != class_Jb)) or (not alfa and (class_Ia != class_Ja)))
                continue;
            if (lists_.detpblk(nJ) == 0)
                continue;

            zero_block(TL, alfa, lists_, class_Ja, class_Jb);

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
                            TL[I * maxL + idx] += integral * tr[I * maxL + idx];
                        }
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
                    for (size_t idx{0}; idx != maxL; ++idx) {
                        TL[J * maxL + idx] += sign * integral1 * TR[I * maxL + idx];
                    }
                }
            }
            scatter_block(TL, sigma, alfa, lists_, class_Ja, class_Jb);
        }
    }
}

void CISigmaBuilder::H2_aabb(std::span<double> basis, std::span<double> sigma) const {
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
                for (size_t Ja{0}; Ja < maxJa; ++Ja) {
                    for (size_t idx{0}; idx < beta_list_size; ++idx)
                        TL[Ja * beta_list_size + idx] = 0.0;
                }

                // Gather cols of C into CR with the correct sign
                for (size_t Ia{0}; Ia < maxIa; ++Ia) {
                    for (size_t idx{0}; const auto& [sign, Ib, _] : vo_beta_list) {
                        TR[Ia * beta_list_size + idx] = basis[C_offset + Ia * maxIb + Ib] * sign;
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
                        sigma[HC_offset + Ja * maxJb + Jb] += TL[Ja * beta_list_size + idx];
                        idx++;
                    }
                }
            }
        }
    }
}

// void CISigmaBuilder::H2_aabb_string_driven(std::span<double> basis, std::span<double> sigma)
// const {
//     auto v = V_.view();
//     const auto& mo_sym = lists_.string_class()->mo_sym();

//     // Loop over blocks of matrix HC
//     for (const auto& [nJ, class_Ja, class_Jb] : lists_.determinant_classes()) {
//         if (lists_.detpblk(nJ) == 0)
//             continue;

//         auto h_Jb = lists_.string_class()->beta_string_classes()[class_Jb].second;
//         const auto HC_offset = lists_.block_offset(nJ);
//         const auto maxJa = lists_.alfa_address()->strpcls(class_Ja);
//         const auto maxJb = lists_.beta_address()->strpcls(class_Jb);

//         // Loop over blocks of matrix C
//         for (const auto& [nI, class_Ia, class_Ib] : lists_.determinant_classes()) {
//             if (lists_.detpblk(nI) == 0)
//                 continue;

//             auto h_Ib = lists_.string_class()->beta_string_classes()[class_Ib].second;
//             const auto C_offset = lists_.block_offset(nI);
//             const size_t maxIb = lists_.beta_address()->strpcls(class_Ib);
//             const size_t maxIa = lists_.alfa_address()->strpcls(class_Ia);

//             const auto& pq_vo_alfa = lists_.get_alfa_vo_list(class_Ia, class_Ja);
//             const auto& rs_vo_beta = lists_.get_beta_vo_list(class_Ib, class_Jb);

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

//                 // Zero the block of CL used to store the sigma
//                 for (size_t Ja{0}; Ja < maxJa; ++Ja) {
//                     for (size_t idx{0}; idx < beta_list_size; ++idx)
//                         TL[Ja * beta_list_size + idx] = 0.0;
//                 }

//                 // Gather cols of C into CR with the correct sign
//                 for (size_t Ia{0}; Ia < maxIa; ++Ia) {
//                     for (size_t idx{0}; const auto& [sign, Ib, _] : vo_beta_list) {
//                         TR[Ia * beta_list_size + idx] = basis[C_offset + Ia * maxIb + Ib] * sign;
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
//                     const auto integral = v(p, r, q, s);

//                     for (const auto& [sign_a, Ia, Ja] : vo_alfa_list) {
//                         const auto factor = integral * sign_a;
//                         for (size_t idx{0}; idx != beta_list_size; ++idx) {
//                             TL[Ja * beta_list_size + idx] += factor * TR[Ia * beta_list_size +
//                             idx];
//                         }
//                     }
//                 } // End loop over p,q

//                 // Scatter cols of CL into HC (the sign was included before in the gathering)
//                 for (size_t Ja{0}; Ja < maxJa; ++Ja) {
//                     for (size_t idx{0}; const auto& [_1, _2, Jb] : vo_beta_list) {
//                         sigma[HC_offset + Ja * maxJb + Jb] += TL[Ja * beta_list_size + idx];
//                         idx++;
//                     }
//                 }
//             }
//         }
//     }
// }

void CISigmaBuilder::H1_aa_gemm(std::span<double> basis, std::span<double> sigma, bool alfa) const {
    auto v = V_.view();

    const size_t norb = lists_.norb();

    const auto na = lists_.na();
    const auto nb = lists_.nb();
    if ((alfa and (na < 1)) or ((!alfa) and (nb < 1)))
        return;

    const auto& alfa_address = lists_.alfa_address();
    const auto& beta_address = lists_.beta_address();
    int num_1h_classes =
        alfa ? lists_.alfa_address_1h()->nclasses() : lists_.beta_address_1h()->nclasses();

    for (int class_K = 0; class_K < num_1h_classes; ++class_K) {
        size_t maxK = alfa ? lists_.alfa_address_1h()->strpcls(class_K)
                           : lists_.beta_address_1h()->strpcls(class_K);
        // loop over blocks of matrix C
        for (const auto& [nI, class_Ia, class_Ib] : lists_.determinant_classes()) {
            if (lists_.detpblk(nI) == 0)
                continue;
            size_t maxL = alfa ? beta_address->strpcls(class_Ib) : alfa_address->strpcls(class_Ia);

            if (maxL > 0) {
                // We gather the block of C into TR

                local_timer ta;
                const size_t dimKL = maxK * maxL;
                // This block requires a temp_dim = norb * maxK * maxL matrix
                const auto temp_dim = norb * dimKL;
                if (TR.size() < temp_dim) {
                    TR.resize(temp_dim);
                    TL.resize(temp_dim);
                }
                // We use TL to store the result of the transformation to the 2h basis
                std::fill_n(TL.begin(), temp_dim, 0.0);
                std::fill_n(TR.begin(), temp_dim, 0.0);

                auto tr = gather_block(basis, TR, alfa, lists_, class_Ia, class_Ib);

                for (size_t K = 0; K < maxK; ++K) {
                    auto& Krlist = alfa ? lists_.get_alfa_1h_list(class_K, K, class_Ia)
                                        : lists_.get_beta_1h_list(class_K, K, class_Ib);
                    for (const auto& [sign_K, q, I] : Krlist) {
                        for (size_t idx{0}; idx != maxL; ++idx) {
                            TL[q * dimKL + K * maxL + idx] += sign_K * tr[I * maxL + idx];
                        }
                    }
                }
                matrix_product('N', 'N', norb, dimKL, norb, 1.0, h_pq.data(), norb, TL.data(),
                               dimKL, 0.0, TR.data(), dimKL);
                std::fill_n(TL.begin(), temp_dim, 0.0);

                for (size_t K = 0; K < maxK; ++K) {
                    auto& Krlist = alfa ? lists_.get_alfa_1h_list(class_K, K, class_Ia)
                                        : lists_.get_beta_1h_list(class_K, K, class_Ib);
                    for (const auto& [sign_K, p, I] : Krlist) {
                        for (size_t idx{0}; idx != maxL; ++idx) {
                            TL[I * maxL + idx] += sign_K * TR[p * dimKL + K * maxL + idx];
                        }
                    }
                }
                scatter_block(TL, sigma, alfa, lists_, class_Ia, class_Ib);
            }
        }
    }
}

void CISigmaBuilder::H2_aaaa_gemm(std::span<double> basis, std::span<double> sigma,
                                  bool alfa) const {
    if ((alfa and (lists_.na() < 2)) or ((!alfa) and (lists_.nb() < 2)))
        return;

    const size_t norb = lists_.norb();
    const size_t npairs = norb * (norb - 1) / 2;

    const auto& alfa_address = lists_.alfa_address();
    const auto& beta_address = lists_.beta_address();

    int num_2h_classes =
        alfa ? lists_.alfa_address_2h()->nclasses() : lists_.beta_address_2h()->nclasses();

    for (int class_K = 0; class_K < num_2h_classes; ++class_K) {
        size_t maxK = alfa ? lists_.alfa_address_2h()->strpcls(class_K)
                           : lists_.beta_address_2h()->strpcls(class_K);

        // loop over blocks of matrix C
        for (const auto& [nI, class_Ia, class_Ib] : lists_.determinant_classes()) {
            if (lists_.detpblk(nI) == 0)
                continue;

            size_t maxL = alfa ? beta_address->strpcls(class_Ib) : alfa_address->strpcls(class_Ia);

            if (maxL > 0) {
                // We gather the block of C into TR
                const size_t dimKL = maxK * maxL;

                // This block requires a temp_dim = npairs * maxK * maxL matrix
                const auto temp_dim = npairs * dimKL;
                if (TR.size() < temp_dim) {
                    TR.resize(temp_dim);
                    TL.resize(temp_dim);
                }
                // We use TL to store the result of the transformation to the 2h basis
                std::fill_n(TL.begin(), temp_dim, 0.0);

                auto tr = gather_block(basis, TR, alfa, lists_, class_Ia, class_Ib);

                for (size_t K = 0; K < maxK; ++K) {
                    auto& Krlist = alfa ? lists_.get_alfa_2h_list(class_K, K, class_Ia)
                                        : lists_.get_beta_2h_list(class_K, K, class_Ib);
                    for (const auto& [sign_K, q, s, I] : Krlist) {
                        const size_t qs_index = q * (q - 1) / 2 + s;
                        for (size_t idx{0}; idx != maxL; ++idx) {
                            TL[qs_index * dimKL + K * maxL + idx] += sign_K * tr[I * maxL + idx];
                        }
                    }
                }
                matrix_product('N', 'N', npairs, dimKL, npairs, 1.0, v_pr_qs_a.data(), npairs,
                               TL.data(), dimKL, 0.0, TR.data(), dimKL);

                std::fill_n(TL.begin(), temp_dim, 0.0);

                for (size_t K = 0; K < maxK; ++K) {
                    auto& Krlist = alfa ? lists_.get_alfa_2h_list(class_K, K, class_Ia)
                                        : lists_.get_beta_2h_list(class_K, K, class_Ib);
                    for (const auto& [sign_K, p, r, I] : Krlist) {
                        const size_t pr_index = p * (p - 1) / 2 + r;
                        for (size_t idx{0}; idx != maxL; ++idx) {
                            TL[I * maxL + idx] += sign_K * TR[pr_index * dimKL + K * maxL + idx];
                        }
                    }
                }
                scatter_block(TL, sigma, alfa, lists_, class_Ia, class_Ib);
            }
        }
    }
}

void CISigmaBuilder::H2_aabb_gemm(std::span<double> basis, std::span<double> sigma) const {
    if ((lists_.na() < 1) or (lists_.nb() < 1))
        return;

    size_t norb = lists_.norb();
    const auto norb2 = norb * norb;

    const int num_1h_class_Ka = lists_.alfa_address_1h()->nclasses();
    const int num_1h_class_Kb = lists_.beta_address_1h()->nclasses();

    // loop over blocks of N-2 space
    for (int class_Ka = 0; class_Ka < num_1h_class_Ka; ++class_Ka) {
        const auto maxKa = lists_.alfa_address_1h()->strpcls(class_Ka);
        for (int class_Kb = 0; class_Kb < num_1h_class_Kb; ++class_Kb) {
            const auto maxKb = lists_.beta_address_1h()->strpcls(class_Kb);

            const auto Kdim = maxKa * maxKb;
            const auto temp_dim = norb2 * Kdim;

            // This block requires a temp_dim = norb * norb * maxKa * maxKb matrix
            if (TR.size() < temp_dim) {
                TR.resize(temp_dim);
                TL.resize(temp_dim);
            }
            std::fill_n(TR.begin(), temp_dim, 0.0);

            // D([qs],[Ka Kb]) = \sum_{Ia,Ib} B^{Ka,Kb,Ia,Ib}_{pq} C_{Ia,Ib}
            for (const auto& [nI, class_Ia, class_Ib] : lists_.determinant_classes()) {
                if (lists_.detpblk(nI) == 0)
                    continue;
                const auto maxIb = lists_.beta_address()->strpcls(class_Ib);
                const auto Cr_offset = lists_.block_offset(nI);
                for (size_t Ka = 0; Ka < maxKa; ++Ka) {
                    auto& Ka_right_list = lists_.get_alfa_1h_list(class_Ka, Ka, class_Ia);
                    for (size_t Kb = 0; Kb < maxKb; ++Kb) {
                        auto& Kb_right_list = lists_.get_beta_1h_list(class_Kb, Kb, class_Ib);
                        const auto Kidx = Ka * maxKb + Kb;
                        for (const auto& [sign_q, q, Ia] : Ka_right_list) {
                            const size_t qnorb = q * norb;
                            const size_t b_offset = Cr_offset + Ia * maxIb;
                            for (const auto& [sign_s, s, Ib] : Kb_right_list) {
                                const size_t qs_index = qnorb + s;
                                TR[qs_index * Kdim + Kidx] = sign_q * sign_s * basis[b_offset + Ib];
                            }
                        }
                    }
                }
            }

            matrix_product('N', 'N', norb2, Kdim, norb2, 1.0, v_pr_qs.data(), norb2, TR.data(),
                           Kdim, 0.0, TL.data(), Kdim);

            // D([qs],[Ka Kb]) = \sum_{Ia,Ib} B^{Ka,Kb,Ia,Ib}_{pq} C_{Ia,Ib}
            for (const auto& [nI, class_Ia, class_Ib] : lists_.determinant_classes()) {
                if (lists_.detpblk(nI) == 0)
                    continue;
                const auto maxIb = lists_.beta_address()->strpcls(class_Ib);
                const auto Cr_offset = lists_.block_offset(nI);

                for (size_t Ka = 0; Ka < maxKa; ++Ka) {
                    const auto& Ka_right_list = lists_.get_alfa_1h_list(class_Ka, Ka, class_Ia);
                    for (size_t Kb = 0; Kb < maxKb; ++Kb) {
                        const auto& Kb_right_list = lists_.get_beta_1h_list(class_Kb, Kb, class_Ib);
                        const auto Kidx = Ka * maxKb + Kb;
                        for (const auto& [sign_p, p, Ia] : Ka_right_list) {
                            const size_t pnorb = p * norb;
                            const size_t s_offset = Cr_offset + Ia * maxIb;
                            for (const auto& [sign_r, r, Ib] : Kb_right_list) {
                                const size_t pr_index = pnorb + r;
                                sigma[s_offset + Ib] +=
                                    sign_p * sign_r * TL[pr_index * Kdim + Kidx];
                            }
                        }
                    }
                }
            }
        }
    }
}

} // namespace forte2
