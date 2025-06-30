#include <algorithm>
#include <future>
#include <iostream>
#include <thread>
#include <vector>

#include "helpers/timer.hpp"
#include "helpers/np_vector_functions.h"
#include "helpers/indexing.hpp"
#include "helpers/blas.h"

#include "ci_sigma_builder.h"

namespace forte2 {

void CISigmaBuilder::H1_aa_gemm(std::span<double> basis, std::span<double> sigma, bool alfa,
                                std::span<double> h) const {
    const size_t norb = lists_.norb();

    const auto na = lists_.na();
    const auto nb = lists_.nb();
    if ((alfa and (na < 1)) or ((!alfa) and (nb < 1)))
        return;

    const auto& alfa_address = lists_.alfa_address();
    const auto& beta_address = lists_.beta_address();
    int num_1h_classes =
        alfa ? lists_.alfa_address_1h()->nclasses() : lists_.beta_address_1h()->nclasses();

    std::vector<double> TR, TL;

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
                // We use TL to store the result of the transformation to the 2h
                // basis
                std::fill_n(TL.begin(), temp_dim, 0.0);
                std::fill_n(TR.begin(), temp_dim, 0.0);

                auto tr = gather_block(basis, TR, alfa, lists_, class_Ia, class_Ib);

                for (size_t K = 0; K < maxK; ++K) {
                    auto& Klist = alfa ? lists_.get_alfa_1h_list(class_K, K, class_Ia)
                                       : lists_.get_beta_1h_list(class_K, K, class_Ib);
                    for (const auto& [sign_K, q, I] : Klist) {
                        add(maxL, sign_K, &tr[I * maxL], 1, &TL[q * dimKL + K * maxL], 1);
                    }
                }

                matrix_product('N', 'N', norb, dimKL, norb, 1.0, h.data(), norb, TL.data(), dimKL,
                               0.0, TR.data(), dimKL);

                for (const auto& [nJ, class_Ja, class_Jb] : lists_.determinant_classes()) {
                    if ((alfa) and (class_Ib != class_Jb) or ((!alfa) and (class_Ia != class_Ja)))
                        continue;

                    std::fill_n(TL.begin(), temp_dim, 0.0);
                    for (size_t K = 0; K < maxK; ++K) {
                        auto& Klist = alfa ? lists_.get_alfa_1h_list(class_K, K, class_Ja)
                                           : lists_.get_beta_1h_list(class_K, K, class_Jb);
                        for (const auto& [sign_K, p, I] : Klist) {
                            add(maxL, sign_K, &TR[p * dimKL + K * maxL], 1, &TL[I * maxL], 1);
                        }
                    }
                    scatter_block(TL, sigma, alfa, lists_, class_Ja, class_Jb);
                }
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
                // We use TL to store the result of the transformation to the 2h
                // basis
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

                for (const auto& [nJ, class_Ja, class_Jb] : lists_.determinant_classes()) {
                    if ((alfa) and (class_Ib != class_Jb) or ((!alfa) and (class_Ia != class_Ja)))
                        continue;

                    std::fill_n(TL.begin(), temp_dim, 0.0);
                    for (size_t K = 0; K < maxK; ++K) {
                        auto& Klist = alfa ? lists_.get_alfa_2h_list(class_K, K, class_Ja)
                                           : lists_.get_beta_2h_list(class_K, K, class_Jb);
                        for (const auto& [sign_K, p, r, I] : Klist) {
                            const size_t pr_index = p * (p - 1) / 2 + r;
                            add(maxL, sign_K, &TR[pr_index * dimKL + K * maxL], 1, &TL[I * maxL],
                                1);
                        }
                    }
                    scatter_block(TL, sigma, alfa, lists_, class_Ja, class_Jb);
                }

                // std::fill_n(TL.begin(), temp_dim, 0.0);

                // for (size_t K = 0; K < maxK; ++K) {
                //     auto& Krlist = alfa ? lists_.get_alfa_2h_list(class_K, K, class_Ia)
                //                         : lists_.get_beta_2h_list(class_K, K, class_Ib);
                //     for (const auto& [sign_K, p, r, I] : Krlist) {
                //         const size_t pr_index = p * (p - 1) / 2 + r;
                //         for (size_t idx{0}; idx != maxL; ++idx) {
                //             TL[I * maxL + idx] += sign_K * TR[pr_index * dimKL + K * maxL + idx];
                //         }
                //     }
                // }
                // scatter_block(TL, sigma, alfa, lists_, class_Ia, class_Ib);
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

    std::vector<double> TR_local;
    std::vector<double> TL_local;

    // loop over blocks of N-2 space
    for (int class_Ka = 0; class_Ka < num_1h_class_Ka; ++class_Ka) {
        for (int class_Kb = 0; class_Kb < num_1h_class_Kb; ++class_Kb) {
            const auto maxKa = lists_.alfa_address_1h()->strpcls(class_Ka);
            const auto maxKb = lists_.beta_address_1h()->strpcls(class_Kb);

            // We gather the block of C into TR
            size_t Kb_block_start = 0;
            size_t Kb_block_end = maxKb;
            size_t Kb_block_size = maxKb;
            const size_t Ka_block_size = std::min(134217728 / (maxKb * norb2), maxKa);
            for (size_t Ka_block_start = 0; Ka_block_start < maxKa;
                 Ka_block_start += Ka_block_size) {
                size_t Ka_block_end = std::min(Ka_block_start + Ka_block_size, maxKa);
                size_t Ka_block_size = Ka_block_end - Ka_block_start;
                const auto Kdim = Ka_block_size * (Kb_block_end - Kb_block_start);
                const auto temp_dim = norb2 * Kdim;

                // This block requires a temp_dim = norb * norb * maxKa * maxKb matrix
                if (TR_local.size() < temp_dim) {
                    TR_local.resize(temp_dim);
                    TL_local.resize(temp_dim);
                }
                std::fill_n(TR_local.begin(), temp_dim, 0.0);

                // D([qs],[Ka Kb]) = \sum_{Ia,Ib} B^{Ka,Kb,Ia,Ib}_{pq} C_{Ia,Ib}
                for (const auto& [nI, class_Ia, class_Ib] : lists_.determinant_classes()) {
                    if (lists_.detpblk(nI) == 0)
                        continue;
                    const auto maxIb = lists_.beta_address()->strpcls(class_Ib);
                    const auto Cr_offset = lists_.block_offset(nI);
                    const auto& Ka_right_list =
                        lists_.get_alfa_1h_list2(class_Ka, class_Ia); // Ka_block_start + Ka,
                    const auto& Kb_right_list =
                        lists_.get_beta_1h_list2(class_Kb, class_Ib); // Kb_block_start + Kb,
                    if (Ka_right_list.empty() || Kb_right_list.empty())
                        continue;
                    for (size_t Ka = 0; Ka < Ka_block_size; ++Ka) {
                        const auto& KaL = Ka_right_list[Ka_block_start + Ka];
                        for (size_t Kb = 0; Kb < Kb_block_size; ++Kb) {
                            const auto& KbL = Kb_right_list[Kb_block_start + Kb];
                            const auto Kidx = Ka * Kb_block_size + Kb;
                            for (const auto& [sign_q, q, Ia] : KaL) {
                                const size_t qnorb = q * norb;
                                const size_t b_offset = Cr_offset + Ia * maxIb;
                                for (const auto& [sign_s, s, Ib] : KbL) {
                                    const size_t qs_index = qnorb + s;
                                    TR_local[qs_index * Kdim + Kidx] =
                                        sign_q * sign_s * basis[b_offset + Ib];
                                }
                            }
                        }
                    }
                }

                matrix_product('N', 'N', norb2, Kdim, norb2, 1.0, v_pr_qs.data(), norb2,
                               TR_local.data(), Kdim, 0.0, TL_local.data(), Kdim);

                // D([qs],[Ka Kb]) = \sum_{Ia,Ib} B^{Ka,Kb,Ia,Ib}_{pq} C_{Ia,Ib}
                for (const auto& [nI, class_Ia, class_Ib] : lists_.determinant_classes()) {
                    if (lists_.detpblk(nI) == 0)
                        continue;
                    const auto maxIb = lists_.beta_address()->strpcls(class_Ib);
                    const auto Cr_offset = lists_.block_offset(nI);
                    const auto& Ka_right_list =
                        lists_.get_alfa_1h_list2(class_Ka, class_Ia); // Ka_block_start + Ka,
                    const auto& Kb_right_list =
                        lists_.get_beta_1h_list2(class_Kb, class_Ib); // Kb_block_start + Kb,
                    if (Ka_right_list.empty() || Kb_right_list.empty())
                        continue;
                    for (size_t Ka = 0; Ka < Ka_block_size; ++Ka) {
                        const auto& KaL = Ka_right_list[Ka_block_start + Ka];
                        for (size_t Kb = 0; Kb < Kb_block_size; ++Kb) {
                            const auto& KbL = Kb_right_list[Kb_block_start + Kb];
                            const auto Kidx = Ka * Kb_block_size + Kb;
                            for (const auto& [sign_p, p, Ia] : KaL) {
                                const size_t pnorb = p * norb;
                                const size_t s_offset = Cr_offset + Ia * maxIb;
                                for (const auto& [sign_r, r, Ib] : KbL) {
                                    const size_t pr_index = pnorb + r;
                                    sigma[s_offset + Ib] +=
                                        sign_p * sign_r * TL_local[pr_index * Kdim + Kidx];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

} // namespace forte2
