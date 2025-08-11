#include <algorithm>
#include <vector>

#include "helpers/timer.hpp"
#include "helpers/np_vector_functions.h"
#include "helpers/indexing.hpp"
#include "helpers/blas.h"

#include "ci_sigma_builder.h"

namespace forte2 {

void CISigmaBuilder::H1_hz(std::span<double> basis, std::span<double> sigma, Spin spin,
                           std::span<double> h) const {
    const size_t norb = lists_.norb();
    const auto na = lists_.na();
    const auto nb = lists_.nb();

    // skip this block if there is no electron with equal spin to that on which this operator acts
    if ((is_alpha(spin) and (na < 1)) or (is_beta(spin) and (nb < 1)))
        return;

    const auto& alfa_address = lists_.alfa_address();
    const auto& beta_address = lists_.beta_address();
    const int num_1h_classes = is_alpha(spin) ? lists_.alfa_address_1h()->nclasses()
                                              : lists_.beta_address_1h()->nclasses();

    // |K>|L> = ± a_p |I>|L>
    for (int class_K = 0; class_K < num_1h_classes; ++class_K) {
        const size_t maxK = is_alpha(spin) ? lists_.alfa_address_1h()->strpcls(class_K)
                                           : lists_.beta_address_1h()->strpcls(class_K);

        if (maxK == 0)
            continue;

        // loop over blocks of matrix C
        for (const auto& [nI, class_Ia, class_Ib] : lists_.determinant_classes()) {
            // skip this block if it is empty
            if (lists_.block_size(nI) == 0)
                continue;

            // size of the strings with opposite spin to the one on which we act
            const size_t maxL =
                is_alpha(spin) ? beta_address->strpcls(class_Ib) : alfa_address->strpcls(class_Ia);

            if (maxL > 0) {
                // Grab the temporary buffers that will hold intermediates D(i,[K L])
                // and return the maximum size of a chunk of K indices
                auto [Kblock1, Kblock2, K_chunk_size] = get_Kblock_spans(norb * maxL, maxK);

                // number of elements in the temporary buffer
                const auto temp_dim = norb * K_chunk_size * maxL;

                // Grab a span with the C coefficients
                auto tr = gather_block(basis, TR, spin, lists_, class_Ia, class_Ib);

                // Loop over ranges of K indices in chuncks of size K_chunk_size
                for (size_t K_start = 0; K_start < maxK; K_start += K_chunk_size) {
                    const size_t K_end = std::min(K_start + K_chunk_size, maxK);
                    // size of the K range, which may be smaller than K_chunk_size
                    const size_t K_size = K_end - K_start;
                    // number of columns in the chunk we are processing
                    const size_t dimKL = K_size * maxL;

                    std::fill_n(Kblock2.begin(), temp_dim, 0.0);

                    // Loop over the K indices in the chunk
                    for (size_t K = 0; K < K_size; ++K) {
                        const auto& Klist =
                            is_alpha(spin)
                                ? lists_.get_alfa_1h_list(class_K, K_start + K, class_Ia)
                                : lists_.get_beta_1h_list(class_K, K_start + K, class_Ib);
                        // D(q,[K L]) += <K|a_q|I> C(I,L)
                        for (const auto& [sign_K, q, I] : Klist) {
                            add(maxL, sign_K, &tr[I * maxL], 1, &Kblock2_[q * dimKL + K * maxL], 1);
                        }
                    }

                    // E(p,[K L]) = sum_q h(p,q) D(q,[K L])
                    matrix_product('N', 'N', norb, dimKL, norb, 1.0, h.data(), norb,
                                   Kblock2_.data(), dimKL, 0.0, Kblock1_.data(), dimKL);

                    // sigma(J L) += <J|a^+_p|K> E(p,[K L])
                    for (const auto& [nJ, class_Ja, class_Jb] : lists_.determinant_classes()) {
                        if ((is_alpha(spin) and (class_Ib != class_Jb)) or
                            (is_beta(spin) and (class_Ia != class_Ja)))
                            continue;

                        // zero the temporary buffer that will hold the result
                        std::fill_n(TL.begin(), TL.size(), 0.0);
                        for (size_t K = 0; K < K_size; ++K) {
                            const auto& Klist =
                                is_alpha(spin)
                                    ? lists_.get_alfa_1h_list(class_K, K_start + K, class_Ja)
                                    : lists_.get_beta_1h_list(class_K, K_start + K, class_Jb);
                            for (const auto& [sign_K, p, I] : Klist) {
                                add(maxL, sign_K, &Kblock1_[p * dimKL + K * maxL], 1, &TL[I * maxL],
                                    1);
                            }
                        }
                        scatter_block(TL, sigma, spin, lists_, class_Ja, class_Jb);
                    }
                }
            }
        }
    }
}

void CISigmaBuilder::H2_hz_same_spin(std::span<double> basis, std::span<double> sigma,
                                     Spin spin) const {
    if ((is_alpha(spin) and (lists_.na() < 2)) or (is_beta(spin) and (lists_.nb() < 2)))
        return;

    const size_t norb = lists_.norb();
    const size_t npairs = norb * (norb - 1) / 2;

    const auto& alfa_address = lists_.alfa_address();
    const auto& beta_address = lists_.beta_address();

    const int num_2h_classes = is_alpha(spin) ? lists_.alfa_address_2h()->nclasses()
                                              : lists_.beta_address_2h()->nclasses();

    for (int class_K = 0; class_K < num_2h_classes; ++class_K) {
        const size_t maxK = is_alpha(spin) ? lists_.alfa_address_2h()->strpcls(class_K)
                                           : lists_.beta_address_2h()->strpcls(class_K);

        if (maxK == 0)
            continue;

        // loop over blocks of matrix C
        for (const auto& [nI, class_Ia, class_Ib] : lists_.determinant_classes()) {
            if (lists_.block_size(nI) == 0)
                continue;

            const size_t maxL =
                is_alpha(spin) ? beta_address->strpcls(class_Ib) : alfa_address->strpcls(class_Ia);

            if (maxL > 0) {
                // grab temporary buffers and the maximum size of a chunk of K indices
                auto [Kblock1, Kblock2, K_chunk_size] = get_Kblock_spans(npairs * maxL, maxK);

                // number of elements in the temporary blocks
                const auto temp_dim = npairs * K_chunk_size * maxL;

                auto tr = gather_block(basis, TR, spin, lists_, class_Ia, class_Ib);

                // Loop over ranges of K indices in chuncks of size K_chunk_size
                for (size_t K_start = 0; K_start < maxK; K_start += K_chunk_size) {
                    const size_t K_end = std::min(K_start + K_chunk_size, maxK);
                    // size of the K range, which may be smaller than K_chunk_size
                    const size_t K_size = K_end - K_start;
                    const size_t dimKL = K_size * maxL;

                    std::fill_n(Kblock2.begin(), temp_dim, 0.0);

                    for (size_t K = 0; K < K_size; ++K) {
                        const auto& Krlist =
                            is_alpha(spin)
                                ? lists_.get_alfa_2h_list(class_K, K + K_start, class_Ia)
                                : lists_.get_beta_2h_list(class_K, K + K_start, class_Ib);
                        for (const auto& [sign_K, q, s, I] : Krlist) {
                            const size_t qs_index = pair_index_gt(q, s);
                            add(maxL, sign_K, &tr[I * maxL], 1,
                                &Kblock2_[qs_index * dimKL + K * maxL], 1);
                        }
                    }

                    matrix_product('N', 'N', npairs, dimKL, npairs, 1.0, v_pr_qs_a.data(), npairs,
                                   Kblock2_.data(), dimKL, 0.0, Kblock1_.data(), dimKL);

                    for (const auto& [nJ, class_Ja, class_Jb] : lists_.determinant_classes()) {
                        if (((is_alpha(spin) and (class_Ib != class_Jb)) or
                             (is_beta(spin) and (class_Ia != class_Ja))))
                            continue;

                        std::fill_n(TL.begin(), TL.size(), 0.0);
                        for (size_t K = 0; K < K_size; ++K) {
                            const auto& Klist =
                                is_alpha(spin)
                                    ? lists_.get_alfa_2h_list(class_K, K + K_start, class_Ja)
                                    : lists_.get_beta_2h_list(class_K, K + K_start, class_Jb);
                            for (const auto& [sign_K, p, r, I] : Klist) {
                                const size_t pr_index = pair_index_gt(p, r);
                                add(maxL, sign_K, &Kblock1_[pr_index * dimKL + K * maxL], 1,
                                    &TL[I * maxL], 1);
                            }
                        }
                        scatter_block(TL, sigma, spin, lists_, class_Ja, class_Jb);
                    }
                }
            }
        }
    }
}

void CISigmaBuilder::H2_hz_opposite_spin(std::span<double> basis, std::span<double> sigma) const {
    if ((lists_.na() < 1) or (lists_.nb() < 1))
        return;

    size_t norb = lists_.norb();
    const auto norb2 = norb * norb;

    const int num_1h_class_Ka = lists_.alfa_address_1h()->nclasses();
    const int num_1h_class_Kb = lists_.beta_address_1h()->nclasses();

    // loop over blocks of N-2 space
    for (int class_Ka = 0; class_Ka < num_1h_class_Ka; ++class_Ka) {
        for (int class_Kb = 0; class_Kb < num_1h_class_Kb; ++class_Kb) {
            const auto maxKa = lists_.alfa_address_1h()->strpcls(class_Ka);
            const auto maxKb = lists_.beta_address_1h()->strpcls(class_Kb);

            if ((maxKa == 0) or (maxKb == 0))
                continue;

            // We gather the block of C into TR
            const size_t Kb_block_start = 0;
            const size_t Kb_block_end = maxKb;
            const size_t Kb_block_size = maxKb;

            auto [Kblock1, Kblock2, Ka_block_size] = get_Kblock_spans(norb2 * maxKb, maxKa);

            for (size_t Ka_block_start = 0; Ka_block_start < maxKa;
                 Ka_block_start += Ka_block_size) {
                size_t Ka_block_end = std::min(Ka_block_start + Ka_block_size, maxKa);
                size_t Ka_block_size = Ka_block_end - Ka_block_start;
                const auto Kdim = Ka_block_size * Kb_block_size;
                const auto temp_dim = norb2 * Kdim;

                std::fill_n(Kblock1.begin(), temp_dim, 0.0);

                // D([qs],[Ka Kb]) = \sum_{Ia,Ib} B^{Ka,Kb,Ia,Ib}_{pq} C_{Ia,Ib}
                for (const auto& [nI, class_Ia, class_Ib] : lists_.determinant_classes()) {
                    if (lists_.block_size(nI) == 0)
                        continue;
                    const auto maxIb = lists_.beta_address()->strpcls(class_Ib);
                    const auto Cr_offset = lists_.block_offset(nI);
                    const auto& Ka_right_list = lists_.get_alfa_1h_list2(class_Ka, class_Ia);
                    const auto& Kb_right_list = lists_.get_beta_1h_list2(class_Kb, class_Ib);
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
                                    Kblock1[qs_index * Kdim + Kidx] =
                                        sign_q * sign_s * basis[b_offset + Ib];
                                }
                            }
                        }
                    }
                }

                matrix_product('N', 'N', norb2, Kdim, norb2, 1.0, v_pr_qs.data(), norb2,
                               Kblock1.data(), Kdim, 0.0, Kblock2.data(), Kdim);

                // D([qs],[Ka Kb]) = \sum_{Ia,Ib} B^{Ka,Kb,Ia,Ib}_{pq} C_{Ia,Ib}
                for (const auto& [nI, class_Ia, class_Ib] : lists_.determinant_classes()) {
                    if (lists_.block_size(nI) == 0)
                        continue;
                    const auto maxIb = lists_.beta_address()->strpcls(class_Ib);
                    const auto Cr_offset = lists_.block_offset(nI);
                    const auto& Ka_right_list = lists_.get_alfa_1h_list2(class_Ka, class_Ia);
                    const auto& Kb_right_list = lists_.get_beta_1h_list2(class_Kb, class_Ib);
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
                                        sign_p * sign_r * Kblock2[pr_index * Kdim + Kidx];
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
