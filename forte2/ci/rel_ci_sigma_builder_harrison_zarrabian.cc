#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>

#include "helpers/timer.hpp"
#include "helpers/np_vector_functions.h"
#include "helpers/indexing.hpp"
#include "helpers/blas.h"
#include "helpers/parallel.h"

#include "rel_ci_sigma_builder.h"

namespace forte2 {

namespace {
// Threading the gather/scatter loops helps only once there is enough work to amortize the overhead
constexpr size_t kRelHzParallelThreshold = 16384;

template <typename F> inline void gated_parallel_for_chunked(size_t count, F&& func) {
    if (count < kRelHzParallelThreshold)
        func(size_t{0}, count);
    else
        parallel_for_chunked(count, std::forward<F>(func));
}
} // namespace

std::tuple<std::span<std::complex<double>>, std::span<std::complex<double>>, size_t>
RelCISigmaBuilder::get_Kblock_spans(size_t nrows, size_t ncols) const {
    if ((nrows == 0) or (ncols == 0)) {
        throw std::invalid_argument("K-block dimensions must be nonzero.");
    }

    const size_t max_elements_per_buffer = memory_size_ / (2 * sizeof(std::complex<double>));
    if (max_elements_per_buffer < nrows) {
        throw std::runtime_error("The CI builder memory limit (" + std::to_string(memory_size_) +
                                 " bytes) is too small for one complex K-block column of " +
                                 std::to_string(nrows) + " elements per buffer.");
    }
    const size_t cols_chunk_size = std::min({max_elements_per_buffer / nrows, ncols,
                                             static_cast<size_t>(std::numeric_limits<int>::max())});
    const size_t block_size = nrows * cols_chunk_size;

    // Keep a larger live size: shrinking here makes a later larger kernel value-initialize the
    // same scratch storage again. The returned spans still expose only block_size elements.
    const bool can_reuse = Kblock1_.size() >= block_size && Kblock2_.size() >= block_size &&
                           Kblock1_.capacity() <= max_elements_per_buffer &&
                           Kblock2_.capacity() <= max_elements_per_buffer;
    if (!can_reuse) {
        // Drop both allocations before growing either one. This prevents geometric vector growth
        // from exceeding the pair budget and leaves a failed partial allocation safe to retry.
        std::vector<std::complex<double>>{}.swap(Kblock1_);
        std::vector<std::complex<double>>{}.swap(Kblock2_);
        Kblock1_ = std::vector<std::complex<double>>(block_size);
        Kblock2_ = std::vector<std::complex<double>>(block_size);
    }

    return {std::span<std::complex<double>>{Kblock1_.data(), block_size},
            std::span<std::complex<double>>{Kblock2_.data(), block_size}, cols_chunk_size};
}

void RelCISigmaBuilder::H1_hz(std::span<std::complex<double>> basis,
                              std::span<std::complex<double>> sigma,
                              std::span<std::complex<double>> h) const {
    // Two-component CI acts only on alpha spinors (nb == 0), so the beta space is the single
    // vacuum string and the opposite-spin spectator count maxL == 1. The [K, L] composite index
    // therefore collapses to just K
    const size_t norb = lists_.norb();
    if (lists_.na() < 1)
        return;

    const int num_1h_classes = lists_.alpha_address_1h()->nclasses();

    // |K> = ± a_q |I>
    for (int class_K = 0; class_K < num_1h_classes; ++class_K) {
        const size_t maxK = lists_.alpha_address_1h()->strpcls(class_K);

        if (maxK == 0)
            continue;

        // loop over blocks of matrix C
        for (const auto& [nI, class_Ia, class_Ib] : lists_.determinant_classes()) {
            // skip this block if it is empty
            if (lists_.block_size(nI) == 0)
                continue;

            // Grab the temporary buffers that will hold intermediates D(i, K)
            // and return the maximum size of a chunk of K indices
            auto [Kblock1, Kblock2, K_chunk_size] = get_Kblock_spans(norb, maxK);

            // Grab a span with the C coefficients
            auto tr = gather_block(basis, TR, Spin::Alpha, lists_, class_Ia, class_Ib);

            // Loop over ranges of K indices in chunks of size K_chunk_size
            for (size_t K_start = 0; K_start < maxK; K_start += K_chunk_size) {
                const size_t K_end = std::min(K_start + K_chunk_size, maxK);
                // size of the K range, which may be smaller than K_chunk_size
                const size_t K_size = K_end - K_start;

                std::fill_n(Kblock2.begin(), norb * K_size, 0.0);

                // D(q, K) = <K|a_q|I> C(I)
                // Parallelize over K: each K writes its own column of Kblock2, so no write race.
                gated_parallel_for_chunked(K_size, [&](size_t kb, size_t ke) {
                    for (size_t K = kb; K < ke; ++K) {
                        const auto& Klist =
                            lists_.get_alpha_1h_list(class_K, K_start + K, class_Ia);
                        for (const auto& [sign_K, q, I] : Klist)
                            Kblock2[q * K_size + K] += static_cast<double>(sign_K) * tr[I];
                    }
                });

                // E(p, K) = sum_q h(p,q) D(q, K)
                matrix_product('N', 'N', norb, K_size, norb, 1.0, h.data(), norb, Kblock2.data(),
                               K_size, 0.0, Kblock1.data(), K_size);

                // sigma(J) += <J|a^+_p|K> E(p, K)
                for (const auto& [nJ, class_Ja, class_Jb] : lists_.determinant_classes()) {
                    if (class_Ib != class_Jb)
                        continue;

                    const size_t maxJ = lists_.alpha_address()->strpcls(class_Ja);
                    // zero the temporary buffer that will hold the result
                    std::fill_n(TL.begin(), TL.size(), 0.0);
                    // The scatter accumulates into TL(J) (a reduction over K), so parallelize over
                    // the output string J, each thread owns a stripe of J and
                    // scans all K, accumulating only into rows it owns.
                    gated_parallel_for_chunked(maxJ, [&](size_t J_begin, size_t J_end) {
                        for (size_t K = 0; K < K_size; ++K) {
                            const auto& Klist =
                                lists_.get_alpha_1h_list(class_K, K_start + K, class_Ja);
                            for (const auto& [sign_K, p, J] : Klist) {
                                // this thread does not own this J
                                if (J < J_begin || J >= J_end)
                                    continue;
                                TL[J] += static_cast<double>(sign_K) * Kblock1[p * K_size + K];
                            }
                        }
                    });
                    scatter_block(TL, sigma, Spin::Alpha, lists_, class_Ja, class_Jb);
                }
            }
        }
    }
}

void RelCISigmaBuilder::H2_hz_same_spin(std::span<std::complex<double>> basis,
                                        std::span<std::complex<double>> sigma) const {
    if (lists_.na() < 2)
        return;

    const size_t norb = lists_.norb();
    const size_t npairs = norb * (norb - 1) / 2;

    const int num_2h_classes = lists_.alpha_address_2h()->nclasses();

    for (int class_K = 0; class_K < num_2h_classes; ++class_K) {
        const size_t maxK = lists_.alpha_address_2h()->strpcls(class_K);

        if (maxK == 0)
            continue;

        // loop over blocks of matrix C
        for (const auto& [nI, class_Ia, class_Ib] : lists_.determinant_classes()) {
            if (lists_.block_size(nI) == 0)
                continue;

            // grab temporary buffers and the maximum size of a chunk of K indices
            auto [Kblock1, Kblock2, K_chunk_size] = get_Kblock_spans(npairs, maxK);

            auto tr = gather_block(basis, TR, Spin::Alpha, lists_, class_Ia, class_Ib);

            // Loop over ranges of K indices in chunks of size K_chunk_size
            for (size_t K_start = 0; K_start < maxK; K_start += K_chunk_size) {
                const size_t K_end = std::min(K_start + K_chunk_size, maxK);
                // size of the K range, which may be smaller than K_chunk_size
                const size_t K_size = K_end - K_start;

                std::fill_n(Kblock2.begin(), npairs * K_size, 0.0);

                // D([qs], K) = <K|a_q a_s|I> C(I)
                // Parallelize over K: each K writes its own column of Kblock2, so no write race.
                gated_parallel_for_chunked(K_size, [&](size_t kb, size_t ke) {
                    for (size_t K = kb; K < ke; ++K) {
                        const auto& Krlist =
                            lists_.get_alpha_2h_list(class_K, K + K_start, class_Ia);
                        for (const auto& [sign_K, q, s, I] : Krlist) {
                            const size_t qs_index = pair_index_gt(q, s);
                            Kblock2[qs_index * K_size + K] += static_cast<double>(sign_K) * tr[I];
                        }
                    }
                });

                // E([pr], K) = sum_{qs} v([pr],[qs]) D([qs], K)
                matrix_product('N', 'N', npairs, K_size, npairs, 1.0, v_pr_qs.data(), npairs,
                               Kblock2.data(), K_size, 0.0, Kblock1.data(), K_size);

                for (const auto& [nJ, class_Ja, class_Jb] : lists_.determinant_classes()) {
                    if (class_Ib != class_Jb)
                        continue;

                    const size_t maxJ = lists_.alpha_address()->strpcls(class_Ja);
                    std::fill_n(TL.begin(), TL.size(), 0.0);
                    // sigma(J) += <J|a^+_p a^+_r|K> E([pr], K). The scatter accumulates into TL(J)
                    // (a reduction over K), so parallelize over the output string J
                    // (owner-computes): each thread owns a stripe of J and scans all K, writing
                    // only rows it owns.
                    gated_parallel_for_chunked(maxJ, [&](size_t J_begin, size_t J_end) {
                        for (size_t K = 0; K < K_size; ++K) {
                            const auto& Klist =
                                lists_.get_alpha_2h_list(class_K, K + K_start, class_Ja);
                            for (const auto& [sign_K, p, r, J] : Klist) {
                                // this thread does now own this J
                                if (J < J_begin || J >= J_end)
                                    continue;
                                const size_t pr_index = pair_index_gt(p, r);
                                TL[J] +=
                                    static_cast<double>(sign_K) * Kblock1[pr_index * K_size + K];
                            }
                        }
                    });
                    scatter_block(TL, sigma, Spin::Alpha, lists_, class_Ja, class_Jb);
                }
            }
        }
    }
}

} // namespace forte2
