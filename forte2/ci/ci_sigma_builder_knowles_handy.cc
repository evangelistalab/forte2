#include <algorithm>
#include <future>
#include <iostream>

#include "helpers/timer.hpp"
#include "helpers/blas.h"
#include "helpers/logger.h"
#include "helpers/memory.h"

#include "ci_sigma_builder.h"

namespace forte2 {

namespace {
void transpose_23(std::span<double> in, std::span<double> out, size_t dim1, size_t dim2,
                  size_t dim3);
void gather_alpha_block(const CIStrings& lists, size_t class_Ka, size_t class_Kb, size_t Ka_start,
                        size_t Ka_size, size_t Kdim, size_t maxKb, std::span<const double> basis,
                        std::span<double> Kblock);
void gather_beta_block(const CIStrings& lists, size_t class_Ka, size_t class_Kb, size_t Ka_start,
                       size_t Ka_size, size_t Kdim, size_t maxKb, std::span<double> basis,
                       std::span<double> TR, std::span<double> Kblock2);
void scatter_beta_block(const CIStrings& lists, size_t class_Ka, size_t class_Kb, size_t Ka_start,
                        size_t Ka_size, size_t Kdim, size_t maxKb, std::span<const double> Kblock1,
                        std::span<double> TR, std::span<double> sigma);
void scatter_alpha_block(const CIStrings& lists, size_t class_Ka, size_t class_Kb, size_t Ka_start,
                         size_t Ka_size, size_t Kdim, size_t maxKb, std::span<const double> Kblock2,
                         std::span<double> sigma);
} // namespace

void CISigmaBuilder::H1_kh(std::span<double> basis, std::span<double> sigma, Spin spin) const {
    size_t norb = lists_.norb();
    // loop over blocks of matrix C
    for (const auto& [nI, class_Ia, class_Ib] : lists_.determinant_classes()) {
        if (lists_.block_size(nI) == 0)
            continue;
        auto Cr = gather_block(basis, TR, spin, lists_, class_Ia, class_Ib);

        for (const auto& [nJ, class_Ja, class_Jb] : lists_.determinant_classes()) {
            // For alpha operator, the beta string classes of the result must be the same
            if (is_alpha(spin) and (class_Ib != class_Jb))
                continue;
            // For beta operator, the alpha string classes of the result must be the same
            if (is_beta(spin) and (class_Ia != class_Ja))
                continue;
            if (lists_.block_size(nJ) == 0)
                continue;

            std::fill_n(TL.begin(), lists_.block_size(nJ), 0.0);

            const size_t maxL = is_alpha(spin) ? lists_.beta_address()->strpcls(class_Ib)
                                               : lists_.alfa_address()->strpcls(class_Ia);

            const auto& pq_vo_list = is_alpha(spin) ? lists_.get_alfa_vo_list(class_Ia, class_Ja)
                                                    : lists_.get_beta_vo_list(class_Ib, class_Jb);

            for (const auto& [pq, vo_list] : pq_vo_list) {
                const auto& [p, q] = pq;
                const double Hpq = h_kh[p * norb + q];
                for (const auto& [sign, I, J] : vo_list) {
                    add(maxL, sign * Hpq, &Cr[I * maxL], 1, &TL[J * maxL], 1);
                }
            }
            scatter_block(TL, sigma, spin, lists_, class_Ja, class_Jb);
        }
    }
}

void CISigmaBuilder::H2_kh(std::span<double> basis, std::span<double> sigma) const {
    size_t norb = lists_.norb();
    const auto npairs = norb * (norb + 1) / 2; // Number of pairs (p, r) with p >= r

    const int num_class_Ka = lists_.alfa_address_1h1p()->nclasses();
    const int num_class_Kb = lists_.beta_address_1h1p()->nclasses();

    // Loop over the Ka and Kb string classes of the D([qs],[Ka Kb]) matrix
    for (size_t class_Ka = 0; class_Ka < num_class_Ka; ++class_Ka) {
        for (size_t class_Kb = 0; class_Kb < num_class_Kb; ++class_Kb) {
            const auto maxKa = lists_.alfa_address_1h1p()->strpcls(class_Ka);
            const auto maxKb = lists_.beta_address_1h1p()->strpcls(class_Kb);

            // Set the size of the Kb range. Here we process all Kb values.
            const size_t Kb_start = 0;
            const size_t Kb_end = maxKb;
            const size_t Kb_size = Kb_end - Kb_start;
            const size_t block_dim = npairs * Kb_size;

            // Skip this chunk if it is empty
            if (block_dim * maxKa == 0)
                continue;

            // Grab the temporary buffers that will hold intermediates like
            // D([i>=j],[Ka Kb]) where the indices range as:
            // - [i>=j] all values
            // - Ka in chunks of maximum size Ka_max_size
            // - Kb in [0, maxKb)
            // Ka_max_size is the maximum size of the Ka range that we can
            // process in one go without exceeding the memory limit, set via
            // the set_memory() function.
            auto [Kblock1, Kblock2, Ka_max_size] = get_Kblock_spans(block_dim, maxKa);

            // Loop over ranges of Ka indices in chuncks of size Ka_max_size
            for (size_t Ka_start = 0; Ka_start < maxKa; Ka_start += Ka_max_size) {
                size_t Ka_end = std::min(Ka_start + Ka_max_size, maxKa);
                // size of the Ka range, which may be smaller than Ka_max_size
                size_t Ka_size = Ka_end - Ka_start;
                // dimensions of the matrix we are going to use for this Ka
                // chunk
                const size_t Kdim = Ka_size * Kb_size;
                // dimension of the D([i<=j],[Ka Kb]) tensor
                const size_t temp_dim = npairs * Kdim;

                // skip empty blocks
                if (temp_dim == 0)
                    continue;

                // zero out the block that will hold the D([i>=j],[Ka Kb]) tensor
                std::fill_n(Kblock1.begin(), temp_dim, 0.0);

                // alpha contribution to the D matrix D([i>=j],[Ka Kb])
                gather_alpha_block(lists_, class_Ka, class_Kb, Ka_start, Ka_size, Kdim, maxKb,
                                   basis, std::span{Kblock1.data(), temp_dim});

                // cyclic transpose of the D matrix D([i>=j],[Ka Kb]) to D([i>=j],[Kb Ka])
                transpose_23(Kblock1, Kblock2, npairs, Ka_size, maxKb);

                // beta contribution to the D matrix D([i>=j],[Kb Ka])
                gather_beta_block(lists_, class_Ka, class_Kb, Ka_start, Ka_size, Kdim, maxKb, basis,
                                  TR, std::span{Kblock2.data(), temp_dim});

                // matrix-matrix multiplication 0.5 * V([k>=l][i>=j]) * D([i>=j],[Kb Ka])
                // The result is the matrix E([k>=l],[Kb Ka])
                // [note that the Ka/Kb indices are still transposed]
                matrix_product('N', 'N', npairs, Kdim, npairs, 0.5, v_ijkl_hk.data(), npairs,
                               Kblock2.data(), Kdim, 0.0, Kblock1.data(), Kdim);

                // beta contribution from the matrix E([k>=l],[Kb Ka]) to
                // sigma([Ib Ia])
                scatter_beta_block(lists_, class_Ka, class_Kb, Ka_start, Ka_size, Kdim, maxKb,
                                   std::span{Kblock1.data(), temp_dim}, TR, sigma);

                // cyclic transpose of E([k>=l],[Kb Ka]) to E([k>=l],[Ka Kb])
                transpose_23(Kblock1, Kblock2, npairs, maxKb, Ka_size);

                // alpha contribution from the matrix E([k>=l],[Ka Kb]) to
                // sigma([Ia Ib])
                scatter_alpha_block(lists_, class_Ka, class_Kb, Ka_start, Ka_size, Kdim, maxKb,
                                    std::span{Kblock2.data(), temp_dim}, sigma);
            }
        }
    }
}

std::tuple<std::span<double>, std::span<double>, size_t>
CISigmaBuilder::get_Kblock_spans(size_t nrows, size_t ncols) const {
    // Find the maximum size of the temporary block to allocate. This is either set by the full
    // size of the block (nrows * ncols) or by the available memory size, whichever is smaller
    std::size_t block_size = std::min(nrows * ncols, memory_size_ / (2 * sizeof(double)));

    // Find the corresponding chunk size for K
    size_t cols_chunk_size = std::min(block_size / nrows, ncols);

    // If the chunk size is too small to store one row, resize it
    bool need_resize = false;
    if (cols_chunk_size < 1) {
        // resize to a reasonable minimum and update block_size
        cols_chunk_size = std::min(static_cast<size_t>(64), ncols);
        block_size = nrows * cols_chunk_size;
        need_resize = true;
    }

    // If the temporary buffers are too small, resize them
    if (Kblock1_.size() < block_size) {
        Kblock1_.resize(block_size);
        Kblock2_.resize(block_size);
        if (need_resize) {
            auto block_size_MB = to_mb<double>(2 * block_size);
            LOG(log_level_) << "Available memory is too small to hold a row of the CI buffers.\n"
                               "Block size has been adjusted to 2 x "
                            << block_size << " (" << block_size_MB << " MB) to hold "
                            << cols_chunk_size
                            << " columns.\n"
                               "For best performance, increase the memory limit.";
        }
    }

    return {std::span<double>{Kblock1_.data(), block_size},
            std::span<double>{Kblock2_.data(), block_size}, cols_chunk_size};
}

// The following functions are used to gather and scatter blocks of data and
// are relevant only to this file. They live in the anonymous namespace to
// avoid polluting the global namespace.
namespace {
/// @brief Cyclic multithreaded transpose of indices 2 and 3 of a 3D tensor
/// in[i][j][k] -> out[i][k][j]
/// @param in Input tensor as a 1D span
/// @param out Output tensor as a 1D span
/// @param dim1 Size of the input first dimension [i]
/// @param dim2 Size of the input second dimension [j]
/// @param dim3 Size of the input third dimension [k]
void transpose_23(std::span<double> in, std::span<double> out, size_t dim1, size_t dim2,
                  size_t dim3) {
    // Kernel for transposing indices 2 and 3 of a 3D tensor for all i in the
    // range [i_begin, i_end)
    auto kernel = [=, &in, &out](size_t i_begin, size_t i_end) {
        const size_t block_size = dim2 * dim3;
        for (size_t i = i_begin; i < i_end; ++i) {
            const double* src = in.data() + i * block_size;
            double* dst = out.data() + i * block_size;
            for (size_t j{0}; j < dim2; ++j) {
                for (size_t k{0}; k < dim3; ++k) {
                    dst[k * dim2 + j] = src[j * dim3 + k];
                }
            }
        }
    };

    assert(in.size() == out.size());
    unsigned int nthreads = std::thread::hardware_concurrency() / 2;
    if (nthreads == 0)
        nthreads = 1;
    size_t chunk = (dim1 + nthreads - 1) / nthreads;

    std::vector<std::future<void>> futures;
    futures.reserve(nthreads);

    for (unsigned int t = 0; t < nthreads; ++t) {
        size_t b_begin = t * chunk;
        size_t b_end = std::min(b_begin + chunk, dim1);
        if (b_begin >= b_end)
            break;
        futures.emplace_back(std::async(std::launch::async, kernel, b_begin, b_end));
    }

    for (auto& f : futures) {
        f.get();
    }
}

void gather_alpha_block(const CIStrings& lists, size_t class_Ka, size_t class_Kb, size_t Ka_start,
                        size_t Ka_size, size_t Kdim, size_t maxKb, std::span<const double> basis,
                        std::span<double> Kblock) {

    const auto Kb_occ = lists.gas_beta_1h1p_occupations()[class_Kb / lists.nirrep()];

    for (auto const& [nI, class_Ia, class_Ib] : lists.determinant_classes()) {
        const auto Ib_occ = lists.gas_beta_occupations()[class_Ib / lists.nirrep()];

        if ((Ib_occ != Kb_occ) or (lists.block_size(nI) == 0))
            continue;

        // get all Ka/Ia pairs connected to the current class_Ka and class_Ia
        const auto alfa_vo_list = lists.get_alfa_vo_list2(class_Ka, class_Ia);
        if (alfa_vo_list.empty())
            continue;

        size_t maxIb = lists.beta_address()->strpcls(class_Ib);
        size_t basis_offset = lists.block_offset(nI);

        // add contributions to the Kblock
        for (size_t Ka = 0; Ka < Ka_size; ++Ka) {
            const auto& vo_alist = alfa_vo_list[Ka + Ka_start];
            for (auto const& [sign_ij, ij, Ia] : vo_alist) {
                size_t K_offset = ij * Kdim + Ka * maxKb;
                size_t basis_Ia_offset = basis_offset + Ia * maxIb;
                add(maxKb, sign_ij, &basis[basis_Ia_offset], 1, &Kblock[K_offset], 1);
            }
        }
    }
}

void gather_beta_block(const CIStrings& lists, size_t class_Ka, size_t class_Kb, size_t Ka_start,
                       size_t Ka_size, size_t Kdim, size_t maxKb, std::span<double> basis,
                       std::span<double> TR, std::span<double> Kblock2) {

    const auto Ka_occ = lists.gas_alfa_1h1p_occupations()[class_Ka / lists.nirrep()];
    for (auto const& [nI, class_Ia, class_Ib] : lists.determinant_classes()) {
        const auto Ia_occ = lists.gas_alfa_occupations()[class_Ia / lists.nirrep()];
        if ((Ia_occ != Ka_occ) or (lists.block_size(nI) == 0))
            continue;

        // get all Kb/Ib pairs connected to the current class_Kb and class_Ib
        const auto beta_vo_list = lists.get_beta_vo_list2(class_Kb, class_Ib);
        if (beta_vo_list.empty())
            continue;

        size_t maxIa = lists.alfa_address()->strpcls(class_Ia);

        // transposes the basis vector into TR
        auto basis_tr = gather_block(basis, TR, Spin::Beta, lists, class_Ia, class_Ib);

        // add contributions to the Kblock
        for (size_t Kb = 0; Kb < maxKb; ++Kb) {
            auto const& vo_blist = beta_vo_list[Kb];
            for (auto const& [sign_ij, ij, Ib] : vo_blist) {
                size_t K_offset = ij * Kdim + Kb * Ka_size;
                size_t basis_Ib_offset = Ib * maxIa + Ka_start;
                add(Ka_size, sign_ij, &basis_tr[basis_Ib_offset], 1, &Kblock2[K_offset], 1);
            }
        }
    }
}

void scatter_beta_block(const CIStrings& lists, size_t class_Ka, size_t class_Kb, size_t Ka_start,
                        size_t Ka_size, size_t Kdim, size_t maxKb, std::span<const double> Kblock1,
                        std::span<double> TR, std::span<double> sigma) {
    const auto Ka_occ = lists.gas_alfa_1h1p_occupations()[class_Ka / lists.nirrep()];
    for (const auto& [nI, class_Ia, class_Ib] : lists.determinant_classes()) {
        const auto Ia_occ = lists.gas_alfa_occupations()[class_Ia / lists.nirrep()];
        if ((Ia_occ != Ka_occ) or (lists.block_size(nI) == 0))
            continue;

        // get all Kb/Ib pairs connected to the current class_Kb and class_Ib
        const auto beta_vo_list = lists.get_beta_vo_list2(class_Kb, class_Ib);
        if (beta_vo_list.empty())
            continue;

        const auto maxIa = lists.alfa_address()->strpcls(class_Ia);

        // zero out the TR temporary vector for the scatter operation
        zero_block(TR, Spin::Beta, lists, class_Ia, class_Ib);

        // scatter contributions from the Kblock into the TR vector
        for (size_t Kb = 0; Kb < maxKb; ++Kb) {
            const auto& vo_blist = beta_vo_list[Kb];
            for (const auto& [sign_ij, ij, Ib] : vo_blist) {
                const size_t K_offset = ij * Kdim + Kb * Ka_size;
                const size_t sigma_offset = Ib * maxIa + Ka_start;
                add(Ka_size, sign_ij, &Kblock1[K_offset], 1, &TR[sigma_offset], 1);
            }
        }

        // scatter the TR([Kb Ka]) vector to the sigma vector
        scatter_block(TR, sigma, Spin::Beta, lists, class_Ia, class_Ib);
    }
}

void scatter_alpha_block(const CIStrings& lists, size_t class_Ka, size_t class_Kb, size_t Ka_start,
                         size_t Ka_size, size_t Kdim, size_t maxKb, std::span<const double> Kblock2,
                         std::span<double> sigma) {
    const auto Kb_occ = lists.gas_beta_1h1p_occupations()[class_Kb / lists.nirrep()];
    for (auto const& [nI, class_Ia, class_Ib] : lists.determinant_classes()) {
        const auto Ib_occ = lists.gas_beta_occupations()[class_Ib / lists.nirrep()];
        if ((Ib_occ != Kb_occ) or (lists.block_size(nI) == 0))
            continue;
        // get all Ia/Ka pairs connected to the current class_Ka and class_Ia
        const auto alfa_vo_list = lists.get_alfa_vo_list2(class_Ka, class_Ia);
        if (alfa_vo_list.empty())
            continue;

        size_t maxIb = lists.beta_address()->strpcls(class_Ib);
        size_t sigma_offset = lists.block_offset(nI);

        // scatter contributions from the Kblock into the sigma vector
        for (size_t Ka = 0; Ka < Ka_size; ++Ka) {
            const auto& vo_alist = alfa_vo_list[Ka + Ka_start];
            for (auto const& [sign_kl, kl, Ia] : vo_alist) {
                size_t K_offset = kl * Kdim + Ka * maxKb;
                size_t sigma_Ia_offset = sigma_offset + Ia * maxIb;
                add(maxKb, sign_kl, &Kblock2[K_offset], 1, &sigma[sigma_Ia_offset], 1);
            }
        }
    }
}
} // namespace
} // namespace forte2
