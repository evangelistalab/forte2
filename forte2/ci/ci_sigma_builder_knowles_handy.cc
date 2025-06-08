#include <vector>
#include <future>
#include <iostream>
#include <cassert>
#include <algorithm>
#include <thread>
#include <algorithm>

#include "helpers/timer.hpp"
#include "helpers/np_vector_functions.h"
#include "helpers/indexing.hpp"
#include "helpers/blas.h"

#include "ci_sigma_builder.h"

namespace forte2 {

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
    // Kernel for transposing indices 2 and 3 of a 3D tensor for all i in the range [i_begin, i_end)
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
    for (auto const& [nI, class_Ia, class_Ib] : lists.determinant_classes()) {
        if ((class_Ib != class_Kb) or (lists.detpblk(nI) == 0))
            continue;

        // get all Ka/Ia pairs connected to the current class_Ka and class_Ia
        auto alfa_vo_list = lists.get_alfa_vo_list2(class_Ka, class_Ia);
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
    for (auto const& [nI, class_Ia, class_Ib] : lists.determinant_classes()) {
        if ((class_Ia != class_Ka) or (lists.detpblk(nI) == 0))
            continue;

        // get all Kb/Ib pairs connected to the current class_Kb and class_Ib
        auto beta_vo_list = lists.get_beta_vo_list2(class_Kb, class_Ib);
        if (beta_vo_list.empty())
            continue;

        size_t maxIa = lists.alfa_address()->strpcls(class_Ia);

        // transposes the basis vector into TR
        auto basis_tr = gather_block(basis, TR, false, lists, class_Ia, class_Ib);

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
    for (const auto& [nI, class_Ia, class_Ib] : lists.determinant_classes()) {
        if ((class_Ia != class_Ka) or (lists.detpblk(nI) == 0))
            continue;

        // get all Kb/Ib pairs connected to the current class_Kb and class_Ib
        const auto beta_vo_list = lists.get_beta_vo_list2(class_Kb, class_Ib);
        if (beta_vo_list.empty())
            continue;

        const auto maxIa = lists.alfa_address()->strpcls(class_Ia);

        // zero out the TR temporary vector for the scatter operation
        zero_block(TR, false, lists, class_Ia, class_Ib);

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
        scatter_block(TR, sigma, false, lists, class_Ia, class_Ib);
    }
}

void scatter_alpha_block(const CIStrings& lists, size_t class_Ka, size_t class_Kb, size_t Ka_start,
                         size_t Ka_size, size_t Kdim, size_t maxKb, std::span<const double> Kblock2,
                         std::span<double> sigma) {
    for (auto const& [nI, class_Ia, class_Ib] : lists.determinant_classes()) {
        if ((class_Ib != class_Kb) or (lists.detpblk(nI) == 0))
            continue;
        // get all Ia/Ka pairs connected to the current class_Ka and class_Ia
        auto alfa_vo_list = lists.get_alfa_vo_list2(class_Ka, class_Ia);
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

std::tuple<std::span<double>, std::span<double>, size_t>
CISigmaBuilder::get_Kblock_spans(size_t dim, size_t maxKa) const {
    // Ensure that Kblock1_ and Kblock2_ are allocated with at least requested memory size
    std::size_t size = memory_size_ / (2 * sizeof(double));
    if (Kblock1_.size() < size) {
        std::cout << "Resizing Knowles-Handy temporary buffers to " << 2 * size << " elements ("
                  << 2 * size * sizeof(double) / (1024 * 1024) << " MB).\n\n";
        Kblock1_.resize(size);
        Kblock2_.resize(size);
    }

    // Derive Ka_max_size from our preallocated buffers:
    size_t available = Kblock1_.size(); // assume Kblock2_ same size
    size_t Ka_max_size = std::min(available / dim, maxKa);
    if (Ka_max_size < 1) {
        // too small for even one row of Ka => resize to hold a reasonable minimum
        Ka_max_size = std::min(static_cast<size_t>(64), maxKa); // 64 is a reasonable minimum size
        size_t new_dim = Ka_max_size * dim;
        Kblock1_.resize(new_dim);
        Kblock2_.resize(new_dim);
        auto available_MB = 2 * available / (1024 * 1024 * sizeof(double));
        auto new_dim_MB = 2 * new_dim / (1024 * 1024 * sizeof(double));
        std::cerr << "Warning: Knowles-Handy temporary buffers too small (" << 2 * available
                  << " elements; " << available_MB << " MB); resized to " << 2 * new_dim
                  << " elements; " << new_dim_MB << " MB.\n"
                  << "For best performance, set the memory size via the "
                     "CISigmaBuilder::set_memory() function.\n";
    }

    size_t max_temp_dim = dim * Ka_max_size;

    return {std::span<double>{Kblock1_.data(), max_temp_dim},
            std::span<double>{Kblock2_.data(), max_temp_dim}, Ka_max_size};
}

void CISigmaBuilder::H2(std::span<double> basis, std::span<double> sigma) const {
    size_t norb = lists_.norb();
    const auto npairs = norb * (norb + 1) / 2; // Number of pairs (p, r) with p >= r

    const int num_class_Ka = lists_.alfa_address()->nclasses();
    const int num_class_Kb = lists_.beta_address()->nclasses();

    // Loop over the Ka and Kb classes of the D([qs],[Ka Kb]) matrix
    for (size_t class_Ka = 0; class_Ka < num_class_Ka; ++class_Ka) {
        for (size_t class_Kb = 0; class_Kb < num_class_Kb; ++class_Kb) {
            const auto maxKa = lists_.alfa_address()->strpcls(class_Ka);
            const auto maxKb = lists_.beta_address()->strpcls(class_Kb);

            // We fix the size of the Kb range
            const size_t Kb_start = 0;
            const size_t Kb_end = maxKb;
            const size_t Kb_size = Kb_end - Kb_start;

            // // Derive Ka_max_size from our preallocated buffers:
            // size_t available = Kblock1_.size(); // assume Kblock2_ same size
            // size_t Ka_max_size = available / (npairs * Kb_size);
            // if (Ka_max_size < 1) {
            //     // too small for even one Ka => resize to hold 1
            //     Ka_max_size = 64; // 64 is a reasonable minimum size
            //     size_t new_dim = npairs * Ka_max_size * Kb_size;
            //     Kblock1_.resize(new_dim);
            //     Kblock2_.resize(new_dim);
            //     auto available_MB = 2 * available / (1024 * 1024 * sizeof(double));
            //     auto new_dim_MB = 2 * new_dim / (1024 * 1024 * sizeof(double));
            //     std::cerr << "Warning: temporary buffers too small (" << available_MB
            //               << " MB); resized to " << new_dim_MB << " MB\n";
            // }
            // // Ensure Ka_max_size is not larger than maxKa
            // Ka_max_size = std::min(Ka_max_size, maxKa);

            // size_t max_temp_dim = npairs * Ka_max_size * Kb_size;
            // std::span<double> Kblock1{Kblock1_.data(), max_temp_dim};
            // std::span<double> Kblock2{Kblock2_.data(), max_temp_dim};

            auto [Kblock1, Kblock2, Ka_max_size] = get_Kblock_spans(npairs * Kb_size, maxKa);

            // Loop over values of Ka in blocks
            for (size_t Ka_start = 0; Ka_start < maxKa; Ka_start += Ka_max_size) {
                size_t Ka_end = std::min(Ka_start + Ka_max_size, maxKa);
                size_t Ka_size = Ka_end - Ka_start;

                // dimensions of the matrix we are going to build
                const auto Kdim = Ka_size * Kb_size;
                // dimension of the D([i<=j],[Ka Kb]) tensor
                const auto temp_dim = npairs * Kdim;

                // skip empty blocks
                if (temp_dim == 0)
                    continue;

                // Zero out the blocks
                std::fill_n(Kblock1.begin(), temp_dim, 0.0);

                // Alpha contribution to the D matrix D([i>=j],[Ka Kb])
                gather_alpha_block(lists_, class_Ka, class_Kb, Ka_start, Ka_size, Kdim, maxKb,
                                   basis, std::span{Kblock1.data(), temp_dim});

                // Cyclic transpose of the D matrix D([i>=j],[Ka Kb]) to D([i>=j],[Kb Ka])
                transpose_23(Kblock1, Kblock2, npairs, Ka_size, maxKb);

                // Beta contribution to the D matrix D([i>=j],[Kb Ka])
                gather_beta_block(lists_, class_Ka, class_Kb, Ka_start, Ka_size, Kdim, maxKb, basis,
                                  TR, std::span{Kblock2.data(), temp_dim});

                // Perform the matrix product 0.5 * V([k>=l][i>=j]) * D([i>=j],[Kb Ka])
                // The result is the matrix E([k>=l],[Kb Ka])
                // [note that the Ka/Kb indices are transposed]
                matrix_product('N', 'N', npairs, Kdim, npairs, 0.5, v_ijkl_hk.data(), npairs,
                               Kblock2.data(), Kdim, 0.0, Kblock1.data(), Kdim);

                // Beta contribution from the matrix E([k>=l],[Kb Ka]) to sigma([Ib Ia])
                scatter_beta_block(lists_, class_Ka, class_Kb, Ka_start, Ka_size, Kdim, maxKb,
                                   std::span{Kblock1.data(), temp_dim}, TR, sigma);

                // Cyclic transpose of E([k>=l],[Kb Ka]) to E([k>=l],[Ka Kb])
                transpose_23(Kblock1, Kblock2, npairs, maxKb, Ka_size);

                // Alpha contribution from the matrix E([k>=l],[Ka Kb]) to sigma([Ia Ib])
                scatter_alpha_block(lists_, class_Ka, class_Kb, Ka_start, Ka_size, Kdim, maxKb,
                                    std::span{Kblock2.data(), temp_dim}, sigma);
            }
        }
    }
}

} // namespace forte2
