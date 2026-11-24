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
#include "ci_hole_contractions.h"

namespace forte2 {

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
    contract_same_spin_sigma<1, std::complex<double>>(
        lists_, Spin::Alpha, basis, sigma, {}, {}, orbital_rows_, h,
        [this](size_t rows, size_t columns) { return get_Kblock_spans(rows, columns); });
}

void RelCISigmaBuilder::H2_hz_same_spin(std::span<std::complex<double>> basis,
                                        std::span<std::complex<double>> sigma) const {
    contract_same_spin_sigma<2, std::complex<double>>(
        lists_, Spin::Alpha, basis, sigma, {}, {}, spinor_pairs_, v_pr_qs,
        [this](size_t rows, size_t columns) { return get_Kblock_spans(rows, columns); });
}

} // namespace forte2
