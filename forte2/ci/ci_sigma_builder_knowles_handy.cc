#include <algorithm>
#include <limits>
#include <stdexcept>

#include "helpers/timer.hpp"
#include "helpers/blas.h"
#include "helpers/logger.h"
#include "helpers/parallel.h"

#include "ci_sigma_builder.h"
#include "ci_string_substitutions.h"
#include "helpers/indexing.hpp"

namespace forte2 {

namespace {
size_t Kblock_column_capacity(size_t nrows, size_t ncols, size_t memory_size) {
    if ((nrows == 0) or (ncols == 0)) {
        throw std::invalid_argument("K-block dimensions must be nonzero.");
    }

    // The memory setting covers both equally sized K-block buffers.
    const size_t max_elements_per_buffer = memory_size / (2 * sizeof(double));
    if (max_elements_per_buffer < nrows) {
        throw std::runtime_error("The CI builder memory limit (" + std::to_string(memory_size) +
                                 " bytes) is too small for one K-block column of " +
                                 std::to_string(nrows) + " elements per buffer.");
    }

    // BLAS dimensions are 32-bit integers, so a single contraction must not exceed INT_MAX
    // columns even when the configured memory limit is larger.
    return std::min({max_elements_per_buffer / nrows, ncols,
                     static_cast<size_t>(std::numeric_limits<int>::max())});
}

struct ExcitationConnection {
    const DeterminantBlock& determinant;
    Spin spin;
    ExcitationSubstitutions substitutions;
};

auto excitation_connections(const CIStrings& lists) {
    std::map<std::pair<int, int>, std::vector<ExcitationConnection>> groups;
    for (const auto& block : lists.blocks()) {
        for (const auto spin : {Spin::Alpha, Spin::Beta}) {
            const auto other = is_alpha(spin) ? Spin::Beta : Spin::Alpha;
            const auto& spectator = lists.strings(other);
            const auto& acting = lists.strings(spin);
            const auto& excited = lists.excited_strings(spin);
            const int L = block.spectator_class(spin);
            const int I = block.string_class(spin);
            const int K_spectator = lists.excited_strings(other)
                                        .find_class(spectator.occupation(L), spectator.irrep(L))
                                        .value();
            const std::span occupation(&acting.occupation(I), 1);
            for (const auto& candidate : generate_1h1p_occupations(occupation, lists.gas_size())) {
                const int first_class = excited.find_class(candidate, 0).value();
                for (int h = 0; h < lists.nirrep(); ++h) {
                    const int K = first_class + h;
                    if (excited.size(K) == 0)
                        continue;
                    ExcitationSubstitutions substitutions(excited, K, acting, I);
                    if (substitutions.empty())
                        continue;
                    const auto classes =
                        is_alpha(spin) ? std::pair{K, K_spectator} : std::pair{K_spectator, K};
                    groups[classes].push_back({block, spin, std::move(substitutions)});
                }
            }
        }
    }
    return groups;
}

/// Visit the spectator slices connected to a tile of D([pq], Ka * beta_size + Kb).
/// For an alpha excitation the spectator slice is contiguous. For a beta excitation
/// the CI block and the tile use their respective beta dimensions as strides.
template <typename Visit>
void for_each_excitation_slice(const ExcitationConnection& connection, size_t beta_size,
                               size_t start, size_t width, std::span<LocalStringAddress> work,
                               Visit&& visit) {
    const auto& [block, spin, substitutions] = connection;
    if (is_alpha(spin)) {
        const auto end = start + width;
        for (size_t first = start; first < end;) {
            const size_t local = first % beta_size;
            const size_t length = std::min(beta_size - local, end - first);
            substitutions.for_each_excitation(
                first / beta_size, work, [&](auto p, auto q, int sign, size_t rank) {
                    visit(pair_index_geq(p, q), sign * (p == q ? 2 : 1), first - start,
                          block.offset + rank * block.beta_size + local, length, 1, 1);
                });
            first += length;
        }
    } else {
        const auto count = std::min(width, beta_size);
        for (size_t col = 0; col < count; ++col) {
            const size_t first = start + col;
            const size_t length = 1 + (width - col - 1) / beta_size;
            substitutions.for_each_excitation(
                first % beta_size, work, [&](auto p, auto q, int sign, size_t rank) {
                    visit(pair_index_geq(p, q), sign * (p == q ? 2 : 1), col,
                          block.offset + (first / beta_size) * block.beta_size + rank, length,
                          length == 1 ? 1 : beta_size, length == 1 ? 1 : block.beta_size);
                });
        }
    }
}

} // namespace

void CISigmaBuilder::H1_kh(std::span<double> basis, std::span<double> sigma, Spin spin) const {
    const auto norb = lists_.norb();
    const auto& strings = lists_.strings(spin);
    std::vector<LocalStringAddress> work(lists_.ngas_spaces());
    for (const auto& input : lists_.blocks()) {
        const auto spectators = input.spectator_size(spin);
        const auto right = gather_block(basis, std::span(TR), spin, input);
        for (const auto& output : lists_.blocks()) {
            if (input.spectator_class(spin) != output.spectator_class(spin))
                continue;
            const int first = input.string_class(spin);
            const ExcitationSubstitutions substitutions(strings, first, strings,
                                                        output.string_class(spin));
            if (substitutions.empty())
                continue;

            auto left = sigma.subspan(output.offset, output.size());
            if (!is_alpha(spin)) {
                left = std::span(TL).first(output.size());
                std::ranges::fill(left, 0.0);
            }
            for (size_t i = 0; i < strings.size(first); ++i) {
                substitutions.for_each_excitation(i, work, [&](auto p, auto q, int sign, size_t j) {
                    add(spectators, sign * h_kh[p * norb + q], right.data() + i * spectators, 1,
                        left.data() + j * spectators, 1);
                });
            }
            if (!is_alpha(spin))
                scatter_block(left, sigma, spin, output);
        }
    }
}

void CISigmaBuilder::H2_kh(std::span<double> basis, std::span<double> sigma) const {
    const auto& alpha = lists_.excited_strings(Spin::Alpha);
    const auto& beta = lists_.excited_strings(Spin::Beta);
    std::vector<LocalStringAddress> work(lists_.ngas_spaces());
    for (const auto& [classes, links] : excitation_connections(lists_)) {
        const auto [ka, kb] = classes;
        const auto& block = excitation_pairs_->for_intermediate(lists_, ka, kb);
        std::vector<size_t> rows;
        for (const auto& link : links)
            for (const auto& [p, q, sign] : link.substitutions.orbitals())
                rows.push_back(pair_index_geq(p, q));
        const CIOrbitalMatrixBlock<double> matrix(*excitation_pairs_, block, v_ijkl_hk,
                                                  std::move(rows));
        const auto nrows = matrix.size();
        const auto beta_size = beta.size(kb);
        const auto columns = checked_ci_product(alpha.size(ka), beta_size);
        const auto [D, E, chunk] = get_Kblock_spans(nrows, columns);
        for (size_t start = 0; start < columns;) {
            const size_t width = std::min(chunk, columns - start);
            std::fill_n(D.begin(), nrows * width, 0.0);
            for (const auto& link : links) {
                for_each_excitation_slice(
                    link, beta_size, start, width, work,
                    [&](size_t pair, int sign, size_t column, size_t coefficient, size_t length,
                        size_t column_stride, size_t coefficient_stride) {
                        const auto row = matrix.row(pair);
                        add(length, sign, basis.data() + coefficient, coefficient_stride,
                            D.data() + row * width + column, column_stride);
                    });
            }

            matrix_product('N', 'N', nrows, width, nrows, 0.5, matrix.data(), nrows, D.data(),
                           width, 0.0, E.data(), width);

            for (const auto& link : links) {
                for_each_excitation_slice(
                    link, beta_size, start, width, work,
                    [&](size_t pair, int sign, size_t column, size_t coefficient, size_t length,
                        size_t column_stride, size_t coefficient_stride) {
                        const auto row = matrix.row(pair);
                        add(length, sign, E.data() + row * width + column, column_stride,
                            sigma.data() + coefficient, coefficient_stride);
                    });
            }
            start += width;
        }
    }
}

std::tuple<std::span<double>, std::span<double>, size_t>
CISigmaBuilder::get_Kblock_spans(size_t nrows, size_t ncols) const {
    const size_t cols_chunk_size = Kblock_column_capacity(nrows, ncols, memory_size_);
    const size_t block_size = nrows * cols_chunk_size;
    const size_t max_elements_per_buffer = memory_size_ / (2 * sizeof(double));

    // Keep a larger live size: shrinking here makes a later larger kernel value-initialize the
    // same scratch storage again. The returned spans still expose only block_size elements.
    const bool can_reuse = Kblock1_.size() >= block_size && Kblock2_.size() >= block_size &&
                           Kblock1_.capacity() <= max_elements_per_buffer &&
                           Kblock2_.capacity() <= max_elements_per_buffer;
    if (!can_reuse) {
        // Drop both allocations before growing either one.
        Kblock1_ = std::vector<double>(block_size);
        Kblock2_ = std::vector<double>(block_size);
    }

    return {std::span<double>{Kblock1_.data(), block_size},
            std::span<double>{Kblock2_.data(), block_size}, cols_chunk_size};
}

size_t CISigmaBuilder::acquire_local_Kblock_buffers(std::vector<double>& Kblock1,
                                                    std::vector<double>& Kblock2, size_t nrows,
                                                    size_t ncols) const {
    // Transfer ownership so sigma-build cache and RDM-local scratch never coexist. The local
    // vectors release the storage when the RDM call returns, avoiding permanent cache inflation.
    Kblock1.swap(Kblock1_);
    Kblock2.swap(Kblock2_);

    const size_t cols_chunk_size = Kblock_column_capacity(nrows, ncols, memory_size_);
    const size_t block_size = nrows * cols_chunk_size;
    const size_t max_elements_per_buffer = memory_size_ / (2 * sizeof(double));

    const bool can_reuse = Kblock1.capacity() >= block_size && Kblock2.capacity() >= block_size &&
                           Kblock1.capacity() <= max_elements_per_buffer &&
                           Kblock2.capacity() <= max_elements_per_buffer;
    if (can_reuse) {
        Kblock1.resize(block_size);
        Kblock2.resize(block_size);
    } else {
        // Release both old allocations before growing either one, keeping peak scratch within the
        // configured pair budget even when the limit was lowered after a sigma build.
        std::vector<double>{}.swap(Kblock1);
        std::vector<double>{}.swap(Kblock2);
        Kblock1 = std::vector<double>(block_size);
        Kblock2 = std::vector<double>(block_size);
    }
    return cols_chunk_size;
}

} // namespace forte2
