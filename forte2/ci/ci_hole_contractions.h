#pragma once

#include <algorithm>
#include <map>
#include <vector>

#include "ci/ci_orbital_blocks.h"
#include "ci/ci_string_substitutions.h"
#include "helpers/blas.h"
#include "helpers/indexing.hpp"
#include "helpers/parallel.h"

namespace forte2 {

/// Connections from one determinant block to a hole product in one spin sector.
template <size_t Rank> struct HoleConnection {
    const DeterminantBlock& determinant;
    HoleSubstitutions<Rank> substitutions;
};

template <size_t Rank>
auto hole_connections(const CIStrings& lists, Spin spin, const DeterminantBlock& determinant) {
    const auto& holes = lists.strings(spin, Rank);
    const auto& strings = lists.strings(spin);
    const auto string_class = determinant.string_class(spin);
    std::vector<std::pair<int, HoleSubstitutions<Rank>>> connections;

    // Remove a multiset of electrons from this occupation. A fixed order of GAS slots
    // generates each reachable hole occupation once, even when several holes are in one slot.
    auto occupation = strings.occupation(string_class);
    const auto append = [&](this const auto& self, size_t remaining, size_t first) -> void {
        if (remaining == 0) {
            const int first_class = holes.find_class(occupation, 0).value();
            for (int h = 0; h < lists.nirrep(); ++h) {
                const int k = first_class + h;
                if (holes.size(k) == 0)
                    continue;
                HoleSubstitutions<Rank> substitutions(holes, k, strings, string_class);
                if (!substitutions.empty())
                    connections.emplace_back(k, std::move(substitutions));
            }
            return;
        }
        for (size_t g = first; g < occupation.size(); ++g) {
            if (occupation[g] == 0)
                continue;
            --occupation[g];
            self(remaining - 1, g);
            ++occupation[g];
        }
    };
    append(Rank, 0);
    return connections;
}

/// Group links by (hole class, spectator class) so all contributions to one intermediate
/// share the same gather, orbital contraction, and scatter.
template <size_t Rank> auto same_spin_hole_connections(const CIStrings& lists, Spin spin) {
    std::map<std::pair<int, int>, std::vector<HoleConnection<Rank>>> groups;
    for (const auto& determinant : lists.blocks()) {
        for (auto& [hole_class, substitutions] : hole_connections<Rank>(lists, spin, determinant)) {
            groups[{hole_class, determinant.spectator_class(spin)}].push_back(
                {determinant, std::move(substitutions)});
        }
    }
    return groups;
}

/// Generate only alpha/beta hole products that share a variational determinant block.
template <size_t AlphaRank, size_t BetaRank> auto mixed_hole_connections(const CIStrings& lists) {
    struct Connection {
        const DeterminantBlock& determinant;
        HoleSubstitutions<AlphaRank> alpha;
        HoleSubstitutions<BetaRank> beta;
    };
    std::map<std::pair<int, int>, std::vector<Connection>> groups;
    for (const auto& determinant : lists.blocks()) {
        const auto alpha = hole_connections<AlphaRank>(lists, Spin::Alpha, determinant);
        const auto beta = hole_connections<BetaRank>(lists, Spin::Beta, determinant);
        for (const auto& [a, a_substitutions] : alpha)
            for (const auto& [b, b_substitutions] : beta)
                groups[{a, b}].push_back({determinant, a_substitutions, b_substitutions});
    }
    return groups;
}

template <size_t Rank> size_t hole_orbital_index(const std::array<uint8_t, Rank>& orbitals) {
    if constexpr (Rank == 1)
        return orbitals[0];
    else if constexpr (Rank == 2)
        return pair_index_gt(orbitals[0], orbitals[1]);
    else {
        static_assert(Rank == 3);
        return triplet_index_gt(orbitals[0], orbitals[1], orbitals[2]);
    }
}

template <typename Scalar>
void add_spectator_slice(size_t size, int sign, const Scalar* source, Scalar* target) {
    if (size == 1)
        *target += static_cast<double>(sign) * *source;
    else
        add(size, Scalar(sign), source, 1, target, 1);
}

/// Visit a tile of the product of two hole classes. Expand each alpha or beta list at most
/// once in the tile, then combine the substitutions within that determinant block.
template <size_t AlphaRank, size_t BetaRank, typename Visit>
void for_each_mixed_hole_substitution(const HoleSubstitutions<AlphaRank>& alpha,
                                      const HoleSubstitutions<BetaRank>& beta, size_t beta_size,
                                      size_t start, size_t size, std::span<LocalStringAddress> work,
                                      Visit&& visit) {
    struct BetaEntry {
        typename HoleSubstitutions<BetaRank>::Orbitals orbitals;
        int sign;
        size_t rank;
    };
    std::vector<BetaEntry> entries;
    const size_t period = std::min(size, beta_size);
    std::vector<size_t> offsets;
    offsets.reserve(period + 1);
    offsets.push_back(0);
    size_t beta_rank = start % beta_size;
    for (size_t b = 0; b < period; ++b) {
        beta.for_each_hole_substitution(beta_rank, work, [&](auto orbitals, int sign, size_t rank) {
            entries.push_back({orbitals, sign, rank});
        });
        offsets.push_back(entries.size());
        if (++beta_rank == beta_size)
            beta_rank = 0;
    }

    for (size_t col = 0; col < size;) {
        const size_t a_rank = (start + col) / beta_size;
        const size_t row_size = std::min(beta_size - (start + col) % beta_size, size - col);
        alpha.for_each_hole_substitution(a_rank, work, [&](auto orbitals, int sign, size_t rank) {
            for (size_t c = col; c < col + row_size; ++c) {
                const size_t b = c % beta_size;
                for (size_t j = offsets[b]; j < offsets[b + 1]; ++j) {
                    const auto& entry = entries[j];
                    visit(c, orbitals, entry.orbitals, sign * entry.sign, rank, entry.rank);
                }
            }
        });
        col += row_size;
    }
}

/// Apply an HZ one- or two-hole contraction for either real orbitals or complex spinors.
template <size_t Rank, typename Scalar, typename Workspace>
void contract_same_spin_sigma(const CIStrings& lists, Spin spin, std::span<Scalar> basis,
                              std::span<Scalar> sigma, std::span<Scalar> right,
                              std::span<Scalar> left, const CIOrbitalIrrepBlocks& orbitals,
                              std::span<const Scalar> integrals, Workspace&& workspace) {
    const auto& holes = lists.strings(spin, Rank);
    std::vector<LocalStringAddress> work(lists.ngas_spaces());
    for (const auto& [classes, connections] : same_spin_hole_connections<Rank>(lists, spin)) {
        const auto [hole_class, spectator] = classes;
        const auto& block = orbitals.for_intermediate(lists, hole_class, spectator);
        std::vector<size_t> rows;
        for (const auto& connection : connections)
            for (const auto& tuple : connection.substitutions.orbitals())
                rows.push_back(hole_orbital_index(tuple));
        const CIOrbitalMatrixBlock matrix(orbitals, block, integrals, std::move(rows));
        const auto nrows = matrix.size();
        const auto spectators = connections.front().determinant.spectator_size(spin);
        const auto count = holes.size(hole_class);
        const auto [D, E, chunk] = workspace(checked_ci_product(nrows, spectators), count);

        for (size_t start = 0; start < count;) {
            const size_t width = std::min(chunk, count - start);
            const size_t columns = width * spectators;
            std::fill_n(D.begin(), nrows * columns, Scalar{});

            for (const auto& [determinant, substitutions] : connections) {
                const auto coefficients = gather_block(basis, right, spin, determinant);
                const auto gather = [&](size_t first, size_t end,
                                        std::span<LocalStringAddress> coords) {
                    for (size_t k = first; k < end; ++k) {
                        substitutions.for_each_hole_substitution(
                            start + k, coords, [&](const auto& tuple, int sign, size_t i) {
                                const auto row = matrix.row(hole_orbital_index(tuple));
                                add_spectator_slice(spectators, sign,
                                                    coefficients.data() + i * spectators,
                                                    D.data() + row * columns + k * spectators);
                            });
                    }
                };
                if (width >= 16384) {
                    parallel_for_chunked(width, [&](size_t first, size_t end) {
                        std::vector<LocalStringAddress> coords(lists.ngas_spaces());
                        gather(first, end, coords);
                    });
                } else {
                    gather(0, width, work);
                }
            }

            matrix_product('N', 'N', nrows, columns, nrows, Scalar(1), matrix.data(), nrows,
                           D.data(), columns, Scalar(0), E.data(), columns);

            for (const auto& [determinant, substitutions] : connections) {
                auto coefficients = sigma.subspan(determinant.offset, determinant.size());
                if (!is_alpha(spin)) {
                    coefficients = left.first(determinant.size());
                    std::ranges::fill(coefficients, Scalar{});
                }
                for (size_t k = 0; k < width; ++k) {
                    substitutions.for_each_hole_substitution(
                        start + k, work, [&](const auto& tuple, int sign, size_t i) {
                            const auto row = matrix.row(hole_orbital_index(tuple));
                            add_spectator_slice(spectators, sign,
                                                E.data() + row * columns + k * spectators,
                                                coefficients.data() + i * spectators);
                        });
                }
                if (!is_alpha(spin))
                    scatter_block(coefficients, sigma, spin, determinant);
            }
            start += width;
        }
    }
}

/// Accumulate the same-spin real density over holes and the unchanged opposite spin.
template <size_t Rank>
void accumulate_same_spin_rdm(const CIStrings& lists, Spin spin, std::span<double> left,
                              std::span<double> right, std::span<double> left_scratch,
                              std::span<double> right_scratch, double* rdm, size_t dimension) {
    struct Substitution {
        typename HoleSubstitutions<Rank>::Orbitals orbitals;
        int sign;
        size_t rank;
    };
    std::vector<LocalStringAddress> left_work(lists.ngas_spaces()), right_work(lists.ngas_spaces());
    std::vector<Substitution> right_substitutions;
    const auto& holes = lists.strings(spin, Rank);
    for (const auto& [classes, links] : same_spin_hole_connections<Rank>(lists, spin)) {
        const size_t count = holes.size(classes.first);
        for (const auto& bra : links) {
            const auto spectators = bra.determinant.spectator_size(spin);
            const auto cl = gather_block(left, left_scratch, spin, bra.determinant);
            for (const auto& ket : links) {
                const auto cr = gather_block(right, right_scratch, spin, ket.determinant);
                for (size_t k = 0; k < count; ++k) {
                    right_substitutions.clear();
                    ket.substitutions.for_each_hole_substitution(
                        k, right_work, [&](const auto& tuple, int sign, size_t j) {
                            right_substitutions.push_back({tuple, sign, j});
                        });
                    bra.substitutions.for_each_hole_substitution(
                        k, left_work, [&](const auto& tuple, int sign, size_t i) {
                            const auto row = hole_orbital_index(tuple) * dimension;
                            for (const auto& [indices, ket_sign, j] : right_substitutions) {
                                const auto col = hole_orbital_index(indices);
                                rdm[row + col] += sign * ket_sign *
                                                  dot(spectators, cl.data() + i * spectators, 1,
                                                      cr.data() + j * spectators, 1);
                            }
                        });
                }
            }
        }
    }
}

} // namespace forte2
