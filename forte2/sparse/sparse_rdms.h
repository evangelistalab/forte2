#pragma once

#include <array>
#include <cstdint>
#include <string_view>
#include <type_traits>
#include <utility>

#include "sparse/sparse_state.h"

namespace forte2 {

enum class Accum { Real, Hermitian };

/// @brief Convert a spin pattern such as "aab" into a bitmask with bit k set when the index at
///        position k carries beta spin.
consteval std::uint32_t spin_mask(std::string_view pattern) {
    std::uint32_t mask = 0;
    for (std::size_t k{0}; k < pattern.size(); ++k)
        if (pattern[k] == 'b')
            mask |= (1u << k);
    return mask;
}

namespace detail {

template <bool IsBeta> inline double destroy_spin(Determinant& J, std::size_t p) {
    if constexpr (IsBeta)
        return J.destroy_beta(p);
    else
        return J.destroy_alpha(p);
}

template <bool IsBeta> inline double create_spin(Determinant& J, std::size_t p) {
    if constexpr (IsBeta)
        return J.create_beta(p);
    else
        return J.create_alpha(p);
}

/// @brief Apply a^+_{p_0} ... a^+_{p_{N-1}} a_{q_{N-1}} ... a_{q_0} to J in place, where the
///        creation indices p are idx[0..N) and the annihilation indices q are idx[N..2N).
/// @return The accumulated sign, or 0 if any operator annihilates the determinant.
template <int N, std::uint32_t SpinMask, std::size_t... K>
inline double apply_excitation(Determinant& J, const std::array<std::size_t, 2 * N>& idx,
                               std::index_sequence<K...>) {
    constexpr std::size_t rank = N;
    double sign = 1.0;
    // at compile time, this unrolls into the equivalent of
    // for i in range(N): sign *= destroy_spin<isbeta(mask(i))>(J, idx[rank + i]);
    ((sign *= destroy_spin<((SpinMask >> K) & 1u) != 0>(J, idx[rank + K])), ...);
    ((sign *= create_spin<((SpinMask >> (rank - 1 - K)) & 1u) != 0>(J, idx[rank - 1 - K])), ...);
    return sign;
}

} // namespace detail

/// @brief Compute a reduced density matrix of rank N between two SparseStates.
/// @tparam N The RDM rank, that is, the number of creation/annihilation operator pairs
/// @tparam SpinMask The spin pattern as a bitmask; see spin_mask
/// @tparam A Whether to accumulate a real or a Hermitian RDM
/// @return gamma[p_0]...[p_{N-1}][q_0]...[q_{N-1}] =
///         <L| a^+_{p_0} ... a^+_{p_{N-1}} a_{q_{N-1}} ... a_{q_0} |R> as a tensor with 2N
///         axes of length norb. For N = 2 this reads gamma2[p][q][r][s] =
///         <L| a^+_p a^+_q a_s a_r |R>, antisymmetric in p <-> q and r <-> s.
/// @note This is a reference implementation: it visits all norb^(2N) index tuples and every
///       determinant of the ket, so it is only usable for small cases.
template <int N, std::uint32_t SpinMask, Accum A>
auto compute_nrdm(const SparseState& state_left, const SparseState& state_right, std::size_t norb) {
    static_assert(N >= 1, "the RDM rank must be at least 1");
    static_assert(SpinMask < (1u << N), "the spin pattern has bits beyond rank N");

    using scalar_t = std::conditional_t<A == Accum::Hermitian, sparse_scalar_t, double>;

    std::array<std::size_t, 2 * N> shape;
    shape.fill(norb);
    auto g = make_zeros<nb::numpy, scalar_t, 2 * N>(shape);
    auto* g_data = g.data();
    const auto size = math::product(shape);

    std::array<std::size_t, 2 * N> idx{};
    Determinant J;

    for (std::size_t flat{0}; flat < size; ++flat) {
        scalar_t rdm = 0.0;
        for (const auto& [I, c_I] : state_right) {
            J = I;
            const double sign =
                detail::apply_excitation<N, SpinMask>(J, idx, std::make_index_sequence<N>{});
            if (sign != 0) {
                auto it = state_left.find(J);
                if (it != state_left.end()) {
                    if constexpr (A == Accum::Hermitian) {
                        rdm += sign * std::conj(it->second) * c_I;
                    } else {
                        rdm += sign * to_double(it->second * c_I);
                    }
                }
            }
        }
        g_data[flat] = rdm;
        // bump index up by one with carryover
        for (std::size_t d = 2 * N; d-- > 0;) {
            if (++idx[d] < norb)
                break;
            idx[d] = 0;
        }
    }
    return g;
}

// == One-component RDMs ==

inline auto compute_a_1rdm(const SparseState& state_left, const SparseState& state_right,
                           std::size_t norb) {
    return compute_nrdm<1, spin_mask("a"), Accum::Real>(state_left, state_right, norb);
}

inline auto compute_b_1rdm(const SparseState& state_left, const SparseState& state_right,
                           std::size_t norb) {
    return compute_nrdm<1, spin_mask("b"), Accum::Real>(state_left, state_right, norb);
}

inline auto compute_aa_2rdm(const SparseState& state_left, const SparseState& state_right,
                            std::size_t norb) {
    return compute_nrdm<2, spin_mask("aa"), Accum::Real>(state_left, state_right, norb);
}

inline auto compute_ab_2rdm(const SparseState& state_left, const SparseState& state_right,
                            std::size_t norb) {
    return compute_nrdm<2, spin_mask("ab"), Accum::Real>(state_left, state_right, norb);
}

inline auto compute_bb_2rdm(const SparseState& state_left, const SparseState& state_right,
                            std::size_t norb) {
    return compute_nrdm<2, spin_mask("bb"), Accum::Real>(state_left, state_right, norb);
}

inline auto compute_aaa_3rdm(const SparseState& state_left, const SparseState& state_right,
                             std::size_t norb) {
    return compute_nrdm<3, spin_mask("aaa"), Accum::Real>(state_left, state_right, norb);
}

inline auto compute_aab_3rdm(const SparseState& state_left, const SparseState& state_right,
                             std::size_t norb) {
    return compute_nrdm<3, spin_mask("aab"), Accum::Real>(state_left, state_right, norb);
}

inline auto compute_abb_3rdm(const SparseState& state_left, const SparseState& state_right,
                             std::size_t norb) {
    return compute_nrdm<3, spin_mask("abb"), Accum::Real>(state_left, state_right, norb);
}

inline auto compute_bbb_3rdm(const SparseState& state_left, const SparseState& state_right,
                             std::size_t norb) {
    return compute_nrdm<3, spin_mask("bbb"), Accum::Real>(state_left, state_right, norb);
}

inline auto compute_aaaa_4rdm(const SparseState& state_left, const SparseState& state_right,
                              std::size_t norb) {
    return compute_nrdm<4, spin_mask("aaaa"), Accum::Real>(state_left, state_right, norb);
}

inline auto compute_aaab_4rdm(const SparseState& state_left, const SparseState& state_right,
                              std::size_t norb) {
    return compute_nrdm<4, spin_mask("aaab"), Accum::Real>(state_left, state_right, norb);
}

inline auto compute_aabb_4rdm(const SparseState& state_left, const SparseState& state_right,
                              std::size_t norb) {
    return compute_nrdm<4, spin_mask("aabb"), Accum::Real>(state_left, state_right, norb);
}

inline auto compute_abbb_4rdm(const SparseState& state_left, const SparseState& state_right,
                              std::size_t norb) {
    return compute_nrdm<4, spin_mask("abbb"), Accum::Real>(state_left, state_right, norb);
}

inline auto compute_bbbb_4rdm(const SparseState& state_left, const SparseState& state_right,
                              std::size_t norb) {
    return compute_nrdm<4, spin_mask("bbbb"), Accum::Real>(state_left, state_right, norb);
}

// == Two-component RDMs ==
//
// The relativistic (two-component) CI stores every active electron in the alpha string, so the
// spin pattern is all-alpha and the indices are spinors rather than spatial orbitals. These
// accumulate Hermitian RDMs, unlike the one-component versions above.

inline auto compute_1rdm_2c(const SparseState& state_left, const SparseState& state_right,
                            std::size_t norb) {
    return compute_nrdm<1, spin_mask("a"), Accum::Hermitian>(state_left, state_right, norb);
}

inline auto compute_2rdm_2c(const SparseState& state_left, const SparseState& state_right,
                            std::size_t norb) {
    return compute_nrdm<2, spin_mask("aa"), Accum::Hermitian>(state_left, state_right, norb);
}

inline auto compute_3rdm_2c(const SparseState& state_left, const SparseState& state_right,
                            std::size_t norb) {
    return compute_nrdm<3, spin_mask("aaa"), Accum::Hermitian>(state_left, state_right, norb);
}

} // namespace forte2
