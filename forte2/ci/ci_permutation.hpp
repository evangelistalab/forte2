#pragma once

#include <complex>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include "helpers/ndarray.h"
#include "helpers/math_structures.h"
#include "ci/ci_strings.h"

namespace forte2 {

namespace detail {
/// @brief Gets, for every string I of one spin, the address of the string that occupies the
/// orbitals perm[u] for each u in I, and the factor (-1)^kappa_I prod_{u in I} conj(phase[u]).
/// Creating the orbitals perm[u] in descending order of u also accumulates the correct fermionic
/// sign.
template <typename T>
void map_strings(const StringAddress& address, const auto& perm, const auto& phase, int norb,
                 std::vector<std::vector<size_t>>& pi, std::vector<std::vector<T>>& epsilon) {
    pi.resize(address.nclasses());
    epsilon.resize(address.nclasses());
    std::vector<LocalStringAddress> work(address.basis().ngas());
    occupation_t occupation(address.basis().ngas());
    for (int h = 0; h < address.nclasses(); ++h) {
        pi[h].resize(address.size(h));
        epsilon[h].resize(address.size(h));
        for (size_t i = 0; i < address.size(h); ++i) {
            const auto source = address.string_at(h, i, work);
            auto target = String::zero();
            T factor = 1.0;
            for (int u = norb - 1; u >= 0; --u) {
                if (source.get_bit(u))
                    factor *= conjugate(phase(u)) * target.create(perm(u));
            }
            const auto j = address.find(target, work, occupation);
            if (!j || j->string_class != h)
                throw std::runtime_error("The permutation moves a string out of its class.");
            pi[h][i] = j->rank;
            epsilon[h][i] = factor;
        }
    }
}
} // namespace detail

/// @brief Re-expresses the determinants of a CI space in orbitals relabeled by a phased (or signed)
/// permutation, where orbital u of the permuted set is orbital index perm[u] of the original set
/// times phase[u].
/// Determinant I of the permuted orbitals is, up to the factor epsilon_I, the
/// determinant pi(I) of the original orbitals that occupies the orbitals perm[u] for each u in I,
/// so the permutation maps the CI coefficients as c_I -> epsilon_I c_{pi(I)}.
/// @param lists The strings of the CI space. The permutation must keep each string in its class.
/// @param perm The original orbital that each permuted orbital is made of
/// @param phase The phase (sign in the real case) of each permuted orbital
/// @return pi(I) and epsilon_I for every determinant I
template <typename T>
std::pair<nb::ndarray<nb::numpy, int64_t, nb::ndim<1>>, nb::ndarray<nb::numpy, T, nb::ndim<1>>>
permutation_map(const CIStrings& lists, np_vector_int perm,
                nb::ndarray<nb::numpy, T, nb::ndim<1>> phase) {
    if (perm.shape(0) != lists.norb() or phase.shape(0) != lists.norb()) {
        throw std::runtime_error("perm and phase must have one entry per orbital.");
    }
    const auto perm_v = perm.view();
    const auto phase_v = phase.view();
    const int norb = static_cast<int>(lists.norb());
    std::vector<std::vector<size_t>> pi_a, pi_b;
    std::vector<std::vector<T>> epsilon_a, epsilon_b;
    detail::map_strings<T>(lists.strings(Spin::Alpha), perm_v, phase_v, norb, pi_a, epsilon_a);
    detail::map_strings<T>(lists.strings(Spin::Beta), perm_v, phase_v, norb, pi_b, epsilon_b);

    auto pi = make_zeros<nb::numpy, int64_t, 1>({lists.ndet()});
    auto epsilon = make_zeros<nb::numpy, T, 1>({lists.ndet()});
    auto pi_v = pi.view();
    auto epsilon_v = epsilon.view();
    for (const auto& block : lists.blocks()) {
        const int ac = block.alpha_class;
        const int bc = block.beta_class;
        for (size_t a = 0; a < block.alpha_size; ++a) {
            for (size_t b = 0; b < block.beta_size; ++b) {
                const auto index = block.offset + a * block.beta_size + b;
                pi_v(index) = static_cast<int64_t>(block.offset + pi_a[ac][a] * block.beta_size +
                                                   pi_b[bc][b]);
                epsilon_v(index) = epsilon_a[ac][a] * epsilon_b[bc][b];
            }
        }
    }
    return {pi, epsilon};
}

} // namespace forte2
