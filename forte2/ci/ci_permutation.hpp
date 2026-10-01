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
void map_strings(const StringList& strings, const StringAddress& address, const auto& perm,
                 const auto& phase, int norb, std::vector<std::vector<size_t>>& pi,
                 std::vector<std::vector<T>>& epsilon) {
    pi.resize(strings.size());
    epsilon.resize(strings.size());
    for (size_t h = 0; h < strings.size(); ++h) {
        pi[h].resize(strings[h].size());
        epsilon[h].resize(strings[h].size());
        for (size_t I = 0; I < strings[h].size(); ++I) {
            auto J = String::zero();
            T f = 1.0;
            for (int u = norb - 1; u >= 0; --u) {
                if (strings[h][I].get_bit(u)) {
                    f *= conjugate(phase(u)) * J.create(perm(u));
                }
            }
            const auto& [add_J, class_J] = address.address_and_class(J);
            if (class_J != h) {
                throw std::runtime_error("The permutation moves a string out of its class.");
            }
            pi[h][I] = add_J;
            epsilon[h][I] = f;
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
    detail::map_strings<T>(lists.alpha_strings(), *lists.alpha_address(), perm_v, phase_v, norb,
                           pi_a, epsilon_a);
    detail::map_strings<T>(lists.beta_strings(), *lists.beta_address(), perm_v, phase_v, norb, pi_b,
                           epsilon_b);

    auto pi = make_zeros<nb::numpy, int64_t, 1>({lists.ndet()});
    auto epsilon = make_zeros<nb::numpy, T, 1>({lists.ndet()});
    auto pi_v = pi.view();
    auto epsilon_v = epsilon.view();
    // the alpha and beta strings stay in their classes, so each determinant stays in its block
    lists.for_each_element([&](const auto block, const auto class_Ia, const auto class_Ib,
                               const auto Ia, const auto Ib, const auto idx) {
        const size_t nIb = lists.beta_address()->strpcls(class_Ib);
        pi_v(idx) = static_cast<int64_t>(lists.block_offset(block) + pi_a[class_Ia][Ia] * nIb +
                                         pi_b[class_Ib][Ib]);
        epsilon_v(idx) = epsilon_a[class_Ia][Ia] * epsilon_b[class_Ib][Ib];
    });
    return {pi, epsilon};
}

} // namespace forte2
