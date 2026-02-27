#pragma once

#include <libint2.hpp>

#include "integrals/basis.h"

#include "helpers/ndarray.h"

namespace forte2 {

// Define a no-params struct to use as a template parameter
struct NoParams {};

template <libint2::Operator Op, std::size_t M, typename Params = NoParams>
[[nodiscard]] auto
compute_one_electron_deriv_pos_indep(const Basis& basis1, const Basis& basis2, const np_matrix& dm,
                                     std::vector<std::pair<double, std::array<double, 3>>>& charges,
                                     Params const& params = Params{}) -> np_vector {
    // Get the number of basis functions in each basis
    const std::size_t nb1 = basis1.size();
    const std::size_t nb2 = basis2.size();

    // Assert that dm has the correct shape
    if (dm.shape(0) != nb1 || dm.shape(1) != nb2) {
        throw std::invalid_argument(
            "compute_one_electron_deriv: density matrix has incorrect shape");
    }

    // Initialize libint2
    libint2::initialize();

    // Prepare engine
    const auto max_nprim = std::max(basis1.max_nprim(), basis2.max_nprim());
    const auto max_l = std::max(basis1.max_l(), basis2.max_l());
    libint2::Engine engine(Op, max_nprim, max_l, 1); // '1' means that we will get first derivatives

    if constexpr (not std::is_same_v<Params, NoParams>) {
        engine.set_params(params);
    }

    const auto& results = engine.results();

    // Loop over shell pairs and fill each buffer
    auto nshells1 = basis1.nshells();
    auto nshells2 = basis2.nshells();

    // Get arrays of indices of the first basis in a shell and the size of each shell
    const auto first_size1 = basis1.shell_first_and_size();
    const auto first_size2 = basis2.shell_first_and_size();

    const auto natoms = charges.size();
    // [TODO] need to optimize this
    const auto shell_to_atom1 = basis1.center_first_and_last(true);
    const auto shell_to_atom2 = basis2.center_first_and_last(true);
    std::unordered_map<std::size_t, std::size_t> shell_to_atom_map1;
    std::unordered_map<std::size_t, std::size_t> shell_to_atom_map2;
    for (std::size_t n = 0; n < shell_to_atom1.size(); ++n) {
        const auto [first, last] = shell_to_atom1[n];
        for (std::size_t s = first; s <= last; ++s) {
            shell_to_atom_map1[s] = n;
        }
    }
    for (std::size_t n = 0; n < shell_to_atom2.size(); ++n) {
        const auto [first, last] = shell_to_atom2[n];
        for (std::size_t s = first; s <= last; ++s) {
            shell_to_atom_map2[s] = n;
        }
    }

    // Allocate the result gradient vector
    np_vector grad = make_zeros<nb::numpy, double, 1>({natoms * 3});

    const auto dm_view = dm.view();
    auto grad_view = grad.view();

    for (std::size_t s1 = 0; s1 < nshells1; ++s1) {
        const auto& shell1 = basis1[s1];
        const auto [f1, n1] = first_size1[s1];
        const auto atom1 = shell_to_atom_map1[s1];

        for (std::size_t s2 = 0; s2 < nshells2; ++s2) {
            const auto& shell2 = basis2[s2];
            const auto [f2, n2] = first_size2[s2];
            const auto atom2 = shell_to_atom_map2[s2];

            // Compute the integrals for this shell pair
            engine.compute(shell1, shell2);

            auto buf_x1 = results[0];
            auto buf_y1 = results[1];
            auto buf_z1 = results[2];
            auto buf_x2 = results[3];
            auto buf_y2 = results[4];
            auto buf_z2 = results[5];
            if (buf_x1 || buf_y1 || buf_z1 || buf_x2 || buf_y2 || buf_z2) {
                for (std::size_t i = 0, ij = 0; i != n1; ++i) {
                    for (std::size_t j = 0; j != n2; ++j, ++ij) {
                        const auto dm_val = dm_view(f1 + i, f2 + j);
                        grad_view(atom1 * 3 + 0) +=
                            static_cast<double>(buf_x1[ij]) * dm_val;
                        grad_view(atom1 * 3 + 1) +=
                            static_cast<double>(buf_y1[ij]) * dm_val;
                        grad_view(atom1 * 3 + 2) +=
                            static_cast<double>(buf_z1[ij]) * dm_val;
                        grad_view(atom2 * 3 + 0) +=
                            static_cast<double>(buf_x2[ij]) * dm_val;
                        grad_view(atom2 * 3 + 1) +=
                            static_cast<double>(buf_y2[ij]) * dm_val;
                        grad_view(atom2 * 3 + 2) +=
                            static_cast<double>(buf_z2[ij]) * dm_val;
                    }
                }
            }
        }
    }

    libint2::finalize();

    return grad;
}
} // namespace forte2