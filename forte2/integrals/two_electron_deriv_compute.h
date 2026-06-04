#pragma once

#include <algorithm>
#include <stdexcept>
#include <string>

#include <libint2.hpp>

#include "helpers/ndarray.h"
#include "integrals/basis.h"
#include "integrals/deriv_utils.h"

namespace forte2 {

template <libint2::Operator Op, typename Weights>
[[nodiscard]] auto
compute_two_electron_3c_deriv(const Basis& basis1, const Basis& basis2, const Basis& basis3,
                              const Weights& weights,
                              const std::vector<std::pair<double, std::array<double, 3>>>& charges)
    -> np_vector {
    const std::size_t nb1 = basis1.size();
    const std::size_t nb2 = basis2.size();
    const std::size_t nb3 = basis3.size();

    validate_deriv_weight_shape(weights, std::array<std::size_t, 3>{nb1, nb2, nb3}, "W3",
                                "compute_two_electron_3c_deriv");
    validate_deriv_centers(basis1, charges, "basis1", "compute_two_electron_3c_deriv");
    validate_deriv_centers(basis2, charges, "basis2", "compute_two_electron_3c_deriv");
    validate_deriv_centers(basis3, charges, "basis3", "compute_two_electron_3c_deriv");

    libint2::initialize();

    const auto max_nprim = std::max({basis1.max_nprim(), basis2.max_nprim(), basis3.max_nprim()});
    const auto max_l = std::max({basis1.max_l(), basis2.max_l(), basis3.max_l()});
    libint2::Engine engine(Op, max_nprim, max_l, 1);
    engine.set(libint2::BraKet::xs_xx);

    const auto& results = engine.results();
    const auto first_size1 = basis1.shell_first_and_size();
    const auto first_size2 = basis2.shell_first_and_size();
    const auto first_size3 = basis3.shell_first_and_size();
    const auto atom1_to_shell = basis1.center_first_and_last(true);
    const auto atom2_to_shell = basis2.center_first_and_last(true);
    const auto atom3_to_shell = basis3.center_first_and_last(true);

    const std::size_t natoms = charges.size();
    auto grad = make_zeros<nb::numpy, double, 1>({3 * natoms});
    auto grad_view = grad.view();
    const auto weight_view = weights.view();

    for (std::size_t a1 = 0; a1 < natoms; ++a1) {
        const auto [first_shell1, last_shell1] = atom1_to_shell[a1];
        for (std::size_t a2 = 0; a2 < natoms; ++a2) {
            const auto [first_shell2, last_shell2] = atom2_to_shell[a2];
            for (std::size_t a3 = 0; a3 < natoms; ++a3) {
                const auto [first_shell3, last_shell3] = atom3_to_shell[a3];

                for (std::size_t s1 = first_shell1; s1 < last_shell1; ++s1) {
                    const auto& shell1 = basis1[s1];
                    const auto [f1, n1] = first_size1[s1];
                    for (std::size_t s2 = first_shell2; s2 < last_shell2; ++s2) {
                        const auto& shell2 = basis2[s2];
                        const auto [f2, n2] = first_size2[s2];
                        for (std::size_t s3 = first_shell3; s3 < last_shell3; ++s3) {
                            const auto& shell3 = basis3[s3];
                            const auto [f3, n3] = first_size3[s3];

                            engine.compute(shell1, shell2, shell3);

                            for (std::size_t component = 0; component < 9; ++component) {
                                const auto buf = results[component];
                                if (!buf) {
                                    continue;
                                }
                                const std::size_t center_idx = component / 3;
                                const std::size_t atom_idx =
                                    center_idx == 0 ? a1 : (center_idx == 1 ? a2 : a3);
                                const std::size_t cart_idx = component % 3;

                                for (std::size_t i = 0, ijk = 0; i != n1; ++i) {
                                    for (std::size_t j = 0; j != n2; ++j) {
                                        for (std::size_t k = 0; k != n3; ++k, ++ijk) {
                                            const double w = real_deriv_weight(
                                                weight_view(f1 + i, f2 + j, f3 + k));
                                            grad_view(atom_idx * 3 + cart_idx) +=
                                                static_cast<double>(buf[ijk]) * w;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    libint2::finalize();
    return grad;
}

template <libint2::Operator Op, typename Weights>
[[nodiscard]] auto
compute_two_electron_2c_deriv(const Basis& basis1, const Basis& basis2, const Weights& weights,
                              const std::vector<std::pair<double, std::array<double, 3>>>& charges)
    -> np_vector {
    const std::size_t nb1 = basis1.size();
    const std::size_t nb2 = basis2.size();

    validate_deriv_weight_shape(weights, std::array<std::size_t, 2>{nb1, nb2}, "W2",
                                "compute_two_electron_2c_deriv");
    validate_deriv_centers(basis1, charges, "basis1", "compute_two_electron_2c_deriv");
    validate_deriv_centers(basis2, charges, "basis2", "compute_two_electron_2c_deriv");

    libint2::initialize();

    const auto max_nprim = std::max(basis1.max_nprim(), basis2.max_nprim());
    const auto max_l = std::max(basis1.max_l(), basis2.max_l());
    libint2::Engine engine(Op, max_nprim, max_l, 1);
    engine.set(libint2::BraKet::xs_xs);

    const auto& results = engine.results();
    const auto first_size1 = basis1.shell_first_and_size();
    const auto first_size2 = basis2.shell_first_and_size();
    const auto atom1_to_shell = basis1.center_first_and_last(true);
    const auto atom2_to_shell = basis2.center_first_and_last(true);

    const std::size_t natoms = charges.size();
    auto grad = make_zeros<nb::numpy, double, 1>({3 * natoms});
    auto grad_view = grad.view();
    const auto weight_view = weights.view();

    for (std::size_t a1 = 0; a1 < natoms; ++a1) {
        const auto [first_shell1, last_shell1] = atom1_to_shell[a1];
        for (std::size_t a2 = 0; a2 < natoms; ++a2) {
            const auto [first_shell2, last_shell2] = atom2_to_shell[a2];

            for (std::size_t s1 = first_shell1; s1 < last_shell1; ++s1) {
                const auto& shell1 = basis1[s1];
                const auto [f1, n1] = first_size1[s1];
                for (std::size_t s2 = first_shell2; s2 < last_shell2; ++s2) {
                    const auto& shell2 = basis2[s2];
                    const auto [f2, n2] = first_size2[s2];

                    engine.compute(shell1, shell2);

                    for (std::size_t component = 0; component < 6; ++component) {
                        const auto buf = results[component];
                        if (!buf) {
                            continue;
                        }
                        const std::size_t atom_idx = component / 3 == 0 ? a1 : a2;
                        const std::size_t cart_idx = component % 3;

                        for (std::size_t i = 0, ij = 0; i != n1; ++i) {
                            for (std::size_t j = 0; j != n2; ++j, ++ij) {
                                const double w = real_deriv_weight(weight_view(f1 + i, f2 + j));
                                grad_view(atom_idx * 3 + cart_idx) +=
                                    static_cast<double>(buf[ij]) * w;
                            }
                        }
                    }
                }
            }
        }
    }

    libint2::finalize();
    return grad;
}

} // namespace forte2
