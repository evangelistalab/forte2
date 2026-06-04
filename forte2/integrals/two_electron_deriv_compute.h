#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <stdexcept>
#include <string>
#include <vector>

#include <libint2.hpp>

#include "helpers/ndarray.h"
#include "helpers/parallel.h"
#include "integrals/basis.h"
#include "integrals/deriv_utils.h"

namespace forte2 {

inline void
validate_deriv_shell_slices(const std::array<std::pair<std::size_t, std::size_t>, 3>& shell_slices,
                            const std::array<std::size_t, 3>& nshells,
                            const std::string& caller_name) {
    for (std::size_t i = 0; i < 3; ++i) {
        if (shell_slices[i].first >= shell_slices[i].second) {
            throw std::invalid_argument(caller_name +
                                        ": shell_slices indices must be in the form of "
                                        "(start, end) with start < end");
        }
        if (shell_slices[i].second > nshells[i]) {
            throw std::invalid_argument(caller_name +
                                        ": shell_slices indices must be within the number "
                                        "of shells in each basis");
        }
    }
}

inline void
validate_deriv_shell_slices(const std::array<std::pair<std::size_t, std::size_t>, 2>& shell_slices,
                            const std::array<std::size_t, 2>& nshells,
                            const std::string& caller_name) {
    for (std::size_t i = 0; i < 2; ++i) {
        if (shell_slices[i].first >= shell_slices[i].second) {
            throw std::invalid_argument(caller_name +
                                        ": shell_slices indices must be in the form of "
                                        "(start, end) with start < end");
        }
        if (shell_slices[i].second > nshells[i]) {
            throw std::invalid_argument(caller_name +
                                        ": shell_slices indices must be within the number "
                                        "of shells in each basis");
        }
    }
}

inline auto shell_to_center_index(const Basis& basis) -> std::vector<std::size_t> {
    const auto center_to_shell = basis.center_first_and_last(true);
    std::vector<std::size_t> shell_to_center(basis.nshells());
    for (std::size_t atom = 0; atom < center_to_shell.size(); ++atom) {
        const auto [first_shell, last_shell] = center_to_shell[atom];
        for (std::size_t shell = first_shell; shell < last_shell; ++shell) {
            shell_to_center[shell] = atom;
        }
    }
    return shell_to_center;
}

template <libint2::Operator Op, typename Weights>
[[nodiscard]] auto compute_two_electron_3c_deriv_by_shell(
    const Basis& basis1, const Basis& basis2, const Basis& basis3,
    const std::array<std::pair<std::size_t, std::size_t>, 3>& shell_slices, const Weights& weights,
    const std::vector<std::pair<double, std::array<double, 3>>>& charges) -> np_vector {
    validate_deriv_centers(basis1, charges, "basis1", "compute_two_electron_3c_deriv_by_shell");
    validate_deriv_centers(basis2, charges, "basis2", "compute_two_electron_3c_deriv_by_shell");
    validate_deriv_centers(basis3, charges, "basis3", "compute_two_electron_3c_deriv_by_shell");
    validate_deriv_shell_slices(
        shell_slices,
        std::array<std::size_t, 3>{basis1.nshells(), basis2.nshells(), basis3.nshells()},
        "compute_two_electron_3c_deriv_by_shell");

    const auto [s1_begin, s1_end] = shell_slices[0];
    const auto [s2_begin, s2_end] = shell_slices[1];
    const auto [s3_begin, s3_end] = shell_slices[2];

    const auto offsets1 = basis1.shell_offsets();
    const auto offsets2 = basis2.shell_offsets();
    const auto offsets3 = basis3.shell_offsets();
    const std::size_t first1 = offsets1[s1_begin];
    const std::size_t first2 = offsets2[s2_begin];
    const std::size_t first3 = offsets3[s3_begin];
    const std::size_t nb1_slice = offsets1[s1_end] - first1;
    const std::size_t nb2_slice = offsets2[s2_end] - first2;
    const std::size_t nb3_slice = offsets3[s3_end] - first3;
    validate_deriv_weight_shape(weights,
                                std::array<std::size_t, 3>{nb1_slice, nb2_slice, nb3_slice}, "W3",
                                "compute_two_electron_3c_deriv_by_shell");

    libint2::initialize();

    const auto max_nprim = std::max({basis1.max_nprim(), basis2.max_nprim(), basis3.max_nprim()});
    const auto max_l = std::max({basis1.max_l(), basis2.max_l(), basis3.max_l()});

    const auto first_size1 = basis1.shell_first_and_size();
    const auto first_size2 = basis2.shell_first_and_size();
    const auto first_size3 = basis3.shell_first_and_size();
    const auto shell1_to_atom = shell_to_center_index(basis1);
    const auto shell2_to_atom = shell_to_center_index(basis2);
    const auto shell3_to_atom = shell_to_center_index(basis3);

    const std::size_t natoms = charges.size();
    auto grad = make_zeros<nb::numpy, double, 1>({3 * natoms});
    auto grad_view = grad.view();
    const auto weight_view = weights.view();
    // Find the number of threads to use (no more than the number of first basis slices)
    const std::size_t max_chunks = std::min(get_num_threads(), s1_end - s1_begin);
    std::vector<std::vector<double>> local_grads(max_chunks, std::vector<double>(3 * natoms, 0.0));
    // Atomic counter to assign a local gradient to each thread without overlap
    std::atomic<std::size_t> next_chunk{0};

    auto kernel = [&](std::size_t s1_chunk_begin, std::size_t s1_chunk_end) {
        libint2::Engine engine(Op, max_nprim, max_l, 1);
        engine.set(libint2::BraKet::xs_xx);
        const auto& results = engine.results();

        const std::size_t chunk_idx = next_chunk.fetch_add(1);
        auto& local_grad = local_grads[chunk_idx];

        for (std::size_t s1 = s1_chunk_begin; s1 < s1_chunk_end; ++s1) {
            const auto& shell1 = basis1[s1];
            const auto [f1, n1] = first_size1[s1];
            const std::size_t atom1 = shell1_to_atom[s1];

            for (std::size_t s2 = s2_begin; s2 < s2_end; ++s2) {
                const auto& shell2 = basis2[s2];
                const auto [f2, n2] = first_size2[s2];
                const std::size_t atom2 = shell2_to_atom[s2];

                for (std::size_t s3 = s3_begin; s3 < s3_end; ++s3) {
                    const auto& shell3 = basis3[s3];
                    const auto [f3, n3] = first_size3[s3];
                    const std::size_t atom3 = shell3_to_atom[s3];

                    engine.compute(shell1, shell2, shell3);

                    for (std::size_t component = 0; component < 9; ++component) {
                        const auto buf = results[component];
                        if (!buf) {
                            continue;
                        }
                        const std::size_t center_idx = component / 3;
                        const std::size_t atom_idx =
                            center_idx == 0 ? atom1 : (center_idx == 1 ? atom2 : atom3);
                        const std::size_t cart_idx = component % 3;

                        for (std::size_t i = 0, ijk = 0; i != n1; ++i) {
                            for (std::size_t j = 0; j != n2; ++j) {
                                for (std::size_t k = 0; k != n3; ++k, ++ijk) {
                                    const double w = real_deriv_weight(weight_view(
                                        f1 - first1 + i, f2 - first2 + j, f3 - first3 + k));
                                    local_grad[atom_idx * 3 + cart_idx] +=
                                        static_cast<double>(buf[ijk]) * w;
                                }
                            }
                        }
                    }
                }
            }
        }
    };

    parallel_for_chunked(s1_begin, s1_end, kernel);

    for (const auto& local_grad : local_grads) {
        for (std::size_t i{0}, maxi{local_grad.size()}; i < maxi; ++i) {
            grad_view(i) += local_grad[i];
        }
    }

    libint2::finalize();
    return grad;
}

template <libint2::Operator Op, typename Weights>
[[nodiscard]] auto
compute_two_electron_3c_deriv(const Basis& basis1, const Basis& basis2, const Basis& basis3,
                              const Weights& weights,
                              const std::vector<std::pair<double, std::array<double, 3>>>& charges)
    -> np_vector {
    return compute_two_electron_3c_deriv_by_shell<Op>(
        basis1, basis2, basis3,
        std::array<std::pair<std::size_t, std::size_t>, 3>{
            std::pair<std::size_t, std::size_t>{0, basis1.nshells()},
            std::pair<std::size_t, std::size_t>{0, basis2.nshells()},
            std::pair<std::size_t, std::size_t>{0, basis3.nshells()}},
        weights, charges);
}

template <libint2::Operator Op, typename Weights>
[[nodiscard]] auto compute_two_electron_2c_deriv_by_shell(
    const Basis& basis1, const Basis& basis2,
    const std::array<std::pair<std::size_t, std::size_t>, 2>& shell_slices, const Weights& weights,
    const std::vector<std::pair<double, std::array<double, 3>>>& charges) -> np_vector {
    validate_deriv_centers(basis1, charges, "basis1", "compute_two_electron_2c_deriv_by_shell");
    validate_deriv_centers(basis2, charges, "basis2", "compute_two_electron_2c_deriv_by_shell");
    validate_deriv_shell_slices(shell_slices,
                                std::array<std::size_t, 2>{basis1.nshells(), basis2.nshells()},
                                "compute_two_electron_2c_deriv_by_shell");

    const auto [s1_begin, s1_end] = shell_slices[0];
    const auto [s2_begin, s2_end] = shell_slices[1];

    const auto offsets1 = basis1.shell_offsets();
    const auto offsets2 = basis2.shell_offsets();
    const std::size_t first1 = offsets1[s1_begin];
    const std::size_t first2 = offsets2[s2_begin];
    const std::size_t nb1_slice = offsets1[s1_end] - first1;
    const std::size_t nb2_slice = offsets2[s2_end] - first2;
    validate_deriv_weight_shape(weights, std::array<std::size_t, 2>{nb1_slice, nb2_slice}, "W2",
                                "compute_two_electron_2c_deriv_by_shell");

    libint2::initialize();

    const auto max_nprim = std::max(basis1.max_nprim(), basis2.max_nprim());
    const auto max_l = std::max(basis1.max_l(), basis2.max_l());

    const auto first_size1 = basis1.shell_first_and_size();
    const auto first_size2 = basis2.shell_first_and_size();
    const auto shell1_to_atom = shell_to_center_index(basis1);
    const auto shell2_to_atom = shell_to_center_index(basis2);

    const std::size_t natoms = charges.size();
    auto grad = make_zeros<nb::numpy, double, 1>({3 * natoms});
    auto grad_view = grad.view();
    const auto weight_view = weights.view();
    // Find the number of threads to use (no more than the number of first basis slices)
    const std::size_t max_chunks = std::min(get_num_threads(), s1_end - s1_begin);
    // Each thread will have its own local gradient to avoid race conditions
    std::vector<std::vector<double>> local_grads(max_chunks, std::vector<double>(3 * natoms, 0.0));
    // Atomic counter to assign a local gradient to each thread without overlap
    std::atomic<std::size_t> next_chunk{0};

    auto kernel = [&](std::size_t s1_chunk_begin, std::size_t s1_chunk_end) {
        libint2::Engine engine(Op, max_nprim, max_l, 1);
        engine.set(libint2::BraKet::xs_xs);
        const auto& results = engine.results();

        const std::size_t chunk_idx = next_chunk.fetch_add(1);
        auto& local_grad = local_grads[chunk_idx];

        for (std::size_t s1 = s1_chunk_begin; s1 < s1_chunk_end; ++s1) {
            const auto& shell1 = basis1[s1];
            const auto [f1, n1] = first_size1[s1];
            const std::size_t atom1 = shell1_to_atom[s1];

            for (std::size_t s2 = s2_begin; s2 < s2_end; ++s2) {
                const auto& shell2 = basis2[s2];
                const auto [f2, n2] = first_size2[s2];
                const std::size_t atom2 = shell2_to_atom[s2];

                engine.compute(shell1, shell2);

                for (std::size_t component = 0; component < 6; ++component) {
                    const auto buf = results[component];
                    if (!buf) {
                        continue;
                    }
                    const std::size_t atom_idx = component / 3 == 0 ? atom1 : atom2;
                    const std::size_t cart_idx = component % 3;

                    for (std::size_t i = 0, ij = 0; i != n1; ++i) {
                        for (std::size_t j = 0; j != n2; ++j, ++ij) {
                            const double w =
                                real_deriv_weight(weight_view(f1 - first1 + i, f2 - first2 + j));
                            local_grad[atom_idx * 3 + cart_idx] += static_cast<double>(buf[ij]) * w;
                        }
                    }
                }
            }
        }
    };

    parallel_for_chunked(s1_begin, s1_end, kernel);

    for (const auto& local_grad : local_grads) {
        for (std::size_t i{0}, maxi{local_grad.size()}; i < maxi; ++i) {
            grad_view(i) += local_grad[i];
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
    return compute_two_electron_2c_deriv_by_shell<Op>(
        basis1, basis2,
        std::array<std::pair<std::size_t, std::size_t>, 2>{
            std::pair<std::size_t, std::size_t>{0, basis1.nshells()},
            std::pair<std::size_t, std::size_t>{0, basis2.nshells()}},
        weights, charges);
}

} // namespace forte2
