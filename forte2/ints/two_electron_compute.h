#pragma once

#include <libint2.hpp>

#include "ints/basis.h"

#include "helpers/ndarray.h"

namespace forte2 {

// Define a no-params struct to use as a template parameter
struct NoParams {};

template <libint2::Operator Op, typename Params = NoParams>
[[nodiscard]] auto compute_two_electron_4c_multi(const Basis& basis1, const Basis& basis2,
                                                 const Basis& basis3, const Basis& basis4,
                                                 Params const& params = Params{})
    -> nb::ndarray<nb::numpy, double, nb::ndim<4>> {
    const auto start = std::chrono::high_resolution_clock::now();

    // Initialize libint2
    libint2::initialize();

    // Prepare engine
    const auto max_nprim = std::max(std::max(basis1.max_nprim(), basis2.max_nprim()),
                                    std::max(basis3.max_nprim(), basis4.max_nprim()));
    const auto max_l = std::max(std::max(basis1.max_l(), basis2.max_l()),
                                std::max(basis3.max_l(), basis4.max_l()));
    libint2::Engine engine(Op, max_nprim, max_l);

    if constexpr (not std::is_same_v<Params, NoParams>) {
        engine.set_params(params);
    }

    const auto& results = engine.results();

    // Get the number of basis functions in each basis
    const std::size_t nb1 = basis1.size();
    const std::size_t nb2 = basis2.size();
    const std::size_t nb3 = basis3.size();
    const std::size_t nb4 = basis4.size();
    const std::size_t nb1234 = nb1 * nb2 * nb3 * nb4;
    const std::size_t nb34 = nb3 * nb4;
    const std::size_t nb234 = nb2 * nb3 * nb4;

    // Get the number of shells in each basis
    auto nshells1 = basis1.nshells();
    auto nshells2 = basis2.nshells();
    auto nshells3 = basis3.nshells();
    auto nshells4 = basis4.nshells();

    // Get arrays of indices of the first basis in a shell and the size of each shell
    const auto first_size1 = basis1.shell_first_and_size();
    const auto first_size2 = basis2.shell_first_and_size();
    const auto first_size3 = basis3.shell_first_and_size();
    const auto first_size4 = basis4.shell_first_and_size();

    // Allocate a flat buffer
    auto buffer = std::make_unique<std::vector<double>>(nb1234, 0.0);
    auto& data = *buffer;

    // Loop over shell quartets and fill each buffer
    for (std::size_t s1 = 0; s1 < nshells1; ++s1) {
        const auto& shell1 = basis1[s1];
        const auto [f1, n1] = first_size1[s1];

        for (std::size_t s2 = 0; s2 < nshells2; ++s2) {
            const auto& shell2 = basis2[s2];
            const auto [f2, n2] = first_size2[s2];

            for (std::size_t s3 = 0; s3 < nshells3; ++s3) {
                const auto& shell3 = basis3[s3];
                const auto [f3, n3] = first_size3[s3];

                for (std::size_t s4 = 0; s4 < nshells4; ++s4) {
                    const auto& shell4 = basis4[s4];

                    // Compute the integrals for this shell pair
                    engine.compute(shell1, shell2, shell3, shell4);

                    // Loop over the components of this operator and fill the buffers
                    if (const auto buf = results[0]; buf) {
                        const auto [f4, n4] = first_size4[s4];
                        for (std::size_t i = 0, ijkl = 0; i != n1; ++i) {
                            for (std::size_t j = 0; j != n2; ++j) {
                                for (std::size_t k = 0; k != n3; ++k) {
                                    const auto f1234 =
                                        (f1 + i) * nb234 + (f2 + j) * nb34 + (f3 + k) * nb4 + f4;
                                    for (std::size_t l = 0; l != n4; ++l, ++ijkl) {
                                        data[f1234 + l] = static_cast<double>(buf[ijkl]);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Finalize libint2
    libint2::finalize();

    const auto end = std::chrono::high_resolution_clock::now();
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "[forte2] Two-electron integrals timing: " << elapsed.count() << " ms\n";

    return make_ndarray<nb::numpy, double, 4>(std::move(buffer),
                                              std::array<std::size_t, 4>{nb1, nb2, nb3, nb4});
}

template <libint2::Operator Op, typename Params = NoParams>
[[nodiscard]] auto compute_two_electron_3c_multi(const Basis& basis1, const Basis& basis2,
                                                 const Basis& basis3,
                                                 Params const& params = Params{})
    -> nb::ndarray<nb::numpy, double, nb::ndim<3>> {
    const auto start = std::chrono::high_resolution_clock::now();

    // Initialize libint2
    libint2::initialize();

    // Prepare engine
    const auto max_nprim =
        std::max(std::max(basis1.max_nprim(), basis2.max_nprim()), basis3.max_nprim());
    const auto max_l = std::max(std::max(basis1.max_l(), basis2.max_l()), basis3.max_l());
    libint2::Engine engine(Op, max_nprim, max_l);
    engine.set(libint2::BraKet::xs_xx);

    if constexpr (not std::is_same_v<Params, NoParams>) {
        engine.set_params(params);
    }

    const auto& results = engine.results();

    // Get the number of basis functions in each basis
    const std::size_t nb1 = basis1.size();
    const std::size_t nb2 = basis2.size();
    const std::size_t nb3 = basis3.size();

    // Loop over shell pairs and fill each buffer
    auto nshells1 = basis1.nshells();
    auto nshells2 = basis2.nshells();
    auto nshells3 = basis3.nshells();

    // Get arrays of indices of the first basis in a shell and the size of each shell
    const auto first_size1 = basis1.shell_first_and_size();
    const auto first_size2 = basis2.shell_first_and_size();
    const auto first_size3 = basis3.shell_first_and_size();

    // Allocate three index tensor
    auto ints = make_ndarray<nb::numpy, double, 3>({nb1, nb2, nb3});
    auto v = ints.view();

    // Loop over the shell triplets and fill the buffer
    for (std::size_t s1 = 0; s1 < nshells1; ++s1) {
        const auto& shell1 = basis1[s1];
        const auto [f1, n1] = first_size1[s1];

        for (std::size_t s2 = 0; s2 < nshells2; ++s2) {
            const auto& shell2 = basis2[s2];
            const auto [f2, n2] = first_size2[s2];

            for (std::size_t s3 = 0; s3 < nshells3; ++s3) {
                const auto& shell3 = basis3[s3];

                // Compute the integrals for this shell triplet
                engine.compute(shell1, shell2, shell3);

                // Loop over the components of this operator and fill the buffers
                if (const auto buf = results[0]; buf) {
                    const auto [f3, n3] = first_size3[s3];
                    for (std::size_t i = 0, ijk = 0; i != n1; ++i) {
                        for (std::size_t j = 0; j != n2; ++j) {
                            for (std::size_t k = 0; k != n3; ++k, ++ijk) {
                                v(f1 + i, f2 + j, f3 + k) = static_cast<double>(buf[ijk]);
                            }
                        }
                    }
                }
            }
        }
    }

    // Finalize libint2
    libint2::finalize();

    const auto end = std::chrono::high_resolution_clock::now();
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "[forte2] Three-center two-electron integrals timing: " << elapsed.count() / 1000.0
              << " s\n";

    return ints;
}

template <libint2::Operator Op, typename Params = NoParams>
[[nodiscard]] auto compute_two_electron_2c_multi(const Basis& basis1, const Basis& basis2,
                                                 Params const& params = Params{})
    -> nb::ndarray<nb::numpy, double, nb::ndim<2>> {
    const auto start = std::chrono::high_resolution_clock::now();

    // Initialize libint2
    libint2::initialize();

    // Prepare engine
    const auto max_nprim = std::max(basis1.max_nprim(), basis2.max_nprim());
    const auto max_l = std::max(basis1.max_l(), basis2.max_l());
    libint2::Engine engine(Op, max_nprim, max_l);
    engine.set(libint2::BraKet::xs_xs);

    if constexpr (not std::is_same_v<Params, NoParams>) {
        engine.set_params(params);
    }

    const auto& results = engine.results();

    // Get the number of basis functions in each basis
    const std::size_t nb1 = basis1.size();
    const std::size_t nb2 = basis2.size();

    // Loop over shell pairs and fill each buffer
    auto nshells1 = basis1.nshells();
    auto nshells2 = basis2.nshells();

    // Get arrays of indices of the first basis in a shell and the size of each shell
    const auto first_size_1 = basis1.shell_first_and_size();
    const auto first_size_2 = basis2.shell_first_and_size();

    auto ints = make_ndarray<nb::numpy, double, 2>({nb1, nb2});
    auto v = ints.view();

    // Loop over the shell pairs and fill the buffer
    for (std::size_t s1 = 0; s1 < nshells1; ++s1) {
        const auto& shell1 = basis1[s1];
        const auto [f1, n1] = first_size_1[s1];

        // Loop over the shells in basis2
        for (std::size_t s2 = 0; s2 < nshells2; ++s2) {
            const auto& shell2 = basis2[s2];

            // Compute the integrals for this shell triplet
            engine.compute(shell1, shell2);

            // Loop over the components of this operator and fill the buffers
            if (const auto buf = results[0]; buf) {
                const auto [f2, n2] = first_size_2[s2];
                for (std::size_t i = 0, ij = 0; i != n1; ++i) {
                    for (std::size_t j = 0; j != n2; ++j, ++ij) {
                        v(f1 + i, f2 + j) = static_cast<double>(buf[ij]);
                    }
                }
            }
        }
    }

    // Finalize libint2
    libint2::finalize();

    const auto end = std::chrono::high_resolution_clock::now();
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "[forte2] Two-center two-electron integrals timing: " << elapsed.count() / 1000.0
              << " s\n";

    return ints;
}

} // namespace forte2