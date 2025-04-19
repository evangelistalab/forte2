#pragma once

#include <libint2.hpp>

#include "ints/basis.h"

#include "helpers/ndarray.h"

namespace forte2 {

// Define a no-params struct to use as a template parameter
struct NoParams {};

template <libint2::Operator Op, std::size_t M, typename Params = NoParams>
[[nodiscard]] auto compute_one_electron_multi(const Basis& basis1, const Basis& basis2,
                                              Params const& params = Params{})
    -> std::array<nb::ndarray<nb::numpy, double, nb::ndim<2>>, M> {
    const auto start = std::chrono::high_resolution_clock::now();

    // Initialize libint2
    libint2::initialize();

    // Prepare engine
    const auto max_nprim = std::max(basis1.max_nprim(), basis2.max_nprim());
    const auto max_l = std::max(basis1.max_l(), basis2.max_l());
    libint2::Engine engine(Op, max_nprim, max_l);

    if constexpr (not std::is_same_v<Params, NoParams>) {
        engine.set_params(params);
    }

    const auto& results = engine.results();

    // Get the number of basis functions in each basis
    const std::size_t nb1 = basis1.size();
    const std::size_t nb2 = basis2.size();

    // Allocate M separate flat buffers
    std::array<std::unique_ptr<std::vector<double>>, M> buffers;
    for (auto& ptr : buffers) {
        ptr = std::make_unique<std::vector<double>>(nb1 * nb2, 0.0);
    }

    // Loop over shell pairs and fill each buffer
    std::size_t off1 = 0;
    auto nshells1 = basis1.nshells();
    auto nshells2 = basis2.nshells();

    for (std::size_t s1 = 0; s1 < nshells1; ++s1) {
        const auto& shell1 = basis1[s1];
        const std::size_t n1 = shell1.size();
        std::size_t off2 = 0;

        for (std::size_t s2 = 0; s2 < nshells2; ++s2) {
            const auto& shell2 = basis2[s2];
            const std::size_t n2 = shell2.size();

            // Compute the integrals for this shell pair
            engine.compute(shell1, shell2);

            // Loop over the components of this operator and fill the buffers
            for (std::size_t comp = 0; comp < M; ++comp) {
                const auto buf = results[comp];
                if (buf) {
                    auto& data = *buffers[comp];
                    for (std::size_t i = 0; i < n1; ++i) {
                        for (std::size_t j = 0; j < n2; ++j) {
                            data[(off1 + i) * nb2 + (off2 + j)] =
                                static_cast<double>(buf[i * n2 + j]);
                        }
                    }
                }
            }
            off2 += n2;
        }
        off1 += n1;
    }

    libint2::finalize();
    const auto end = std::chrono::high_resolution_clock::now();
    const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "[forte2] One-electron integrals timing: " << elapsed.count() << " µs\n";

    // Wrap each buffer into a Python-owned ndarray
    std::array<nb::ndarray<nb::numpy, double, nb::ndim<2>>, M> mats;
    for (std::size_t k = 0; k < M; ++k) {
        mats[k] = make_ndarray<nb::numpy, double, 2>(std::move(buffers[k]),
                                                     std::array<std::size_t, 2>{nb1, nb2});
    }
    return mats;
}
} // namespace forte2