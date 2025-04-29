#include <algorithm>

#include "basis.h"
#include "real_spherical_harmonics.hpp"

namespace forte2 {

void evaluate_shell(const libint2::Shell& shell, const std::array<double, 3>& point,
                    double* buffer) {
    // Evaluate the shell at the given points
    const auto& contr = shell.contr[0];
    const auto& [x0, y0, z0] = shell.O;
    const auto& [x, y, z] = point;
    const auto& exponents = shell.alpha;
    const auto nprim = shell.nprim();
    const auto l = contr.l;
    const auto& coeffs = contr.coeff;
    const auto size = contr.size();

    if (l > 7) {
        throw std::runtime_error("[forte2] evaluate_shell with l > 7 is not supported");
    }

    const auto r2 = (x - x0) * (x - x0) + (y - y0) * (y - y0) + (z - z0) * (z - z0);

    double g_sum = 0.0; // sum of Gaussian functions
    for (std::size_t i = 0; i < nprim; ++i) {
        g_sum += coeffs[i] * std::exp(-exponents[i] * r2);
    }

    // Use the appropriate function to compute the spherical harmonics
    // based on the angular momentum l
    ints::compute_real_spherical_harmonic[l](x - x0, y - y0, z - z0, buffer);

    for (std::size_t i = 0; i < size; ++i) {
        buffer[i] *= g_sum;
    }
}

np_matrix basis_at_points(const Basis& basis, const std::vector<std::array<double, 3>>& points) {
    auto values = make_zeros<nb::numpy, double, 2>({points.size(), basis.size()});
    auto v = values.view();
    std::vector<double> buffer(64);
    std::size_t first_basis = 0;
    for (const auto& shell : basis.shells()) {
        const auto shell_size = shell.size();
        if (shell.contr[0].pure == false) {
            throw std::runtime_error("[forte2] basis_at_points: Only pure shells are supported");
        }
        std::size_t p = 0;
        for (const auto& point : points) {
            evaluate_shell(shell, point, buffer.data());
            for (std::size_t i = 0; i < shell_size; ++i) {
                v(p, first_basis + i) += buffer[i];
            }
            ++p;
        }
        first_basis += shell_size;
    }
    return values;
}

np_matrix orbitals_at_points(const Basis& basis, const std::vector<std::array<double, 3>>& points,
                             np_matrix C) {
    auto norb = C.shape(1);
    auto values = make_zeros<nb::numpy, double, 2>({points.size(), norb});
    auto v = values.view();
    std::vector<double> buffer(16);
    std::size_t first_basis = 0;
    for (const auto& shell : basis.shells()) {
        if (shell.contr[0].pure == false) {
            throw std::runtime_error("[forte2] orbitals_at_points: Only pure shells are supported");
        }
        const auto shell_size = shell.size();
        std::size_t p = 0;
        for (const auto& point : points) {
            evaluate_shell(shell, point, buffer.data());
            for (std::size_t i = 0; i < shell_size; ++i) {
                for (std::size_t j = 0; j < norb; ++j) {
                    v(p, j) += buffer[i] * C(first_basis + i, j);
                }
            }
            ++p;
        }
        first_basis += shell_size;
    }
    return values;
}

} // namespace forte2