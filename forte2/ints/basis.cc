#include <algorithm>

#include "basis.h"
#include "real_spherical_harmonics.hpp"

namespace forte2 {
Basis::Basis() {}

Basis::Basis(const std::vector<libint2::Shell>& shells) : shells_(shells) {}

void Basis::add(const libint2::Shell& shell) {
    if (shell.contr.size() != 1) {
        throw std::runtime_error("Only single shells are supported");
    }
    if (shell.contr[0].pure == false) {
        std::cout
            << "[forte2] Warning: Cartesian Gaussians have limited support.\n"
               "         It is recommended to use Solid-Harmonic Gaussians instead (is_pure=True)."
            << std::endl;
    }
    shells_.push_back(shell);
    max_l_ = std::max(max_l_, shell.contr[0].l);
    max_nprim_ = std::max(max_nprim_, shell.nprim());
    size_ += shell.contr[0].size();
}

std::size_t Basis::size() const { return size_; }

std::size_t Basis::nshells() const { return shells_.size(); }

const libint2::Shell& Basis::operator[](size_t i) const {
    if (i >= shells_.size()) {
        throw std::out_of_range("Index out of range");
    }
    return shells_[i];
}

int Basis::max_l() const { return max_l_; }

std::size_t Basis::max_nprim() const { return max_nprim_; }

std::vector<std::pair<std::size_t, std::size_t>> Basis::shell_first_and_size() const {
    std::vector<std::pair<std::size_t, std::size_t>> result;
    result.reserve(shells_.size());
    std::size_t first = 0;
    for (const auto& shell : shells_) {
        result.emplace_back(first, shell.size());
        first += shell.size();
    }
    return result;
}

std::vector<std::pair<std::size_t, std::size_t>> Basis::center_first_and_last() const {
    std::vector<std::pair<std::size_t, std::size_t>> result;
    if (shells_.empty()) {
        return result;
    }

    std::size_t first = 0;
    std::size_t last = 0;
    auto [x0, y0, z0] = shells_[0].O;
    for (const auto& shell : shells_) {
        auto [x, y, z] = shell.O;
        if (double shell_dist = std::hypot(x - x0, y - y0, z - z0); shell_dist > 1e-8) {
            // if the center is different from the previous one, add the previous shell
            result.emplace_back(first, last);
            // update the center and the first index
            x0 = x;
            y0 = y;
            z0 = z;
            first = last;
        }
        last += shell.size();
    }
    result.emplace_back(first, last);
    return result;
}

np_matrix Basis::value_at_points(const std::vector<std::array<double, 3>>& points) const {
    auto values = make_zeros<nb::numpy, double, 2>({points.size(), size()});
    auto v = values.view();
    std::vector<double> buffer(64);
    std::size_t first_basis = 0;
    for (const auto& shell : shells_) {
        const auto shell_size = shell.size();
        if (shell.contr[0].pure == false) {
            throw std::runtime_error("[forte2] value_at_points: Only pure shells are supported");
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

np_matrix Basis::value_at_points_C(const std::vector<std::array<double, 3>>& points,
                                   np_matrix C) const {
    auto norb = C.shape(1);
    auto values = make_zeros<nb::numpy, double, 2>({points.size(), norb});
    auto v = values.view();
    std::vector<double> buffer(16);
    std::size_t first_basis = 0;
    for (const auto& shell : shells_) {
        if (shell.contr[0].pure == false) {
            throw std::runtime_error("[forte2] value_at_points_C: Only pure shells are supported");
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

} // namespace forte2