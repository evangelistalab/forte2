#include <algorithm>

#include "basis.h"
#include "real_spherical_harmonics.hpp"

namespace forte2 {
Basis::Basis() {}

Basis::Basis(const std::vector<libint2::Shell>& shells) : shells_(shells) {}

void Basis::add(const libint2::Shell& shell) {
    if (shell.contr.size() != 1) {
        throw std::runtime_error("Only pure shells are supported");
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

void Basis::value_at_points(const std::vector<std::array<double, 3>>& points,
                            std::vector<double>& out) const {
    if (out.size() < size() * points.size()) {
        throw std::runtime_error("Output vector is too small");
    }
    std::fill(out.begin(), out.end(), 0.0);
    for (const auto& shell : shells_) {
        evaluate_shell(shell, points, out);
    }
}

void evaluate_shell(const libint2::Shell& shell, const std::vector<std::array<double, 3>>& points,
                    std::vector<double>& buffer) { // Evaluate the shell at the given points
    const auto& contr = shell.contr[0];
    const auto& [x0, y0, z0] = shell.O;
    const auto l = contr.l;
    const auto& exponents = shell.alpha;
    const auto& coeffs = contr.coeff;
    const auto nprim = shell.nprim();

    auto compute = ints::compute_real_spherical_harmonic[l];

    std::size_t k = 0;
    for (const auto& point : points) {
        double g_val = 0.0;
        const auto& [x, y, z] = point;
        const auto r2 = (x - x0) * (x - x0) + (y - y0) * (y - y0) + (z - z0) * (z - z0);
        for (std::size_t i = 0; i < nprim; ++i) {
            g_val += coeffs[i] * std::exp(-exponents[i] * r2);
        }
        compute(g_val, x - x0, y - y0, z - z0, buffer.data() + k);
        k += nprim;
    }
}

} // namespace forte2