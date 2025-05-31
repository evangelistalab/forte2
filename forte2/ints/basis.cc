#include <algorithm>

#include "basis.h"

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

const std::vector<libint2::Shell>& Basis::shells() const { return shells_; }

const libint2::Shell& Basis::operator[](size_t i) const {
    if (i >= shells_.size()) {
        throw std::out_of_range("Index out of range");
    }
    return shells_[i];
}

int Basis::max_l() const { return max_l_; }

std::string Basis::name() const { return name_; }

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

std::string shell_label(int l, int idx) {
    static const std::vector<std::vector<std::string>> labels{
        {"s"},
        {"py", "pz", "px"},
        {"dxy", "dyz", "dz2", "dxz", "dx2-y2"},
        {"fy(3x2-y2)", "fxyz", "fyz2", "fz3", "fxz2", "fz(x2-y2)", "fx(x2-3y2)"}};
    static const std::vector<std::string> general_labels{"s", "p", "d", "f", "g", "h",
                                                         "i", "j", "k", "l", "m", "n"};
    // Validate that the provided indices are within the expected ranges.
    if (l < 0) {
        throw std::out_of_range("Invalid angular momentum index: " + std::to_string(l));
    }
    if (idx < 0) {
        throw std::out_of_range("Invalid index for angular momentum " + std::to_string(l) + ": " +
                                std::to_string(idx));
    }
    if (l < static_cast<int>(labels.size()) and idx < static_cast<int>(labels[l].size())) {
        return labels[l][idx];
    }
    return general_labels[l] + "(" + std::to_string(idx) + ")";
}

} // namespace forte2