#include <algorithm>
#include <cmath>
#include <numeric>

#include "helpers/string_algorithms.h"
#include "ci/determinant.hpp"
#include "ci/sparse_state.h"

namespace forte2 {

std::string SparseState::str(int n) const {
    if (n == 0) {
        n = Determinant::norb();
    }
    std::string s;
    for (const auto& [det, c] : elements()) {
        if (std::abs(c) > 1.0e-8) {
            s += forte2::str(det, n) + " * " + to_string_with_precision(c, 8) + "\n";
        }
    }
    return s;
}

} // namespace forte2
