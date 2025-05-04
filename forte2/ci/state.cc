#include <algorithm>
#include <cmath>
#include <numeric>
#include <string>

#include "ci/determinant.h"
#include "ci/state.h"

namespace forte2 {

// std::string State::str(int n) const {
//     if (n == 0) {
//         n = Determinant::norb;
//     }
//     std::string s;
//     for (const auto& [det, c] : elements()) {
//         if (std::abs(c) > 1.0e-8) {
//             s += forte2::str(det, n) + " * " + std::to_string(c.real) + "\n";
//         }
//     }
//     return s;
// }

} // namespace forte2
