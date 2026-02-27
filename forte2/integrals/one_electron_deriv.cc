#include <span>

#include "integrals/one_electron_deriv.h"
#include "integrals/one_electron_deriv_compute.h"

namespace forte2 {
np_vector overlap_deriv(const Basis& b1, const Basis& b2, const np_matrix& dm,
                        std::vector<std::pair<double, std::array<double, 3>>>& charges) {
    return compute_one_electron_deriv_pos_indep<libint2::Operator::overlap, 1>(b1, b2, dm, charges);
}

} // namespace forte2