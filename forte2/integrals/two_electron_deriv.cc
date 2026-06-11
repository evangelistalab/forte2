#include "integrals/two_electron_deriv.h"
#include "integrals/two_electron_deriv_compute.h"

namespace forte2 {

np_vector coulomb_3c_deriv(const Basis& b1, const Basis& b2, const Basis& b3,
                           const np_tensor3_c& W3,
                           const std::vector<std::pair<double, std::array<double, 3>>>& charges) {
    return compute_two_electron_3c_deriv<libint2::Operator::coulomb>(b1, b2, b3, W3, charges);
}

np_vector coulomb_3c_deriv(const Basis& b1, const Basis& b2, const Basis& b3,
                           const np_tensor3_complex_c& W3,
                           const std::vector<std::pair<double, std::array<double, 3>>>& charges) {
    return compute_two_electron_3c_deriv<libint2::Operator::coulomb>(b1, b2, b3, W3, charges);
}

np_vector coulomb_2c_deriv(const Basis& b1, const Basis& b2, const np_matrix_c& W2,
                           const std::vector<std::pair<double, std::array<double, 3>>>& charges) {
    return compute_two_electron_2c_deriv<libint2::Operator::coulomb>(b1, b2, W2, charges);
}

np_vector coulomb_2c_deriv(const Basis& b1, const Basis& b2, const np_matrix_complex_c& W2,
                           const std::vector<std::pair<double, std::array<double, 3>>>& charges) {
    return compute_two_electron_2c_deriv<libint2::Operator::coulomb>(b1, b2, W2, charges);
}

} // namespace forte2
