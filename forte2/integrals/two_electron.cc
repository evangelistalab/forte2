#include "integrals/two_electron.h"
#include "integrals/two_electron_compute.h"

namespace forte2 {

np_tensor4 coulomb_4c(const Basis& b1, const Basis& b2, const Basis& b3, const Basis& b4) {
    return compute_two_electron_4c_multi<libint2::Operator::coulomb>(b1, b2, b3, b4);
}

np_tensor4 coulomb_4c_by_shell_slices(
    const Basis& b1, const Basis& b2, const Basis& b3, const Basis& b4,
    const std::vector<std::pair<std::size_t, std::size_t>>& shell_slices) {
    return compute_two_electron_4c_by_shell_slices<libint2::Operator::coulomb>(
        b1, b2, b3, b4, shell_slices);
}

np_tensor3 coulomb_3c(const Basis& b1, const Basis& b2, const Basis& b3) {
    return compute_two_electron_3c_multi_async<libint2::Operator::coulomb>(b1, b2, b3);
}

np_matrix coulomb_2c(const Basis& b1, const Basis& b2) {
    return compute_two_electron_2c_multi<libint2::Operator::coulomb>(b1, b2);
}

np_tensor3 erf_coulomb_3c(const Basis& b1, const Basis& b2, const Basis& b3, double omega) {
    return compute_two_electron_3c_multi<libint2::Operator::erf_coulomb>(b1, b2, b3, omega);
}

np_matrix erf_coulomb_2c(const Basis& b1, const Basis& b2, double omega) {
    return compute_two_electron_2c_multi<libint2::Operator::erf_coulomb>(b1, b2, omega);
}

np_tensor3 erfc_coulomb_3c(const Basis& b1, const Basis& b2, const Basis& b3, double omega) {
    return compute_two_electron_3c_multi<libint2::Operator::erfc_coulomb>(b1, b2, b3, omega);
}

np_matrix erfc_coulomb_2c(const Basis& b1, const Basis& b2, double omega) {
    return compute_two_electron_2c_multi<libint2::Operator::erfc_coulomb>(b1, b2, omega);
}

} // namespace forte2