#include <span>

#include "integrals/one_electron_deriv.h"
#include "integrals/one_electron_deriv_compute.h"

namespace forte2 {
np_vector overlap_deriv(const Basis& b1, const Basis& b2, const np_matrix& dm,
                        const std::vector<std::pair<double, std::array<double, 3>>>& charges) {
    return compute_one_electron_deriv<libint2::Operator::overlap, 1>(b1, b2, dm, charges);
}

np_vector kinetic_deriv(const Basis& b1, const Basis& b2, const np_matrix& dm,
                        const std::vector<std::pair<double, std::array<double, 3>>>& charges) {
    return compute_one_electron_deriv<libint2::Operator::kinetic, 1>(b1, b2, dm, charges);
}

np_vector nuclear_deriv(const Basis& basis1, const Basis& basis2, const np_matrix& dm,
                        const std::vector<std::pair<double, std::array<double, 3>>>& charges) {
    return compute_one_electron_deriv<libint2::Operator::nuclear, 1>(basis1, basis2, dm, charges);
}

np_tensor3
overlap_deriv_matrices(const Basis& basis1, const Basis& basis2,
                       const std::vector<std::pair<double, std::array<double, 3>>>& charges) {
    return compute_one_electron_deriv_matrices<libint2::Operator::overlap>(basis1, basis2, charges);
}

np_tensor3
kinetic_deriv_matrices(const Basis& basis1, const Basis& basis2,
                       const std::vector<std::pair<double, std::array<double, 3>>>& charges) {
    return compute_one_electron_deriv_matrices<libint2::Operator::kinetic>(basis1, basis2, charges);
}

np_tensor3
nuclear_deriv_matrices(const Basis& basis1, const Basis& basis2,
                       const std::vector<std::pair<double, std::array<double, 3>>>& charges) {
    return compute_one_electron_deriv_matrices<libint2::Operator::nuclear>(basis1, basis2, charges);
}

} // namespace forte2
