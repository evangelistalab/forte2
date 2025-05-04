#pragma once

#include <complex>

namespace forte2 {

// Define the scalar type used in the SparseOperator and State objects
using sparse_scalar_t = std::complex<double>;

// For double
inline double to_double(const double& input) { return input; }

// For std::complex<double>
inline double to_double(const std::complex<double>& input) { return std::real(input); }

} // namespace forte2