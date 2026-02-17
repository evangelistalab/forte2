#pragma once

#include <complex>

#include "helpers/ndarray.h"
#include "sparse/sparse_operator.h"

namespace forte2 {
SparseOperator sparse_operator_hamiltonian(double scalar_energy,
                                           ndarray<double, 2> one_electron_integrals,
                                           ndarray<double, 4> two_electron_integrals,
                                           double screen_thresh);
SparseOperator sparse_operator_hamiltonian(double scalar_energy,
                                           ndarray<std::complex<double>, 2> one_electron_integrals,
                                           ndarray<std::complex<double>, 4> two_electron_integrals,
                                           double screen_thresh);
} // namespace forte2
