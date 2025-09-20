#pragma once

#include <complex>

#include "helpers/ndarray.h"
#include "sparse/sparse_operator.h"

namespace forte2 {
SparseOperator sparse_operator_hamiltonian(double scalar_energy, np_matrix one_electron_integrals,
                                           np_tensor4 two_electron_integrals, double screen_thresh);
SparseOperator sparse_operator_hamiltonian(double scalar_energy,
                                           np_matrix_complex one_electron_integrals,
                                           np_tensor4_complex two_electron_integrals,
                                           double screen_thresh);
} // namespace forte2
