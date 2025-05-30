#pragma once

#include "helpers/ndarray.h"

#include "ci/sparse_operator.h"

namespace forte2 {
SparseOperator sparse_operator_hamiltonian(int nmo, double scalar_energy,
                                           np_matrix one_electron_integrals,
                                           np_tensor4 two_electron_integrals, double screen_thresh);
} // namespace forte2
