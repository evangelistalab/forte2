#pragma once

#include <array>

#include "helpers/ndarray.h"

namespace nb = nanobind;

namespace forte2 {
class Basis;

np_vector overlap_deriv(const Basis& basis1, const Basis& basis2, const np_matrix& dm,
                       std::vector<std::pair<double, std::array<double, 3>>>& charges);

np_vector kinetic_deriv(const Basis& basis1, const Basis& basis2, const np_matrix& dm, 
                       std::vector<std::pair<double, std::array<double, 3>>>& charges);

} // namespace forte2