#pragma once

#include <array>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;

namespace forte2 {
class Basis;

/// @brief Compute the coulomb integrals (b1 b2|b3 b4).
/// @param basis1 The basis set in the bra.
/// @param basis2 The basis set in the ket.
/// @return A 2D ndarray of shape (n1, n2), where n1 is the number of basis functions in
///         basis1 and n2 is the number of basis functions in basis2.
nb::ndarray<nb::numpy, double, nb::ndim<4>> coulomb_4c(const Basis& basis1, const Basis& basis2,
                                                       const Basis& basis3, const Basis& basis4);

nb::ndarray<nb::numpy, double, nb::ndim<3>> coulomb_3c(const Basis& basis1, const Basis& basis2,
                                                       const Basis& basis3);

nb::ndarray<nb::numpy, double, nb::ndim<2>> coulomb_2c(const Basis& b1, const Basis& b2);

} // namespace forte2