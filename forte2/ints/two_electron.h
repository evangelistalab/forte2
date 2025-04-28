#pragma once

#include <array>

#include "helpers/ndarray.h"

namespace nb = nanobind;

namespace forte2 {
class Basis;

/// @brief Compute the coulomb integrals (b1 b2 | 1  / r_12 | b3 b4).
/// @param basis1 The basis set in the bra for electron 1 center 1.
/// @param basis2 The basis set in the bra for electron 1 center 2.
/// @param basis3 The basis set in the ket for electron 2 center 3.
/// @param basis4 The basis set in the ket for electron 2 center 4.
/// @return A 4D ndarray of shape (n1, n2, n3, n4), where ni is the number of basis functions in
///         basisi
np_tensor4 coulomb_4c(const Basis& basis1, const Basis& basis2, const Basis& basis3,
                      const Basis& basis4);

/// @brief Compute the coulomb integrals (b1 | 1  / r_12 | b2 b3).
/// @param basis1 The basis set in the bra for electron 1 center 1.
/// @param basis2 The basis set in the ket for electron 2 center 2.
/// @param basis3 The basis set in the ket for electron 2 center 3.
/// @return A 3D ndarray of shape (n1, n2, n3), where ni is the number of basis functions in
///         basisi
np_tensor3 coulomb_3c(const Basis& basis1, const Basis& basis2, const Basis& basis3);

/// @brief Compute the coulomb integrals (b1 | 1  / r_12 | b2).
/// @param basis1 The basis set in the bra for electron 1 center 1.
/// @param basis2 The basis set in the ket for electron 2 center 2.
/// @return A 2D ndarray of shape (n1, n2), where ni is the number of basis functions in
///         basisi
np_matrix coulomb_2c(const Basis& b1, const Basis& b2);

/// @brief Compute the ERF-coulomb integrals (b1 | erf(omega r_12) / r_12 | b3 b4).
/// @param basis1 The basis set in the bra for electron 1 center 1.
/// @param basis2 The basis set in the bra for electron 2 center 2.
/// @param basis3 The basis set in the ket for electron 2 center 3.
/// @param omega The attenuation parameter (>= 0).
/// @return A 3D ndarray of shape (n1, n2, n3), where ni is the number of basis functions in
///         basisi
np_tensor3 erf_coulomb_3c(const Basis& basis1, const Basis& basis2, const Basis& basis3,
                          double omega);

/// @brief Compute the ERF-coulomb integrals (b1 | erf(omega r_12) / r_12 | b2).
/// @param basis1 The basis set in the bra for electron 1 center 1.
/// @param basis2 The basis set in the ket for electron 2 center 2.
/// @param omega The attenuation parameter (>= 0).
/// @return A 2D ndarray of shape (n1, n2), where ni is the number of basis functions in
///         basisi
np_matrix erf_coulomb_2c(const Basis& basis1, const Basis& basis2, double omega);

/// @brief Compute the ERFC-coulomb integrals (b1 | erfc(omega r_12) / r_12 | b3 b4).
/// @param basis1 The basis set in the bra for electron 1 center 1.
/// @param basis2 The basis set in the bra for electron 2 center 2.
/// @param basis3 The basis set in the ket for electron 2 center 3.
/// @param omega The attenuation parameter (>= 0).
/// @return A 3D ndarray of shape (n1, n2, n3), where ni is the number of basis functions in
///         basisi
np_tensor3 erfc_coulomb_3c(const Basis& basis1, const Basis& basis2, const Basis& basis3,
                           double omega);

/// @brief Compute the ERFC-coulomb integrals (b1 | erfc(omega r_12) / r_12 | b2).
/// @param basis1 The basis set in the bra for electron 1 center 1.
/// @param basis2 The basis set in the ket for electron 2 center 2.
/// @param omega The attenuation parameter (>= 0).
/// @return A 2D ndarray of shape (n1, n2), where ni is the number of basis functions in
///         basisi
np_matrix erfc_coulomb_2c(const Basis& basis1, const Basis& basis2, double omega);

} // namespace forte2