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

/// @brief Compute the diagonal (mn|mn) of the four-center two-electron Coulomb integral matrix.
/// @param basis The orbital basis set.
/// @return A 1D ndarray of length (n * n) laid out row-major over the AO-pair index p = m * n + n',
///         where n is the number of basis functions in basis.
np_vector coulomb_4c_diagonal(const Basis& basis);

/// @brief Compute a dense block (AB|CD) of the four-center two-electron Coulomb integral matrix for
///        a list of bra shell-pairs (rows) and ket shell-pairs (columns).
/// @param basis The orbital basis set.
/// @param bra_pairs An (n_bra, 2) integer array of (shellA, shellB) indices defining the rows.
/// @param ket_pairs An (n_ket, 2) integer array of (shellC, shellD) indices defining the columns.
/// @return A 2D ndarray of shape (n_bra_ao_pairs, n_ket_ao_pairs), row-major. Within a shell-pair
///         the AO-pair order is iA * nB + iB; blocks are concatenated in the given shell-pair order.
np_matrix coulomb_4c_pair_block(const Basis& basis, const np_matrix_int& bra_pairs,
                                const np_matrix_int& ket_pairs);

/// @brief Compute the coulomb integrals (b1 | 1  / r_12 | b2), by shells.
/// @param basis1 The basis set in the bra for electron 1 center 1.
/// @param basis2 The basis set in the ket for electron 2 center 2.
/// @param basis3 The basis set in the ket for electron 2 center 3.
/// @param shell_slices A 3-array of tuples specifying the shell ranges to compute for each basis
/// @return A 3D ndarray of shape (n1, n2, n3), where ni is the number of basis functions in basisi
np_tensor3_c coulomb_3c_by_shell(const Basis& basis1, const Basis& basis2, const Basis& basis3, 
                               const std::array<std::pair<std::size_t, std::size_t>, 3>& shell_slices);

/// @brief Compute the coulomb integrals (b1 | 1  / r_12 | b2), by shells, into a provided buffer.
/// @param basis1 The basis set in the bra for electron 1 center 1.
/// @param basis2 The basis set in the ket for electron 2 center 2.
/// @param basis3 The basis set in the ket for electron 2 center 3.
/// @param shell_slices A 3-array of tuples specifying the shell ranges to compute for each basis
/// @param buffer A 3D ndarray of shape of at least (n1, n2, n3) to store the results, where ni is the number of basis functions in basisi
void coulomb_3c_by_shell(const Basis& basis1, const Basis& basis2, const Basis& basis3,
                         const std::array<std::pair<std::size_t, std::size_t>, 3>& shell_slices,
                         np_tensor3_c& buffer);

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