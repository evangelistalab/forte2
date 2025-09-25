#pragma once

#include <vector>

#include "ci/determinant.h"

namespace psi {
class Matrix;
} // namespace psi

namespace forte2 {

/// @brief Create a single excitation from alpha orbital i to alpha orbital a
/// @param det The original determinant
/// @param i The index of the occupied alpha orbital
/// @param a The index of the virtual alpha orbital
/// @return The new determinant with the single excitation
Determinant create_single_a_excitation(const Determinant& det, size_t i, size_t a);

/// @brief Create a single excitation from beta orbital i to beta orbital a
/// @param det The original determinant
/// @param i The index of the occupied beta orbital
/// @param a The index of the virtual beta orbital
/// @return The new determinant with the single excitation
Determinant create_single_b_excitation(const Determinant& det, size_t i, size_t a);

/// @brief Create a double excitation from alpha orbitals i,j to alpha orbitals a,b
/// @param det The original determinant
/// @param i The index of the first occupied alpha orbital
/// @param j The index of the second occupied alpha orbital
/// @param a The index of the first virtual alpha orbital
/// @param b The index of the second virtual alpha orbital
/// @return The new determinant with the double excitation
Determinant create_double_aa_excitation(const Determinant& det, size_t i, size_t j, size_t a,
                                        size_t b);

/// @brief Create a double excitation from beta orbitals i,j to beta orbitals a,b
/// @param det The original determinant
/// @param i The index of the first occupied beta orbital
/// @param j The index of the second occupied beta orbital
/// @param a The index of the first virtual beta orbital
/// @param b The index of the second virtual beta orbital
/// @return The new determinant with the double excitation
Determinant create_double_bb_excitation(const Determinant& det, size_t i, size_t j, size_t a,
                                        size_t b);

/// @brief Create a double excitation from alpha orbital i and beta orbital j to
///        alpha orbital a and beta orbital b
/// @param det The original determinant
/// @param i The index of the occupied alpha orbital
/// @param j The index of the occupied beta orbital
/// @param a The index of the virtual alpha orbital
/// @param b The index of the virtual beta orbital
Determinant create_double_ab_excitation(const Determinant& det, size_t i, size_t j, size_t a,
                                        size_t b);

/// @brief Build the S^2 operator matrix in the given basis of determinants (multithreaded)
/// @param dets A vector of determinants
/// @return A matrix of size (num_dets, num_dets) with the S^2 operator matrix
// std::shared_ptr<psi::Matrix> make_s2_matrix(const std::vector<Determinant>& dets);

/// @brief Build the Hamiltonian operator matrix in the given basis of determinants (multithreaded)
/// @param dets A vector of determinants
/// @param as_ints A pointer to the ActiveSpaceIntegrals object
/// @return A matrix of size (num_dets, num_dets) with the Hamiltonian operator matrix
// std::shared_ptr<psi::Matrix> make_hamiltonian_matrix(const std::vector<Determinant>& dets,
//                                                      std::shared_ptr<ActiveSpaceIntegrals>
//                                                      as_ints);

/// @brief Generate all strings of n orbitals and k electrons in each irrep
/// @param n The number of orbitals
/// @param k The number of electrons
/// @param nirrep The number of irreps
/// @param mo_symmetry The symmetry of the MOs
/// @return A vector of vectors of strings, one vector for each irrep
std::vector<std::vector<String>> make_strings(int n, int k, size_t nirrep,
                                              const std::vector<int>& mo_symmetry);

/// @brief Generate the Hilbert space for a given number of electrons and orbitals
/// @param nmo The number of orbitals
/// @param na The number of alpha electrons
/// @param nb The number of beta electrons
/// @param nirrep The number of irreps (optional)
/// @param mo_symmetry The symmetry of the MOs (optional)
/// @param symmetry The symmetry of the determinants (optional)
/// @return A vector of determinants
std::vector<Determinant> make_hilbert_space(size_t nmo, size_t na, size_t nb, size_t nirrep = 1,
                                            std::vector<int> mo_symmetry = std::vector<int>(),
                                            int symmetry = 0);

/// @brief Generate the Hilbert space for a given number of electrons and orbitals
/// @param nmo The number of orbitals
/// @param na The number of alpha electrons
/// @param nb The number of beta electrons
/// @param ref The reference determinant
/// @param truncation The excitation level truncation
/// @param nirrep The number of irreps (optional)
/// @param mo_symmetry The symmetry of the MOs (optional)
/// @param symmetry The symmetry of the determinants (optional)
/// @return A vector of determinants
std::vector<Determinant> make_hilbert_space(size_t nmo, size_t na, size_t nb, Determinant ref,
                                            int truncation, size_t nirrep = 1,
                                            std::vector<int> mo_symmetry = std::vector<int>(),
                                            int symmetry = 0);

} // namespace forte2
