#pragma once

#include <vector>

#include "ci/determinant.h"

namespace psi {
class Matrix;
} // namespace psi

namespace forte2 {

// class ActiveSpaceIntegrals;

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
