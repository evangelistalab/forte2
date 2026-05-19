#pragma once

#include <vector>

#include "determinant/determinant.h"

namespace psi {
class Matrix;
} // namespace psi

namespace forte2 {

/// @brief Create a single excitation from orbital i to orbital a
/// @param str The original string
/// @param i The index of the occupied orbital
/// @param a The index of the virtual orbital
/// @return The new string with the single excitation
std::pair<String, double> create_single_excitation(const String& str, size_t i, size_t a);

/// @brief Create a double excitation from orbitals i,j to orbitals a,b
/// @param str The original string
/// @param i The index of the first occupied orbital
/// @param j The index of the second occupied orbital
/// @param a The index of the first virtual orbital
/// @param b The index of the second virtual orbital
/// @return The new string with the double excitation
std::pair<String, double> create_double_excitation(const String& str, size_t i, size_t j, size_t a,
                                                   size_t b);

/// @brief Create a single excitation from orbital i to orbital a using the fast creation and
///        destruction methods that assume the excitation is valid (i is occupied and a is virtual)
/// @param str The original string
/// @param i The index of the occupied orbital
/// @param a The index of the virtual orbital
/// @return The new string with the single excitation
std::pair<String, double> create_single_excitation_unchecked(const String& str, size_t i, size_t a);

/// @brief Create a double excitation from orbitals i,j to orbitals a,b using the fast creation and
///        destruction methods that assume the excitation is valid (i,j are occupied and a,b are
///        virtual)
/// @param str The original string
/// @param i The index of the first occupied orbital
/// @param j The index of the second occupied orbital
/// @param a The index of the first virtual orbital
/// @param b The index of the second virtual orbital
/// @return The new string with the double excitation
std::pair<String, double> create_double_excitation_unchecked(const String& str, size_t i, size_t j,
                                                             size_t a, size_t b);

/// @brief Create a single excitation from alpha orbital i to alpha orbital a
/// @param det The original determinant
/// @param i The index of the occupied alpha orbital
/// @param a The index of the virtual alpha orbital
/// @return The new determinant with the single excitation
std::pair<Determinant, double> create_single_a_excitation(const Determinant& det, size_t i,
                                                          size_t a);

/// @brief Create a single excitation from beta orbital i to beta orbital a
/// @param det The original determinant
/// @param i The index of the occupied beta orbital
/// @param a The index of the virtual beta orbital
/// @return The new determinant with the single excitation
std::pair<Determinant, double> create_single_b_excitation(const Determinant& det, size_t i,
                                                          size_t a);

/// @brief Create a double excitation from alpha orbitals i,j to alpha orbitals a,b
/// @param det The original determinant
/// @param i The index of the first occupied alpha orbital
/// @param j The index of the second occupied alpha orbital
/// @param a The index of the first virtual alpha orbital
/// @param b The index of the second virtual alpha orbital
/// @return The new determinant with the double excitation
std::pair<Determinant, double> create_double_aa_excitation(const Determinant& det, size_t i,
                                                           size_t j, size_t a, size_t b);

/// @brief Create a double excitation from beta orbitals i,j to beta orbitals a,b
/// @param det The original determinant
/// @param i The index of the first occupied beta orbital
/// @param j The index of the second occupied beta orbital
/// @param a The index of the first virtual beta orbital
/// @param b The index of the second virtual beta orbital
/// @return The new determinant with the double excitation
std::pair<Determinant, double> create_double_bb_excitation(const Determinant& det, size_t i,
                                                           size_t j, size_t a, size_t b);

/// @brief Create a double excitation from alpha orbital i and beta orbital j to
///        alpha orbital a and beta orbital b
/// @param det The original determinant
/// @param i The index of the occupied alpha orbital
/// @param j The index of the occupied beta orbital
/// @param a The index of the virtual alpha orbital
/// @param b The index of the virtual beta orbital
std::pair<Determinant, double> create_double_ab_excitation(const Determinant& det, size_t i,
                                                           size_t j, size_t a, size_t b);

/// @brief Build the S^2 operator matrix in the given basis of determinants (multithreaded)
/// @param dets A vector of determinants
/// @return A matrix of size (num_dets, num_dets) with the S^2 operator matrix
// std::shared_ptr<psi::Matrix> make_s2_matrix(const std::vector<Determinant>& dets);

/// @brief Build the Hamiltonian operator matrix in the given basis of determinants
/// (multithreaded)
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

/// @brief Given a vector of occupied orbitals, compute the list of virtual orbitals
/// @param occ The occupied orbitals (must be sorted in ascending order)
/// @param vir The virtual orbital vector (will be filled and must be of size n - occ.size())
/// @param n The total number of orbitals
void collect_virtual_orbitals(const std::vector<size_t>& occ, std::vector<size_t>& vir,
                              const size_t n);

/// @brief Apply a general operator to this determinant without checking applicability.
///
/// This function assumes the caller has already verified that the operator can be applied, for
/// example with can_apply_operator(cre, ann). Calling it on an inapplicable determinant may
/// produce a determinant and sign for a different algebraic operation.
///
/// @param d the determinant
/// @param new_d the new determinant
/// @param cre the creation operator
/// @param ann the annihilation operator
/// @param sign the sign mask (precomputed by the user) of the operator
/// @return the sign of the final determinant (+1, -1)
///
/// Example:
///
///   Determinant det, new_det, cre, ann, sign_mask;
///   // test if the operator can be applied
///   if (det.can_apply_operator(cre,ann)) {
///       // compute the sign mask
///       compute_sign_mask(cre, ann, sign_mask);
///       auto value = apply_operator_to_det_unchecked(det, new_det, cre, ann, sign_mask);
///       // do something with value and new_det
///   }
///
double apply_operator_to_det_unchecked(const Determinant& d, Determinant& new_d,
                                       const Determinant& cre, const Determinant& ann,
                                       const Determinant& sign);

/// @brief Compute the matrix element of the S^2 operator between two determinants. The S^2 operator
/// is defined as S^2 = S- S+ + Sz (Sz + 1), where S- and S+ are the spin lowering and raising
/// operators, and Sz is the spin z-component operator. The matrix element is computed using the
/// formula: S^2 = Sz (Sz + 1) + Nbeta + Npairs - sum_pq' a+(qa) a+(pb) a-(qb) a-(pa), where Nbeta
/// is the number of beta electrons, Npairs is the number of alpha/beta pairs, and the sum is over
/// pairs of orbitals p and q such that p is occupied in the beta string and unoccupied in the alpha
/// string, and q is occupied in the alpha string and unoccupied in the beta string. The function
/// assumes that the determinants are properly normalized and that the orbital indices are within
/// bounds. The function returns 0 if the determinants have different spin multiplicities (i.e.
/// different numbers of alpha and beta electrons), since in that case the S^2 matrix element is
/// zero. The function does not check for other types of invalid input, such as determinants that
/// differ by more than a double excitation, so the caller is responsible for ensuring that the
/// input determinants are valid for the intended use case.
/// @tparam N The number of spin-orbital slots in the determinants, which must be a multiple of 128
/// @param lhs the left-hand side determinant
/// @param rhs the right-hand side determinant
/// @return the matrix element of the S^2 operator between the two determinants
double spin2(const Determinant& lhs, const Determinant& rhs);

/// @brief Describe the excitation connection of a determinant d, relative to this one. The
/// excitation connection is defined as the creation and annihilation operators that need to be
/// applied to this determinant to obtain d. The excitation connection is a vector of 4 vectors:
/// [[alpha annihilation], [alpha creation], [beta annihilation], [beta creation]]
/// @param d the determinant to compare to
/// @return the excitation connection of d relative to this determinant
std::vector<std::vector<size_t>> excitation_connection(const Determinant lhs,
                                                       const Determinant rhs);

} // namespace forte2
