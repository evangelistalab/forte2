#pragma once

#include <functional>
#include <vector>
#include <cmath>
#include <span>
#include <complex>

#include "helpers/ndarray.h"
#include "helpers/memory.h"

#include "ci/ci_strings.h"
#include "ci/slater_rules.h"
#include "ci/ci_sigma_builder.h"

namespace forte2 {

class RelCISigmaBuilder {
  public:
    // == Class Constructor ==
    RelCISigmaBuilder(const CIStrings& lists, double E, np_matrix_complex& H, np_tensor4_complex& V,
                      int log_level = 3);

    // == Class Public Functions ==

    /// @brief Set the memory size for temporary buffers in bytes. Use before calling Hamiltonian().
    /// @param mb Memory size in megabytes (default is 1 GB)
    void set_memory(int mb);

    /// @brief Set the CI algorithm to use for building the Hamiltonian
    /// @param algorithm The CI algorithm to use (default is "knowles-handy")
    /// Supported algorithms: "kh", "hz", "knowles-handy", "harrison-zarrabian"
    void set_algorithm(const std::string& algorithm);

    /// @brief Get the name of the current sigma build algorithm
    /// @return The name of the current sigma build algorithm
    std::string get_algorithm() const;

    /// @brief Set the one and two-electron integrals for the Hamiltonian
    void set_Hamiltonian(double E, np_matrix_complex H, np_tensor4_complex V);

    /// @brief Set the logging level for the class
    void set_log_level(int level) { log_level_ = level; }

    /// @brief Form the diagonal of the Hamiltonian matrix in the CI basis
    /// @param dets The list of determinants
    /// @param spin_adapter The spin adapter for the CSF
    /// @param spin_adapt_full_preconditioner If true, use the exact diagonal elements,
    ///        otherwise use approximate diagonal elements.
    /// @return The diagonal elements of the Hamiltonian matrix
    np_vector_complex form_Hdiag(const std::vector<Determinant>& dets) const;

    /// @brief Compute the Hamiltonian matrix element between two determinants
    /// @param dets The list of determinants
    /// @param I The index of the first determinant
    /// @param J The index of the second determinant
    /// @return The Hamiltonian matrix element <I|H|J>
    std::complex<double> slater_rules(const std::vector<Determinant>& dets, size_t I,
                                      size_t J) const;

    /// @brief Apply the Hamiltonian to the wave function
    /// @param basis The basis vector
    /// @param sigma The resulting sigma vector |sigma> = H |basis>
    void Hamiltonian(np_vector_complex basis, np_vector_complex sigma) const;

    /// @brief Compute the spin-dependent one-electron reduced density matrix
    /// @param C_left The left-hand side coefficients
    /// @param C_right The right-hand side coefficients
    /// @param spin The spin component to compute
    /// @return The one-electron reduced density matrix stored as
    ///        gamma(spin)[p][q] = <L| a^+_p a_q |R> with p,q orbitals of spin alpha/beta
    /// @note If the number of orbitals is 0, a matrix of shape (0, 0) is returned
    np_matrix_complex compute_1rdm(np_vector_complex C_left, np_vector_complex C_right) const;

    /// @brief Compute the same-spin two-electron reduced density matrix
    /// @param C_left The left-hand side coefficients
    /// @param C_right The right-hand side coefficients
    /// @param spin The spin component to compute
    /// @return The two-electron same-spin reduced density matrix stored as a matrix
    ///        gamma(sigma)[p>q][r>s] = <L| a^+_p a^+_q a_s a_r |R>
    ///        with p > q, and r > s orbitals of spin sigma
    np_tensor4_complex compute_2rdm(np_vector_complex C_left, np_vector_complex C_right) const;

    /// @brief Compute the cumulant of the spin-free two-electron reduced density matrix
    /// @param C_left The left-hand side coefficients
    /// @param C_right The right-hand side coefficients
    /// @return The cumulant of the two-electron spin-free reduced density matrix stored as a tensor
    ///        lambda[p][q][r][s] = gamma[p][q][r][s] - gamma[p][r] * gamma[q][s] +
    ///                              0.5 * gamma[p][s] * gamma[q][r]
    np_tensor4_complex compute_2cumulant(np_vector_complex C_left, np_vector_complex C_right) const;

    /// @brief Compute the three-electron same-spin reduced density matrix
    /// @param C_left The left-hand side coefficients
    /// @param C_right The right-hand side coefficients
    /// @param spin The spin component to compute
    /// @return The three-electron same-spin reduced density matrix stored as a matrix
    ///        gamma(sigma)[p>q>r][s>t>u] = <L| a^+_p a^+_q a^+_r a_u a_t a_s |R>
    ///        with p > q > r, and s > t > u orbitals of spin sigma
    np_tensor6_complex compute_3rdm(np_vector_complex C_left, np_vector_complex C_right) const;

    /// @brief Compute the cumulant of the spin-free three-electron reduced density matrix
    /// @param C_left The left-hand side coefficients
    /// @param C_right The right-hand side coefficients
    /// @return The cumulant of the three-electron spin-free reduced density matrix stored as a
    /// tensor
    ///        lambda[p][q][r][s][t][u] = gamma[p][q][r][s][t][u] + ...
    np_tensor6_complex compute_3cumulant(np_vector_complex C_left, np_vector_complex C_right) const;

    np_matrix_complex compute_1rdm_debug(np_vector_complex C_left, np_vector_complex C_right) const;
    np_tensor4_complex compute_2rdm_debug(np_vector_complex C_left,
                                          np_vector_complex C_right) const;
    np_tensor6_complex compute_3rdm_debug(np_vector_complex C_left,
                                          np_vector_complex C_right) const;

  private:
    // == Class Private Variables ==

    /// @brief The CI algorithm to use for building the Hamiltonian
    CIAlgorithm algorithm_ = CIAlgorithm::Knowles_Handy; // Default to Knowles-Handy algorithm
    /// @brief The CIStrings object containing the determinant classes and their properties
    const CIStrings& lists_;
    /// @brief The scalar energy
    double E_;
    /// @brief One-electron integrals in the form of a matrix H[p][q] = <p|H|q> = h_pq
    np_matrix_complex H_;
    /// @brief Two-electron integrals in the form of a tensor V[p][q][r][s] = <pq|rs> = (pr|qs)
    np_tensor4_complex V_;
    /// @brief Object for computing the energy and Slater determinants
    RelSlaterRules rel_slater_rules_;
    /// @brief Memory size for temporary buffers in bytes (default 1 GB)
    size_t memory_size_ = 1073741824;
    /// @brief logging level for the class
    int log_level_ = 3;

    // == Class Private Functions/Data ==

    /// @brief Temporary vectors used for gathering and scattering blocks of the CI matrix
    /// These vectors are allocated when the class is constructed and resized as needed
    mutable std::vector<std::complex<double>> TR;
    mutable std::vector<std::complex<double>> TL;

    /// @brief Temporary vectors used to store blocks of data of the form
    /// L(op,K,L) = <K|op|I> C_{IL}
    /// where `op` is an operator, K is a state in N, N-1, or N-2 electron strings
    /// while I and L are alpha/beta string
    /// These vectors are allocated on the first call to the Hamiltonian function
    /// and resized as needed
    mutable std::vector<std::complex<double>> Kblock1_;
    mutable std::vector<std::complex<double>> Kblock2_;

    /// @brief Scalar contribution to the sigma vector |sigma> = E |basis>
    void H0(std::span<std::complex<double>> basis, std::span<std::complex<double>> sigma) const;

    // -- Harrison-Zarrabian Algorithm Functions/Data ---

    /// @brief One-electron integrals: H[p][q] = <p|H|q> = h_pq
    mutable std::vector<std::complex<double>> h_hz;
    /// @brief Two-electron integrals: V[p][q][r][s] = <pq||rs> = (pr|qs) - (ps|qr)
    mutable std::vector<std::complex<double>> v_pr_qs;

    /// @brief  One-electron contribution to the sigma vector |sigma> = H |basis>
    /// @param alfa If true, compute the alpha contribution, otherwise the beta
    /// @param h The one-electron integrals
    void H1_hz(std::span<std::complex<double>> basis, std::span<std::complex<double>> sigma,
               Spin spin, std::span<std::complex<double>> h) const;

    /// @brief  Two-electron same-spin contribution to the sigma vector |sigma> = H |basis>
    /// @param alfa If true, compute the alpha contribution, otherwise the beta
    void H2_hz_same_spin(std::span<std::complex<double>> basis,
                         std::span<std::complex<double>> sigma, Spin spin) const;

    /// @brief  Two-electron mixed-spin contribution to the sigma vector |sigma> = H |basis>
    /// @param basis The basis vector
    /// @param sigma The resulting sigma vector
    void H2_hz_opposite_spin(std::span<std::complex<double>> basis,
                             std::span<std::complex<double>> sigma) const;

    // -- Knowles-Handy Algorithm Functions/Data --

    // Modified one-electron integrals used in the Knowles-Handy algorithm
    mutable std::vector<std::complex<double>> h_kh;
    // Modified two-electron integrals used in the Knowles-Handy algorithm
    // mutable std::vector<std::complex<double>> v_ijkl_hk;

    /// @brief Builds the one-electron contribution to the sigma vector using the Knowles-Handy
    /// algorithm.
    void H1_kh(std::span<std::complex<double>> basis, std::span<std::complex<double>> sigma) const;

    /// @brief Builds the two-electron contribution to the sigma vector using the Knowles-Handy
    /// algorithm.
    void H2_kh(std::span<std::complex<double>> basis, std::span<std::complex<double>> sigma) const;

    std::tuple<std::span<std::complex<double>>, std::span<std::complex<double>>, size_t>
    get_Kblock_spans(size_t dim, size_t maxKa) const;
};

[[nodiscard]] std::span<std::complex<double>> gather_block(std::span<std::complex<double>> source,
                                                           std::span<std::complex<double>> dest,
                                                           Spin spin, const CIStrings& lists,
                                                           int class_Ia, int class_Ib);

void zero_block(std::span<std::complex<double>> dest, Spin spin, const CIStrings& lists,
                int class_Ia, int class_Ib);

void scatter_block(std::span<std::complex<double>> source, std::span<std::complex<double>> dest,
                   Spin spin, const CIStrings& lists, int class_Ia, int class_Ib);

} // namespace forte2
