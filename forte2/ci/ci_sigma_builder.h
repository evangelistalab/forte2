#pragma once

#include <functional>
#include <vector>
#include <cmath>
#include <span>

#include "helpers/ndarray.h"

#include "ci/ci_string_lists.h"
#include "ci/slater_rules.h"
#include "ci/ci_spin_adapter.h"

namespace forte2 {

enum class CIAlgorithm {
    Knowles_Handy,     // Knowles-Handy algorithm
    Harrison_Zarrabian // Harrison-Zarrabian algorithm
};

class CISigmaBuilder {
  public:
    // == Class Constructor ==
    CISigmaBuilder(const CIStrings& lists, double E, np_matrix& H, np_tensor4& V,
                   int log_level = 3);
    ~CISigmaBuilder();

    // == Class Public Functions ==

    /// @brief Set the memory size for temporary buffers in bytes. Use before calling Hamiltonian().
    /// @param mb Memory size in megabytes (default is 1 GB)
    void set_memory(int mb);

    /// @brief Set the CI algorithm to use for building the Hamiltonian
    /// @param algorithm The CI algorithm to use (default is "knowles-handy")
    /// Supported algorithms: "kh", "hz", "knowles-handy", "harrison-zarrabian"
    void set_algorithm(const std::string& algorithm);

    /// @brief Set the one and two-electron integrals for the Hamiltonian
    void set_Hamiltonian(double E, np_matrix H, np_tensor4 V);

    /// @brief Set the logging level for the class
    void set_log_level(int level) { log_level_ = level; }

    /// @brief Form the diagonal of the Hamiltonian matrix in the CI basis
    /// @return The diagonal elements of the Hamiltonian matrix
    np_vector form_Hdiag_csf(const std::vector<Determinant>& dets,
                             const CISpinAdapter& spin_adapter,
                             bool spin_adapt_full_preconditioner) const;

    /// @brief Compute the Slater rules for the CSF matrix element
    /// @param dets The list of determinants
    /// @param spin_adapter The spin adapter for the CSF
    /// @param I The index of the first CSF
    /// @param J The index of the second CSF
    /// @return The matrix element <I|H|J> in the CSF basis
    double slater_rules_csf(const std::vector<Determinant>& dets, const CISpinAdapter& spin_adapter,
                            size_t I, size_t J) const;

    /// @brief Apply the Hamiltonian to the wave function
    /// @param basis The basis vector
    /// @param sigma The resulting sigma vector |sigma> = H |basis>
    void Hamiltonian(np_vector basis, np_vector sigma) const;

    /// @brief Return the average build time for the Hamiltonian components
    std::vector<double> avg_build_time() const {
        if (build_count_ == 0) {
            return {0.0, 0.0, 0.0, 0.0};
        } else {
            return {hdiag_timer_ / static_cast<double>(build_count_),
                    haabb_timer_ / static_cast<double>(build_count_),
                    haaaa_timer_ / static_cast<double>(build_count_),
                    hbbbb_timer_ / static_cast<double>(build_count_)};
        }
    }

    /// @brief Compute the one-electron reduced density matrix
    /// @param C_left The left-hand side coefficients
    /// @param C_right The right-hand side coefficients
    /// @param alfa If true, compute the alpha contribution, otherwise the beta
    /// @return The one-electron reduced density matrix stored as
    ///        gamma(sigma)[p][q] = <L| a^+_p a_q |R> with p,q orbitals of spin sigma
    np_matrix compute_1rdm_same_irrep(np_vector C_left, np_vector C_right, bool alfa);

    /// @brief Compute the spin-free one-electron reduced density matrix
    /// @param C_left The left-hand side coefficients
    /// @param C_right The right-hand side coefficients
    /// @return The spin-free one-electron reduced density matrix stored as
    ///        Gamma[p][q] = gamma(alpha)[p][q] + gamma(beta)[p][q]
    np_matrix compute_sf_1rdm_same_irrep(np_vector C_left, np_vector C_right);

    /// @brief Compute the two-electron same-spin reduced density matrix
    /// @param C_left The left-hand side coefficients
    /// @param C_right The right-hand side coefficients
    /// @param alfa If true, compute the alpha contribution, otherwise the beta
    /// @return The two-electron same-spin reduced density matrix stored as a matrix
    ///        gamma(sigma)[p][q][r][s] = <L| a^+_p a^+_q a_s a_r |R>
    ///        with p > q, and r > s orbitals of spin sigma
    np_matrix compute_2rdm_aa_same_irrep(np_vector C_left, np_vector C_right, bool alfa) const;

    /// @brief Compute the two-electron same-spin two-electron reduced density matrix
    /// @param C_left The left-hand side coefficients
    /// @param C_right The right-hand side coefficients
    /// @param alfa If true, compute the alpha contribution, otherwise the beta
    /// @return The two-electron same-spin reduced density matrix stored as a tensor
    ///        gamma[p][q][r][s] = <L| a^+_p a^+_q a_s a_r |R>
    np_tensor4 compute_2rdm_aa_same_irrep_full(np_vector C_left, np_vector C_right,
                                               bool alfa) const;

    /// @brief Compute the mixed-spin two-electron reduced density matrix
    /// @param C_left The left-hand side coefficients
    /// @param C_right The right-hand side coefficients
    /// @return The two-electron mixed-spin reduced density matrix stored as a tensor
    ///        gamma[p][q][r][s] = <L| a^+_p a^+_q a_s a_r |R>
    ///        with p,r orbitals of spin alpha and q,s orbitals of spin beta
    np_tensor4 compute_2rdm_ab_same_irrep(np_vector C_left, np_vector C_right);

    /// @brief Compute the spin-free two-electron reduced density matrix
    /// @param C_left The left-hand side coefficients
    /// @param C_right The right-hand side coefficients
    /// @return The two-electron spin-free reduced density matrix stored as a tensor
    ///        gamma[p][q][r][s] = gamma(alpha)[p][q][r][s] +
    ///                            gamma(beta)[p][q][r][s] +
    ///                            gamma[p][q][r][s]
    ///                            gamma[q][p][s][r]
    ///        with p,r orbitals of spin alpha and q,s orbitals of spin beta
    np_tensor4 compute_sf_2rdm_same_irrep(np_vector C_left, np_vector C_right);

  private:
    // == Class Private Variables ==

    /// @brief The CI algorithm to use for building the Hamiltonian
    CIAlgorithm algorithm_ = CIAlgorithm::Harrison_Zarrabian; // Default to Knowles-Handy algorithm
    /// @brief The CIStrings object containing the determinant classes and their properties
    const CIStrings& lists_;
    /// @brief The scalar energy
    double E_;
    /// @brief One-electron integrals in the form of a matrix H[p][q] = <p|H|q> = h_pq
    np_matrix H_;
    /// @brief Two-electron integrals in the form of a tensor V[p][q][r][s] = <pq|rs> = (pr|qs)
    np_tensor4 V_;
    /// @brief Object for computing the energy and Slater determinants
    SlaterRules slater_rules_;
    /// @brief Memory size for temporary buffers in bytes (default 1 GB)
    size_t memory_size_ = 1073741824;
    /// @brief logging level for the class
    int log_level_ = 3;

    mutable double hdiag_timer_ = 0.0;
    mutable double haaaa_timer_ = 0.0;
    mutable double haabb_timer_ = 0.0;
    mutable double hbbbb_timer_ = 0.0;
    mutable double rdm1_timer_ = 0.0;
    mutable double rdm2_aa_timer_ = 0.0;
    mutable double rdm2_ab_timer_ = 0.0;
    mutable int build_count_ = 0;

    // == Class Private Functions/Data ==

    /// @brief Temporary vectors used for gathering and scattering blocks of the CI matrix
    /// These vectors are allocated when the class is constructed and resized as needed
    mutable std::vector<double> TR;
    mutable std::vector<double> TL;

    /// @brief Scalar contribution to the sigma vector |sigma> = E |basis>
    void H0(std::span<double> basis, std::span<double> sigma) const;

    // -- Harrison-Zarrabian Algorithm Functions/Data ---

    /// @brief One-electron integrals: H[p][q] = <p|H|q> = h_pq
    mutable std::vector<double> h_hz;
    /// @brief Two-electron integrals: V[p][q][r][s] = <pq|rs> = (pr|qs)
    mutable std::vector<double> v_pr_qs;
    /// @brief Two-electron integrals: V[p][q][r][s] = <pq||rs> = (pr|qs) - (ps|qr)
    mutable std::vector<double> v_pr_qs_a;

    /// @brief  One-electron contribution to the sigma vector |sigma> = H |basis>
    /// @param alfa If true, compute the alpha contribution, otherwise the beta
    /// @param h The one-electron integrals
    void H1_aa_gemm(std::span<double> basis, std::span<double> sigma, bool alfa,
                    std::span<double> h) const;

    /// @brief  Two-electron same-spin contribution to the sigma vector |sigma> = H |basis>
    /// @param alfa If true, compute the alpha contribution, otherwise the beta
    void H2_aaaa_gemm(std::span<double> basis, std::span<double> sigma, bool alfa) const;

    /// @brief  Two-electron mixed-spin contribution to the sigma vector |sigma> = H |basis>
    /// @param basis The basis vector
    /// @param sigma The resulting sigma vector
    void H2_aabb_gemm(std::span<double> basis, std::span<double> sigma) const;

    // -- Knowles-Handy Algorithm Functions/Data --
    // Modified one-electron integrals used in the Knowles-Handy algorithm
    mutable std::vector<double> h_kh;
    // Modified two-electron integrals used in the Knowles-Handy algorithm
    mutable std::vector<double> v_ijkl_hk;
    /// @brief Temporary vectors used for the Knowles-Handy algorithm
    /// These vectors are allocated on the first call to the Hamiltonian function
    /// and resized as needed
    mutable std::vector<double> Kblock1_;
    mutable std::vector<double> Kblock2_;

    /// @brief Builds the one-electron contribution to the sigma vector using the Knowles-Handy
    /// algorithm.
    void H1_kh(std::span<double> basis, std::span<double> sigma, bool alpha) const;

    /// @brief Builds the two-electron contribution to the sigma vector using the Knowles-Handy
    /// algorithm.
    void H2_kh(std::span<double> basis, std::span<double> sigma) const;

    std::tuple<std::span<double>, std::span<double>, size_t> get_Kblock_spans(size_t dim,
                                                                              size_t maxKa) const;
};

[[nodiscard]] std::span<double> gather_block(std::span<double> source, std::span<double> dest,
                                             bool alfa, const CIStrings& lists, int class_Ia,
                                             int class_Ib);

void zero_block(std::span<double> dest, bool alfa, const CIStrings& lists, int class_Ia,
                int class_Ib);

void scatter_block(std::span<double> source, std::span<double> dest, bool alfa,
                   const CIStrings& lists, int class_Ia, int class_Ib);

} // namespace forte2
