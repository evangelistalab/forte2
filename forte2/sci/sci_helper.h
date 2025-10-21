#pragma once

#include <functional>
#include <vector>
#include <cmath>
#include <span>

#include "helpers/ndarray.h"

#include "ci/determinant.h"
#include "ci/slater_rules.h"

#include "sci/sci_strings.h"

namespace forte2 {

/// @brief A hash map from determinants to double values
using DetMap = ankerl::unordered_dense::map<Determinant, double, Determinant::Hash>;

/// @brief Screening criteria for selected CI
enum class ScreeningCriterion { HBCI, eHBCI };

/// @brief Energy correction method for selected CI
enum class EnergyCorrection { PT2, Variational };

/// @brief A helper class for selected CI methods such as CIPSI and HBCI
/// This class manages selection and sigma vector computation for selected CI methods.
class SelectedCIHelper {
  public:
    // == Class Constructor ==
    /// @brief Construct a SelectedCIHelper object
    /// @param norb Number of orbitals
    /// @param dets The initial determinants in the variational space
    /// @param c The initial CI coefficients for the determinants (shape: (n_dets, n_roots))
    /// @param E Hamiltonian scalar energy
    /// @param H One-electron integrals (shape: (n_orb, n_orb))
    /// @param V Two-electron integrals in physicist notation (shape: (n_orb, n_orb, n_orb, n_orb))
    /// @param log_level Logging level
    SelectedCIHelper(size_t norb, const std::vector<Determinant>& dets, np_matrix& c, double E,
                     np_matrix& H, np_tensor4& V, int log_level = 3);

    // == Class Public Methods ==

    /// @brief Return the determinants in the variational space
    const std::vector<Determinant>& variational_dets() const { return dets_; }

    /// @brief Set the Hamiltonian integrals
    /// @param E Hamiltonian scalar energy
    /// @param H One-electron integrals
    /// @param V Two-electron integrals in physicist notation
    void set_Hamiltonian(double E, np_matrix H, np_tensor4 V);

    /// @brief Set the CI coefficients
    /// @param c The CI coefficients for the determinants (shape: (n_dets, n_roots))
    void set_c(np_matrix& c);

    /// @brief Set the energy of the roots (as computed externally by the Davidson solver)
    void set_energies(np_vector e);

    /// @brief Set the number of threads to use for parallelization
    void set_num_threads(size_t n) { num_threads_ = n; }

    /// @brief Set the number of batches per thread for parallelization
    void set_num_batches_per_thread(size_t n) { num_batches_per_thread_ = n; }

    /// @brief Set the orbital that should be excluded from the list of creation operators
    void set_frozen_creation(const std::vector<size_t>& frozen_creation);

    /// @brief Set the screening criterion
    void set_screening_criterion(std::string criterion);

    /// @return The energies of the roots
    const std::vector<double>& energies() const { return root_energies_; }

    /// @return The PT2 energy contributions of the roots due to the new variational determinants
    const std::vector<double>& ept2_var() const { return ept2_var_; }

    /// @return The PT2 energy contributions of the roots due to the determinat excluded
    const std::vector<double>& ept2_pt() const { return ept2_pt_; }

    /// @return The number of determinants in the variational space
    const size_t num_dets_var() const { return dets_.size(); }

    /// @return The number new variational determinants added in the last selection step
    size_t num_new_dets_var() const { return num_new_dets_var_; }

    /// @return The number new PT2 determinants added in the last selection step
    size_t num_new_dets_pt2() const { return num_new_dets_pt2_; }

    /// @return The time taken for the last selection step
    double selection_time() const { return selection_time_; }

    /// @brief Perform HBCI selection with a reference implementation
    /// @param var_threshold The threshold for variational selection
    /// @param pt2_threshold The threshold for PT2 selection
    void select_hbci_ref(double var_threshold, double pt2_threshold);

    /// @brief Perform HBCI selection with an optimized batch implementation
    /// @param var_threshold The threshold for variational selection
    /// @param pt2_threshold The threshold for PT2 selection
    void select_hbci(double var_threshold, double pt2_threshold);

    /// @brief Compute the diagonal of the Hamiltonian matrix
    /// @return A vector of the diagonal elements of the Hamiltonian matrix
    np_vector Hdiag() const;

    /// @brief Apply the Hamiltonian to a given basis and sigma vectors
    /// @param basis The coefficients of the determinants
    /// @param sigma The coefficients of the sigma vector
    void Hamiltonian(np_vector basis, np_vector sigma) const;

    /// @brief Compute the expectation value of S^2 for each root
    /// @return A vector of <S^2> values for each root
    std::vector<double> compute_spin2() const;

    /// @brief Compute the alpha one-electron reduced density matrix
    /// @param left_root The left-hand side root index
    /// @param right_root The right-hand side root index
    /// @return The one-electron reduced density matrix stored as
    ///        gamma(alpha)[p][q] = <L| a^+_p a_q |R> with p,q orbitals of spin alpha
    /// @note If the number of orbitals is 0, a matrix of shape (0, 0) is returned
    np_matrix compute_a_1rdm(size_t left_root, size_t right_root) const;

    /// @brief Compute the beta one-electron reduced density matrix
    /// @param left_root The left-hand side root index
    /// @param right_root The right-hand side root index
    /// @return The one-electron reduced density matrix stored as
    ///        gamma(beta)[p][q] = <L| b^+_p b_q |R> with p,q orbitals of spin beta
    /// @note If the number of orbitals is 0, a matrix of shape (0, 0) is returned
    np_matrix compute_b_1rdm(size_t left_root, size_t right_root) const;

    /// @brief Compute the spin-free one-electron reduced density matrix
    /// @param left_root The left-hand side root index
    /// @param right_root The right-hand side root index
    /// @return The spin-free one-electron reduced density matrix stored as
    ///        Gamma[p][q] = gamma(alpha)[p][q] + gamma(beta)[p][q]
    np_matrix compute_sf_1rdm(size_t left_root, size_t right_root) const;

  private:
    // == Class Private Methods ==
    /// @brief Compute the energies of all determinants in the variational space
    void compute_det_energies();

    /// @brief Prepare the string lists for fast Hamiltonian application
    void prepare_strings();

    /// @brief Update the orbital energies, one- and two-electron integrals
    void update_orbital_energies();

    /// @brief Update the sorted two-electron integrals for fast Hamiltonian application
    void update_hbci_ints();

    /// @brief Compute the energy contribution for a given determinant
    double compute_delta_ept2(double delta, double v) const;

    /// @brief Apply the Hamiltonian operator H0, H1a, H1b, H2a, H2b, H2ab to the basis and
    /// accumulate the result in sigma
    /// @param basis
    /// @param sigma
    void H0(std::span<double> basis, std::span<double> sigma) const;
    void H1a(std::span<double> basis, std::span<double> sigma) const;
    void H1b(std::span<double> basis, std::span<double> sigma) const;
    void H2a(std::span<double> basis, std::span<double> sigma) const;
    void H2b(std::span<double> basis, std::span<double> sigma) const;
    void H2ab(std::span<double> basis, std::span<double> sigma) const;

    /// @brief Find matching determinants for the given excitation and accumulate their
    /// contributions to the sigma vector
    void find_matching_dets(std::span<double> basis, std::span<double> sigma,
                            const SelectedCIStrings& list, size_t i, size_t j,
                            double int_sign) const;

    /// @brief Select new variational and PT2 determinants using a batch approach
    /// @param var_threshold The threshold for variational selection
    /// @param pt2_threshold The threshold for PT2 selection
    /// @param num_batches The total number of batches
    /// @param batch_id The batch index to process
    std::pair<std::vector<DetMap>, std::vector<DetMap>> select_hbci_batch(double var_threshold,
                                                                          double pt2_threshold,
                                                                          size_t num_batches,
                                                                          size_t batch_id);

    /// @brief Compute the expectation value of S^2 for a given batch of determinants
    /// @param num_batches The total number of batches
    /// @param batch_id The batch index to process
    std::vector<double> spin2_batch(size_t num_batches, size_t batch_id) const;

    /// @brief Find matching determinants for the given excitation and compute their
    /// contributions to the one-electron reduced density matrix
    double find_matching_dets_1rdm(size_t left_root, size_t right_root,
                                   const SelectedCIStrings& list, size_t i, size_t j,
                                   double sign) const;

    // == Class Private Variables ==

    /// @brief Threshold for integral screening
    static constexpr double integral_threshold = 1e-12;

    /// @brief Screening criterion for selected CI
    ScreeningCriterion screening_criterion_ = ScreeningCriterion::HBCI; // Default to HBCI screening

    /// @brief Energy correction method for selected CI
    EnergyCorrection energy_correction_ = EnergyCorrection::PT2; // Default to PT2 correction

    /// @brief logging level for the class
    int log_level_ = 3;

    /// @brief Number of threads to use for parallelization
    size_t num_threads_ = 1;

    /// @brief Number of batches per thread for parallelization
    size_t num_batches_per_thread_ = 1;

    /// @brief Number of orbitals
    const size_t norb_;

    /// @brief Number of spatial orbitals squared
    const size_t norb2_;
    /// @brief Number of spatial orbitals cubed
    const size_t norb3_;

    /// @brief Number of alpha electrons
    size_t na_;
    /// @brief Number of beta electrons
    size_t nb_;

    /// @brief The scalar energy
    double E_;
    /// @brief One-electron integrals in the form of a matrix H[p][q] = <p|H|q> = h_pq
    np_matrix H_;
    /// @brief Two-electron integrals in the form of a tensor V[p][q][r][s] = <pq|rs> = (pr|qs)
    np_tensor4 V_;

    /// @brief The Slater rules for the current set of determinants
    SlaterRules slater_rules_;

    /// @brief Do frozen creation
    /// @brief The mask that controls frozen creation
    String frozen_creation_mask_;

    /// @brief Orbital energies: e[p] = <p|H|p>
    std::vector<double> epsilon_;
    /// @brief One-electron integrals: H[p][q] = <p|H|q> = h_pq
    std::vector<double> h_;
    /// @brief Two-electron integrals: V[p][q][r][s] = <pq|rs> = (pr|qs)
    std::vector<double> v_;
    /// @brief Two-electron integrals: V[p][q][r][s] = <pq||rs> = (pr|qs) - (ps|qr)
    std::vector<double> v_a_;

    /// @brief Sorted two-electron integrals for fast HBCI selection (two hole indices)
    std::vector<std::vector<std::tuple<double, double, u_int32_t, u_int32_t>>> v_sorted_;

    /// @brief Sorted antisymmetrized two-electron integrals for fast HBCI selection (two hole
    /// indices)
    std::vector<std::vector<std::tuple<double, double, u_int32_t, u_int32_t>>> va_sorted_;

    /// @brief Sorted antisymmetrized two-electron integrals for HBCI selection (hole-particle
    /// indices)
    std::vector<std::vector<std::tuple<double, double, u_int32_t, u_int32_t>>> vab_sorted_;

    /// @return Return the one-electron integral <i|h|j>
    inline const double& h(std::size_t i, std::size_t j) const noexcept {
        return h_[i * norb_ + j];
    }

    /// @return Return the two-electron integral <pq|rs>
    inline const double& V(std::size_t p, std::size_t q, std::size_t r,
                           std::size_t s) const noexcept {
        return v_[p * norb3_ + q * norb2_ + r * norb_ + s];
    }

    /// @return Return the antisymmetrized two-electron integral <pq||rs>
    inline const double& Va(std::size_t p, std::size_t q, std::size_t r,
                            std::size_t s) const noexcept {
        return v_a_[p * norb3_ + q * norb2_ + r * norb_ + s];
    }

    /// @brief The number of roots to compute
    size_t nroots_;
    /// @brief The initial guess for the CI coefficients
    np_matrix c_guess_;
    /// @brief The initial guess for the determinants
    std::vector<Determinant> guess_dets_;

    /// @brief The variational determinants
    std::vector<Determinant> dets_;
    /// @brief The energies of the determinants in the variational space
    std::vector<double> det_energies_;
    /// @brief The CI coefficients of the determinants in the variational space
    /// Stored as a flat vector of size dets_.size() * nroots_, where the coefficients for each
    /// root are stored contiguously. E.g., the coefficient for determinant i and root r is at
    /// index i * nroots_ + r.
    std::vector<double> c_;

    /// @brief The alpha and beta strings for the determinants in the variational space
    SelectedCIStrings ab_list_;
    /// @brief The beta and alpha strings for the determinants in the variational space
    SelectedCIStrings ba_list_;

    /// @brief The energies of the roots
    std::vector<double> root_energies_;
    /// @brief The perturbative energy contributionsdue to the new variational determinants
    std::vector<double> ept2_var_;
    /// @brief The perturbative energy contributions due to the determinants excluded
    std::vector<double> ept2_pt_;

    /// @brief The number of new variational determinants added in the last selection step
    size_t num_new_dets_var_ = 0;
    /// @brief The number of new PT2 determinants added in the last selection step
    size_t num_new_dets_pt2_ = 0;

    /// @brief Time taken for the last selection step
    double selection_time_ = 0.0;
};

} // namespace forte2
