#pragma once

#include <functional>
#include <vector>
#include <cmath>
#include <span>

#include "helpers/ndarray.h"
#include "determinant.h"

#include "ci/slater_rules.h"
#include "ci/sci_strings.h"

namespace forte2 {

class SelectedCIHelper {
  public:
    // == Class Constructor ==
    SelectedCIHelper(size_t norb, const std::vector<Determinant>& dets, np_matrix& c, double E,
                     np_matrix& H, np_tensor4& V, int log_level = 3);

    // == Class Public Methods ==

    /// @brief Return the determinants in the variational space
    const std::vector<Determinant>& get_variational_dets() const { return dets_; }

    /// @brief Set the Hamiltonian integrals
    void set_Hamiltonian(double E, np_matrix H, np_tensor4 V);

    void set_c(np_matrix& c);

    /// @brief Perform CIPSI selection with the given threshold
    void select_cipsi(double threshold);

    /// @brief Perform HBCI selection with the given threshold
    void select_hbci(double threshold);

    /// @brief Perform HBCI selection with the given threshold
    void select_hbci2(double threshold);

    void Hamiltonian(np_vector basis, np_vector sigma) const;

    np_vector Hdiag() const;

    np_matrix compute_sf_1rdm(size_t left_root, size_t right_root) const;

    np_matrix compute_a_1rdm(size_t left_root, size_t right_root) const;
    np_matrix compute_b_1rdm(size_t left_root, size_t right_root) const;

  private:
    // == Class Private Methods ==
    /// @brief Compute the energies of all determinants in the variational space
    void compute_det_energies();
    /// @brief Prepare the string lists for fast Hamiltonian application
    void prepare_strings();
    void update_hbci_ints();

    void H0(std::span<double> basis, std::span<double> sigma) const;
    void H1a(std::span<double> basis, std::span<double> sigma) const;
    void H1b(std::span<double> basis, std::span<double> sigma) const;
    void H2a(std::span<double> basis, std::span<double> sigma) const;
    void H2b(std::span<double> basis, std::span<double> sigma) const;
    void H2ab(std::span<double> basis, std::span<double> sigma) const;
    void find_matching_dets(std::span<double> basis, std::span<double> sigma,
                            const SelectedCIStrings& list, size_t i, size_t j,
                            double int_sign) const;

    double find_matching_dets_1rdm(size_t left_root, size_t right_root,
                                   const SelectedCIStrings& list, size_t i, size_t j,
                                   double sign) const;

    // == Class Private Variables ==

    static constexpr double integral_threshold = 1e-12;

    /// @brief logging level for the class
    int log_level_ = 3;

    /// @brief Number of orbitals
    const size_t norb_;
    const size_t norb2_;
    const size_t norb3_;
    size_t na_;
    size_t nb_;

    /// @brief The scalar energy
    double E_;
    /// @brief One-electron integrals in the form of a matrix H[p][q] = <p|H|q> = h_pq
    np_matrix H_;
    /// @brief Two-electron integrals in the form of a tensor V[p][q][r][s] = <pq|rs> = (pr|qs)
    np_tensor4 V_;

    SlaterRules slater_rules_;

    /// @brief Orbital energies: e[p] = <p|H|p>
    std::vector<double> epsilon_;
    /// @brief One-electron integrals: H[p][q] = <p|H|q> = h_pq
    std::vector<double> h_;
    /// @brief Two-electron integrals: V[p][q][r][s] = <pq|rs> = (pr|qs)
    std::vector<double> v_;
    /// @brief Two-electron integrals: V[p][q][r][s] = <pq||rs> = (pr|qs) - (ps|qr)
    std::vector<double> v_a_;

    std::vector<std::vector<std::tuple<double, size_t, size_t>>> v_sorted_;
    std::vector<std::vector<std::tuple<double, size_t, size_t>>> va_sorted_;

    inline const double& h(std::size_t i, std::size_t j) const noexcept {
        return h_[i * norb_ + j];
    }

    inline const double& V(std::size_t i, std::size_t j, std::size_t a,
                           std::size_t b) const noexcept {
        return v_[i * norb3_ + j * norb2_ + a * norb_ + b];
    }

    inline const double& Va(std::size_t i, std::size_t j, std::size_t a,
                            std::size_t b) const noexcept {
        return v_a_[i * norb3_ + j * norb2_ + a * norb_ + b];
    }

    /// @brief The determinants in the reference space
    size_t nroots_;
    /// @brief The initial guess for the CI coefficients
    np_matrix c_guess_;
    /// @brief The initial guess for the determinants
    std::vector<Determinant> guess_dets_;

    /// @brief The determinants in the variational space
    std::vector<Determinant> dets_;
    /// @brief The energies of the determinants in the variational space
    std::vector<double> det_energies_;
    /// @brief The CI coefficients of the determinants in the variational space
    /// Stored as a flat vector of size dets_.size() * nroots_, where the coefficients for each
    /// root are stored contiguously. E.g., the coefficient for determinant i and root r is at
    /// index i * nroots_ + r.
    std::vector<double> c_;

    SelectedCIStrings ab_list_;
    SelectedCIStrings ba_list_;
};

} // namespace forte2
