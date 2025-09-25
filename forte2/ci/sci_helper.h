#pragma once

#include <functional>
#include <vector>
#include <cmath>
#include <span>

#include "helpers/ndarray.h"
#include "determinant.h"
// #include "ci/slater_rules.h"

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

  private:
    // == Class Private Variables ==

    /// @brief logging level for the class
    int log_level_ = 3;

    /// @brief Number of orbitals
    const size_t norb_;
    const size_t norb2_;
    const size_t norb3_;

    /// @brief The scalar energy
    double E_;
    /// @brief One-electron integrals in the form of a matrix H[p][q] = <p|H|q> = h_pq
    np_matrix H_;
    /// @brief Two-electron integrals in the form of a tensor V[p][q][r][s] = <pq|rs> = (pr|qs)
    np_tensor4 V_;

    /// @brief Orbital energies: e[p] = <p|H|p>
    std::vector<double> epsilon_;
    /// @brief One-electron integrals: H[p][q] = <p|H|q> = h_pq
    std::vector<double> h_;
    /// @brief Two-electron integrals: V[p][q][r][s] = <pq|rs> = (pr|qs)
    std::vector<double> v_;
    /// @brief Two-electron integrals: V[p][q][r][s] = <pq||rs> = (pr|qs) - (ps|qr)
    std::vector<double> v_a_;

    /// @brief The determinants in the reference space
    size_t nroots_;
    std::vector<Determinant> dets_;
    np_matrix c_guess_;
    std::vector<double> c_;
};

} // namespace forte2
