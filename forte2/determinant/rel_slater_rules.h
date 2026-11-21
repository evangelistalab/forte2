#pragma once

#include <complex>
#include <optional>
#include <vector>

#include "helpers/ndarray.h"

#include "determinant/determinant.h"

namespace forte2 {

/// @brief Class to compute Hamiltonian matrix elements of determinants using Slater rules for the
/// relativistic case (i.e., using spinors instead of spatial orbitals).

class RelSlaterRules {
  public:
    // ==> Class Constructors <==

    /// @brief Construct a RelSlaterRules object.
    /// @param nspinor Number of spinorbitals.
    /// @param scalar_energy Scalar energy term.
    /// @param one_electron_integrals One-electron integrals h[p,q] = <p|h|q>
    /// @param two_electron_integrals Two-electron integrals in physicist's notation
    ///     V[p,q,r,s] = <pq|rs>. They are antisymmetrized on the fly.
    RelSlaterRules(int nspinor, double scalar_energy, np_matrix_complex one_electron_integrals,
                   np_tensor4_complex two_electron_integrals);

    // ==> Class Interface <==

    /// @brief (Re)initialize the integrals used by this object, e.g. to reuse it across CI
    /// iterations without reconstructing it. Any of scalar_energy/one_electron_integrals/
    /// two_electron_integrals left as nullopt keeps its current value.
    /// @param nspinor Number of spinorbitals.
    /// @param scalar_energy New scalar energy, or nullopt to keep the current value.
    /// @param one_electron_integrals New one-electron integrals h[p,q] = <p|h|q>, or nullopt to
    ///        keep the current value.
    /// @param two_electron_integrals New two-electron integrals in physicist's notation
    ///     V[p,q,r,s] = <pq|rs>, or nullopt to keep the current value. Antisymmetrized on the fly.
    /// @note If nspinor differs from the number of spinors this object currently holds, both
    /// one_electron_integrals and two_electron_integrals must be given together: a stale one
    /// left over from a different nspinor would be read out of bounds.
    void update_integrals(int nspinor, std::optional<double> scalar_energy = std::nullopt,
                          std::optional<np_matrix_complex> one_electron_integrals = std::nullopt,
                          std::optional<np_tensor4_complex> two_electron_integrals = std::nullopt);

    /// @brief Compute the energy of a determinant
    /// @param det The determinant for which to compute the energy.
    /// @return The energy of the determinant.
    double energy(const Determinant& det) const;

    /// @brief Compute the energies of a vector of determinants
    /// @param dets The vector of determinants for which to compute the energies.
    /// @return A vector containing the energies of the determinants.
    np_vector energies(const std::vector<Determinant>& dets) const;

    /// @brief Compute the matrix element of the Hamiltonian between two determinants
    /// @param lhs The left-hand side determinant.
    /// @param rhs The right-hand side determinant.
    /// @return The matrix element of the Hamiltonian between the two determinants.
    std::complex<double> slater_rules(const Determinant& lhs, const Determinant& rhs) const;

  private:
    /// @brief Number of spin(or)-orbitals
    int nspinor_ = 0;
    /// @brief Scalar energy term
    double scalar_energy_ = 0.0;
    /// @brief One-electron integrals, owned copy: h_[p*nspinor_+q] = <p|h|q>
    std::vector<std::complex<double>> h_;
    /// @brief Two-electron integrals, owned copy: v_[p,q,r,s] = <pq|rs>
    std::vector<std::complex<double>> v_;

    /// @return The one-electron integral <p|h|q>
    inline std::complex<double> h(std::size_t p, std::size_t q) const noexcept {
        return h_[p * nspinor_ + q];
    }

    /// @return The two-electron integral <pq|rs>
    inline std::complex<double> v(std::size_t p, std::size_t q, std::size_t r,
                                  std::size_t s) const noexcept {
        const auto n = static_cast<std::size_t>(nspinor_);
        return v_[((p * n + q) * n + r) * n + s];
    }
};

} // namespace forte2
