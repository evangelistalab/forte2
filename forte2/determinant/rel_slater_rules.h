#pragma once

#include <complex>
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
    /// @param two_electron_integrals Two-electron integrals V[p,q,r,s]
    /// @param tei_is_asym Flag indicating if the two-electron integrals are antisymmetric.
    /// If true, the two-electron integrals are assumed to be antisymmetric:
    ///     V[p,q,r,s] = <pq|rs> - <pq|sr>.
    /// If false, the two-electron integrals are assumed to be symmetric:
    ///     V[p,q,r,s] = <pq|rs>.
    RelSlaterRules(int nspinor, double scalar_energy, np_matrix_complex one_electron_integrals,
                   np_tensor4_complex two_electron_integrals, bool tei_is_asym = false);

    // ==> Class Interface <==

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
    const int nspinor_;
    /// @brief Scalar energy term
    const double scalar_energy_;
    /// @brief Effective one-electron integrals (restricted)
    const np_matrix_complex one_electron_integrals_;
    /// @brief Two-electron integrals (restricted) in the form V[p,q,r,s] = <pq|rs>
    const np_tensor4_complex two_electron_integrals_;
    /// @brief Flag indicating if the two-electron integrals are antisymmetric (i.e. V[p,q,r,s] =
    /// <pq|rs>
    /// - <pq|sr>)
    const bool tei_is_asym_;
};

} // namespace forte2
