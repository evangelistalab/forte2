#pragma once
#include <complex>
#include "helpers/ndarray.h"

#include "ci/determinant.h"

namespace forte2 {

/// @brief Class to compute the energy and other properties of determinants using Slater rules.
class SlaterRules {
  public:
    // ==> Class Constructors <==

    SlaterRules(int norb, double scalar_energy, np_matrix one_electron_integrals,
                np_tensor4 two_electron_integrals);

    // ==> Class Interface <==

    /// Compute a determinant's energy
    double energy(const Determinant& det) const;

    /// Compute the matrix element of the Hamiltonian between two determinants
    double slater_rules(const Determinant& lhs, const Determinant& rhs) const;

  private:
    /// Number of orbitals
    int norb_;
    /// Scalar energy term
    double scalar_energy_;
    /// Effective one-electron integrals (restricted)
    np_matrix one_electron_integrals_;
    /// Two-electron integrals (restricted) in the form V[p,q,r,s] = <pq|rs>
    np_tensor4 two_electron_integrals_;
};

class RelSlaterRules {
  public:
    // ==> Class Constructors <==

    RelSlaterRules(int nspinor, double scalar_energy, np_matrix_complex one_electron_integrals,
                   np_tensor4_complex two_electron_integrals);

    // ==> Class Interface <==

    /// Compute a determinant's energy
    double energy(const Determinant& det) const;

    /// Compute the matrix element of the Hamiltonian between two determinants
    std::complex<double> slater_rules(const Determinant& lhs, const Determinant& rhs) const;

  private:
    /// Number of spin(or)-orbitals
    int nspinor_;
    /// Scalar energy term
    double scalar_energy_;
    /// Effective one-electron integrals (restricted)
    np_matrix_complex one_electron_integrals_;
    /// Two-electron integrals (restricted) in the form V[p,q,r,s] = <pq|rs>
    np_tensor4_complex two_electron_integrals_;
};

} // namespace forte2
