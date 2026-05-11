#pragma once

#include <complex>
#include <vector>

#include "helpers/ndarray.h"

#include "determinant/determinant.h"

namespace forte2 {

class RelSlaterRules {
  public:
    // ==> Class Constructors <==

    RelSlaterRules(int nspinor, double scalar_energy, np_matrix_complex one_electron_integrals,
                   np_tensor4_complex two_electron_integrals, bool tei_is_asym = false);

    // ==> Class Interface <==

    /// Compute a determinant's energy
    double energy(const Determinant& det) const;

    /// Compute the energies of a vector of determinants
    np_vector energies(const std::vector<Determinant>& dets) const;

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
    /// Flag indicating if the two-electron integrals are antisymmetric (i.e. <pq||rs> = <pq|rs> -
    /// <pq|sr>)
    bool tei_is_asym_;
};

} // namespace forte2
