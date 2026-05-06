#pragma once
#include <complex>
#include "helpers/ndarray.h"

#include "ci/determinant.h"

namespace forte2 {

/// @brief Class to compute the energy and other properties of determinants using Slater rules.
/// This class assumes a restricted formalism, i.e., alpha and beta spin orbitals share the same
/// spatial orbitals and are orthogonal.
/// @details The class copies the one- and two-electron integrals to internal data structures
/// to speed up access.

class SlaterRules {
  public:
    // ==> Class Constructors <==

    /// @brief Construct a SlaterRules object.
    /// @param norb Number of orbitals.
    /// @param scalar_energy Scalar energy term.
    /// @param one_electron_integrals One-electron integrals (restricted) h[p,q] = <p|h|q>
    /// @param two_electron_integrals Two-electron integrals (restricted) in the form V[p,q,r,s] =
    /// <pq|rs>
    SlaterRules(int norb, double scalar_energy, np_matrix one_electron_integrals,
                np_tensor4 two_electron_integrals);

    // ==> Class Interface <==

    /// Compute the energy of a determinant
    double energy(const Determinant& det) const;

    /// Compute the energies of a vector of determinants
    np_vector energies(const std::vector<Determinant>& dets) const;

    /// Compute the matrix element of the Hamiltonian between two determinants
    double slater_rules(const Determinant& lhs, const Determinant& rhs) const;

    // ==> Helper Functions <==

    /// @return The one-electron integral h[p,q] = <p|h|q>
    double h(int p, int q) const noexcept { return h_[p * norb_ + q]; }

    /// @return The two-electron integral V[p,q,r,s] = <pq|rs>
    double v(int p, int q, int r, int s) const noexcept {
        return v_[p * norb3_ + q * norb2_ + r * norb_ + s];
    }

    /// @return The anti-symmetrized two-electron integral Va[p,q,r,s] <pq||rs> = <pq|rs> - <pq|sr>
    double va(int p, int q, int r, int s) const noexcept {
        return va_[p * norb3_ + q * norb2_ + r * norb_ + s];
    }

    /// @return The Coulomb integral J[p,q] = <pq|pq>
    double J(int p, int q) const noexcept { return J_[p * norb_ + q]; }

    /// @return The Coulomb - exchange integral JK[p,q] = <pq|pq> - <pq|qp>
    double JK(int p, int q) const noexcept { return JK_[p * norb_ + q]; }

    /// @return The Fock-Coulomb integral f_J[p,q,r] = <pr|qr>
    double f_J(int p, int q, int r) const noexcept { return f_J_[p * norb2_ + q * norb_ + r]; }

    /// @return The Fock-Coulomb-Exchange integral f_JK[p,q,r] = <pr|qr> - <pr|rq>
    double f_JK(int p, int q, int r) const noexcept { return f_JK_[p * norb2_ + q * norb_ + r]; }

    /// @brief Compute the singles coupling for alpha spin
    double singles_coupling_a(size_t i, size_t a, const Determinant& d) const noexcept;

    /// @brief Compute the singles coupling for beta spin
    double singles_coupling_b(size_t i, size_t a, const Determinant& d) const noexcept;

  private:
    /// @brief Number of orbitals
    const int norb_;
    /// @brief Precomputed values to speed up access to two-electron integrals
    const int norb2_;
    /// @brief Precomputed values to speed up access to two-electron integrals
    const int norb3_;
    /// Scalar energy term
    double scalar_energy_;
    /// Two-electron integrals (restricted) in the form V[p,q,r,s] = <pq|rs>
    np_tensor4 two_electron_integrals_;
    /// One-electron integrals h[p * norb_ + q] = <p|h|q>
    std::vector<double> h_;
    /// Coulomb integrals J[p * norb_ + q] = <pq|pq>
    std::vector<double> J_;
    /// Exchange integrals JK[p * norb_ + q] = <pq|pq> - <pq|qp>
    std::vector<double> JK_;
    /// Fock-Coulomb integrals f_J[p * norb_ * norb_ + q * norb_ + r] = <pr|qr>
    std::vector<double> f_J_;
    /// Fock-Coulomb-Exchange integrals f_JK[p * norb_ * norb_ + q * norb_ + r] = <pr|qr> - <pr|rq>
    std::vector<double> f_JK_;
    /// Two-electron integrals (restricted) in the form V[p,q,r,s] = <pq|rs>
    std::vector<double> v_;
    /// Anti-symmetrized two-electron integrals <pq||rs> = <pq|rs> - <pq|sr>
    std::vector<double> va_;
};

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
