#pragma once
#include <cstddef>
#include <vector>
#include "helpers/ndarray.h"

#include "determinant/determinant.h"

namespace forte2 {

/// @brief Class to compute Hamiltonian matrix elements of determinants using Slater rules.
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
    double slater_rules(const Determinant& lhs, const Determinant& rhs) const;

    /// @brief Compute the singles coupling for alpha spin <D(i->a)|H|D> where D(i->a) is the
    /// determinant obtained by exciting an electron from occupied orbital i to virtual orbital a.
    /// @param i The occupied orbital index.
    /// @param a The virtual orbital index.
    /// @param d The determinant for which to compute the coupling.
    /// @return The singles coupling for alpha spin.
    double singles_coupling_a(size_t i, size_t a, const Determinant& d) const noexcept;

    /// @brief Compute the singles coupling for beta spin <D(i->a)|H|D> where D(i->a) is the
    /// determinant obtained by exciting an electron from occupied orbital i to virtual orbital a.
    /// @param i The occupied orbital index.
    /// @param a The virtual orbital index.
    /// @param d The determinant for which to compute the coupling.
    /// @return The singles coupling for beta spin.
    double singles_coupling_b(size_t i, size_t a, const Determinant& d) const noexcept;

  private:
    // ==> Helper Functions <==

    /// @return The one-electron integral h[p,q] = <p|h|q>
    double h(std::size_t p, std::size_t q) const noexcept { return h_[p * norb_ + q]; }

    /// @return The two-electron integral V[p,q,r,s] = <pq|rs>
    double v(std::size_t p, std::size_t q, std::size_t r, std::size_t s) const noexcept {
        return v_[p * norb3_ + q * norb2_ + r * norb_ + s];
    }

    /// @return The anti-symmetrized two-electron integral Va[p,q,r,s] <pq||rs> = <pq|rs> - <pq|sr>
    double va(std::size_t p, std::size_t q, std::size_t r, std::size_t s) const noexcept {
        return va_[p * norb3_ + q * norb2_ + r * norb_ + s];
    }

    /// @return The Coulomb integral J[p,q] = <pq|pq>
    double J(std::size_t p, std::size_t q) const noexcept { return J_[p * norb_ + q]; }

    /// @return The Coulomb - exchange integral JK[p,q] = <pq|pq> - <pq|qp>
    double JK(std::size_t p, std::size_t q) const noexcept { return JK_[p * norb_ + q]; }

    /// @return The Fock-Coulomb integral f_J[p,q,r] = <pr|qr>
    double f_J(std::size_t p, std::size_t q, std::size_t r) const noexcept {
        return f_J_[p * norb2_ + q * norb_ + r];
    }

    /// @return The Fock-Coulomb-Exchange integral f_JK[p,q,r] = <pr|qr> - <pr|rq>
    double f_JK(std::size_t p, std::size_t q, std::size_t r) const noexcept {
        return f_JK_[p * norb2_ + q * norb_ + r];
    }

    /// @brief Number of orbitals
    const std::size_t norb_;
    /// @brief Precomputed values to speed up access to two-electron integrals
    const std::size_t norb2_;
    /// @brief Precomputed values to speed up access to two-electron integrals
    const std::size_t norb3_;
    /// @brief Scalar energy term
    const double scalar_energy_;
    /// @brief One-electron integrals h[p * norb_ + q] = <p|h|q>
    std::vector<double> h_;
    /// @brief Coulomb integrals J[p * norb_ + q] = <pq|pq>
    std::vector<double> J_;
    /// @brief Exchange integrals JK[p * norb_ + q] = <pq|pq> - <pq|qp>
    std::vector<double> JK_;
    /// @brief Fock-Coulomb integrals f_J[p * norb_ * norb_ + q * norb_ + r] = <pr|qr>
    std::vector<double> f_J_;
    /// @brief Fock-Coulomb-Exchange integrals f_JK[p * norb_ * norb_ + q * norb_ + r] = <pr|qr> -
    /// <pr|rq>
    std::vector<double> f_JK_;
    /// @brief Two-electron integrals (restricted) in the form V[p,q,r,s] = <pq|rs>
    std::vector<double> v_;
    /// @brief Anti-symmetrized two-electron integrals <pq||rs> = <pq|rs> - <pq|sr>
    std::vector<double> va_;
};

} // namespace forte2
