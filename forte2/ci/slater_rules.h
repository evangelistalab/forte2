#pragma once

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

//     /// Compute the matrix element of the Hamiltonian between this determinant
//     /// and a given one
//     double slater_rules(const Determinant& lhs, const Determinant& rhs) const;
//     /// Compute the matrix element of the Hamiltonian between this determinant
//     /// and a given one
//     double slater_rules_single_alpha(const Determinant& det, int i, int a) const;
//     /// Compute the matrix element of the Hamiltonian between this determinant
//     /// and a given one
//     double slater_rules_single_beta(const Determinant& det, int i, int a) const;
//     /// Compute the matrix element of the Hamiltonian between this determinant
//     /// and a given one
//     double slater_rules_single_alpha_abs(const Determinant& det, int i, int a) const;
//     /// Compute the matrix element of the Hamiltonian between this determinant
//     /// and a given one
//     double slater_rules_single_beta_abs(const Determinant& det, int i, int a) const;

//     /// Return the alpha effective one-electron integral
//     double oei_a(size_t p, size_t q) const { return oei_a_[p * nmo_ + q]; }
//     /// Return the beta effective one-electron integral
//     double oei_b(size_t p, size_t q) const { return oei_b_[p * nmo_ + q]; }
//     std::vector<double> oei_a_vector() { return oei_a_; }
//     std::vector<double> oei_b_vector() { return oei_b_; }

//     /// Return the alpha-alpha antisymmetrized two-electron integral <pq||rs>
//     double tei_aa(size_t p, size_t q, size_t r, size_t s) const {
//         return tei_aa_[nmo3_ * p + nmo2_ * q + nmo_ * r + s];
//     }
//     /// Return the alpha-beta two-electron integral <pq|rs>
//     double tei_ab(size_t p, size_t q, size_t r, size_t s) const {
//         return tei_ab_[nmo3_ * p + nmo2_ * q + nmo_ * r + s];
//     }
//     /// Return the beta-beta antisymmetrized two-electron integral <pq||rs>
//     double tei_bb(size_t p, size_t q, size_t r, size_t s) const {
//         return tei_bb_[nmo3_ * p + nmo2_ * q + nmo_ * r + s];
//     }

//     /// Return a vector of alpha-alpha antisymmetrized two-electron integrals
//     const std::vector<double>& tei_aa_vector() const { return tei_aa_; }
//     /// Return a vector of alpha-beta antisymmetrized two-electron integrals
//     const std::vector<double>& tei_ab_vector() const { return tei_ab_; }
//     /// Return a vector of beta-beta antisymmetrized two-electron integrals
//     const std::vector<double>& tei_bb_vector() const { return tei_bb_; }

//     /// Return the alpha-alpha antisymmetrized two-electron integral <pq||pq>
//     double diag_tei_aa(size_t p, size_t q) const {
//         return tei_aa_[p * nmo3_ + q * nmo2_ + p * nmo_ + q];
//     }
//     /// Return the alpha-beta two-electron integral <pq|rs>
//     double diag_tei_ab(size_t p, size_t q) const {
//         return tei_ab_[p * nmo3_ + q * nmo2_ + p * nmo_ + q];
//     }
//     /// Return the beta-beta antisymmetrized two-electron integral <pq||rs>
//     double diag_tei_bb(size_t p, size_t q) const {
//         return tei_bb_[p * nmo3_ + q * nmo2_ + p * nmo_ + q];
//     }
//     IntegralType get_integral_type() { return integral_type_; }
//     /// Set the active integrals
//     void set_active_integrals(const ambit::Tensor& tei_aa, const ambit::Tensor& tei_ab,
//                               const ambit::Tensor& tei_bb);
//     /// Compute the restricted_docc operator
//     /// F^{closed}_{uv} = h_{uv} + \sum_{i = frozen_core}^{restricted_core} 2(uv|ii) - (ui|vi)
//     void compute_restricted_one_body_operator();
//     /// Set the restricted_one_body_operator
//     void set_restricted_one_body_operator(const std::vector<double>& oei_a,
//                                           const std::vector<double>& oei_b) {
//         oei_a_ = oei_a;
//         oei_b_ = oei_b;
//     }

//     /// Streamline the process of setting up active integrals and
//     /// restricted_docc
//     /// Sets active integrals based on active space and restricted_docc
//     /// If you want more control, don't use this function.
//     void set_active_integrals_and_restricted_docc();

//     /// Add (scalar, oei, tei) another ActiveSpaceIntegrals
//     void add(std::shared_ptr<ActiveSpaceIntegrals> as_ints, const double factor = 1.0);

//     /// Print the alpha-alpha integrals
//     void print();

//   private:
//     // ==> Class Private Data <==

//     /// The number of MOs
//     size_t nmo_;
//     /// The number of MOs squared
//     size_t nmo2_;
//     /// The number of MOs cubed
//     size_t nmo3_;
//     /// The number of MOs to the fourth power
//     size_t nmo4_;
//     /// The integral type
//     IntegralType integral_type_;
//     /// The integrals object
//     std::shared_ptr<ForteIntegrals> ints_;
//     /// The frozen core energy
//     double frozen_core_energy_;
//     /// The scalar contribution to the energy
//     double scalar_energy_;
//     /// The alpha one-electron integrals
//     std::vector<double> oei_a_;
//     /// The beta one-electron integrals
//     std::vector<double> oei_b_;
//     /// The alpha-alpha antisymmetrized two-electron integrals in physicist
//     /// notation
//     std::vector<double> tei_aa_;
//     /// The alpha-beta antisymmetrized two-electron integrals in physicist
//     /// notation
//     std::vector<double> tei_ab_;
//     /// The beta-beta antisymmetrized two-electron integrals in physicist
//     /// notation
//     std::vector<double> tei_bb_;
//     /// A vector of indices for the active molecular orbitals
//     std::vector<size_t> active_mo_;
//     /// A vector of the symmetry of the active molecular orbitals
//     std::vector<int> active_mo_symmetry_;
//     /// A Vector of indices for the restricted_docc molecular orbitals
//     std::vector<size_t> restricted_docc_mo_;

//     // ==> Class Private Functions <==

//     inline size_t tei_index(size_t p, size_t q, size_t r, size_t s) const {
//         return nmo3_ * p + nmo2_ * q + nmo_ * r + s;
//     }

//     void startup();

} // namespace forte2
