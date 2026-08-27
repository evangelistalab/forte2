#pragma once

#include <cstddef>
#include <unordered_map>
#include <vector>

#include "determinant/determinant.h"
#include "sparse/sparse.h"
#include "sparse/sparse_state.h"
#include "sparse/sq_operator_string.h"

namespace forte2 {

/// @brief Spin-orbital density cumulants for a sparse multiconfigurational reference.
///
/// Orbital subsets and antisymmetric tensor indices use Determinant bit strings. Fixed core and
/// empty virtual contributions to the one-body density are represented implicitly, while
/// higher-body cumulants contain active indices only.
class CumulantReference {
  public:
    CumulantReference(const SparseState& vacuum, std::size_t norb, int max_cumulant = 2,
                      double screen_thresh = 1.0e-12);

    /// @return The sparse state from which the cumulants were constructed.
    const SparseState& vacuum() const;

    /// @return The number of spatial orbitals represented by this reference.
    std::size_t norb() const;

    /// @return The largest available density-cumulant rank.
    int max_cumulant() const;

    /// @return The threshold used when identifying reference support and storing cumulants.
    double screen_thresh() const;

    /// @return Spin orbitals occupied in every significant determinant.
    const Determinant& core_modes() const;

    /// @return Spin orbitals whose occupations vary in the sparse reference.
    const Determinant& active_modes() const;

    /// @return Spin orbitals unoccupied in every significant determinant.
    const Determinant& virtual_modes() const;

    /// Return gamma^p_q = <a^+_p a_q>. Orbital indices are spatial and spins are explicit.
    sparse_scalar_t gamma(std::size_t p, bool p_alpha, std::size_t q, bool q_alpha) const;

    /// Return eta^p_q = delta^p_q - gamma^p_q.
    sparse_scalar_t eta(std::size_t p, bool p_alpha, std::size_t q, bool q_alpha) const;

    /// Return gamma^p_q using spin-orbital modes in Determinant storage convention.
    sparse_scalar_t gamma_mode(std::size_t p, std::size_t q) const;

    /// Return the rank-k RDM element encoded by creator and annihilator bit strings.
    sparse_scalar_t rdm(const Determinant& cre, const Determinant& ann) const;

    /// Return the rank-k RDM reconstructed with cumulants above max_cumulant set to zero.
    sparse_scalar_t truncated_rdm(const Determinant& cre, const Determinant& ann) const;

    /// Return the rank-k density cumulant encoded by creator and annihilator bit strings.
    sparse_scalar_t cumulant(const Determinant& cre, const Determinant& ann) const;

    /// @return The number of explicitly stored nonzero cumulants of a given rank.
    std::size_t cumulant_size(int rank) const;

  private:
    std::size_t determinant_mode(std::size_t orbital, bool alpha) const;
    std::size_t compact_mode(std::size_t determinant_mode) const;
    void validate_indices(const Determinant& cre, const Determinant& ann) const;
    sparse_scalar_t expectation(const SQOperatorString& term) const;
    sparse_scalar_t rdm_modes(std::vector<std::size_t> upper, std::vector<std::size_t> lower) const;
    sparse_scalar_t truncated_rdm_modes(
        const std::vector<std::size_t>& upper, const std::vector<std::size_t>& lower,
        std::unordered_map<SQOperatorString, sparse_scalar_t, SQOperatorString::Hash>& cache) const;
    sparse_scalar_t cumulant_modes(std::vector<std::size_t> upper,
                                   std::vector<std::size_t> lower) const;
    void build_orbital_spaces();
    void build_one_body_density();
    void build_two_body_cumulant();
    void build_three_body_cumulant();
    void build_four_body_cumulant();

    SparseState vacuum_;
    std::size_t norb_ = 0;
    int max_cumulant_ = 0;
    double screen_thresh_ = 1.0e-12;
    sparse_scalar_t norm_ = 0.0;
    Determinant all_modes_ = Determinant::zero();
    Determinant core_modes_ = Determinant::zero();
    Determinant active_modes_ = Determinant::zero();
    Determinant virtual_modes_ = Determinant::zero();
    std::vector<sparse_scalar_t> gamma_;
    std::vector<std::unordered_map<SQOperatorString, sparse_scalar_t, SQOperatorString::Hash>>
        rdms_;
    std::vector<std::unordered_map<SQOperatorString, sparse_scalar_t, SQOperatorString::Hash>>
        cumulants_;
};

} // namespace forte2
