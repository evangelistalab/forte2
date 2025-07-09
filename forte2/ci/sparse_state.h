#pragma once

#include <vector>
#include <unordered_map>

#include "ci/sparse.h"
#include "ci/determinant.h"
#include "ci/sparse_operator.h"

namespace forte2 {

/// @brief A class to represent general Fock space states
class SparseState
    : public VectorSpace<SparseState, Determinant, sparse_scalar_t, Determinant::Hash> {
  public:
    using base_t = VectorSpace<SparseState, Determinant, sparse_scalar_t, Determinant::Hash>;
    using base_t::base_t; // Make the base class constructors visible

    /// @return a string representation of the object
    /// @param n the number of spatial orbitals to print
    std::string str(int n = 0) const;
};

// Functions to apply operators to a state
/// @brief Apply an operator to a state
/// @param op the operator to apply
/// @param state the state to apply the operator to
/// @param screen_thresh the threshold to screen the operator
/// @return the new state
SparseState apply_operator_lin(const SparseOperator& op, const SparseState& state,
                               double screen_thresh = 1.0e-12);

/// @brief Apply the antihermitian combination of an operator to a state
/// @param op the operator to apply
/// @param state the state to apply the operator to
/// @param screen_thresh the threshold to screen the operator
/// @return the new state
SparseState apply_operator_antiherm(const SparseOperator& op, const SparseState& state,
                                    double screen_thresh = 1.0e-12);

/// compute the projection  <state0 | op | ref>, for each operator op in gop
std::vector<sparse_scalar_t> get_projection(const SparseOperatorList& sop, const SparseState& ref,
                                            const SparseState& state0);

/// apply the number projection operator P^alpha_na P^beta_nb |state>
SparseState apply_number_projector(int na, int nb, const SparseState& state);

/// compute the overlap value <left_state|right_state>
sparse_scalar_t overlap(const SparseState& left_state, const SparseState& right_state);

/// compute the S^2 expectation value
sparse_scalar_t spin2(const SparseState& left_state, const SparseState& right_state);

/// Return the normalized state
SparseState normalize(const SparseState& state);

} // namespace forte2
