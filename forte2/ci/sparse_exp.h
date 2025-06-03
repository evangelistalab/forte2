#pragma once

#include "ci/sparse_state.h"
#include "ci/sparse_operator.h"
#include "ci/determinant.h"
#include "helpers/memory.h"

namespace forte2 {

/**
 * @brief The SparseExp class
 * This class implements an algorithm to apply the exponential of an operator to a state
 *
 *    |state> -> exp(op) |state>
 *
 */
class SparseExp {
    enum class OperatorType { Excitation, Antihermitian };

  public:
    /// @brief Constructor
    /// @param maxk the maximum power of op used in the Taylor expansion of exp(op)
    /// @param screen_thresh a threshold to select which elements of the operator applied to the
    /// state. An operator in the form exp(t ...), where t is an amplitude, will be applied to a
    /// determinant Phi_I with coefficient C_I if the product |t * C_I| > screen_threshold
    SparseExp(int maxk, double screen_thresh);

    /// @brief Compute the exponential applied to a state via a Taylor expansion
    ///
    ///             exp(op) |state>
    ///
    /// This algorithms is useful when applying the exponential repeatedly
    /// to the same state or in an iterative procedure
    /// This function applies only those elements of the operator that satisfy the condition:
    ///     |t * C_I| > screen_threshold
    /// where C_I is the coefficient of a determinant
    ///
    /// @param sop the operator. Each term in this operator is applied in the order provided
    /// @param state the state to which the factorized exponential will be applied
    /// @param scaling_factor A scalar factor that multiplies the operator exponentiated. If set to
    /// -1.0 it allows to compute the inverse of the exponential exp(-op)
    SparseState apply_op(const SparseOperator& sop, const SparseState& state,
                         double scaling_factor = 1.0);

    SparseState apply_op(const SparseOperatorList& sop, const SparseState& state,
                         double scaling_factor = 1.0);

    /// @brief Compute the exponential of the antihermitian of an operator applied to a state via a
    /// Taylor expansion
    ///
    ///             exp(op - op^dagger) |state>
    ///
    /// This algorithms is useful when applying the exponential repeatedly
    /// to the same state or in an iterative procedure
    /// This function applies only those elements of the operator that satisfy the condition:
    ///     |t * C_I| > screen_threshold
    /// where C_I is the coefficient of a determinant
    ///
    /// @param sop the operator. Each term in this operator is applied in the order provided
    /// @param state the state to which the factorized exponential will be applied
    /// @param scaling_factor A scalar factor that multiplies the operator exponentiated. If set to
    /// -1.0 it allows to compute the inverse of the exponential exp(-op)
    SparseState apply_antiherm(const SparseOperator& sop, const SparseState& state,
                               double scaling_factor = 1.0);

    SparseState apply_antiherm(const SparseOperatorList& sop, const SparseState& state,
                               double scaling_factor = 1.0);

  private:
    int maxk_ = 19;
    double screen_thresh_ = 1e-12;

    SparseState apply_exp_operator(OperatorType op_type, const SparseOperator& sop,
                                   const SparseState& state, double scaling_factor);
};

/// @brief This class implements an algorithm to apply a factorized exponential operator to a state

class SparseFactExp {
  public:
    /// @brief Constructor
    /// @param screen_thresh a threshold to select which elements of the operator applied to the
    /// state. An operator in the form exp(t ...), where t is an amplitude, will be applied to a
    /// determinant Phi_I with coefficient C_I if the product |t * C_I| > screen_threshold
    SparseFactExp(double screen_thresh = 1.0e-12);

    /// @brief Compute the factorized exponential applied to a state using an exact algorithm
    ///
    ///             ... exp(op2) exp(op1) |state>
    ///
    /// This algorithm is useful when applying the factorized exponential repeatedly
    /// to the same state or in an iterative procedure
    /// This function applies only those elements of the operator that satisfy the condition:
    ///     |t * C_I| > screen_threshold
    /// where C_I is the coefficient of a determinant
    ///
    /// @param sop the operator. Each term in this operator is applied in the order provided
    /// @param state the state to which the factorized exponential will be applied
    /// @param inverse If true, compute the inverse of the factorized exponential:
    ///
    ///             exp(-op1) exp(-op2) ... |state>
    ///
    /// @param reverse If true, apply the operators in reverse order
    SparseState apply_op(const SparseOperatorList& sop, const SparseState& state,
                         bool inverse = false, bool reverse = false);

    /// @brief Compute the factorized exponential applied to a state using an exact algorithm
    ///
    ///             ... exp(op2 - op2^dagger) exp(op1 - op1^dagger) |state>
    ///
    /// This algorithm is useful when applying the factorized exponential repeatedly
    /// to the same state or in an iterative procedure
    /// This function applies only those elements of the operator that satisfy the condition:
    ///     |t * C_I| > screen_threshold
    /// where C_I is the coefficient of a determinant
    ///
    /// @param sop the operator. Each term in this operator is applied in the order provided
    /// @param state the state to which the factorized exponential will be applied
    /// @param inverse If true, compute the inverse of the factorized exponential
    ///
    ///             exp(-op1 + op1^dagger) exp(-op2 + op2^dagger) ... |state>
    ///
    /// @param reverse If true, apply the operators in reverse order
    SparseState apply_antiherm(const SparseOperatorList& sop, const SparseState& state,
                               bool inverse = false, bool reverse = false);

  private:
    double screen_thresh_;
};

} // namespace forte2