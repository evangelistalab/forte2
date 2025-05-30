#pragma once

#include <vector>
#include <unordered_map>
#include <helpers/math_structures.h>

#include "ci/sparse.h"
#include "ci/sq_operator_string.h"

namespace forte2 {

/// @brief A class to represent general second quantized operators
///
/// An operator is a linear combination of terms, where each term is a numerical factor
/// times a product of second quantized operators (a SQOperatorString object)
///
/// For example:
///   0.1 * [2a+ 0a-] - 0.5 * [2a+ 0a-] + ...
///           Term 0            Term 1
///
/// This class stores operators in each term in the following canonical form
///   a+_p1 a+_p2 ...  a+_P1 a+_P2 ...   ... a-_Q2 a-_Q1   ... a-_q2 a-_q1
///   alpha creation   beta creation    beta annihilation  alpha annihilation
///
/// with indices sorted as
///
///   (p1 < p2 < ...) (P1 < P2 < ...)  (... > Q2 > Q1) (... > q2 > q1)
///
class SparseOperator : public VectorSpace<SparseOperator, SQOperatorString, sparse_scalar_t,
                                          SQOperatorString::Hash> {
  public:
    SparseOperator() = default;
    using base_t =
        VectorSpace<SparseOperator, SQOperatorString, sparse_scalar_t, SQOperatorString::Hash>;
    using base_t::base_t; // Make the base class constructors visible

    /// @brief add a term to this operator
    /// @param str a string that defines the product of operators in the format [... q_2 q_1 q_0]
    /// @param coefficient a coefficient that multiplies the product of second quantized operators
    /// @param allow_reordering if true, the operator will be reordered to canonical form
    /// @details The operator is stored in canonical form
    ///
    ///     coefficient * [... q_2 q_1 q_0]
    ///
    /// where q_0, q_1, ... are second quantized operators. These operators are
    /// passed as string
    ///
    ///     '[... q_2 q_1 q_0]'
    ///
    /// where q_i = <orbital_i><spin_i><type_i> and the quantities in <> are
    ///
    ///     orbital_i: int
    ///     spin_i: 'a' (alpha) or 'b' (beta)
    ///     type_i: '+' (creation) or '-' (annihilation)
    ///
    /// For example, '[0a+ 1b+ 12b- 0a-]'
    ///
    void add_term_from_str(const std::string& str, sparse_scalar_t coefficient,
                           bool allow_reordering = false);

    /// @return a string representation of this operator
    std::vector<std::string> str() const;

    /// @return a latex representation of this operator
    std::string latex() const;
};

/// @return The product of two second quantized operators
SparseOperator operator*(const SparseOperator& lhs, const SparseOperator& rhs);

/// @return The commutator of two second quantized operators
SparseOperator commutator(const SparseOperator& lhs, const SparseOperator& rhs);

} // namespace forte2
