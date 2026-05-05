#pragma once

#include <string>
#include <vector>

#include "determinant/determinant.h"
#include "helpers/math_structures.h"
#include "sparse/sparse.h"
#include "sparse/sparse_state.h"
#include "sparse/sq_operator_string.h"

namespace forte2 {

class SparseOperator;
class SparseState;

/// @brief A normal-ordered string represented by compact creator and annihilator determinants.
///
/// The creation and annihilation determinants encode normal creators and normal annihilators,
/// respectively. Their physical meaning is defined by the determinant reference used when
/// converting to physical operator tuples, strings, or ordinary SparseOperator objects.
class NormalOrderedString {
  public:
    NormalOrderedString();
    NormalOrderedString(const Determinant& cre, const Determinant& ann);

    /// @return The normal creator string.
    const Determinant& cre() const;

    /// @return The normal annihilator string.
    const Determinant& ann() const;

    /// @return The sign mask associated with this normal-ordered string.
    const Determinant& sign_mask() const;

    /// @return True if this string contains no normal operators.
    bool is_identity() const;

    /// @return The number of normal operators in this string.
    int count() const;

    /// @return The many-body rank, defined as ceil(count / 2).
    int many_body_rank() const;

    /// @return A physical operator tuple for a determinant reference.
    op_tuple_t op_tuple(const Determinant& reference) const;

    /// @return A string representation assuming the particle vacuum.
    std::string str() const;

    /// @return A physical string representation for a determinant reference.
    std::string str(const Determinant& reference) const;

    /// @return A LaTeX representation assuming the particle vacuum.
    std::string latex() const;

    /// @return A physical LaTeX representation for a determinant reference.
    std::string latex(const Determinant& reference) const;

    bool operator==(const NormalOrderedString& other) const;
    bool operator<(const NormalOrderedString& other) const;

    struct Hash {
        std::size_t operator()(const NormalOrderedString& str) const;
    };

  private:
    Determinant cre_ = Determinant::zero();
    Determinant ann_ = Determinant::zero();
    mutable Determinant sign_mask_ = Determinant::zero();
    mutable bool sign_mask_valid_ = false;
};

/// @brief A sparse operator in determinant-normal-ordered form.
class NormalOrderedSparseOperator
    : public VectorSpace<NormalOrderedSparseOperator, NormalOrderedString, sparse_scalar_t,
                         NormalOrderedString::Hash> {
  public:
    using base_t = VectorSpace<NormalOrderedSparseOperator, NormalOrderedString, sparse_scalar_t,
                               NormalOrderedString::Hash>;
    using old_container = base_t::old_container;
    using base_t::base_t;

    NormalOrderedSparseOperator();
    explicit NormalOrderedSparseOperator(const Determinant& reference);
    NormalOrderedSparseOperator(const Determinant& reference, const NormalOrderedString& str,
                                sparse_scalar_t coefficient);

    /// @return The determinant reference that defines normal creation and annihilation.
    const Determinant& reference() const;

    /// @return The coefficient of a normal-ordered term.
    sparse_scalar_t coefficient(const NormalOrderedString& str) const;

    /// @return A string representation of this operator.
    std::vector<std::string> str() const;

    /// @return A LaTeX representation of this operator.
    std::string latex() const;

    /// @return A copy with terms above max_rank removed.
    NormalOrderedSparseOperator truncate(int max_rank, double screen_thresh = 1.0e-12) const;

    /// @return This normal-ordered operator expanded as an ordinary SparseOperator.
    SparseOperator to_sparse_operator(double screen_thresh = 1.0e-12) const;

    /// @return The result of applying this normal-ordered operator to a SparseState.
    SparseState apply_to_state(const SparseState& state, double screen_thresh = 1.0e-12) const;

    bool operator==(const NormalOrderedSparseOperator& other) const;

  private:
    Determinant reference_ = Determinant::zero();
};

/// @brief Normal order a SparseOperator with respect to a determinant vacuum.
NormalOrderedSparseOperator normal_order(const SparseOperator& op, const Determinant& reference,
                                         double screen_thresh = 1.0e-12, int max_rank = -1);

/// @brief A sparse operator in generalized normal-ordered form.
///
/// Terms are keyed by physical SQOperatorString objects. The vacuum is a sparse CI state, so
/// contractions are evaluated directly as vacuum expectation values instead of from a determinant
/// occupation pattern.
class GeneralizedNormalOrderedSparseOperator
    : public VectorSpace<GeneralizedNormalOrderedSparseOperator, SQOperatorString, sparse_scalar_t,
                         SQOperatorString::Hash> {
  public:
    using base_t = VectorSpace<GeneralizedNormalOrderedSparseOperator, SQOperatorString,
                               sparse_scalar_t, SQOperatorString::Hash>;
    using old_container = base_t::old_container;
    using base_t::base_t;

    GeneralizedNormalOrderedSparseOperator();
    GeneralizedNormalOrderedSparseOperator(const SparseState& vacuum, std::size_t norb,
                                           int max_cumulant = -1);
    GeneralizedNormalOrderedSparseOperator(const SparseState& vacuum, std::size_t norb,
                                           int max_cumulant, const SQOperatorString& str,
                                           sparse_scalar_t coefficient);

    /// @return The sparse CI vacuum defining generalized contractions.
    const SparseState& vacuum() const;

    /// @return The number of spatial orbitals used for the vacuum contractions.
    std::size_t norb() const;

    /// @return Maximum contracted body rank, or -1 for unbounded contractions.
    int max_cumulant() const;

    /// @return The coefficient of a generalized normal-ordered term.
    sparse_scalar_t coefficient(const SQOperatorString& str) const;

    /// @return A string representation of this operator.
    std::vector<std::string> str() const;

    /// @return A LaTeX representation of this operator.
    std::string latex() const;

    /// @return A copy with terms above max_rank removed.
    GeneralizedNormalOrderedSparseOperator truncate(int max_rank,
                                                    double screen_thresh = 1.0e-12) const;

    /// @return This generalized normal-ordered operator expanded as a SparseOperator.
    SparseOperator to_sparse_operator(double screen_thresh = 1.0e-12) const;

    /// @return The result of applying this generalized normal-ordered operator to a SparseState.
    SparseState apply_to_state(const SparseState& state, double screen_thresh = 1.0e-12) const;

    bool operator==(const GeneralizedNormalOrderedSparseOperator& other) const;

  private:
    SparseState vacuum_;
    std::size_t norb_ = 0;
    int max_cumulant_ = -1;
};

/// @brief Generalized normal order a SparseOperator with respect to a sparse CI vacuum.
GeneralizedNormalOrderedSparseOperator
generalized_normal_order(const SparseOperator& op, const SparseState& vacuum, std::size_t norb,
                         int max_cumulant = -1, double screen_thresh = 1.0e-12, int max_rank = -1);

} // namespace forte2
