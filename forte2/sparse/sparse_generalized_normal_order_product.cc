#include "sparse/sparse_generalized_normal_order_product.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <vector>

#include "helpers/parallel.h"
#include "sparse/sparse_normal_order_internal.h"
#include "sparse/sparse_operator.h"
#include "sparse/sq_operator_string.h"

namespace forte2 {

namespace {

bool sparse_states_equal(const SparseState& lhs, const SparseState& rhs) {
    if (lhs.size() != rhs.size()) {
        return false;
    }
    for (const auto& [det, coefficient] : lhs.elements()) {
        const auto it = rhs.elements().find(det);
        if (it == rhs.elements().end() or it->second != coefficient) {
            return false;
        }
    }
    return true;
}

void validate_compatible(const GeneralizedNormalOrderedSparseOperator& lhs,
                         const GeneralizedNormalOrderedSparseOperator& rhs) {
    if (lhs.norb() != rhs.norb()) {
        throw std::invalid_argument(
            "GeneralizedNormalOrderedProductComputer::commutator: norb values must match");
    }
    if (lhs.max_cumulant() != rhs.max_cumulant()) {
        throw std::invalid_argument("GeneralizedNormalOrderedProductComputer::commutator: "
                                    "max_cumulant values must match");
    }
    if (not sparse_states_equal(lhs.vacuum(), rhs.vacuum())) {
        throw std::invalid_argument(
            "GeneralizedNormalOrderedProductComputer::commutator: vacua must match");
    }
}

} // namespace

GeneralizedNormalOrderedProductComputer::GeneralizedNormalOrderedProductComputer(
    int max_rank, double screen_thresh)
    : max_rank_(max_rank), screen_thresh_(screen_thresh) {
    if (max_rank < 0) {
        throw std::invalid_argument(
            "GeneralizedNormalOrderedProductComputer: max_rank must be non-negative");
    }
    if (screen_thresh < 0.0) {
        throw std::invalid_argument(
            "GeneralizedNormalOrderedProductComputer: screen_thresh must be non-negative");
    }
}

GeneralizedNormalOrderedSparseOperator GeneralizedNormalOrderedProductComputer::commutator(
    const GeneralizedNormalOrderedSparseOperator& lhs,
    const GeneralizedNormalOrderedSparseOperator& rhs) const {
    validate_compatible(lhs, rhs);

    const auto lhs_sparse = lhs.to_sparse_operator(screen_thresh_);
    const auto rhs_sparse = rhs.to_sparse_operator(screen_thresh_);
    const auto& lhs_terms = lhs_sparse.elements_as_vec();
    const auto& rhs_terms = rhs_sparse.elements_as_vec();
    std::vector<double> rhs_coefficient_magnitudes;
    std::vector<Determinant> rhs_supports;
    std::vector<bool> rhs_even_operator_counts;
    rhs_coefficient_magnitudes.reserve(rhs_terms.size());
    rhs_supports.reserve(rhs_terms.size());
    rhs_even_operator_counts.reserve(rhs_terms.size());
    for (const auto& [term, coefficient] : rhs_terms) {
        rhs_coefficient_magnitudes.push_back(std::abs(coefficient));
        rhs_supports.push_back(term.cre() | term.ann());
        rhs_even_operator_counts.push_back(term.count() % 2 == 0);
    }

    // Keep enough independent work to balance irregular contraction costs without creating one
    // result operator per term for the large intermediate operators in a BCH expansion.
    constexpr std::size_t blocks_per_thread = 256;
    constexpr std::size_t max_block_size = 64;
    const auto target_blocks = blocks_per_thread * get_num_threads();
    const auto lhs_block_size =
        std::max(std::size_t{2}, std::min(max_block_size, lhs_terms.size() / target_blocks));
    const std::size_t num_blocks = (lhs_terms.size() + lhs_block_size - 1) / lhs_block_size;
    std::vector<GeneralizedNormalOrderedSparseOperator> block_results;
    block_results.reserve(num_blocks);
    for (std::size_t block = 0; block < num_blocks; ++block) {
        block_results.emplace_back(lhs.vacuum(), lhs.norb(), lhs.max_cumulant());
    }

    parallel_for_dynamic_thread(0, num_blocks, [&](std::size_t block) {
        const auto block_begin = block * lhs_block_size;
        const auto block_end = std::min(block_begin + lhs_block_size, lhs_terms.size());
        GeneralizedNormalOrderComputer normal_orderer(lhs.vacuum(), lhs.norb(), lhs.max_cumulant(),
                                                      screen_thresh_, max_rank_);
        SparseOperator batch;
        batch.reserve(std::min((block_end - block_begin) * rhs_terms.size() * std::size_t{2},
                               std::size_t{500000}));
        SQOperatorProductComputer product_computer;
        const std::function<void(const SQOperatorString&, const sparse_scalar_t)> add_to_batch =
            [&batch, &normal_orderer](const SQOperatorString& term, sparse_scalar_t coefficient) {
                if (normal_orderer.could_contribute(term)) {
                    batch.add(term, coefficient);
                }
            };
        for (std::size_t lhs_index = block_begin; lhs_index < block_end; ++lhs_index) {
            const auto& [lhs_term, lhs_coefficient] = lhs_terms[lhs_index];
            const auto lhs_coefficient_magnitude = std::abs(lhs_coefficient);
            const auto lhs_support = lhs_term.cre() | lhs_term.ann();
            const bool lhs_even_operator_count = lhs_term.count() % 2 == 0;
            for (std::size_t rhs_index = 0; rhs_index < rhs_terms.size(); ++rhs_index) {
                const auto& [rhs_term, rhs_coefficient] = rhs_terms[rhs_index];
                // The two product directions can contribute the same bare string, so 2 |c_l c_r|
                // bounds the coefficient generated by one commutator pair before GNO.
                if (2.0 * lhs_coefficient_magnitude * rhs_coefficient_magnitudes[rhs_index] <=
                    screen_thresh_) {
                    continue;
                }
                // Disjoint strings commute when at least one contains an even number of operators.
                // Precomputed supports reduce this from four determinant intersections per pair to
                // one disjointness check.
                if ((lhs_even_operator_count or rhs_even_operator_counts[rhs_index]) and
                    lhs_support.is_disjoint_from(rhs_supports[rhs_index])) {
                    continue;
                }
                const auto factor = lhs_coefficient * rhs_coefficient;
                product_computer.commutator(lhs_term, rhs_term, factor, add_to_batch);
            }
        }

        auto& block_result = block_results[block];
        for (const auto& [term, coefficient] : batch.elements()) {
            if (coefficient != sparse_scalar_t{0.0}) {
                normal_orderer.add_term(term, coefficient, block_result);
            }
        }
    });

    GeneralizedNormalOrderedSparseOperator result(lhs.vacuum(), lhs.norb(), lhs.max_cumulant());
    std::size_t result_size = 0;
    for (const auto& block_result : block_results) {
        result_size += block_result.size();
    }
    result.reserve(std::min(result_size, std::size_t{1000000}));
    for (const auto& block_result : block_results) {
        result += block_result;
    }
    return result.truncate(max_rank_, screen_thresh_);
}

GeneralizedNormalOrderedSparseOperator
generalized_normal_ordered_commutator(const GeneralizedNormalOrderedSparseOperator& lhs,
                                      const GeneralizedNormalOrderedSparseOperator& rhs,
                                      int max_rank, double screen_thresh) {
    return GeneralizedNormalOrderedProductComputer(max_rank, screen_thresh).commutator(lhs, rhs);
}

} // namespace forte2
