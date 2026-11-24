#pragma once

#include <algorithm>
#include <cstddef>
#include <span>
#include <utility>
#include <vector>

#include "ci/ci_strings.h"

namespace forte2 {

/// @brief Index layout for CI matrices whose orbital rows and columns are grouped by irrep.
/// @details A row represents one orbital, pair, or triplet. Its canonical
/// index follows the original orbital ordering and the conventions in helpers/indexing.hpp.
/// The tuple irrep is the XOR of its orbital irreps. Rows of each irrep retain their relative
/// canonical order, and local_row() maps a canonical index to its row within that irrep.
///
/// The layout stores index maps and matrix offsets. Callers supply the integral or RDM values.
/// A packed matrix concatenates the row-major square blocks in irrep order. For example, with
/// orbital irreps [0, 1, 0], the blocks contain canonical rows [0, 2] and [1]. Their offsets are
/// 0 and 4, respectively, and the packed matrix contains five elements.
class CIOrbitalIrrepBlocks {
  public:
    /// @brief The orbital indices and matrix offset for one irrep.
    /// @details The position in blocks_ gives the irrep. The same indices are used for rows
    /// and columns. An empty block occupies zero elements in the packed matrix.
    struct Block {
        /// Canonical orbital-tuple index for each local row. The column mapping is identical.
        std::vector<size_t> canonical_rows;
        /// Offset in scalar elements from the start of the packed matrix to this block.
        size_t matrix_offset = 0;

        /// Number of rows and columns in this square block.
        size_t size() const { return canonical_rows.size(); }
    };

    /// Single orbitals, indexed by p.
    static CIOrbitalIrrepBlocks orbitals(const CIStrings& lists) {
        CIOrbitalIrrepBlocks result(lists.nirrep());
        for (const int irrep : lists.orbital_irreps())
            result.add_row(irrep);
        result.set_matrix_offsets();
        return result;
    }

    /// Opposite-spin pairs (p,q), with both indices unrestricted.
    static CIOrbitalIrrepBlocks ordered_pairs(const CIStrings& lists) {
        return make_pairs(lists, [norb = lists.norb()](size_t) { return norb; });
    }

    /// Antisymmetric pairs (p,q) with p > q.
    static CIOrbitalIrrepBlocks strict_pairs(const CIStrings& lists) {
        return make_pairs(lists, [](size_t p) { return p; });
    }

    /// Knowles-Handy excitation pairs (p,q) with p >= q.
    static CIOrbitalIrrepBlocks pairs_with_diagonal(const CIStrings& lists) {
        return make_pairs(lists, [](size_t p) { return p + 1; });
    }

    /// Two alpha indices p > q followed by an unrestricted beta index r.
    static CIOrbitalIrrepBlocks aab_triplets(const CIStrings& lists) {
        return make_triplets(
            lists, [](size_t p) { return p; }, [norb = lists.norb()](size_t) { return norb; });
    }

    /// An unrestricted alpha index p followed by two beta indices q > r.
    static CIOrbitalIrrepBlocks abb_triplets(const CIStrings& lists) {
        return make_triplets(
            lists, [norb = lists.norb()](size_t) { return norb; }, [](size_t q) { return q; });
    }

    /// Antisymmetric spinor triplets with p > q > r.
    static CIOrbitalIrrepBlocks spinor_triplets(const CIStrings& lists) {
        return make_triplets(lists, [](size_t p) { return p; }, [](size_t q) { return q; });
    }

    /// Select the orbital irrep required for an intermediate to reach the target CI state.
    /// class_Ka and class_Kb label the intermediate's alpha and beta strings, including any
    /// spectator. The required tuple irrep is the target irrep XOR the two string irreps.
    /// Pass the CIStrings whose orbital labels and irrep count were used to build this layout.
    const Block& for_intermediate(const CIStrings& lists, size_t class_Ka, size_t class_Kb) const {
        // The string-class number is occupation * nirrep + irrep. Irrep products use XOR.
        const auto nirrep = lists.nirrep();
        const auto intermediate_irrep = (class_Ka % nirrep) ^ (class_Kb % nirrep);
        return blocks_[lists.symmetry() ^ intermediate_irrep];
    }

    /// Return the row within the irrep block that contains the given canonical index.
    size_t local_row(size_t canonical_index) const { return local_rows_[canonical_index]; }
    /// Total number of scalar elements in the packed square blocks.
    size_t packed_matrix_size() const { return packed_matrix_size_; }
    /// Maximum row count across the orbital irrep blocks.
    size_t max_rows() const { return max_rows_; }
    /// Whether all rows fit in a single block in canonical order, including an empty layout.
    bool is_dense() const { return max_rows_ == local_rows_.size(); }

    /// Fill the packed matrix with element(row, col) evaluated at canonical indices.
    /// Only rows and columns of the same irrep are passed to element().
    template <typename Scalar, typename Element>
    void pack_matrix(std::vector<Scalar>& packed, Element element) const {
        packed.resize(packed_matrix_size_);
        for (const auto& block : blocks_) {
            size_t offset = block.matrix_offset;
            for (const size_t row : block.canonical_rows) {
                for (const size_t col : block.canonical_rows)
                    packed[offset++] = element(row, col);
            }
        }
    }

    /// Pack a two-electron integral matrix using orbital pairs on both axes.
    /// pair_from_index() inverts the factory's canonical pair ordering. The resulting orbital
    /// pairs are passed to element() in row, column order.
    template <typename Scalar, typename PairFromIndex, typename Element>
    void pack_pair_matrix(std::vector<Scalar>& packed, PairFromIndex pair_from_index,
                          Element element) const {
        std::vector<std::pair<size_t, size_t>> pairs;
        pairs.reserve(local_rows_.size());
        for (size_t row = 0; row < local_rows_.size(); ++row)
            pairs.push_back(pair_from_index(row));

        pack_matrix(packed,
                    [&](size_t row, size_t col) { return element(pairs[row], pairs[col]); });
    }

    /// Copy packed blocks into a dense row-major matrix in canonical order.
    /// Entries between different irreps retain their initial values.
    template <typename Scalar> void unpack_matrix(const Scalar* packed, Scalar* matrix) const {
        const size_t stride = local_rows_.size();
        for (const auto& block : blocks_) {
            size_t offset = block.matrix_offset;
            for (const size_t row : block.canonical_rows) {
                for (const size_t col : block.canonical_rows)
                    matrix[row * stride + col] = packed[offset++];
            }
        }
    }

  private:
    explicit CIOrbitalIrrepBlocks(int nirrep) : blocks_(nirrep) {}

    static CIOrbitalIrrepBlocks make_pairs(const CIStrings& lists, auto end_q) {
        CIOrbitalIrrepBlocks result(lists.nirrep());
        const auto& irreps = lists.orbital_irreps();
        for (size_t p = 0; p < irreps.size(); ++p) {
            for (size_t q = 0; q < end_q(p); ++q)
                result.add_row(irreps[p] ^ irreps[q]);
        }
        result.set_matrix_offsets();
        return result;
    }

    static CIOrbitalIrrepBlocks make_triplets(const CIStrings& lists, auto end_q, auto end_r) {
        CIOrbitalIrrepBlocks result(lists.nirrep());
        const auto& irreps = lists.orbital_irreps();
        for (size_t p = 0; p < irreps.size(); ++p) {
            for (size_t q = 0; q < end_q(p); ++q) {
                const int pair_irrep = irreps[p] ^ irreps[q];
                for (size_t r = 0; r < end_r(q); ++r)
                    result.add_row(pair_irrep ^ irreps[r]);
            }
        }
        result.set_matrix_offsets();
        return result;
    }

    void add_row(int irrep) {
        auto& rows = blocks_[irrep].canonical_rows;
        local_rows_.push_back(rows.size());
        rows.push_back(local_rows_.size() - 1);
    }

    void set_matrix_offsets() {
        for (auto& block : blocks_) {
            block.matrix_offset = packed_matrix_size_;
            packed_matrix_size_ += block.size() * block.size();
            max_rows_ = std::max(max_rows_, block.size());
        }
    }

    /// One block per irrep, including empty blocks.
    std::vector<Block> blocks_;
    /// Maps each canonical index to its local row.
    std::vector<size_t> local_rows_;
    size_t max_rows_ = 0;
    size_t packed_matrix_size_ = 0;
};

/// @brief The integral matrix for the orbital indices that can act on an intermediate.
/// @details Canonical indices are mapped to the restricted row order. The original packed integral
/// block is borrowed when every row is active. A subset is packed into a smaller square
/// matrix using the selected rows on both axes.
/// @tparam Scalar Integral type, double for spatial orbitals or std::complex<double> for spinors.
template <typename Scalar> class CIOrbitalMatrixBlock {
  public:
    CIOrbitalMatrixBlock(const CIOrbitalIrrepBlocks& layout,
                         const CIOrbitalIrrepBlocks::Block& block, std::span<const Scalar> matrix,
                         std::vector<size_t> rows)
        : layout_(layout) {
        std::ranges::sort(rows);
        const auto last = std::unique(rows.begin(), rows.end());
        rows.erase(last, rows.end());
        size_ = rows.size();

        const auto full = matrix.subspan(block.matrix_offset, block.size() * block.size());
        if (size_ == block.size()) {
            full_block_ = full;
            return;
        }
        row_map_.resize(block.size());
        for (size_t r = 0; r < size_; ++r) {
            rows[r] = layout_.local_row(rows[r]);
            row_map_[rows[r]] = r;
        }
        packed_.reserve(size_ * size_);
        for (const auto r : rows)
            for (const auto s : rows)
                packed_.push_back(full[r * block.size() + s]);
    }

    size_t size() const { return size_; }
    const Scalar* data() const { return row_map_.empty() ? full_block_.data() : packed_.data(); }
    size_t row(size_t canonical) const {
        const auto row = layout_.local_row(canonical);
        return row_map_.empty() ? row : row_map_[row];
    }

  private:
    const CIOrbitalIrrepBlocks& layout_;
    std::span<const Scalar> full_block_;
    std::vector<size_t> row_map_;
    std::vector<Scalar> packed_;
    size_t size_ = 0;
};

/// @brief Store the irrep blocks used to accumulate a reduced density matrix.
/// @details The caller owns the layout and a zero-initialized square output matrix indexed by
/// canonical orbital tuples. Both must outlive this object. When rows span several irreps, the
/// object owns a temporary packed matrix, and unpack() copies the accumulated blocks to the
/// output. A single block can accumulate directly into the output because its local rows already
/// follow the canonical order.
/// @tparam Scalar RDM element type, double or std::complex<double>.
template <typename Scalar> class CIIrrepRDM {
  public:
    CIIrrepRDM(const CIOrbitalIrrepBlocks& layout, Scalar* output)
        : layout_(layout), output_(output),
          packed_(layout.is_dense() ? 0 : layout.packed_matrix_size()) {}

    /// Return the start of a nonempty block from this object's layout.
    Scalar* data(const CIOrbitalIrrepBlocks::Block& block) {
        return (packed_.empty() ? output_ : packed_.data()) + block.matrix_offset;
    }

    /// Write the accumulated blocks to the caller's output. Call after the final contraction.
    void unpack() const {
        if (!packed_.empty())
            layout_.unpack_matrix(packed_.data(), output_);
    }

  private:
    /// Map from local rows to canonical indices, owned by the caller.
    const CIOrbitalIrrepBlocks& layout_;
    /// RDM output in canonical order, owned by the caller.
    Scalar* output_;
    /// Temporary storage for the RDM blocks when an unpacking step is needed.
    std::vector<Scalar> packed_;
};

} // namespace forte2
