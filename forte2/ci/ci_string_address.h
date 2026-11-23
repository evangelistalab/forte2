#pragma once

#include <map>
#include <memory>
#include <optional>
#include <span>
#include <vector>

#include "ci/ci_occupation.h"
#include "determinant/determinant.h"

namespace forte2 {

/// A string in one GAS factor. The rank is local to its electron count and irrep.
struct LocalStringAddress {
    size_t rank;
    int irrep;
};

/// The size of each GAS, its orbital labels, and the shared local string tables.
/// A table is built once per (GAS slot, electron count) needed by any spin or operator.
/// The stored bits use active orbital indices, but each string occupies only its own GAS.
class GASStringBasis {
  public:
    struct Sector {
        std::vector<String> strings;
        /// Positions in the original enumeration, before partitioning the strings by irrep.
        std::vector<size_t> positions;
        /// Local target ranks for allowed creation or annihilation, in orbital order.
        std::vector<size_t> creations;
        std::vector<size_t> annihilations;
    };

    struct Factor {
        std::vector<Sector> sectors;              ///< Indexed by irrep.
        std::vector<LocalStringAddress> in_order; ///< Original local enumeration order.
        ankerl::unordered_dense::map<String, LocalStringAddress, String::Hash> addresses;
    };

    GASStringBasis(int nirrep, const std::vector<std::vector<int>>& orbital_symmetry);

    void prepare(std::span<const occupation_t> occupations);
    /// Build the one-orbital substitutions between all prepared adjacent electron counts.
    void prepare_substitutions();

    size_t ngas() const { return sizes_.size(); }
    size_t norb() const { return irreps_.size(); }
    int nirrep() const { return nirrep_; }
    const auto& sizes() const { return sizes_; }
    const auto& irreps() const { return irreps_; }
    const auto& offsets() const { return offsets_; }
    size_t gas_of(size_t orbital) const { return orbital_gas_[orbital]; }
    const String& mask(size_t gas) const { return masks_[gas]; }
    const Factor& factor(size_t gas, int electrons) const { return *factors_[gas][electrons]; }

    /// Apply a local creation or annihilation, returning its sign or zero if forbidden.
    /// The caller accounts for the sign from preceding GAS factors.
    int substitute(size_t gas, int electrons, LocalStringAddress& address, size_t orbital,
                   bool create) const;

  private:
    int nirrep_;
    std::vector<int> sizes_;
    std::vector<int> irreps_;
    std::vector<size_t> offsets_{0};
    std::vector<size_t> orbital_gas_;
    std::vector<String> masks_;
    std::vector<String> orbital_prefix_;
    std::vector<std::vector<std::optional<Factor>>> factors_;
};

/// A spin-string address space with row-major GAS product ordering within each class.
/// A class is (occupation index * nirrep + total irrep). The count for each suffix of the
/// product and each suffix irrep gives the number of admissible completions. This implements
/// the same address order as filtering the lexicographic Cartesian product by total irrep.
class StringAddress {
  public:
    struct Index {
        size_t rank;
        int string_class;
    };

    StringAddress(std::shared_ptr<const GASStringBasis> basis,
                  std::vector<occupation_t> occupations);

    const GASStringBasis& basis() const { return *basis_; }
    int nclasses() const { return nclasses_; }
    int irrep(int string_class) const { return string_class % basis_->nirrep(); }
    const occupation_t& occupation(int string_class) const;
    size_t size(int string_class) const;
    size_t total_size() const { return total_size_; }
    const auto& occupations() const { return occupations_; }
    std::optional<int> find_class(const occupation_t& occupation, int irrep) const;

    /// Coordinates are ordered by GAS slot and the caller provides ngas() entries.
    size_t rank(int string_class, std::span<const LocalStringAddress> coordinates) const;
    void unrank(int string_class, size_t rank, std::span<LocalStringAddress> coordinates) const;

    /// Convert to or from a full active-space string at determinant-facing call sites.
    String string_at(int string_class, size_t rank, std::span<LocalStringAddress> work) const;
    std::optional<Index> find(const String& string, std::span<LocalStringAddress> work,
                              occupation_t& occupation) const;

  private:
    struct Product {
        std::vector<const GASStringBasis::Factor*> factors;
        std::vector<size_t> suffix_counts; ///< Indexed by (GAS boundary, suffix irrep).
    };

    const Product& product(int string_class) const {
        return products_[string_class / basis_->nirrep()];
    }

    std::shared_ptr<const GASStringBasis> basis_;
    std::vector<occupation_t> occupations_;
    std::vector<Product> products_;
    std::map<occupation_t, int> occupation_indices_;
    int nclasses_ = 0;
    size_t total_size_ = 0;
};

/// Checked index arithmetic for product dimensions and coefficient offsets.
size_t checked_ci_product(size_t a, size_t b);
size_t checked_ci_sum(size_t a, size_t b);

} // namespace forte2
