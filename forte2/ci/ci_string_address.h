#pragma once

#include <cassert>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <vector>

#include "ci/ci_occupation.h"
#include "determinant/determinant.h"

namespace forte2 {

/// @brief The irrep and rank of a string in one GAS space.
/// @details The rank indexes strings with the same electron count and irrep in that space.
struct LocalStringAddress {
    size_t rank;
    int irrep;
};

/// @brief Store the GAS orbital layout and the strings and substitutions for each GAS space.
/// @details Each GAS space has one table for each electron count needed by either spin.
/// The variational and intermediate string spaces share these tables. The bits in a local
/// string use active orbital indices and are set only for orbitals in its GAS space.
class GASStringBasis {
  public:
    /// @brief Strings and substitution tables for one GAS space, electron count, and irrep.
    /// @details Each substitution table stores one row per source string. Creation rows are
    /// indexed by vacant orbitals and annihilation rows by occupied orbitals, in orbital order.
    /// The target irrep is the source irrep XOR the orbital irrep.
    struct Sector {
        std::vector<String> strings;
        /// Positions in the original enumeration, before partitioning the strings by irrep.
        std::vector<size_t> positions;
        /// Ranks of the target strings in their respective irreps.
        std::vector<size_t> creations;
        std::vector<size_t> annihilations;
    };

    /// @brief All strings with a given electron count in one GAS space.
    /// @details sectors groups the strings by irrep and addresses maps occupation bit strings
    /// to their local addresses. Product addressing uses in_order and Sector::positions to
    /// recover the local enumeration order when there are several GAS spaces and irreps.
    struct Factor {
        std::vector<Sector> sectors;
        std::vector<LocalStringAddress> in_order;
        ankerl::unordered_dense::map<String, LocalStringAddress, String::Hash> addresses;
    };

    /// @brief A row of the creation table for one local string.
    /// @details ranks refers to the target ranks in the basis, in order of the vacant orbitals
    /// in string. orbital_start is the first orbital of the source GAS space.
    struct CreationRow {
        const GASStringBasis& basis;
        String string;
        std::span<const size_t> ranks;
        size_t orbital_start;
        int irrep;

        /// Set the target local address and return the sign, or zero for an occupied orbital.
        int create(size_t orbital, LocalStringAddress& target) const {
            if (string.get_bit(orbital))
                return 0;
            const size_t occupied = string.intersection_count(basis.orbital_prefix_[orbital]);
            const size_t vacancy = orbital - orbital_start - occupied;
            target = {ranks[vacancy], irrep ^ basis.irreps_[orbital]};
            return occupied % 2 == 0 ? 1 : -1;
        }
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

    CreationRow creation_row(size_t gas, int electrons, LocalStringAddress address) const {
        const auto& sector = factor(gas, electrons).sectors[address.irrep];
        const size_t vacancies = sizes_[gas] - electrons;
        const auto ranks = std::span(sector.creations).subspan(address.rank * vacancies, vacancies);
        return {*this, sector.strings[address.rank], ranks, offsets_[gas], address.irrep};
    }

    /// Apply a local annihilation, returning its sign or zero for a vacant orbital.
    /// The caller accounts for the sign from preceding GAS factors for both operators.
    int annihilate(size_t gas, int electrons, LocalStringAddress& address, size_t orbital) const {
        const auto& sector = factor(gas, electrons).sectors[address.irrep];
        const auto& local = sector.strings[address.rank];
        if (!local.get_bit(orbital))
            return 0;
        const size_t occupied = local.intersection_count(orbital_prefix_[orbital]);
        address.rank = sector.annihilations[address.rank * electrons + occupied];
        address.irrep ^= irreps_[orbital];
        return occupied % 2 == 0 ? 1 : -1;
    }

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

/// @brief Map between the address of a spin string and its local GAS string addresses.
/// @details A string class is indexed by occupation index * nirrep + total irrep. Within
/// each class, strings follow the lexicographic order of the Cartesian product of the local
/// strings with the required total irrep. The counts for the remaining GAS spaces give the
/// number of strings that complete a given choice of local strings.
class StringAddress {
  public:
    /// @brief The class index and the rank of a spin string within that class.
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
    /// With one factor, its irrep and local rank are the product irrep and rank.
    size_t rank(int string_class, std::span<const LocalStringAddress> coordinates) const {
        if (basis_->ngas() == 1) {
            assert(coordinates.front().irrep == irrep(string_class));
            return coordinates.front().rank;
        }
        return rank_product(string_class, coordinates);
    }
    void unrank(int string_class, size_t rank, std::span<LocalStringAddress> coordinates) const {
        if (basis_->ngas() == 1) {
            assert(rank < size(string_class));
            coordinates.front() = {rank, irrep(string_class)};
        } else {
            unrank_product(string_class, rank, coordinates);
        }
    }

    /// Convert to or from a full active-space string at determinant-facing call sites.
    String string_at(int string_class, size_t rank, std::span<LocalStringAddress> work) const;
    std::optional<Index> find(const String& string, std::span<LocalStringAddress> work,
                              occupation_t& occupation) const;

  private:
    /// @brief Local string tables and counts for a single GAS occupation pattern.
    /// @details factors contains one table per GAS space, in input order. The entry
    /// suffix_counts[g * nirrep + h] counts strings formed from GAS spaces g onward with irrep h.
    struct Product {
        std::vector<const GASStringBasis::Factor*> factors;
        std::vector<size_t> suffix_counts;
    };

    const Product& product(int string_class) const {
        return products_[string_class / basis_->nirrep()];
    }
    size_t rank_product(int string_class, std::span<const LocalStringAddress> coordinates) const;
    void unrank_product(int string_class, size_t rank,
                        std::span<LocalStringAddress> coordinates) const;

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
