#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

#include "ci/ci_string_address.h"

namespace forte2 {

/// @brief Connect a hole string class to a variational string class by electron creation.
/// @details The stored orbital tuples have the required GAS occupation and irrep changes.
/// Applying the creation operators in descending orbital order gives the adjoint of the
/// annihilation used in the Harrison-Zarrabian sigma build and the RDMs. Each operator
/// updates the local address in its GAS space.
/// @tparam Rank Number of electrons to create.
template <size_t Rank> class HoleSubstitutions {
  public:
    using Orbitals = std::array<uint8_t, Rank>;

    HoleSubstitutions(const StringAddress& holes, int hole_class, const StringAddress& strings,
                      int string_class)
        : holes_(holes), strings_(strings), hole_class_(hole_class), string_class_(string_class) {
        const auto& basis = holes.basis();
        const auto& from = holes.occupation(hole_class);
        const auto& to = strings.occupation(string_class);
        size_t remaining = Rank;
        int prefix = 0;
        for (size_t g = 0; g < basis.ngas(); ++g) {
            const int change = to[g] - from[g];
            if (change < 0 || static_cast<size_t>(change) > remaining)
                return;
            remaining -= change;
            // Fill from the right so that higher orbitals are created first. The preceding
            // GAS occupations are still those in the hole string at each creation.
            for (int n = 0; n < change; ++n) {
                gases_[remaining + n] = g;
                electrons_[remaining + n] = from[g] + n;
            }
            if ((prefix * change) % 2 != 0)
                gas_sign_ = -gas_sign_;
            prefix += from[g];
        }
        if (remaining != 0)
            return;

        Orbitals orbitals;
        append_orbitals<0>(orbitals, basis.norb(),
                           holes.irrep(hole_class) ^ strings.irrep(string_class));
    }

    bool empty() const { return orbitals_.empty(); }
    const auto& orbitals() const { return orbitals_; }

    /// Visit (orbitals, sign, rank in the filled string class) for one hole product address.
    template <typename Visit>
    void for_each_hole_substitution(size_t hole_rank, std::span<LocalStringAddress> coordinates,
                                    Visit&& visit) const {
        if (empty())
            return;
        holes_.unrank(hole_class_, hole_rank, coordinates);
        visit_creations<0>(0, level_size<0>(), coordinates, gas_sign_, visit);
    }

  private:
    /// @brief A creation operator shared by several orbital tuples.
    /// @details The range [first, last) indexes the next operators in prefixes_, or the full
    /// tuples in orbitals_ at the last step. The full tuples also give the orbital indices
    /// used in the contraction.
    struct CreationPrefix {
        uint8_t orbital;
        size_t first;
        size_t last;
    };

    template <size_t Step> size_t level_size() const {
        if constexpr (Step + 1 == Rank)
            return orbitals_.size();
        else
            return prefixes_[Step].size();
    }

    template <size_t Step> void append_orbitals(Orbitals& orbitals, size_t end, int irrep) {
        const auto& basis = holes_.basis();
        const size_t gas = gases_[Step];
        for (size_t p = basis.offsets()[gas]; p < std::min(end, basis.offsets()[gas + 1]); ++p) {
            orbitals[Step] = static_cast<uint8_t>(p);
            const int next_irrep = irrep ^ basis.irreps()[p];
            if constexpr (Step + 1 == Rank) {
                if (next_irrep == 0)
                    orbitals_.push_back(orbitals);
            } else {
                const size_t first = level_size<Step + 1>();
                append_orbitals<Step + 1>(orbitals, p, next_irrep);
                const size_t last = level_size<Step + 1>();
                if (first != last)
                    prefixes_[Step].push_back({orbitals[Step], first, last});
            }
        }
    }

    template <size_t Step, typename Visit>
    void visit_creations(size_t first, size_t last, std::span<LocalStringAddress> coordinates,
                         int sign, Visit& visit) const {
        const auto& basis = holes_.basis();
        const size_t gas = gases_[Step];
        auto& factor = coordinates[gas];
        const auto before = factor;
        const auto row = basis.creation_row(gas, electrons_[Step], before);
        for (size_t i = first; i < last; ++i) {
            const auto orbital = [&] {
                if constexpr (Step + 1 == Rank)
                    return orbitals_[i][Step];
                else
                    return prefixes_[Step][i].orbital;
            }();
            const int local = row.create(orbital, factor);
            if (local == 0)
                continue;
            if constexpr (Step + 1 == Rank) {
                visit(orbitals_[i], sign * local, strings_.rank(string_class_, coordinates));
            } else {
                const auto& prefix = prefixes_[Step][i];
                visit_creations<Step + 1>(prefix.first, prefix.last, coordinates, sign * local,
                                          visit);
            }
        }
        factor = before;
    }

    const StringAddress& holes_;
    const StringAddress& strings_;
    int hole_class_;
    int string_class_;
    std::array<size_t, Rank> gases_{};
    std::array<int, Rank> electrons_{};
    int gas_sign_ = 1;
    std::array<std::vector<CreationPrefix>, Rank - 1> prefixes_;
    std::vector<Orbitals> orbitals_;
};

/// @brief Store the single excitations connecting two string classes with the same GAS layout.
/// @details Each excitation annihilates an electron in orbital q and creates one in orbital p.
/// The orbital pairs are shared by the source strings and grouped by q so the annihilation
/// can be evaluated once for all its creation partners.
class ExcitationSubstitutions {
  public:
    ExcitationSubstitutions(const StringAddress& source, int source_class,
                            const StringAddress& target, int target_class);

    bool empty() const { return annihilations_.empty(); }

    /// Visit the pairs that can connect the classes, independent of the source string.
    template <typename Visit> void for_each_orbital_pair(Visit&& visit) const {
        for (const auto& removal : annihilations_)
            for (size_t i = removal.first; i < removal.last; ++i)
                visit(creations_[i], removal.q);
    }

    /// Visit (p, q, sign, rank in the target class), changing only the acted-on factors.
    template <typename Visit>
    void for_each_excitation(size_t rank, std::span<LocalStringAddress> coordinates,
                             Visit&& visit) const {
        if (empty())
            return;
        const auto& basis = source_.basis();
        const auto& occupation = source_.occupation(source_class_);
        source_.unrank(source_class_, rank, coordinates);
        for (const auto& [q, gas_sign, first, last] : annihilations_) {
            const size_t from = basis.gas_of(q);
            auto& source = coordinates[from];
            const auto saved_source = source;
            const int annihilation = basis.annihilate(from, occupation[from], source, q);
            if (annihilation == 0)
                continue;

            const size_t to = basis.gas_of(creations_[first]);
            const int electrons = occupation[to] - (to == from);
            auto& target = coordinates[to];
            // When p and q belong to the same GAS, this includes the annihilation.
            const auto saved_target = target;
            const auto row = basis.creation_row(to, electrons, saved_target);
            for (size_t i = first; i < last; ++i) {
                const auto p = creations_[i];
                const int creation = row.create(p, target);
                if (creation != 0)
                    visit(p, q, gas_sign * annihilation * creation,
                          target_.rank(target_class_, coordinates));
            }
            target = saved_target;
            source = saved_source;
        }
    }

  private:
    /// @brief An annihilation operator and its allowed creation partners.
    /// @details The range [first, last) indexes creations_ for orbital q. gas_sign is the
    /// sign from the electrons in the GAS spaces preceding the operators.
    struct Annihilation {
        uint8_t q;
        int gas_sign;
        size_t first;
        size_t last;
    };

    const StringAddress& source_;
    const StringAddress& target_;
    int source_class_;
    int target_class_;
    std::vector<Annihilation> annihilations_;
    std::vector<uint8_t> creations_;
};

} // namespace forte2
