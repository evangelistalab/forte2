#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

#include "ci/ci_string_address.h"

namespace forte2 {

/// Connect two string classes by creating the electrons missing from a hole string.
/// Only operators with the required GAS occupation change and total irrep are stored.
/// The orbitals appear in descending order, so creating them in that order is the adjoint of
/// the annihilation order used for HZ and the RDMs. Untouched factors keep their coordinates.
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
        const auto append = [&](this const auto& self, size_t step, size_t end, int irrep) -> void {
            if (step == Rank) {
                if (irrep == 0)
                    orbitals_.push_back(orbitals);
                return;
            }
            const auto g = gases_[step];
            for (size_t p = basis.offsets()[g]; p < std::min(end, basis.offsets()[g + 1]); ++p) {
                orbitals[step] = static_cast<uint8_t>(p);
                self(step + 1, p, irrep ^ basis.irreps()[p]);
            }
        };
        append(0, basis.norb(), holes.irrep(hole_class) ^ strings.irrep(string_class));
    }

    bool empty() const { return orbitals_.empty(); }
    const auto& orbitals() const { return orbitals_; }

    /// Visit (orbitals, sign, rank in the filled string class) for one hole product address.
    template <typename Visit>
    void for_each_hole_substitution(size_t hole_rank, std::span<LocalStringAddress> coordinates,
                                    Visit&& visit) const {
        if (empty())
            return;
        const auto& basis = holes_.basis();
        holes_.unrank(hole_class_, hole_rank, coordinates);
        for (const auto& orbitals : orbitals_) {
            std::array<LocalStringAddress, Rank> saved;
            size_t step = 0;
            int sign = gas_sign_;
            for (; step < Rank; ++step) {
                auto& factor = coordinates[gases_[step]];
                saved[step] = factor;
                const int local =
                    basis.substitute(gases_[step], electrons_[step], factor, orbitals[step], true);
                if (local == 0)
                    break;
                sign *= local;
            }
            if (step == Rank)
                visit(orbitals, sign, strings_.rank(string_class_, coordinates));
            while (step > 0) {
                --step;
                coordinates[gases_[step]] = saved[step];
            }
        }
    }

  private:
    const StringAddress& holes_;
    const StringAddress& strings_;
    int hole_class_;
    int string_class_;
    std::array<size_t, Rank> gases_{};
    std::array<int, Rank> electrons_{};
    int gas_sign_ = 1;
    std::vector<Orbitals> orbitals_;
};

/// The a†_p a_q operators connecting two classes in the same GAS orbital layout.
/// The list stores only the orbital pairs, shared by all source strings in the class.
class ExcitationSubstitutions {
  public:
    struct Orbitals {
        uint8_t p;
        uint8_t q;
        int gas_sign;
    };

    ExcitationSubstitutions(const StringAddress& source, int source_class,
                            const StringAddress& target, int target_class);

    bool empty() const { return orbitals_.empty(); }
    const auto& orbitals() const { return orbitals_; }

    /// Visit (p, q, sign, rank in the target class), changing only the acted-on factors.
    template <typename Visit>
    void for_each_excitation(size_t rank, std::span<LocalStringAddress> coordinates,
                             Visit&& visit) const {
        if (empty())
            return;
        const auto& basis = source_.basis();
        const auto& occupation = source_.occupation(source_class_);
        source_.unrank(source_class_, rank, coordinates);
        for (const auto& [p, q, gas_sign] : orbitals_) {
            const size_t from = basis.gas_of(q);
            const size_t to = basis.gas_of(p);
            auto& source = coordinates[from];
            auto& target = coordinates[to];
            const auto saved_source = source;
            const auto saved_target = target;
            const int annihilation = basis.substitute(from, occupation[from], source, q, false);
            if (annihilation == 0)
                continue;
            const int creation =
                basis.substitute(to, occupation[to] - (to == from), target, p, true);
            if (creation != 0)
                visit(p, q, gas_sign * annihilation * creation,
                      target_.rank(target_class_, coordinates));
            source = saved_source;
            target = saved_target;
        }
    }

  private:
    const StringAddress& source_;
    const StringAddress& target_;
    int source_class_;
    int target_class_;
    std::vector<Orbitals> orbitals_;
};

} // namespace forte2
