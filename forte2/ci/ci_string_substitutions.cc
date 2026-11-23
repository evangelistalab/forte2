#include "ci_string_substitutions.h"

namespace forte2 {

ExcitationSubstitutions::ExcitationSubstitutions(const StringAddress& source, int source_class,
                                                 const StringAddress& target, int target_class)
    : source_(source), target_(target), source_class_(source_class), target_class_(target_class) {
    const auto& basis = source.basis();
    const auto& source_occ = source.occupation(source_class);
    const auto& target_occ = target.occupation(target_class);
    const int required_irrep = source.irrep(source_class) ^ target.irrep(target_class);

    // For a transfer, exactly one GAS gains an electron and one loses one.
    const auto none = basis.ngas();
    size_t from = none, to = none;
    for (size_t g = 0; g < basis.ngas(); ++g) {
        const auto change = target_occ[g] - source_occ[g];
        if (change == 0)
            continue;
        if (change == -1 && from == none)
            from = g;
        else if (change == 1 && to == none)
            to = g;
        else
            return;
    }
    const auto append_pairs = [&](size_t create_gas, size_t annihilate_gas, int sign) {
        for (size_t q = basis.offsets()[annihilate_gas]; q < basis.offsets()[annihilate_gas + 1];
             ++q) {
            const size_t first = creations_.size();
            for (size_t p = basis.offsets()[create_gas]; p < basis.offsets()[create_gas + 1]; ++p) {
                if ((basis.irreps()[p] ^ basis.irreps()[q]) != required_irrep)
                    continue;
                if (create_gas == annihilate_gas && p != q &&
                    source_occ[create_gas] == basis.sizes()[create_gas])
                    continue;
                creations_.push_back(static_cast<uint8_t>(p));
            }
            if (first != creations_.size())
                annihilations_.push_back({static_cast<uint8_t>(q), sign, first, creations_.size()});
        }
    };
    if (from == none && to == none) {
        for (size_t g = 0; g < basis.ngas(); ++g)
            if (source_occ[g] != 0)
                append_pairs(g, g, 1);
    } else if (from != none && to != none) {
        // Each operator sees the preceding GAS occupations at the time it acts. If the
        // removed electron preceded the creation, its parity contribution has disappeared.
        int preceding = -(from < to);
        for (size_t g = 0; g < std::max(from, to); ++g)
            preceding += source_occ[g] * ((g < from) + (g < to));
        append_pairs(to, from, preceding % 2 == 0 ? 1 : -1);
    }
}

} // namespace forte2
