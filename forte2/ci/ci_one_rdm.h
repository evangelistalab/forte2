#pragma once

#include "ci/ci_string_substitutions.h"
#include "ci/ci_strings.h"
#include "helpers/blas.h"

namespace forte2 {

/// Accumulate <C_left|a†_p a_q|C_right> for two vectors in the same CI space.
template <typename Scalar, typename RDMView>
void accumulate_one_rdm(const CIStrings& lists, Spin spin, std::span<Scalar> left,
                        std::span<Scalar> right, std::span<Scalar> left_scratch,
                        std::span<Scalar> right_scratch, RDMView rdm) {
    const auto& strings = lists.strings(spin);
    std::vector<LocalStringAddress> work(lists.ngas_spaces());
    for (const auto& ket : lists.blocks()) {
        const auto spectators = ket.spectator_size(spin);
        const auto cr = gather_block(right, right_scratch, spin, ket);
        for (const auto& bra : lists.blocks()) {
            if (ket.spectator_class(spin) != bra.spectator_class(spin))
                continue;
            const int first = ket.string_class(spin);
            const ExcitationSubstitutions substitutions(strings, first, strings,
                                                        bra.string_class(spin));
            if (substitutions.empty())
                continue;
            const auto cl = gather_block(left, left_scratch, spin, bra);
            for (size_t i = 0; i < strings.size(first); ++i) {
                substitutions.for_each_excitation(i, work, [&](auto p, auto q, int sign, size_t j) {
                    rdm(p, q) += Scalar(sign) * dot(spectators, cl.data() + j * spectators, 1,
                                                    cr.data() + i * spectators, 1);
                });
            }
        }
    }
}

} // namespace forte2
