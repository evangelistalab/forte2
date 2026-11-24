#include "helpers/timer.hpp"
#include "helpers/np_matrix_functions.h"
#include "helpers/np_vector_functions.h"

#include "ci_sigma_builder.h"
#include "ci_string_substitutions.h"
#include "helpers/blas.h"

namespace forte2 {

np_matrix CISigmaBuilder::compute_s_1trdm(const CISigmaBuilder& sigmabuilder_right,
                                          np_vector C_left, np_vector C_right, Spin spin) const {
    const auto& sigmabuilder_left = *this;

    const auto& lists_left = sigmabuilder_left.lists_;
    const auto& lists_right = sigmabuilder_right.lists_;

    // Get the number of electrons
    const auto na_left = lists_left.na();
    const auto nb_left = lists_left.nb();
    const auto na_right = lists_right.na();
    const auto nb_right = lists_right.nb();
    const auto norb = lists_left.norb();

    if (lists_left.norb() != lists_right.norb()) {
        throw std::runtime_error("FCI transition RDMs: The number of MOs must be the same in "
                                 "the two wave functions.");
    }

    if (na_left != na_right or nb_left != nb_right) {
        throw std::runtime_error("FCI transition RDMs: The number of alpha and beta electrons "
                                 "must be the same in the "
                                 "two wave functions.");
    }
    auto rdm = make_zeros<nb::numpy, double, 2>({norb, norb});
    if ((is_alpha(spin) and (na_left < 1)) or (is_beta(spin) and (nb_left < 1)) or (norb < 1))
        return rdm;

    auto rdm_view = rdm.view();

    const auto other = is_alpha(spin) ? Spin::Beta : Spin::Alpha;
    const auto& active_right = lists_right.strings(spin);
    const auto& active_left = lists_left.strings(spin);
    const auto& spectator_right = lists_right.strings(other);
    const auto& spectator_left = lists_left.strings(other);
    const bool matching_layout = lists_left.orbital_irreps() == lists_right.orbital_irreps() &&
                                 lists_left.nirrep() == lists_right.nirrep() &&
                                 lists_left.gas_size() == lists_right.gas_size();
    std::vector<LocalStringAddress> right_work(lists_right.ngas_spaces());
    std::vector<LocalStringAddress> left_work(lists_left.ngas_spaces());
    occupation_t left_occupation(lists_left.ngas_spaces());
    std::vector<std::pair<size_t, size_t>> spectator_map;

    auto Cl_span = vector::as_span<double>(C_left);
    auto Cr_span = vector::as_span<double>(C_right);

    for (const auto& ket : lists_right.blocks()) {
        const auto cr = gather_block(Cr_span, std::span(sigmabuilder_right.TR), spin, ket);
        const auto nr = ket.spectator_size(spin);
        const auto ir = ket.string_class(spin);
        const auto sr = ket.spectator_class(spin);
        for (const auto& bra : lists_left.blocks()) {
            const auto nl = bra.spectator_size(spin);
            const auto il = bra.string_class(spin);
            const auto sl = bra.spectator_class(spin);

            if (matching_layout) {
                if (spectator_right.irrep(sr) != spectator_left.irrep(sl) ||
                    spectator_right.occupation(sr) != spectator_left.occupation(sl))
                    continue;
                const ExcitationSubstitutions substitutions(active_right, ir, active_left, il);
                if (substitutions.empty())
                    continue;
                const auto cl = gather_block(Cl_span, std::span(TL), spin, bra);
                for (size_t i = 0; i < active_right.size(ir); ++i) {
                    substitutions.for_each_excitation(
                        i, right_work, [&](auto p, auto q, int sign, size_t j) {
                            rdm_view(p, q) +=
                                sign * dot(nr, cl.data() + j * nl, 1, cr.data() + i * nr, 1);
                        });
                }
                continue;
            }

            // An orbital can belong to different GAS slots in the two states. Decode
            // one product at a time and use the destination's own address space.
            spectator_map.clear();
            for (size_t i = 0; i < nr; ++i) {
                const auto string = spectator_right.string_at(sr, i, right_work);
                const auto j = spectator_left.find(string, left_work, left_occupation);
                if (j && j->string_class == sl)
                    spectator_map.emplace_back(i, j->rank);
            }
            if (spectator_map.empty())
                continue;

            const auto cl = gather_block(Cl_span, std::span(TL), spin, bra);
            for (size_t i = 0; i < active_right.size(ir); ++i) {
                const auto string = active_right.string_at(ir, i, right_work);
                string.for_each_set_bit([&](size_t q) {
                    auto hole = string;
                    const auto annihilation = hole.destroy_unchecked(q);
                    for (size_t p = 0; p < norb; ++p) {
                        if (hole.get_bit(p))
                            continue;
                        auto result = hole;
                        const auto sign = annihilation * result.create_unchecked(p);
                        const auto j = active_left.find(result, left_work, left_occupation);
                        if (!j || j->string_class != il)
                            continue;
                        for (const auto& [l, m] : spectator_map)
                            rdm_view(p, q) += sign * cl[j->rank * nl + m] * cr[i * nr + l];
                    }
                });
            }
        }
    }
    return rdm;
}

np_matrix CISigmaBuilder::compute_a_1trdm(const CISigmaBuilder& sigmabuilder_right,
                                          np_vector C_left, np_vector C_right) const {
    return compute_s_1trdm(sigmabuilder_right, C_left, C_right, Spin::Alpha);
}

np_matrix CISigmaBuilder::compute_b_1trdm(const CISigmaBuilder& sigmabuilder_right,
                                          np_vector C_left, np_vector C_right) const {
    return compute_s_1trdm(sigmabuilder_right, C_left, C_right, Spin::Beta);
}

np_matrix CISigmaBuilder::compute_sf_1trdm(const CISigmaBuilder& sigmabuilder_right,
                                           np_vector C_left, np_vector C_right) const {
    auto sf_1trdm = make_zeros<nb::numpy, double, 2>({lists_.norb(), lists_.norb()});
    if (lists_.norb() > 0) {
        auto a_1trdm = compute_a_1trdm(sigmabuilder_right, C_left, C_right);
        auto b_1trdm = compute_b_1trdm(sigmabuilder_right, C_left, C_right);
        matrix::daxpy<double>(1.0, a_1trdm, sf_1trdm);
        matrix::daxpy<double>(1.0, b_1trdm, sf_1trdm);
    }
    return sf_1trdm;
}

} // namespace forte2
