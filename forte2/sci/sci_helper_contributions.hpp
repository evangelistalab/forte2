#pragma once

// Definition of SelectedCIHelper::generate_contributions. It is kept out of sci_helper.h and
// included only by the translation units that instantiate it, so that the many other consumers of
// sci_helper.h do not pay to recompile the loop body.

#include <algorithm>
#include <cmath>
#include <span>
#include <vector>

#include "determinant/determinant_helpers.h"

#include "sci/sci_helper.h"

namespace forte2 {

template <class Acc>
void SelectedCIHelper::generate_contributions(ContributionScratch& s, double prefilter,
                                              size_t num_batches, size_t batch_id,
                                              std::span<const uint64_t> parent_mask,
                                              Acc&& acc) const {
    // size the caller's buffers on first use; later batches reuse them untouched
    s.aocc.resize(na_);
    s.bocc.resize(nb_);
    s.avir.resize(norb_ - na_);
    s.bvir.resize(norb_ - nb_);
    s.abs_c_max.resize(max_block_size_);
    s.c_block.resize(max_block_size_ * nroots_);
    auto& aocc = s.aocc;
    auto& bocc = s.bocc;
    auto& avir = s.avir;
    auto& bvir = s.bvir;
    // the buffers belong to this thread, so they cannot alias c_ or the string lists; without
    // that promise every criterion below reloads abs_c_max[k] from memory
    double* __restrict abs_c_max = s.abs_c_max.data();
    double* __restrict c_block = s.c_block.data();

    size_t noa, nob, nva, nvb;
    const auto a_string_size = ab_list_.first_string_size();

    // norb_mask is used to compute the allowed virtual creation indices
    String norb_mask = String::zero();
    norb_mask.fill_up_to(norb_);

    Determinant new_det;
    // Loop over all unique alpha strings
    for (size_t i{0}; i < a_string_size; ++i) {
        const String& a_str = ab_list_.sorted_first_string(i);
        const auto& second_string_to_det_index = ab_list_.second_string_to_det_index()[i].values();
        const size_t block_size = second_string_to_det_index.size();

        // Gather this alpha string's coefficients into the scratch buffers
        double abs_c_max_block = 0.0; // track the maximum absolute CI coefficient
        for (size_t k{0}; k < block_size; ++k) {
            const size_t det_index = second_string_to_det_index[k].second;
            double abs_c_max_det = 0.0;
            // A parent outside the mask is treated as having no coefficients at all, which drops
            // every contribution it would make without any further test.
            if (parent_mask.empty() or ((parent_mask[det_index >> 6] >> (det_index & 63)) & 1ULL)) {
                for (size_t r{0}; r < nroots_; ++r) {
                    const double c_r = c_[det_index * nroots_ + r];
                    c_block[k * nroots_ + r] = c_r;
                    abs_c_max_det = std::max(abs_c_max_det, std::abs(c_r));
                }
            } else {
                std::fill_n(c_block + k * nroots_, nroots_, 0.0);
            }
            abs_c_max[k] = abs_c_max_det;
            abs_c_max_block = std::max(abs_c_max_block, abs_c_max_det);
        }

        // Every criterion below is proportional to one of these coefficients, so a block whose
        // coefficients are all zero cannot produce a contribution that survives acc.
        if (abs_c_max_block == 0.0)
            continue;

        // find the occupied and virtual orbitals for the current alpha string
        auto a_str_annihilation_masked = a_str & ~frozen_annihilation_mask_;
        // noa is the number of occupied alpha orbitals that we are allowed to annihilate from
        a_str_annihilation_masked.find_set_bits(aocc, noa);
        auto a_str_creation_masked = (~a_str & norb_mask) & ~frozen_creation_mask_;
        // nva is the number of virtual alpha orbitals that we are allowed to create into
        a_str_creation_masked.find_set_bits(avir, nva);

        // spans are more convenient for range-based for loops below
        std::span<size_t> aocc_span(aocc.data(), noa);
        std::span<size_t> avir_span(avir.data(), nva);

        // single alpha excitations
        for (const auto& i : aocc_span) {
            for (const auto& a : avir_span) {
                // *_unchecked avoids checking if i and a are already occupied/unoccupied
                // since we already know they are
                auto [new_a_str, sign] = create_single_excitation_unchecked(a_str, i, a);
                // determine if this determinant belongs to the current batch
                if (batch_of(new_a_str, num_batches) != batch_id) {
                    continue;
                }
                new_det.set_alpha_string(new_a_str);
                // add the occupied orbital contribution
                for (size_t k{0}; k < block_size; ++k) {
                    // singles_coupling_a is expensive, so drop parents that cannot survive first
                    if (abs_c_max[k] == 0.0)
                        continue;
                    const auto& [b_str_idx, det_index] = second_string_to_det_index[k];
                    new_det.set_beta_string(ab_list_.sorted_second_string(b_str_idx));
                    // singles_coupling_a can be expensive to compute
                    // a possible replacement here is h(i, a)
                    const double integral = slater_rules_.singles_coupling_a(i, a, new_det);
                    acc(new_det, c_block + k * nroots_, det_index, sign * integral,
                        std::fabs(integral * abs_c_max[k]));
                }
            }
        }

        // double alpha-alpha excitations
        for (const auto& i : aocc_span) {
            for (const auto& j : aocc_span) {
                if (i >= j)
                    continue;
                const auto& v_list = va_sorted_[i * norb_ + j];
                for (const auto& [coupling, integral, a, b] : v_list) {
                    // break early if the integrals are too small for all determinants
                    if (std::fabs(coupling * abs_c_max_block) <= prefilter)
                        break;

                    if ((a >= b) or a_str.get_bit(a) or a_str.get_bit(b))
                        continue;

                    auto [new_a_str, sign] = create_double_excitation_unchecked(a_str, i, j, a, b);

                    if (batch_of(new_a_str, num_batches) != batch_id) {
                        continue;
                    }

                    for (size_t k{0}; k < block_size; ++k) {
                        const double criterion = std::fabs(coupling * abs_c_max[k]);
                        if (criterion <= prefilter)
                            continue;
                        const auto& [b_str_idx, det_index] = second_string_to_det_index[k];
                        new_det.set_alpha_string(new_a_str);
                        new_det.set_beta_string(ab_list_.sorted_second_string(b_str_idx));
                        acc(new_det, c_block + k * nroots_, det_index, sign * integral, criterion);
                    }
                }
            }
        }

        // double alpha-beta excitations
        for (const auto& i : aocc_span) {
            for (const auto& a : avir_span) {
                // find the new alpha string after excitation and the sign and store it
                auto [new_a_str, a_sign] = create_single_excitation_unchecked(a_str, i, a);

                if (batch_of(new_a_str, num_batches) != batch_id) {
                    continue;
                }

                const auto& v_list = vab_sorted_[i * norb_ + a];
                new_det.set_alpha_string(new_a_str);

                for (const auto& [coupling, integral, j, b] : v_list) {
                    // break early if the integrals are too small
                    if (std::fabs(coupling * abs_c_max_block) <= prefilter)
                        break;
                    for (size_t k{0}; k < block_size; ++k) {
                        const auto& [b_str_idx, det_index] = second_string_to_det_index[k];
                        const String& b_str = ab_list_.sorted_second_string(b_str_idx);

                        // check if the beta excitation is valid
                        if ((not b_str.get_bit(j)) or b_str.get_bit(b)) {
                            continue;
                        }

                        const double criterion = std::fabs(coupling * abs_c_max[k]);
                        if (criterion <= prefilter)
                            continue;

                        auto [new_b_str, b_sign] = create_single_excitation_unchecked(b_str, j, b);
                        new_det.set_beta_string(new_b_str);
                        acc(new_det, c_block + k * nroots_, det_index,
                            a_sign * b_sign * integral, criterion);
                    }
                }
            }
        }

        // beta excitations

        // All beta excitations with a shared a_str share a batch. This test skips the rest of the
        // iteration, so it has to stay the last statement of the alpha-string loop: moving it
        // earlier would silently drop the alpha excitations above for every non-matching batch.
        if (batch_of(a_str, num_batches) != batch_id)
            continue;

        new_det.set_alpha_string(a_str);
        for (size_t k{0}; k < block_size; ++k) {
            if (abs_c_max[k] == 0.0)
                continue;
            const auto& [b_str_idx, det_index] = second_string_to_det_index[k];
            const String& b_str = ab_list_.sorted_second_string(b_str_idx);
            auto b_str_annihilation_masked = b_str & ~frozen_annihilation_mask_;
            b_str_annihilation_masked.find_set_bits(bocc, nob);
            auto b_str_creation_masked = (~b_str & norb_mask) & ~frozen_creation_mask_;
            b_str_creation_masked.find_set_bits(bvir, nvb);
            std::span<size_t> bocc_span(bocc.data(), nob);
            std::span<size_t> bvir_span(bvir.data(), nvb);

            // single beta excitations
            for (const auto& i : bocc_span) {
                for (const auto& a : bvir_span) {
                    new_det.set_beta_string(
                        b_str); // push the current beta string to compute coupling
                    const double integral = slater_rules_.singles_coupling_b(i, a, new_det);
                    const double criterion = std::fabs(integral * abs_c_max[k]);
                    if (criterion <= prefilter)
                        continue;
                    auto [new_b_str, sign] = create_single_excitation_unchecked(b_str, i, a);
                    new_det.set_beta_string(new_b_str); // push the new beta string
                    acc(new_det, c_block + k * nroots_, det_index, sign * integral, criterion);
                }
            }

            // double beta-beta excitations
            for (const auto& i : bocc_span) {
                for (const auto& j : bocc_span) {
                    if (i >= j)
                        continue;
                    const auto& v_list = va_sorted_[i * norb_ + j];
                    for (const auto& [coupling, integral, a, b] : v_list) {
                        const double criterion = std::fabs(coupling * abs_c_max[k]);
                        if (criterion <= prefilter)
                            break;

                        if ((a >= b) or b_str.get_bit(a) or b_str.get_bit(b))
                            continue;

                        auto [new_b_str, sign] =
                            create_double_excitation_unchecked(b_str, i, j, a, b);
                        new_det.set_alpha_string(a_str);
                        new_det.set_beta_string(new_b_str);
                        acc(new_det, c_block + k * nroots_, det_index, sign * integral, criterion);
                    }
                }
            }
        }
    }
}

} // namespace forte2
