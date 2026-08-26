#pragma once

// Definition of SelectedCIHelper::generate_connections. This is kept out of sci_helper.h and
// included only by the translation units that instantiate it (HBCI selection and the PT2 stage),
// so that the many other consumers of sci_helper.h do not pay to recompile it.

#include <cmath>
#include <span>
#include <vector>

#include "determinant/determinant_helpers.h"

#include "sci/sci_helper.h"

namespace forte2 {

template <class Acc>
void SelectedCIHelper::generate_connections(std::span<const double> c_screen,
                                            double screen_threshold, size_t num_batches,
                                            size_t batch_id, const DetSet& existing_dets,
                                            Acc&& acc) const {
    std::vector<size_t> aocc(na_);
    std::vector<size_t> bocc(nb_);
    std::vector<size_t> avir(norb_ - na_);
    std::vector<size_t> bvir(norb_ - nb_);

    size_t noa, nob, nva, nvb;
    const auto a_string_size = ab_list_.first_string_size();

    // precompute the maximum block size for the temporary storage
    std::size_t max_block_size = 0;
    for (size_t i{0}; i < a_string_size; ++i) {
        max_block_size = std::max(max_block_size, ab_list_.second_string_to_det_index()[i].size());
    }

    // allocate the temporary storage for the largest block of alpha strings
    std::vector<double> abs_c_max(max_block_size, 0.0);
    std::vector<double> c_block(max_block_size * nroots_, 0.0);

    // norb_mask is used to compute the allowed virtual creation indices
    String norb_mask = String::zero();
    norb_mask.fill_up_to(norb_);

    Determinant new_det;
    // Loop over all unique alpha strings
    for (size_t i{0}; i < a_string_size; ++i) {
        const String& a_str = ab_list_.sorted_first_string(i);
        const auto& second_string_to_det_index = ab_list_.second_string_to_det_index()[i];

        // grab the screening coefficients for all determinants with the current alpha string for
        // all roots
        double abs_c_max_block = 0.0; // track the maximum absolute CI coefficient
        for (size_t k{0}; const auto& [_, idx] : second_string_to_det_index) {
            double abs_c_max_det = 0.0;
            for (size_t r{0}; r < nroots_; ++r) {
                const double c_r = c_screen[idx * nroots_ + r];
                c_block[k * nroots_ + r] = c_r;
                abs_c_max_block = std::max(abs_c_max_block, std::abs(c_r));
                abs_c_max_det = std::max(abs_c_max_det, std::abs(c_r));
            }
            abs_c_max[k] = abs_c_max_det;
            ++k;
        }

        // Every criterion below is proportional to one of these coefficients, so a block whose
        // coefficients are all zero cannot produce a connection. Callers that restrict the parent
        // space (the stochastic PT2 step) do so by zeroing entries of c_screen, and this is what
        // makes skipping those parents cost O(1) per alpha string.
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
                for (size_t k{0}; const auto& [b_str_idx, det_index] : second_string_to_det_index) {
                    const String& b_str = ab_list_.sorted_second_string(b_str_idx);
                    new_det.set_beta_string(b_str);
                    // singles_coupling_a can be expensive to compute
                    // a possible replacement here is h(i, a)
                    const double integral = slater_rules_.singles_coupling_a(i, a, new_det);
                    const double criterion = std::fabs(integral * abs_c_max[k]);
                    if (criterion > screen_threshold) {
                        // if the determinant is already in the variational space, skip it
                        if (!existing_dets.count(new_det)) {
                            std::span<const double> coeffs(c_block.data() + k * nroots_, nroots_);
                            acc(new_det, coeffs, det_index, sign * integral, criterion);
                        }
                    }
                    k++;
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
                    if (std::fabs(coupling * abs_c_max_block) < screen_threshold)
                        break;

                    if ((a >= b) or a_str.get_bit(a) or a_str.get_bit(b))
                        continue;

                    auto [new_a_str, sign] = create_double_excitation_unchecked(a_str, i, j, a, b);

                    if (batch_of(new_a_str, num_batches) != batch_id) {
                        continue;
                    }

                    for (size_t k{0};
                         const auto& [b_str_idx, det_index] : second_string_to_det_index) {
                        const double criterion = std::fabs(coupling * abs_c_max[k]);
                        if (criterion > screen_threshold) {
                            new_det.set_alpha_string(new_a_str);
                            new_det.set_beta_string(ab_list_.sorted_second_string(b_str_idx));
                            // if the determinant is already in the variational space, skip it
                            if (!existing_dets.count(new_det)) {
                                std::span<const double> coeffs(c_block.data() + k * nroots_,
                                                               nroots_);
                                acc(new_det, coeffs, det_index, sign * integral, criterion);
                            }
                        }
                        k++;
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
                    if (std::fabs(coupling * abs_c_max_block) < screen_threshold)
                        break;
                    for (size_t k{0};
                         const auto& [b_str_idx, det_index] : second_string_to_det_index) {
                        const String& b_str = ab_list_.sorted_second_string(b_str_idx);

                        // check if the beta excitation is valid
                        if ((not b_str.get_bit(j)) or b_str.get_bit(b)) {
                            k++;
                            continue;
                        }

                        const double criterion = std::fabs(coupling * abs_c_max[k]);
                        if (criterion > screen_threshold) {
                            auto [new_b_str, b_sign] =
                                create_single_excitation_unchecked(b_str, j, b);
                            new_det.set_beta_string(new_b_str);
                            if (!existing_dets.count(new_det)) {
                                std::span<const double> coeffs(c_block.data() + k * nroots_,
                                                               nroots_);
                                acc(new_det, coeffs, det_index, a_sign * b_sign * integral,
                                    criterion);
                            }
                        }
                        k++;
                    }
                }
            }
        }

        // Beta excitations leave the alpha string untouched, so the connected determinants belong
        // to the batch of the *parent* alpha string. This test must stay the last statement of the
        // alpha-string loop: it skips the remainder of the iteration, and moving it earlier would
        // silently drop the alpha excitations above for every non-matching batch.
        if (batch_of(a_str, num_batches) != batch_id)
            continue;

        new_det.set_alpha_string(a_str);
        for (size_t k{0}; const auto& [b_str_idx, det_index] : second_string_to_det_index) {
            if (abs_c_max[k] == 0.0) {
                k++;
                continue;
            }
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
                    if (criterion > screen_threshold) {
                        auto [new_b_str, sign] = create_single_excitation_unchecked(b_str, i, a);
                        new_det.set_beta_string(new_b_str); // push the new beta string
                        if (!existing_dets.count(new_det)) {
                            std::span<const double> coeffs(c_block.data() + k * nroots_, nroots_);
                            acc(new_det, coeffs, det_index, sign * integral, criterion);
                        }
                    }
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
                        if (criterion < screen_threshold)
                            break;

                        if ((a >= b) or b_str.get_bit(a) or b_str.get_bit(b))
                            continue;

                        auto [new_b_str, sign] =
                            create_double_excitation_unchecked(b_str, i, j, a, b);
                        new_det.set_alpha_string(a_str);
                        new_det.set_beta_string(new_b_str);
                        if (!existing_dets.count(new_det)) {
                            std::span<const double> coeffs(c_block.data() + k * nroots_, nroots_);
                            acc(new_det, coeffs, det_index, sign * integral, criterion);
                        }
                    }
                }
            }
            k++;
        }
    }
}

} // namespace forte2
