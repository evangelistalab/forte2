#include <atomic>
#include <cmath>
#include <thread>
#include <future>
#include <mutex>

#include "helpers/logger.h"
#include "helpers/timer.hpp"
#include "helpers/sorting.hpp"
#include "helpers/np_matrix_functions.h"

#include "determinant/determinant_helpers.h"
#include "sci_helper.h"

namespace forte2 {

double SelectedCIHelper::compute_delta_ept2(double delta, double v) const {
    if (energy_correction_ == EnergyCorrection::Variational) {
        return -0.5 * (delta + std::sqrt(delta * delta + 4.0 * v * v));
    } else if (energy_correction_ == EnergyCorrection::PT2) {
        if (pt2_regularizer_ == PT2Regularizer::Shift) {
            return v * v / (delta + pt2_regularizer_strength_);
        } else if (pt2_regularizer_ == PT2Regularizer::DSRG) {
            return v * v * regularized_denominator(delta, pt2_regularizer_strength_);
        } else {
            return v * v / delta;
        }
    }
    throw std::runtime_error("Unknown energy correction method");
    return 0.0;
}

void SelectedCIHelper::update_orbital_energies() {
    auto rdm = compute_sf_1rdm(0, 0);
    for (size_t r{1}; r < nroots_; ++r) {
        auto rdm_r = compute_sf_1rdm(r, r);
        matrix::daxpy(1.0, rdm_r, rdm);
    }
    matrix::scale(rdm, 1.0 / nroots_);

    for (size_t i = 0; i < norb_; ++i) {
        epsilon_[i] = h(i, i);
        for (size_t j = 0; j < norb_; ++j) {
            for (size_t k = 0; k < norb_; ++k) {
                epsilon_[i] += rdm(j, k) * (V(i, j, i, k) - 0.5 * V(i, j, k, i));
            }
        }
    }
}

void SelectedCIHelper::select_hbci_ref(double var_threshold, double pt2_threshold) {
    compute_det_energies();
    prepare_strings();

    update_hbci_ints();

    local_timer selection_timer;

    // One coupling per external determinant per root, plus the set of determinants that will
    // join the variational space. The coupling must be complete before it is squared, so it is
    // never split by which side of var_threshold an individual connection falls on.
    std::vector<DetMap> map(nroots_);
    DetSet promoted;

    std::vector<size_t> aocc(na_, 0);
    std::vector<size_t> bocc(nb_, 0);
    std::vector<size_t> avir(norb_ - na_, 0);
    std::vector<size_t> bvir(norb_ - nb_, 0);

    size_t noa, nob;
    for (size_t idx{0}, idx_max{dets_.size()}; idx < idx_max; ++idx) {
        const auto& det = dets_[idx];
        std::span<double> c_det(c_.data() + idx * nroots_, nroots_);
        double max_abs_c = 0.0;
        for (size_t r{0}; r < nroots_; ++r) {
            max_abs_c = std::max(max_abs_c, std::fabs(c_det[r]));
        }

        det.collect_alpha_occupied(aocc, noa);
        det.collect_beta_occupied(bocc, nob);
        collect_virtual_orbitals(aocc, avir, norb_);
        collect_virtual_orbitals(bocc, bvir, norb_);
        size_t nva = norb_ - noa;
        size_t nvb = norb_ - nob;

        std::span<size_t> aocc_span(aocc.data(), noa);
        std::span<size_t> avir_span(avir.data(), nva);
        std::span<size_t> bocc_span(bocc.data(), nob);
        std::span<size_t> bvir_span(bvir.data(), nvb);

        for (const auto& i : aocc_span) {
            if (!annihilation_allowed(i))
                continue;
            for (const auto& a : avir_span) {
                if (!creation_allowed(a))
                    continue;
                // const double integral = h_[i * norb_ + a];
                const double integral = slater_rules_.singles_coupling_a(i, a, det);
                const double criterion = std::fabs(integral * max_abs_c);

                if (criterion <= pt2_threshold)
                    continue;

                const auto [new_det, sign] = create_single_a_excitation(det, i, a);

                if (criterion > var_threshold) {
                    promoted.insert(new_det);
                }
                for (size_t r{0}; r < nroots_; ++r) {
                    map[r][new_det] += sign * integral * c_det[r];
                }
            }
        }

        for (const auto& i : bocc_span) {
            if (!annihilation_allowed(i))
                continue;
            for (const auto& a : bvir_span) {
                if (!creation_allowed(a))
                    continue;
                // const double integral =
                //     slater_rules_.singles_coupling(i, a, bocc, aocc); // h_[i * norb_ + a];
                const double integral = slater_rules_.singles_coupling_b(i, a, det);
                const double criterion = std::fabs(integral * max_abs_c);
                if (criterion <= pt2_threshold)
                    continue;

                const auto [new_det, sign] = create_single_b_excitation(det, i, a);

                if (criterion > var_threshold) {
                    promoted.insert(new_det);
                }
                for (size_t r{0}; r < nroots_; ++r) {
                    map[r][new_det] += sign * integral * c_det[r];
                }
            }
        }

        for (const auto& i : aocc_span) {
            if (!annihilation_allowed(i))
                continue;
            for (const auto& j : aocc_span) {
                if (i >= j || !annihilation_allowed(j))
                    continue;
                for (const auto& a : avir_span) {
                    if (!creation_allowed(a))
                        continue;
                    for (const auto& b : avir_span) {
                        if (a >= b || !creation_allowed(b))
                            continue;

                        const double integral = Va(i, j, a, b);
                        const double criterion = std::fabs(integral * max_abs_c);
                        if (criterion <= pt2_threshold)
                            continue;

                        const auto [new_det, sign] = create_double_aa_excitation(det, i, j, a, b);

                        if (criterion > var_threshold) {
                            promoted.insert(new_det);
                        }
                        for (size_t r{0}; r < nroots_; ++r) {
                            map[r][new_det] += sign * integral * c_det[r];
                        }
                    }
                }
            }
        }

        for (const auto& i : bocc_span) {
            if (!annihilation_allowed(i))
                continue;
            for (const auto& j : bocc_span) {
                if (i >= j || !annihilation_allowed(j))
                    continue;
                for (const auto& a : bvir_span) {
                    if (!creation_allowed(a))
                        continue;
                    for (const auto& b : bvir_span) {
                        if (a >= b || !creation_allowed(b))
                            continue;
                        const double integral = Va(i, j, a, b);
                        const double criterion = std::fabs(integral * max_abs_c);
                        if (criterion <= pt2_threshold)
                            continue;

                        const auto [new_det, sign] = create_double_bb_excitation(det, i, j, a, b);

                        if (criterion > var_threshold) {
                            promoted.insert(new_det);
                        }
                        for (size_t r{0}; r < nroots_; ++r) {
                            map[r][new_det] += sign * integral * c_det[r];
                        }
                    }
                }
            }
        }

        for (const auto& i : aocc_span) {
            if (!annihilation_allowed(i))
                continue;
            for (const auto& j : bocc_span) {
                if (!annihilation_allowed(j))
                    continue;
                for (const auto& a : avir_span) {
                    if (!creation_allowed(a))
                        continue;
                    for (const auto& b : bvir_span) {
                        if (!creation_allowed(b))
                            continue;
                        const double integral = V(i, j, a, b);
                        const double criterion = std::fabs(integral * max_abs_c);
                        if (criterion <= pt2_threshold)
                            continue;

                        const auto [new_det, sign] = create_double_ab_excitation(det, i, j, a, b);

                        if (criterion > var_threshold) {
                            promoted.insert(new_det);
                        }
                        for (size_t r{0}; r < nroots_; ++r) {
                            map[r][new_det] += sign * integral * c_det[r];
                        }
                    }
                }
            }
        }
    }

    // Drop the determinants that are already in the variational space; the correction runs over
    // the determinants outside it
    for (const auto& det : dets_) {
        promoted.erase(det);
        for (size_t r{0}; r < nroots_; ++r) {
            map[r].erase(det);
        }
    }

    // add variational determinants first
    for (const auto& det : promoted) {
        dets_.push_back(det);
    }

    for (size_t r{0}; r < nroots_; ++r) {
        double var = 0.0;
        double pt = 0.0;
        for (const auto& [det, val] : map[r]) {
            const double delta = root_energies_[r] - slater_rules_.energy(det);
            (promoted.count(det) ? var : pt) += compute_delta_ept2(delta, val);
        }
        ept2_var_[r] = var;
        ept2_pt_[r] = pt;
    }

    c_.resize(dets_.size() * nroots_, 0.0);

    compute_det_energies();
    prepare_strings();
}

void SelectedCIHelper::select_hbci(double var_threshold, double pt2_threshold) {
    local_timer selection_timer;

    // The sorted integral lists only changes over iterations if the eHBCI criterion
    // is used, since the epsilon denomiators enter into the (dressed) integral, and those
    // are computed from the Fock matrix of the previous iteration. The plain HBCI
    // just stores |V|, which doesn't change at all.
    if (screening_criterion_ == ScreeningCriterion::eHBCI) {
        update_orbital_energies();
        update_hbci_ints();
    }

    const auto num_threads = get_num_threads();
    const size_t num_batches = num_batches_per_thread_ * num_threads; // total number of batches

    std::atomic<size_t> next_batch(0);

    std::vector<std::vector<Determinant>> thread_new_dets(num_threads);
    std::vector<std::vector<double>> local_ept2_var(num_threads, std::vector<double>(nroots_, 0.0));
    std::vector<std::vector<double>> local_ept2_pt(num_threads, std::vector<double>(nroots_, 0.0));
    std::vector<std::vector<std::tuple<size_t, size_t, double>>> thread_log_data(num_threads);

    DetSet existing_dets(dets_.begin(), dets_.end());

    // worker function for each thread that processes batches of determinants
    auto worker = [&](size_t thread_id) {
        // Persistent storage for this thread, so that re-walking the variational space once per
        // batch does not reallocate. select_hbci_batch clears it on entry but the memory stays.
        SelectHbciScratch s;
        std::vector<Determinant> new_dets_local;

        while (true) {
            // Get the next batch ID for this thread
            size_t batch_id = next_batch.fetch_add(1);
            if (batch_id >= num_batches)
                break;

            local_timer batch_timer;

            select_hbci_batch(s, var_threshold, pt2_threshold, num_batches, batch_id,
                              existing_dets);

            // Each determinant carries its complete coupling to the variational wave function and
            // is attributed to exactly one of the two contributions, so no filtering between them
            // is needed. Determinants already in the variational space were skipped during
            // generation.
            new_dets_local.clear();
            size_t num_pt_dets = 0;
            for (const auto& [det, idx] : s.map) {
                const double energy = slater_rules_.energy(det);
                auto& target = s.promoted[idx / nroots_] ? local_ept2_var[thread_id]
                                                         : local_ept2_pt[thread_id];
                for (size_t r{0}; r < nroots_; ++r) {
                    target[r] += compute_delta_ept2(root_energies_[r] - energy, s.coeffs[idx + r]);
                }
                if (s.promoted[idx / nroots_]) {
                    new_dets_local.push_back(det);
                } else {
                    num_pt_dets++;
                }
            }

            // Append to thread-local container (no locks)
            thread_new_dets[thread_id].insert(thread_new_dets[thread_id].end(),
                                              new_dets_local.begin(), new_dets_local.end());

            thread_log_data[thread_id].push_back(
                {batch_id, num_pt_dets, batch_timer.elapsed_seconds()});
        }
    };

    // launch threads
    std::vector<std::future<void>> workers;
    for (size_t t{0}; t < num_threads; ++t)
        workers.push_back(std::async(std::launch::async, worker, t));

    for (auto& w : workers)
        w.get();

    // combine the local ept2 contributions from all threads
    for (size_t r{0}; r < nroots_; ++r) {
        ept2_var_[r] = 0.0;
        ept2_pt_[r] = 0.0;
        for (size_t t = 0; t < num_threads; ++t) {
            ept2_var_[r] += local_ept2_var[t][r];
            ept2_pt_[r] += local_ept2_pt[t][r];
        }
    }

    // print a summary of each thread's work
    for (size_t t{0}; t < num_threads; ++t) {
        size_t total_batches = thread_log_data[t].size();
        size_t total_dets = 0;
        double total_time = 0.0;
        for (const auto& [batch_id, num_pt_dets, time] : thread_log_data[t]) {
            total_dets += num_pt_dets;
            total_time += time;
        }

        LOG(log_level_) << "Thread " << t << " processed " << total_batches << " batches, found "
                        << total_dets << " new determinants in " << total_time << " seconds (avg "
                        << total_time / total_batches << " s/batch, "
                        << (total_time > 0.0 ? std::to_string(total_dets / total_time) : "N/A")
                        << " dets/s)";
    }

    // count the new determinants
    num_new_dets_var_ = 0;
    for (auto& v : thread_new_dets)
        num_new_dets_var_ += v.size();

    // reserve space to avoid multiple allocations
    dets_.reserve(dets_.size() + num_new_dets_var_);

    // merge all new determinants from different threads (each thread has unique determinants)
    for (auto& v : thread_new_dets) {
        dets_.insert(dets_.end(), v.begin(), v.end());
    }

    c_.resize(dets_.size() * nroots_, 0.0);

    compute_det_energies();
    prepare_strings();

    // print a summary of the selection
    num_new_dets_pt2_ = 0;
    for (size_t t{0}; t < num_threads; ++t) {
        for (const auto& [batch_id, num_pt_dets, time] : thread_log_data[t]) {
            num_new_dets_pt2_ += num_pt_dets;
        }
    }
    selection_time_ = selection_timer.elapsed_seconds();
}

void SelectedCIHelper::select_hbci_batch(SelectHbciScratch& s, double var_threshold,
                                         double pt2_threshold, size_t num_batches, size_t batch_id,
                                         const DetSet& existing_dets) {
    auto& map = s.map;
    auto& coeffs = s.coeffs;
    auto& promoted = s.promoted;
    map.clear();
    coeffs.clear();
    promoted.clear();

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

    // The single place that decides whether a connection survives and what happens to it. Every
    // channel goes through it, so none of them can drift from the others. Channels also test the
    // criterion themselves before building the external determinant, but only to skip work and
    // only against an upper bound on it; this test is the one that decides.
    auto accumulate = [&](const Determinant& det, const double* c_parent, double coupling,
                          double criterion) {
        if (criterion <= pt2_threshold)
            return;
        // if the determinant is already in the variational space, skip it
        if (existing_dets.count(det))
            return;
        // A determinant can be reached by several parents. Every contribution to <det|H|Psi> has
        // to be summed before that sum is squared, so all of them accumulate into one entry of
        // coeffs regardless of which side of var_threshold the individual connection falls on.
        // Whether the determinant is promoted is a property of the determinant, recorded
        // separately: one connection above var_threshold is enough.
        const size_t loc = coeffs.size();
        auto [it, emplaced] = map.try_emplace(det, loc);
        if (emplaced) {
            for (size_t r{0}; r < nroots_; ++r)
                coeffs.push_back(coupling * c_parent[r]);
            promoted.push_back(criterion > var_threshold ? 1 : 0);
        } else {
            const size_t idx = it->second;
            for (size_t r{0}; r < nroots_; ++r)
                coeffs[idx + r] += coupling * c_parent[r];
            if (criterion > var_threshold)
                promoted[idx / nroots_] = 1;
        }
    };

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
            for (size_t r{0}; r < nroots_; ++r) {
                const double c_r = c_[det_index * nroots_ + r];
                c_block[k * nroots_ + r] = c_r;
                abs_c_max_det = std::max(abs_c_max_det, std::abs(c_r));
            }
            abs_c_max[k] = abs_c_max_det;
            abs_c_max_block = std::max(abs_c_max_block, abs_c_max_det);
        }

        // Every criterion below is proportional to one of these coefficients, so a block whose
        // coefficients are all zero cannot produce a connection that survives accumulate.
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
                    accumulate(new_det, c_block + k * nroots_, sign * integral,
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
                    if (std::fabs(coupling * abs_c_max_block) <= pt2_threshold)
                        break;

                    if ((a >= b) or a_str.get_bit(a) or a_str.get_bit(b))
                        continue;

                    auto [new_a_str, sign] = create_double_excitation_unchecked(a_str, i, j, a, b);

                    if (batch_of(new_a_str, num_batches) != batch_id) {
                        continue;
                    }

                    for (size_t k{0}; k < block_size; ++k) {
                        const double criterion = std::fabs(coupling * abs_c_max[k]);
                        if (criterion <= pt2_threshold)
                            continue;
                        const auto& [b_str_idx, det_index] = second_string_to_det_index[k];
                        new_det.set_alpha_string(new_a_str);
                        new_det.set_beta_string(ab_list_.sorted_second_string(b_str_idx));
                        accumulate(new_det, c_block + k * nroots_, sign * integral, criterion);
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
                    if (std::fabs(coupling * abs_c_max_block) <= pt2_threshold)
                        break;
                    for (size_t k{0}; k < block_size; ++k) {
                        const auto& [b_str_idx, det_index] = second_string_to_det_index[k];
                        const String& b_str = ab_list_.sorted_second_string(b_str_idx);

                        // check if the beta excitation is valid
                        if ((not b_str.get_bit(j)) or b_str.get_bit(b)) {
                            continue;
                        }

                        const double criterion = std::fabs(coupling * abs_c_max[k]);
                        if (criterion <= pt2_threshold)
                            continue;

                        auto [new_b_str, b_sign] = create_single_excitation_unchecked(b_str, j, b);
                        new_det.set_beta_string(new_b_str);
                        accumulate(new_det, c_block + k * nroots_, a_sign * b_sign * integral,
                                   criterion);
                    }
                }
            }
        }

        // beta excitations
        
        // All beta excitations with a shared a_str share a batch
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
                    if (criterion <= pt2_threshold)
                        continue;
                    auto [new_b_str, sign] = create_single_excitation_unchecked(b_str, i, a);
                    new_det.set_beta_string(new_b_str); // push the new beta string
                    accumulate(new_det, c_block + k * nroots_, sign * integral, criterion);
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
                        if (criterion <= pt2_threshold)
                            break;

                        if ((a >= b) or b_str.get_bit(a) or b_str.get_bit(b))
                            continue;

                        auto [new_b_str, sign] =
                            create_double_excitation_unchecked(b_str, i, j, a, b);
                        new_det.set_alpha_string(a_str);
                        new_det.set_beta_string(new_b_str);
                        accumulate(new_det, c_block + k * nroots_, sign * integral, criterion);
                    }
                }
            }
        }
    }
}

} // namespace forte2
