#include <atomic>
#include <thread>
#include <future>
#include <mutex>

#include "helpers/logger.h"
#include "helpers/timer.hpp"
#include "helpers/sorting.hpp"
#include "helpers/np_matrix_functions.h"

#include "ci/determinant_helpers.h"
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

    size_t checks_count = 0;

    std::vector<DetMap> V_map(nroots_);
    std::vector<DetMap> PT_map(nroots_);

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

        det.get_fast_a_occ(aocc, noa);
        det.get_fast_b_occ(bocc, nob);
        compute_fast_virtual(aocc, avir, norb_);
        compute_fast_virtual(bocc, bvir, norb_);
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

                if (criterion < pt2_threshold)
                    continue;

                const auto [new_det, sign] = create_single_a_excitation(det, i, a);

                if (criterion > var_threshold) {
                    for (size_t r{0}; r < nroots_; ++r) {
                        V_map[r][new_det] += sign * integral * c_det[r];
                    }
                } else {
                    for (size_t r{0}; r < nroots_; ++r) {
                        PT_map[r][new_det] += sign * integral * c_det[r];
                    }
                }
                checks_count++;
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
                if (criterion < pt2_threshold)
                    continue;

                const auto [new_det, sign] = create_single_b_excitation(det, i, a);

                if (criterion > var_threshold) {
                    for (size_t r{0}; r < nroots_; ++r) {
                        V_map[r][new_det] += sign * integral * c_det[r];
                    }
                } else {
                    for (size_t r{0}; r < nroots_; ++r) {
                        PT_map[r][new_det] += sign * integral * c_det[r];
                    }
                }
                checks_count++;
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
                        if (criterion < pt2_threshold)
                            continue;

                        const auto [new_det, sign] = create_double_aa_excitation(det, i, j, a, b);

                        if (criterion > var_threshold) {
                            for (size_t r{0}; r < nroots_; ++r) {
                                V_map[r][new_det] += sign * integral * c_det[r];
                            }
                        } else {
                            for (size_t r{0}; r < nroots_; ++r) {
                                PT_map[r][new_det] += sign * integral * c_det[r];
                            }
                        }
                        checks_count++;
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
                        if (criterion < pt2_threshold)
                            continue;

                        const auto [new_det, sign] = create_double_bb_excitation(det, i, j, a, b);

                        if (criterion > var_threshold) {
                            for (size_t r{0}; r < nroots_; ++r) {
                                V_map[r][new_det] += sign * integral * c_det[r];
                            }
                        } else {
                            for (size_t r{0}; r < nroots_; ++r) {
                                PT_map[r][new_det] += sign * integral * c_det[r];
                            }
                        }
                        checks_count++;
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
                        if (criterion < pt2_threshold)
                            continue;

                        const auto [new_det, sign] = create_double_ab_excitation(det, i, j, a, b);

                        if (criterion > var_threshold) {
                            for (size_t r{0}; r < nroots_; ++r) {
                                V_map[r][new_det] += sign * integral * c_det[r];
                            }
                        } else {
                            for (size_t r{0}; r < nroots_; ++r) {
                                PT_map[r][new_det] += sign * integral * c_det[r];
                            }
                        }
                        checks_count++;
                    }
                }
            }
        }
    }

    // Remove the coupling to determinants that are already in the variational space
    for (size_t r{0}; r < nroots_; ++r) {
        for (const auto& det : dets_) {
            V_map[r].erase(det);
            PT_map[r].erase(det);
        }
    }

    // Remove the coupling for the PT2 determinants that are already in the variational space
    for (size_t r{0}; r < nroots_; ++r) {
        for (const auto& [det, val] : V_map[r]) {
            PT_map[r].erase(det);
        }
    }

    // add variational determinants first
    for (const auto& [det, val] : V_map[0]) {
        dets_.push_back(det);
    }

    // print all the variational determinants
    for (size_t r{0}; r < nroots_; ++r) {
        double var = 0.0;
        double pt = 0.0;
        for (const auto& [det, val] : V_map[r]) {
            const double delta = root_energies_[r] - slater_rules_.energy(det);
            var += compute_delta_ept2(delta, val);
        }
        for (const auto& [det, val] : PT_map[r]) {
            const double delta = root_energies_[r] - slater_rules_.energy(det);
            pt += compute_delta_ept2(delta, val);
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

    update_orbital_energies();
    update_hbci_ints();

    const size_t num_batches = num_batches_per_thread_ * num_threads_; // total number of batches

    std::atomic<size_t> next_batch(0);
    std::mutex merge_mutex;
    std::mutex log_mutex;

    std::vector<std::vector<Determinant>> thread_new_dets(num_threads_);
    std::vector<std::vector<double>> local_ept2_var(num_threads_,
                                                    std::vector<double>(nroots_, 0.0));
    std::vector<std::vector<double>> local_ept2_pt(num_threads_, std::vector<double>(nroots_, 0.0));
    std::vector<std::vector<std::tuple<size_t, size_t, double>>> thread_log_data(num_threads_);

    // worker function for each thread that processes batches of determinants
    auto worker = [&](size_t thread_id) {
        while (true) {
            // Get the next batch ID for this thread
            size_t batch_id = next_batch.fetch_add(1);
            if (batch_id >= num_batches)
                break;

            local_timer batch_timer;

            auto [V_map, PT_map] =
                select_hbci_batch(var_threshold, pt2_threshold, num_batches, batch_id);

            // Filter out existing determinants
            for (size_t r = 0; r < nroots_; ++r) {
                for (const auto& det : dets_) {
                    V_map[r].erase(det);
                    PT_map[r].erase(det);
                }
                for (const auto& [det, _] : V_map[0])
                    PT_map[r].erase(det);
            }

            std::vector<Determinant> new_dets_local;
            new_dets_local.reserve(V_map[0].size());
            for (const auto& [det, _] : V_map[0])
                new_dets_local.push_back(det);

            // Compute contributions
            for (size_t r = 0; r < nroots_; ++r) {
                double var = 0.0, pt = 0.0;
                for (const auto& [det, val] : V_map[r])
                    var += compute_delta_ept2(root_energies_[r] - slater_rules_.energy(det), val);
                for (const auto& [det, val] : PT_map[r])
                    pt += compute_delta_ept2(root_energies_[r] - slater_rules_.energy(det), val);
                local_ept2_var[thread_id][r] += var;
                local_ept2_pt[thread_id][r] += pt;
            }

            // Append to thread-local container (no locks)
            thread_new_dets[thread_id].insert(thread_new_dets[thread_id].end(),
                                              new_dets_local.begin(), new_dets_local.end());

            thread_log_data[thread_id].push_back(
                {batch_id, PT_map[0].size(), batch_timer.elapsed_seconds()});
        }
    };

    // launch threads
    std::vector<std::future<void>> workers;
    for (size_t t{0}; t < num_threads_; ++t)
        workers.push_back(std::async(std::launch::async, worker, t));

    for (auto& w : workers)
        w.get();

    // combine the local ept2 contributions from all threads
    for (size_t r{0}; r < nroots_; ++r) {
        ept2_var_[r] = 0.0;
        ept2_pt_[r] = 0.0;
        for (size_t t = 0; t < num_threads_; ++t) {
            ept2_var_[r] += local_ept2_var[t][r];
            ept2_pt_[r] += local_ept2_pt[t][r];
        }
    }

    // print a summary of each thread's work
    for (size_t t{0}; t < num_threads_; ++t) {
        size_t total_batches = thread_log_data[t].size();
        size_t total_dets = 0;
        double total_time = 0.0;
        for (const auto& [batch_id, num_pt_dets, time] : thread_log_data[t]) {
            total_dets += num_pt_dets;
            total_time += time;
        }
        std::cout << "Thread " << t << " processed " << total_batches << " batches, found "
                  << total_dets << " new determinants in " << total_time << " seconds (avg "
                  << total_time / total_batches << " s/batch, " << total_dets / total_time
                  << " dets/s)" << std::endl;
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
    for (size_t t{0}; t < num_threads_; ++t) {
        for (const auto& [batch_id, num_pt_dets, time] : thread_log_data[t]) {
            num_new_dets_pt2_ += num_pt_dets;
        }
    }
    selection_time_ = selection_timer.elapsed_seconds();
}

std::pair<std::vector<DetMap>, std::vector<DetMap>>
SelectedCIHelper::select_hbci_batch(double var_threshold, double pt2_threshold, size_t num_batches,
                                    size_t batch_id) {
    std::vector<size_t> aocc(na_);
    std::vector<size_t> bocc(nb_);
    std::vector<size_t> avir(norb_ - na_);
    std::vector<size_t> bvir(norb_ - nb_);

    size_t checks_count = 0;
    double e_pt2 = 0.0;

    size_t noa, nob, nva, nvb;

    std::vector<DetMap> V_map(nroots_);
    std::vector<DetMap> PT_map(nroots_);

    const auto a_string_size = ab_list_.first_string_size();

    // precompute the maximum block size for the temporary storage
    std::size_t max_block_size = 0;
    for (size_t i{0}; i < a_string_size; ++i) {
        max_block_size = std::max(max_block_size, ab_list_.second_string_to_det_index()[i].size());
    }

    // allocate the temporary storage for the largest block of alpha strings
    std::vector<double> abs_c_max(max_block_size, 0.0);
    std::vector<double> c_block(max_block_size * nroots_, 0.0);

    String norb_mask = String::zero();
    norb_mask.fill_up_to(norb_);

    Determinant new_det;
    // Loop over all unique alpha strings
    for (size_t i{0}; i < a_string_size; ++i) {
        const String& a_str = ab_list_.sorted_first_string(i);
        const auto& second_string_to_det_index = ab_list_.second_string_to_det_index()[i];

        // grab the CI coefficients for all determinants with the current alpha string for all
        // roots
        double abs_c_max_block = 0.0; // track the maximum absolute CI coefficient
        for (size_t k{0}; const auto& [_, idx] : second_string_to_det_index) {
            double abs_c_max_det = 0.0;
            for (size_t r{0}; r < nroots_; ++r) {
                const double c_r = c_[idx * nroots_ + r];
                c_block[k * nroots_ + r] = c_r;
                abs_c_max_block = std::max(abs_c_max_block, std::abs(c_r));
                abs_c_max_det = std::max(abs_c_max_det, std::abs(c_r));
            }
            abs_c_max[k] = abs_c_max_det;
            ++k;
        }

        // find the occupied and empty orbitals for the current alpha string
        auto a_str_annihilation_masked = a_str & ~frozen_annihilation_mask_;
        a_str_annihilation_masked.find_set_bits(aocc, noa);
        auto a_str_creation_masked = (~a_str & norb_mask) & ~frozen_creation_mask_;
        a_str_creation_masked.find_set_bits(avir, nva);

        std::span<size_t> aocc_span(aocc.data(), noa);
        std::span<size_t> avir_span(avir.data(), nva);

        // single alpha excitations
        for (const auto& i : aocc_span) {
            for (const auto& a : avir_span) {
                auto [new_a_str, sign] = create_single_excitation(a_str, i, a);
                if (String::Hash()(new_a_str) % num_batches != batch_id) {
                    continue;
                }
                new_det.set_a_string(new_a_str);
                // add the occupied orbital contribution
                for (size_t k{0}; const auto& [b_str_idx, det_index] : second_string_to_det_index) {
                    const String& b_str = ab_list_.sorted_second_string(b_str_idx);
                    new_det.set_b_string(b_str);
                    // const double integral = h_[i * norb_ + a];
                    const double integral = slater_rules_.singles_coupling_a(i, a, new_det);
                    const double criterion = std::fabs(integral * abs_c_max[k]);
                    if (criterion > pt2_threshold) {
                        if (criterion > var_threshold) {
                            for (size_t r{0}; r < nroots_; ++r) {
                                V_map[r][new_det] += sign * integral * c_block[k * nroots_ + r];
                            }
                        } else {
                            for (size_t r{0}; r < nroots_; ++r) {
                                PT_map[r][new_det] += sign * integral * c_block[k * nroots_ + r];
                            }
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
                    if (std::fabs(coupling * abs_c_max_block) < pt2_threshold)
                        break;

                    if ((a >= b) or a_str.get_bit(a) or a_str.get_bit(b) or !creation_allowed(a, b))
                        continue;

                    auto [new_a_str, sign] = create_double_excitation(a_str, i, j, a, b);

                    if (String::Hash()(new_a_str) % num_batches != batch_id) {
                        continue;
                    }

                    for (size_t k{0};
                         const auto& [b_str_idx, det_index] : second_string_to_det_index) {
                        const double criterion = std::fabs(coupling * abs_c_max[k]);
                        if (criterion > pt2_threshold) {
                            new_det.set_a_string(new_a_str);
                            new_det.set_b_string(ab_list_.sorted_second_string(b_str_idx));

                            if (criterion > var_threshold) {
                                for (size_t r{0}; r < nroots_; ++r) {
                                    V_map[r][new_det] += sign * integral * c_block[k * nroots_ + r];
                                }
                            } else {
                                for (size_t r{0}; r < nroots_; ++r) {
                                    PT_map[r][new_det] +=
                                        sign * integral * c_block[k * nroots_ + r];
                                }
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
                auto [new_a_str, a_sign] = create_single_excitation(a_str, i, a);

                if (String::Hash()(new_a_str) % num_batches != batch_id) {
                    continue;
                }

                const auto& v_list = vab_sorted_[i * norb_ + a];
                new_det.set_a_string(new_a_str);

                for (const auto& [coupling, integral, j, b] : v_list) {
                    // break early if the integrals are too small
                    if (std::fabs(coupling * abs_c_max_block) < pt2_threshold)
                        break;
                    if (!creation_allowed(b) || !annihilation_allowed(j))
                        continue;

                    for (size_t k{0};
                         const auto& [b_str_idx, det_index] : second_string_to_det_index) {
                        const String& b_str = ab_list_.sorted_second_string(b_str_idx);

                        // check if the beta excitation is valid
                        if ((not b_str.get_bit(j)) or b_str.get_bit(b)) {
                            k++;
                            continue;
                        }

                        const double criterion = std::fabs(coupling * abs_c_max[k]);
                        if (criterion > pt2_threshold) {
                            auto [new_b_str, b_sign] = create_single_excitation(b_str, j, b);
                            new_det.set_b_string(new_b_str);
                            if (criterion > var_threshold) {
                                for (size_t r{0}; r < nroots_; ++r) {
                                    V_map[r][new_det] +=
                                        a_sign * b_sign * integral * c_block[k * nroots_ + r];
                                }
                            } else {
                                for (size_t r{0}; r < nroots_; ++r) {
                                    PT_map[r][new_det] +=
                                        a_sign * b_sign * integral * c_block[k * nroots_ + r];
                                }
                            }
                        }
                        k++;
                    }
                }
            }
        }

        // beta excitations
        if (String::Hash()(a_str) % num_batches != batch_id)
            continue;

        new_det.set_a_string(a_str);
        for (size_t k{0}; const auto& [b_str_idx, det_index] : second_string_to_det_index) {
            const String& b_str = ab_list_.sorted_second_string(b_str_idx);
            auto b_str_annihilation_masked = b_str & ~frozen_annihilation_mask_;
            b_str_annihilation_masked.find_set_bits(bocc, nob);
            auto b_str_creation_masked = (~b_str & norb_mask) & ~frozen_creation_mask_;
            b_str_creation_masked.find_set_bits(bvir, nvb);
            // compute_fast_virtual(bocc, bvir, norb_);
            std::span<size_t> bocc_span(bocc.data(), nob);
            std::span<size_t> bvir_span(bvir.data(), nvb);

            // single beta excitations
            for (const auto& i : bocc_span) {
                for (const auto& a : bvir_span) {
                    new_det.set_b_string(b_str); // push the current beta string to compute coupling
                    // const double integral = h_[i * norb_ + a];
                    const double integral = slater_rules_.singles_coupling_b(i, a, new_det);
                    const double criterion = std::fabs(integral * abs_c_max[k]);
                    if (criterion > pt2_threshold) {
                        auto [new_b_str, sign] = create_single_excitation(b_str, i, a);
                        new_det.set_b_string(new_b_str); // push the new beta string

                        if (criterion > var_threshold) {
                            for (size_t r{0}; r < nroots_; ++r) {
                                V_map[r][new_det] += sign * integral * c_block[k * nroots_ + r];
                            }
                        } else {
                            for (size_t r{0}; r < nroots_; ++r) {
                                PT_map[r][new_det] += sign * integral * c_block[k * nroots_ + r];
                            }
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
                        if (criterion < pt2_threshold)
                            break;

                        if ((a >= b) or b_str.get_bit(a) or b_str.get_bit(b) or
                            !creation_allowed(a, b))
                            continue;

                        auto [new_b_str, sign] = create_double_excitation(b_str, i, j, a, b);
                        new_det.set_a_string(a_str);
                        new_det.set_b_string(new_b_str);

                        if (criterion > var_threshold) {
                            for (size_t r{0}; r < nroots_; ++r) {
                                V_map[r][new_det] += sign * integral * c_block[k * nroots_ + r];
                            }
                        } else {
                            for (size_t r{0}; r < nroots_; ++r) {
                                PT_map[r][new_det] += sign * integral * c_block[k * nroots_ + r];
                            }
                        }
                    }
                }
            }
            k++;
        }
    }
    return {V_map, PT_map};
}

} // namespace forte2
