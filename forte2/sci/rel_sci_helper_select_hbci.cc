#include <atomic>
#include <complex>
#include <future>
#include <thread>

#include "helpers/logger.h"
#include "helpers/timer.hpp"

#include "determinant/determinant_helpers.h"
#include "rel_sci_helper.h"

namespace forte2 {

// == Complex-coupling convention (the one place the real -> complex port is not mechanical) ==
//
// For an external determinant J connected to a variational determinant I by the excitation
// generated below, the first-order (PT2 / Epstein-Nesbet) numerator is
//     V_J = <J|H|Psi> = sum_I <J|H|I> c_I .
// `singles_coupling_a(i, a, I)` and `Va(i, j, a, b)` evaluate the *I-side* matrix element
// <I|H|J> (up to the excitation sign), so the bra-side element is its complex conjugate:
//     <J|H|I> = sign * conj(integral)          (sign is real, +/-1).
// (conj(Va(i,j,a,j)) = Va(a,j,i,j) and the differing self-terms vanish by antisymmetry, so the
// conjugate of the I-side coupling is exactly the J-side coupling.) We therefore accumulate
// `sign * conj(integral) * c_I`. Only |V_J| enters `compute_delta_ept2`, and the screening
// criterion uses |integral| (conjugation invariant), so this matters only when several complex
// c_I contribute to the same J -- but it is the correct PT2 numerator.

void RelSelectedCIHelper::select_hbci_ref(double var_threshold, double pt2_threshold) {
    compute_det_energies();
    prepare_strings();

    update_hbci_ints();

    local_timer selection_timer;

    std::vector<RelDetMap> V_map(nroots_);
    std::vector<RelDetMap> PT_map(nroots_);

    std::vector<size_t> aocc(na_, 0);
    std::vector<size_t> avir(norb_ - na_, 0);

    for (size_t idx{0}, idx_max{dets_.size()}; idx < idx_max; ++idx) {
        const auto& det = dets_[idx];
        std::span<std::complex<double>> c_det(c_.data() + idx * nroots_, nroots_);
        double max_abs_c = 0.0;
        for (size_t r{0}; r < nroots_; ++r) {
            max_abs_c = std::max(max_abs_c, std::abs(c_det[r]));
        }

        size_t noa;
        det.collect_alpha_occupied(aocc, noa);
        collect_virtual_orbitals(aocc, avir, norb_);
        size_t nva = norb_ - noa;

        std::span<size_t> aocc_span(aocc.data(), noa);
        std::span<size_t> avir_span(avir.data(), nva);

        // single alpha excitations
        for (const auto& i : aocc_span) {
            if (!annihilation_allowed(i))
                continue;
            for (const auto& a : avir_span) {
                if (!creation_allowed(a))
                    continue;
                const std::complex<double> integral = singles_coupling_a(i, a, det);
                const double criterion = std::abs(integral * max_abs_c);

                if (criterion < pt2_threshold)
                    continue;

                const auto [new_det, sign] = create_single_a_excitation(det, i, a);
                const std::complex<double> coupling = sign * std::conj(integral);

                if (criterion > var_threshold) {
                    for (size_t r{0}; r < nroots_; ++r) {
                        V_map[r][new_det] += coupling * c_det[r];
                    }
                } else {
                    for (size_t r{0}; r < nroots_; ++r) {
                        PT_map[r][new_det] += coupling * c_det[r];
                    }
                }
            }
        }

        // double alpha-alpha excitations
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

                        const std::complex<double> integral = Va(i, j, a, b);
                        const double criterion = std::abs(integral * max_abs_c);
                        if (criterion < pt2_threshold)
                            continue;

                        const auto [new_det, sign] = create_double_aa_excitation(det, i, j, a, b);
                        const std::complex<double> coupling = sign * std::conj(integral);

                        if (criterion > var_threshold) {
                            for (size_t r{0}; r < nroots_; ++r) {
                                V_map[r][new_det] += coupling * c_det[r];
                            }
                        } else {
                            for (size_t r{0}; r < nroots_; ++r) {
                                PT_map[r][new_det] += coupling * c_det[r];
                            }
                        }
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

    // Remove the coupling for the PT2 determinants that are already in the new variational space
    for (size_t r{0}; r < nroots_; ++r) {
        for (const auto& [det, val] : V_map[r]) {
            PT_map[r].erase(det);
        }
    }

    // add variational determinants first (root 0 defines the variational space)
    for (const auto& [det, val] : V_map[0]) {
        dets_.push_back(det);
    }

    // accumulate the perturbative energy contributions for each root
    for (size_t r{0}; r < nroots_; ++r) {
        double var = 0.0;
        double pt = 0.0;
        for (const auto& [det, val] : V_map[r]) {
            const double delta = root_energies_[r] - slater_rules_.energy(det);
            var += compute_delta_ept2(delta, std::abs(val));
        }
        for (const auto& [det, val] : PT_map[r]) {
            const double delta = root_energies_[r] - slater_rules_.energy(det);
            pt += compute_delta_ept2(delta, std::abs(val));
        }
        ept2_var_[r] = var;
        ept2_pt_[r] = pt;
    }

    // number of new determinants added this cycle (root 0 defines the variational space)
    num_new_dets_var_ = V_map[0].size();
    num_new_dets_pt2_ = PT_map[0].size();

    c_.resize(dets_.size() * nroots_, 0.0);

    compute_det_energies();
    prepare_strings();

    selection_time_ = selection_timer.elapsed_seconds();
}

void RelSelectedCIHelper::select_hbci(double var_threshold, double pt2_threshold) {
    local_timer selection_timer;

    // Plain HBCI stores |V|, which does not change across iterations, so the sorted integral
    // lists are built once (in the constructor / set_Hamiltonian). eHBCI, which would fold in the
    // orbital-energy denominators, is not supported in the two-component helper.

    const size_t num_batches = num_batches_per_thread_ * num_threads_; // total number of batches

    std::atomic<size_t> next_batch(0);

    std::vector<std::vector<Determinant>> thread_new_dets(num_threads_);
    std::vector<std::vector<double>> local_ept2_var(num_threads_,
                                                    std::vector<double>(nroots_, 0.0));
    std::vector<std::vector<double>> local_ept2_pt(num_threads_, std::vector<double>(nroots_, 0.0));
    std::vector<std::vector<std::tuple<size_t, size_t, double>>> thread_log_data(num_threads_);

    DetSet existing_dets(dets_.begin(), dets_.end());

    // worker function for each thread that processes batches of determinants
    auto worker = [&](size_t thread_id) {
        // persistent storage for this thread to avoid repeated allocations. The maps are cleared
        // at the beginning of select_hbci_batch, but the underlying memory is reused.
        RelDetRootMap V_map, PT_map;
        std::vector<std::complex<double>> V_coeffs, PT_coeffs;
        std::vector<Determinant> new_dets_local;

        while (true) {
            // Get the next batch ID for this thread
            size_t batch_id = next_batch.fetch_add(1);
            if (batch_id >= num_batches)
                break;

            local_timer batch_timer;

            select_hbci_batch(V_map, PT_map, V_coeffs, PT_coeffs, var_threshold, pt2_threshold,
                              num_batches, batch_id, existing_dets);

            // Filter out determinants in PT_map that are already in the new variational space
            // (V_map). Both have already been filtered against the existing variational space in
            // select_hbci_batch.
            for (const auto& [det, _] : V_map) {
                PT_map.erase(det);
            }

            // Compute energy contributions for new variational and PT2 determinants.
            for (const auto& [det, idx] : V_map) {
                const double energy = slater_rules_.energy(det);
                for (size_t r{0}; r < nroots_; ++r) {
                    local_ept2_var[thread_id][r] +=
                        compute_delta_ept2(root_energies_[r] - energy, std::abs(V_coeffs[idx + r]));
                }
            }

            for (const auto& [det, idx] : PT_map) {
                const double energy = slater_rules_.energy(det);
                for (size_t r{0}; r < nroots_; ++r) {
                    local_ept2_pt[thread_id][r] +=
                        compute_delta_ept2(root_energies_[r] - energy, std::abs(PT_coeffs[idx + r]));
                }
            }

            // Append to thread-local container (no locks)
            new_dets_local.clear();
            new_dets_local.reserve(V_map.size());
            for (const auto& [det, _] : V_map)
                new_dets_local.push_back(det);

            thread_new_dets[thread_id].insert(thread_new_dets[thread_id].end(),
                                              new_dets_local.begin(), new_dets_local.end());

            thread_log_data[thread_id].push_back(
                {batch_id, PT_map.size(), batch_timer.elapsed_seconds()});
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

        LOG(log_level_) << "Thread " << t << " processed " << total_batches << " batches, found "
                        << total_dets << " new determinants in " << total_time << " seconds (avg "
                        << (total_batches > 0 ? total_time / total_batches : 0.0) << " s/batch, "
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
    for (size_t t{0}; t < num_threads_; ++t) {
        for (const auto& [batch_id, num_pt_dets, time] : thread_log_data[t]) {
            num_new_dets_pt2_ += num_pt_dets;
        }
    }
    selection_time_ = selection_timer.elapsed_seconds();
}

void RelSelectedCIHelper::select_hbci_batch(RelDetRootMap& V_map, RelDetRootMap& PT_map,
                                            std::vector<std::complex<double>>& V_coeffs,
                                            std::vector<std::complex<double>>& PT_coeffs,
                                            double var_threshold, double pt2_threshold,
                                            size_t num_batches, size_t batch_id,
                                            const DetSet& existing_dets) {
    V_map.clear();
    PT_map.clear();
    V_coeffs.clear();
    PT_coeffs.clear();

    // Accumulate `prefactor * c_I` for the determinant `det`, appending a new coefficient block for
    // each root if the determinant is seen for the first time. `prefactor` already carries the
    // conjugated, signed coupling <J|H|I> (see the note at the top of this file).
    auto accumulate = [&](RelDetRootMap& map, std::vector<std::complex<double>>& coeffs,
                          const Determinant& det, std::complex<double> prefactor,
                          const std::span<std::complex<double>>& c) {
        size_t loc = coeffs.size();
        auto [it, emplaced] = map.try_emplace(det, loc);
        if (emplaced) {
            // new determinant, need to append to coeffs vector
            for (auto& c_r : c) {
                coeffs.push_back(prefactor * c_r);
            }
        } else {
            // idx is the starting index in the coeffs vector for this determinant
            size_t idx = it->second;
            for (size_t r{0}; auto& c_r : c) {
                coeffs[idx + r] += prefactor * c_r;
                r++;
            }
        }
    };

    std::vector<size_t> aocc(na_);
    std::vector<size_t> avir(norb_ - na_);

    const auto a_string_size = ab_list_.first_string_size();

    // norb_mask is used to compute the allowed virtual creation indices
    String norb_mask = String::zero();
    norb_mask.fill_up_to(norb_);

    Determinant new_det;
    auto hash = String::Hash();
    // Loop over all unique alpha strings. With nb == 0 each alpha string maps to exactly one
    // determinant (the beta string is the single empty spectator), so the per-alpha-string
    // coefficient "block" of the general helper collapses to one coefficient vector and the
    // block-max collapses to that determinant's max |c|.
    for (size_t i{0}; i < a_string_size; ++i) {
        const String& a_str = ab_list_.sorted_first_string(i);
        const auto& [b_str_idx, det_index] =
            *ab_list_.second_string_to_det_index()[i].begin(); // single entry

        // fix the (empty) beta spectator string for this determinant; only the alpha string is
        // mutated by the excitations below
        new_det.set_beta_string(ab_list_.sorted_second_string(b_str_idx));

        // CI coefficients of this determinant for all roots, viewed directly in c_ (no gather)
        std::span<std::complex<double>> c_det(c_.data() + det_index * nroots_, nroots_);
        double abs_c_max_det = 0.0;
        for (size_t r{0}; r < nroots_; ++r)
            abs_c_max_det = std::max(abs_c_max_det, std::abs(c_det[r]));

        // find the occupied and virtual orbitals for the current alpha string
        auto a_str_annihilation_masked = a_str & ~frozen_annihilation_mask_;
        // noa is the number of occupied alpha orbitals that we are allowed to annihilate from
        size_t noa, nva;
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
                auto [new_a_str, sign] = create_single_excitation_unchecked(a_str, i, a);
                // determine if this determinant belongs to the current batch
                if (hash(new_a_str) % num_batches != batch_id) {
                    continue;
                }
                new_det.set_alpha_string(new_a_str);
                const std::complex<double> integral = singles_coupling_a(i, a, new_det);
                const double criterion = std::abs(integral * abs_c_max_det);
                if (criterion > pt2_threshold) {
                    // if the determinant is already in the variational space, skip it
                    if (!existing_dets.count(new_det)) {
                        const std::complex<double> coupling = sign * std::conj(integral);
                        if (criterion > var_threshold) {
                            accumulate(V_map, V_coeffs, new_det, coupling, c_det);
                        } else {
                            accumulate(PT_map, PT_coeffs, new_det, coupling, c_det);
                        }
                    }
                }
            }
        }

        // double alpha-alpha excitations
        for (const auto& i : aocc_span) {
            for (const auto& j : aocc_span) {
                if (i >= j)
                    continue;
                const auto& v_list = va_sorted_[i * norb_ + j];
                for (const auto& [key, integral, a, b] : v_list) {
                    // break early if the integrals are too small (the sorted real key monotonically
                    // decreases, so no later term can pass the threshold)
                    if (std::fabs(key * abs_c_max_det) < pt2_threshold)
                        break;

                    if ((a >= b) or a_str.get_bit(a) or a_str.get_bit(b))
                        continue;

                    auto [new_a_str, sign] = create_double_excitation_unchecked(a_str, i, j, a, b);

                    if (hash(new_a_str) % num_batches != batch_id) {
                        continue;
                    }

                    const double criterion = std::abs(integral * abs_c_max_det);
                    if (criterion > pt2_threshold) {
                        new_det.set_alpha_string(new_a_str);
                        // if the determinant is already in the variational space, skip it
                        if (!existing_dets.count(new_det)) {
                            const std::complex<double> coupling = sign * std::conj(integral);
                            if (criterion > var_threshold) {
                                accumulate(V_map, V_coeffs, new_det, coupling, c_det);
                            } else {
                                accumulate(PT_map, PT_coeffs, new_det, coupling, c_det);
                            }
                        }
                    }
                }
            }
        }
    }
}

} // namespace forte2
