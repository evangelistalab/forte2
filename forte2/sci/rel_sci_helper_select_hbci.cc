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
// `singles_coupling(i, a, I)` and `Va(i, j, a, b)` evaluate the *I-side* matrix element
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

    // One coupling per external determinant per root, plus the set of determinants that will
    // join the variational space. The coupling must be complete before it is squared, so it is
    // never split by which side of var_threshold an individual connection falls on.
    std::vector<RelDetMap> map(nroots_);
    DetSet promoted;

    std::vector<size_t> occ(nel_, 0);
    std::vector<size_t> vir(norb_ - nel_, 0);

    for (size_t idx{0}, idx_max{dets_.size()}; idx < idx_max; ++idx) {
        const auto& det = dets_[idx];
        std::span<std::complex<double>> c_det(c_.data() + idx * nroots_, nroots_);
        double max_abs_c = 0.0;
        for (size_t r{0}; r < nroots_; ++r) {
            max_abs_c = std::max(max_abs_c, std::abs(c_det[r]));
        }

        const SpinorString str(det);
        size_t nocc;
        det.find_set_bits(occ, nocc);
        collect_virtual_orbitals(occ, vir, norb_);
        size_t nvir = norb_ - nocc;

        std::span<size_t> occ_span(occ.data(), nocc);
        std::span<size_t> vir_span(vir.data(), nvir);

        // single excitations
        for (const auto& i : occ_span) {
            if (!annihilation_allowed(i))
                continue;
            for (const auto& a : vir_span) {
                if (!creation_allowed(a))
                    continue;
                const std::complex<double> integral = singles_coupling(i, a, det);
                const double criterion = std::abs(integral * max_abs_c);

                if (criterion <= pt2_threshold)
                    continue;

                const auto [new_str, sign] = create_single_excitation_unchecked(str, i, a);
                const Determinant new_det(new_str);
                const std::complex<double> coupling = sign * std::conj(integral);

                if (criterion > var_threshold) {
                    promoted.insert(new_det);
                }
                for (size_t r{0}; r < nroots_; ++r) {
                    map[r][new_det] += coupling * c_det[r];
                }
            }
        }

        // double excitations
        for (const auto& i : occ_span) {
            if (!annihilation_allowed(i))
                continue;
            for (const auto& j : occ_span) {
                if (i >= j || !annihilation_allowed(j))
                    continue;
                for (const auto& a : vir_span) {
                    if (!creation_allowed(a))
                        continue;
                    for (const auto& b : vir_span) {
                        if (a >= b || !creation_allowed(b))
                            continue;

                        const std::complex<double> integral = Va(i, j, a, b);
                        const double criterion = std::abs(integral * max_abs_c);
                        if (criterion <= pt2_threshold)
                            continue;

                        const auto [new_str, sign] =
                            create_double_excitation_unchecked(str, i, j, a, b);
                        const Determinant new_det(new_str);
                        const std::complex<double> coupling = sign * std::conj(integral);

                        if (criterion > var_threshold) {
                            promoted.insert(new_det);
                        }
                        for (size_t r{0}; r < nroots_; ++r) {
                            map[r][new_det] += coupling * c_det[r];
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

    // accumulate the perturbative energy contributions for each root
    for (size_t r{0}; r < nroots_; ++r) {
        double var = 0.0;
        double pt = 0.0;
        for (const auto& [det, val] : map[r]) {
            const double delta = root_energies_[r] - slater_rules_.energy(det);
            (promoted.count(det) ? var : pt) += compute_delta_ept2(delta, std::abs(val));
        }
        ept2_var_[r] = var;
        ept2_pt_[r] = pt;
    }

    // number of new determinants added this cycle (root 0 defines the variational space)
    num_new_dets_var_ = promoted.size();
    num_new_dets_pt2_ = map[0].size() - promoted.size();

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
        RelSelectHbciScratch s;
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
                    target[r] +=
                        compute_delta_ept2(root_energies_[r] - energy, std::abs(s.coeffs[idx + r]));
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
    for (size_t t{0}; t < num_threads; ++t) {
        for (const auto& [batch_id, num_pt_dets, time] : thread_log_data[t]) {
            num_new_dets_pt2_ += num_pt_dets;
        }
    }
    selection_time_ = selection_timer.elapsed_seconds();
}

void RelSelectedCIHelper::select_hbci_batch(RelSelectHbciScratch& s, double var_threshold,
                                            double pt2_threshold, size_t num_batches,
                                            size_t batch_id, const DetSet& existing_dets) {
    auto& map = s.map;
    auto& coeffs = s.coeffs;
    auto& promoted = s.promoted;
    map.clear();
    coeffs.clear();
    promoted.clear();

    // size the caller's buffers on first use; later batches reuse them untouched
    s.occ.resize(nel_);
    s.vir.resize(norb_ - nel_);
    auto& occ = s.occ;
    auto& vir = s.vir;

    // The single place that decides whether a connection survives and what happens to it. Both
    // channels go through it, so neither can drift from the other, and it applies the same test
    // as select_hbci_ref. Channels also test the criterion themselves before building the
    // external determinant, but only to skip work and only against an upper bound on it; this
    // test is the one that decides.
    //
    // `coupling` already carries the conjugated, signed matrix element <J|H|I> (see the note at
    // the top of this file).
    auto accumulate = [&](const Determinant& det, std::span<const std::complex<double>> c_parent,
                          std::complex<double> coupling, double criterion) {
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

    SpinorString norb_mask = SpinorString::zero();
    norb_mask.fill_up_to(norb_);

    const auto& perm = ab_list_.det_permutation();
    // each determinant is its own first string, so string i is determinant perm[i]
    for (size_t i{0}, nstrings{ab_list_.first_string_size()}; i < nstrings; ++i) {
        const SpinorString& str = ab_list_.sorted_first_string(i);
        const size_t det_index = perm[i];

        // CI coefficients of this determinant for all roots, viewed directly in c_ (no gather)
        std::span<const std::complex<double>> c_det(c_.data() + det_index * nroots_, nroots_);
        double abs_c_max_det = 0.0;
        for (size_t r{0}; r < nroots_; ++r)
            abs_c_max_det = std::max(abs_c_max_det, std::abs(c_det[r]));

        // Every criterion below is proportional to this coefficient, so a determinant whose
        // coefficients are all zero cannot produce a connection that survives accumulate.
        if (abs_c_max_det == 0.0)
            continue;

        // occupied spinors that may be annihilated and virtual spinors that may be created
        size_t nocc, nvir;
        (str & ~frozen_annihilation_mask_).find_set_bits(occ, nocc);
        ((~str & norb_mask) & ~frozen_creation_mask_).find_set_bits(vir, nvir);

        std::span<size_t> occ_span(occ.data(), nocc);
        std::span<size_t> vir_span(vir.data(), nvir);

        // single excitations
        for (const auto& i : occ_span) {
            for (const auto& a : vir_span) {
                const auto [new_str, sign] = create_single_excitation_unchecked(str, i, a);
                if (batch_of(new_str, num_batches) != batch_id) {
                    continue;
                }
                const Determinant new_det(new_str);
                const std::complex<double> integral = singles_coupling(i, a, new_det);
                accumulate(new_det, c_det, sign * std::conj(integral),
                           std::abs(integral * abs_c_max_det));
            }
        }

        // double excitations
        for (const auto& i : occ_span) {
            for (const auto& j : occ_span) {
                if (i >= j)
                    continue;
                const auto& v_list = va_sorted_[i * norb_ + j];
                for (const auto& [key, integral, a, b] : v_list) {
                    // break early if the integrals are too small (the sorted real key monotonically
                    // decreases, so no later term can pass the threshold)
                    if (std::fabs(key * abs_c_max_det) <= pt2_threshold)
                        break;

                    if (str.get_bit(a) or str.get_bit(b))
                        continue;

                    const auto [new_str, sign] =
                        create_double_excitation_unchecked(str, i, j, a, b);

                    if (batch_of(new_str, num_batches) != batch_id) {
                        continue;
                    }

                    const double criterion = std::abs(integral * abs_c_max_det);
                    if (criterion <= pt2_threshold)
                        continue;
                    accumulate(Determinant(new_str), c_det, sign * std::conj(integral), criterion);
                }
            }
        }
    }
}

} // namespace forte2
