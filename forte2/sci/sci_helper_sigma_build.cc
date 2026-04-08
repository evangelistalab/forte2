#include "helpers/logger.h"
#include "helpers/timer.hpp"
#include "helpers/ndarray.h"
#include "helpers/np_vector_functions.h"
#include "helpers/np_matrix_functions.h"

#include <future>

#include "sci_helper.h"

namespace forte2 {

namespace {

template <typename WorkFn>
void run_parallel_indices(size_t count, size_t num_threads, WorkFn&& work) {
    if (num_threads <= 1 || count == 0) {
        for (size_t i{0}; i < count; ++i)
            work(i);
        return;
    }

    std::vector<std::future<void>> workers;
    workers.reserve(num_threads);
    for (size_t t{0}; t < num_threads; ++t) {
        workers.push_back(std::async(std::launch::async, [count, num_threads, t, &work]() {
            for (size_t i{t}; i < count; i += num_threads)
                work(i);
        }));
    }

    for (auto& w : workers)
        w.get();
}

} // namespace

void SelectedCIHelper::compute_det_energies() {
    // compute the energy of all the determinants
    const auto istart = det_energies_.size();
    det_energies_.resize(dets_.size());
    for (size_t i{istart}, n{dets_.size()}; i < n; ++i) {
        det_energies_[i] = slater_rules_.energy(dets_[i]);
    }
}

void SelectedCIHelper::prepare_strings() {
    // create the sorted string lists for alpha-beta and beta-alpha
    // Alpha-Beta
    std::vector<Determinant> sorted_dets = dets_;
    ab_list_ = SelectedCIStrings(norb_, sorted_dets);

    // Beta-Alpha (flip alpha and beta strings)
    for (size_t i{0}, n{dets_.size()}; i < n; ++i) {
        sorted_dets[i] = dets_[i].spin_flip();
    }
    ba_list_ = SelectedCIStrings(norb_, sorted_dets);
}

void SelectedCIHelper::Hamiltonian(np_vector basis, np_vector sigma) const {
    local_timer t;
    vector::zero<double>(sigma);
    auto b_span = vector::as_span<double>(basis);
    auto s_span = vector::as_span<double>(sigma);

    H0(b_span, s_span);
    H1a(b_span, s_span);
    H1b(b_span, s_span);
    H2a(b_span, s_span);
    H2b(b_span, s_span);
    H2ab(b_span, s_span);
}

void SelectedCIHelper::H0(std::span<double> basis, std::span<double> sigma) const {
    // H0 is diagonal in the determinant basis
    for (size_t i{0}, i_max{dets_.size()}; i < i_max; ++i) {
        sigma[i] = det_energies_[i] * basis[i];
    }
}

void SelectedCIHelper::find_matching_dets(std::span<double> basis, std::span<double> sigma,
                                          const SelectedCIStrings& list, size_t i, size_t j,
                                          double int_sign) const {
    // Find the range of determinants with the current alpha string
    const auto& [istart, iend] = list.range(i);
    const auto& [jstart, jend] = list.range(j);
    const auto& det_permutation = list.det_permutation();

    // Here we choose to loop over the smaller range and look up the deteminants in the larger range
    // by using the hash map
    if (iend - istart >= jend - jstart) {
        const auto& i_map = list.second_string_to_det_index()[i];
        for (size_t jj{jstart}; jj < jend; ++jj) {
            const auto idx_j = list.sorted_dets_second_string(jj);
            if (const auto it = i_map.find(idx_j); it != i_map.end()) {
                sigma[it->second] += int_sign * basis[det_permutation[jj]];
            }
        }
    } else {
        const auto& j_map = list.second_string_to_det_index()[j];
        for (size_t ii{istart}; ii < iend; ++ii) {
            const auto idx_i = list.sorted_dets_second_string(ii);
            if (const auto it = j_map.find(idx_i); it != j_map.end()) {
                sigma[det_permutation[ii]] += int_sign * basis[it->second];
            }
        }
    }
}

void SelectedCIHelper::H1a(std::span<double> basis, std::span<double> sigma) const {
    const auto first_string_size = ab_list_.first_string_size();

    if (num_threads_ <= 1) {
        // Loop over all unique alpha strings
        for (size_t i{0}; i < first_string_size; ++i) {
            const auto& sublist = ab_list_.one_hole_first_string_list()[i];
            for (const auto& [p, hole_idx, sign_p] : sublist) {
                const auto& inv_sublist = ab_list_.one_hole_first_string_list_inv()[hole_idx];
                for (const auto& [q, j, sign_q] : inv_sublist) {
                    if (p == q)
                        continue; // skip diagonal contribution
                    const double h_pq = h(p, q);
                    if (std::fabs(h_pq) < integral_threshold)
                        continue;
                    const double sign = sign_p * sign_q;
                    find_matching_dets(basis, sigma, ab_list_, i, j, h_pq * sign);
                }
            }
        }
        return;
    }

    run_parallel_indices(first_string_size, num_threads_, [&](size_t i) {
        const auto& sublist = ab_list_.one_hole_first_string_list()[i];
        for (const auto& [p, hole_idx, sign_p] : sublist) {
            const auto& inv_sublist = ab_list_.one_hole_first_string_list_inv()[hole_idx];
            for (const auto& [q, j, sign_q] : inv_sublist) {
                if (p == q)
                    continue; // skip diagonal contribution
                const double h_pq = h(p, q);
                if (std::fabs(h_pq) < integral_threshold)
                    continue;
                const double sign = sign_p * sign_q;
                find_matching_dets(basis, sigma, ab_list_, i, j, h_pq * sign);
            }
        }
    });
}

void SelectedCIHelper::H1b(std::span<double> basis, std::span<double> sigma) const {
    const auto first_string_size = ba_list_.first_string_size();

    if (num_threads_ <= 1) {
        // Loop over all unique beta strings
        for (size_t i{0}; i < first_string_size; ++i) {
            const auto& sublist = ba_list_.one_hole_first_string_list()[i];
            for (const auto& [p, hole_idx, sign_p] : sublist) {
                const auto& inv_sublist = ba_list_.one_hole_first_string_list_inv()[hole_idx];
                for (const auto& [q, j, sign_q] : inv_sublist) {
                    if (p == q)
                        continue; // skip diagonal contribution
                    const double h_pq = h(p, q);
                    if (std::fabs(h_pq) < integral_threshold)
                        continue;
                    const double sign = sign_p * sign_q;
                    find_matching_dets(basis, sigma, ba_list_, i, j, h_pq * sign);
                }
            }
        }
        return;
    }

    run_parallel_indices(first_string_size, num_threads_, [&](size_t i) {
        const auto& sublist = ba_list_.one_hole_first_string_list()[i];
        for (const auto& [p, hole_idx, sign_p] : sublist) {
            const auto& inv_sublist = ba_list_.one_hole_first_string_list_inv()[hole_idx];
            for (const auto& [q, j, sign_q] : inv_sublist) {
                if (p == q)
                    continue; // skip diagonal contribution
                const double h_pq = h(p, q);
                if (std::fabs(h_pq) < integral_threshold)
                    continue;
                const double sign = sign_p * sign_q;
                find_matching_dets(basis, sigma, ba_list_, i, j, h_pq * sign);
            }
        }
    });
}

void SelectedCIHelper::H2a(std::span<double> basis, std::span<double> sigma) const {
    const auto first_string_size = ab_list_.first_string_size();

    if (num_threads_ <= 1) {
        // Loop over all unique alpha strings
        for (size_t i{0}; i < first_string_size; ++i) {
            const auto& sublist = ab_list_.two_hole_string_list()[i];
            for (const auto& [p, q, hole_idx, sign_pq] : sublist) { // (p < q)
                const auto& inv_sublist = ab_list_.two_hole_string_list_inv()[hole_idx];
                for (const auto& [r, s, j, sign_rs] : inv_sublist) { // (r < s)
                    if ((p == r) and (q == s))
                        continue; // skip diagonal contribution
                    const double v_pqrs = Va(p, q, r, s);
                    if (std::fabs(v_pqrs) < integral_threshold)
                        continue;
                    const double sign = sign_pq * sign_rs;
                    find_matching_dets(basis, sigma, ab_list_, i, j, v_pqrs * sign);
                }
            }
        }
        return;
    }

    run_parallel_indices(first_string_size, num_threads_, [&](size_t i) {
        const auto& sublist = ab_list_.two_hole_string_list()[i];
        for (const auto& [p, q, hole_idx, sign_pq] : sublist) { // (p < q)
            const auto& inv_sublist = ab_list_.two_hole_string_list_inv()[hole_idx];
            for (const auto& [r, s, j, sign_rs] : inv_sublist) { // (r < s)
                if ((p == r) and (q == s))
                    continue; // skip diagonal contribution
                const double v_pqrs = Va(p, q, r, s);
                if (std::fabs(v_pqrs) < integral_threshold)
                    continue;
                const double sign = sign_pq * sign_rs;
                find_matching_dets(basis, sigma, ab_list_, i, j, v_pqrs * sign);
            }
        }
    });
}

void SelectedCIHelper::H2b(std::span<double> basis, std::span<double> sigma) const {
    const auto first_string_size = ba_list_.first_string_size();

    if (num_threads_ <= 1) {
        // Loop over all unique beta strings
        for (size_t i{0}; i < first_string_size; ++i) {
            const auto& sublist = ba_list_.two_hole_string_list()[i];
            for (const auto& [p, q, hole_idx, sign_pq] : sublist) { // (p < q)
                const auto& inv_sublist = ba_list_.two_hole_string_list_inv()[hole_idx];
                for (const auto& [r, s, j, sign_rs] : inv_sublist) { // (r < s)
                    if ((p == r) and (q == s))
                        continue; // skip diagonal contribution
                    const double v_pqrs = Va(p, q, r, s);
                    if (std::fabs(v_pqrs) < integral_threshold)
                        continue;
                    const double sign = sign_pq * sign_rs;
                    find_matching_dets(basis, sigma, ba_list_, i, j, v_pqrs * sign);
                }
            }
        }
        return;
    }

    run_parallel_indices(first_string_size, num_threads_, [&](size_t i) {
        const auto& sublist = ba_list_.two_hole_string_list()[i];
        for (const auto& [p, q, hole_idx, sign_pq] : sublist) { // (p < q)
            const auto& inv_sublist = ba_list_.two_hole_string_list_inv()[hole_idx];
            for (const auto& [r, s, j, sign_rs] : inv_sublist) { // (r < s)
                if ((p == r) and (q == s))
                    continue; // skip diagonal contribution
                const double v_pqrs = Va(p, q, r, s);
                if (std::fabs(v_pqrs) < integral_threshold)
                    continue;
                const double sign = sign_pq * sign_rs;
                find_matching_dets(basis, sigma, ba_list_, i, j, v_pqrs * sign);
            }
        }
    });
}

void SelectedCIHelper::H2ab(std::span<double> basis, std::span<double> sigma) const {
    const auto first_string_size = ab_list_.first_string_size();
    const auto& det_permutation = ab_list_.det_permutation();

    if (num_threads_ <= 1) {
        // Loop over all unique alpha strings
        for (size_t i{0}; i < first_string_size; ++i) {
            const auto& i_map = ab_list_.second_string_to_det_index()[i];
            const auto& sublist_a = ab_list_.one_hole_first_string_list()[i];
            for (const auto& [p, hole_idx, sign_p] : sublist_a) {
                const auto& inv_sublist_a = ab_list_.one_hole_first_string_list_inv()[hole_idx];
                for (const auto& [q, j, sign_q] : inv_sublist_a) {
                    // At this point we have a+_p a_q acting on the alpha string
                    const auto& [jstart, jend] = ab_list_.range(j);
                    // Loop over all the beta strings with the same alpha string
                    for (size_t jj{jstart}; jj < jend; ++jj) {
                        const auto idx_j = ab_list_.sorted_dets_second_string(jj);
                        // Now loop over single excitations in the beta string
                        const auto& sublist_b = ab_list_.one_hole_second_string_list()[idx_j];
                        for (const auto& [r, hole_idx_b, sign_r] : sublist_b) {
                            const auto& inv_sublist_b =
                                ab_list_.one_hole_second_string_list_inv()[hole_idx_b];
                            for (const auto& [s, k, sign_s] : inv_sublist_b) {
                                if ((p == q) and (r == s))
                                    continue; // skip diagonal contribution
                                const double v_pqrs = V(p, r, q, s);
                                if (std::fabs(v_pqrs) < integral_threshold)
                                    continue;
                                const double sign = sign_p * sign_q * sign_r * sign_s;
                                // Check if the determinant with the new beta string exists
                                if (const auto it = i_map.find(k); it != i_map.end()) {
                                    sigma[it->second] += v_pqrs * sign * basis[det_permutation[jj]];
                                }
                            }
                        }
                    }
                }
            }
        }
        return;
    }

    run_parallel_indices(first_string_size, num_threads_, [&](size_t i) {
        const auto& i_map = ab_list_.second_string_to_det_index()[i];
        const auto& sublist_a = ab_list_.one_hole_first_string_list()[i];
        for (const auto& [p, hole_idx, sign_p] : sublist_a) {
            const auto& inv_sublist_a = ab_list_.one_hole_first_string_list_inv()[hole_idx];
            for (const auto& [q, j, sign_q] : inv_sublist_a) {
                // At this point we have a+_p a_q acting on the alpha string
                const auto& [jstart, jend] = ab_list_.range(j);
                // Loop over all the beta strings with the same alpha string
                for (size_t jj{jstart}; jj < jend; ++jj) {
                    const auto idx_j = ab_list_.sorted_dets_second_string(jj);
                    // Now loop over single excitations in the beta string
                    const auto& sublist_b = ab_list_.one_hole_second_string_list()[idx_j];
                    for (const auto& [r, hole_idx_b, sign_r] : sublist_b) {
                        const auto& inv_sublist_b =
                            ab_list_.one_hole_second_string_list_inv()[hole_idx_b];
                        for (const auto& [s, k, sign_s] : inv_sublist_b) {
                            if ((p == q) and (r == s))
                                continue; // skip diagonal contribution
                            const double v_pqrs = V(p, r, q, s);
                            if (std::fabs(v_pqrs) < integral_threshold)
                                continue;
                            const double sign = sign_p * sign_q * sign_r * sign_s;
                            // Check if the determinant with the new beta string exists
                            if (const auto it = i_map.find(k); it != i_map.end()) {
                                sigma[it->second] += v_pqrs * sign * basis[det_permutation[jj]];
                            }
                        }
                    }
                }
            }
        }
    });
}

double SelectedCIHelper::find_matching_dets_1rdm(size_t left_root, size_t right_root,
                                                 const SelectedCIStrings& list, size_t i, size_t j,
                                                 double sign) const {
    double result = 0.0;

    // Find the range of determinants with the current alpha string
    const auto& [istart, iend] = list.range(i);
    const auto& [jstart, jend] = list.range(j);
    const auto& det_permutation = list.det_permutation();

    // Here we choose to loop over the smaller range and look up the deteminants in the larger range
    // by using the hash map
    if (iend - istart >= jend - jstart) {
        const auto& i_map = list.second_string_to_det_index()[i];
        for (size_t jj{jstart}; jj < jend; ++jj) {
            const auto idx_j = list.sorted_dets_second_string(jj);
            if (const auto it = i_map.find(idx_j); it != i_map.end()) {
                result += sign * c_[nroots_ * it->second + left_root] *
                          c_[nroots_ * det_permutation[jj] + right_root];
            }
        }
    } else {
        const auto& j_map = list.second_string_to_det_index()[j];
        for (size_t ii{istart}; ii < iend; ++ii) {
            const auto idx_i = list.sorted_dets_second_string(ii);
            if (const auto it = j_map.find(idx_i); it != j_map.end()) {
                result += sign * c_[nroots_ * det_permutation[ii] + left_root] *
                          c_[nroots_ * it->second + right_root];
            }
        }
    }

    return result;
}

np_matrix SelectedCIHelper::compute_a_1rdm(size_t left_root, size_t right_root) const {
    auto rdm = make_zeros<nb::numpy, double, 2>({norb_, norb_});
    double* rdm_data = rdm.data();

    const auto first_string_size = ab_list_.first_string_size();
    const auto& one_hole_first_strings = ab_list_.one_hole_first_strings();
    // Loop over all unique alpha strings
    for (size_t i{0}; i < first_string_size; ++i) {
        const auto& sublist = ab_list_.one_hole_first_string_list()[i];
        for (const auto& [p, hole_idx, sign_p] : sublist) {
            const auto& inv_sublist = ab_list_.one_hole_first_string_list_inv()[hole_idx];
            for (const auto& [q, j, sign_q] : inv_sublist) {
                const double sign = sign_p * sign_q;
                rdm_data[p * norb_ + q] +=
                    find_matching_dets_1rdm(left_root, right_root, ab_list_, i, j, sign);
            }
        }
    }

    return rdm;
}

np_matrix SelectedCIHelper::compute_b_1rdm(size_t left_root, size_t right_root) const {
    auto rdm = make_zeros<nb::numpy, double, 2>({norb_, norb_});
    double* rdm_data = rdm.data();

    const auto first_string_size = ba_list_.first_string_size();
    const auto& one_hole_first_strings = ba_list_.one_hole_first_strings();
    // Loop over all unique beta strings
    for (size_t i{0}; i < first_string_size; ++i) {
        const auto& sublist = ba_list_.one_hole_first_string_list()[i];
        for (const auto& [p, hole_idx, sign_p] : sublist) {
            const auto& inv_sublist = ba_list_.one_hole_first_string_list_inv()[hole_idx];
            for (const auto& [q, j, sign_q] : inv_sublist) {
                const double sign = sign_p * sign_q;
                rdm_data[p * norb_ + q] +=
                    find_matching_dets_1rdm(left_root, right_root, ba_list_, i, j, sign);
            }
        }
    }

    return rdm;
}

np_matrix SelectedCIHelper::compute_sf_1rdm(size_t left_root, size_t right_root) const {
    auto sf_1rdm = make_zeros<nb::numpy, double, 2>({norb_, norb_});
    if (norb_ > 0) {
        auto a_1rdm = compute_a_1rdm(left_root, right_root);
        auto b_1rdm = compute_b_1rdm(left_root, right_root);
        matrix::daxpy<double>(1.0, a_1rdm, sf_1rdm);
        matrix::daxpy<double>(1.0, b_1rdm, sf_1rdm);
    }
    return sf_1rdm;
}

} // namespace forte2
