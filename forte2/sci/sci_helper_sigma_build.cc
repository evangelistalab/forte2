#include "helpers/logger.h"
#include "helpers/timer.hpp"
#include "helpers/ndarray.h"
#include "helpers/np_vector_functions.h"
#include "helpers/np_matrix_functions.h"
#include "helpers/spin.h"

#include <future>

#include "sci_helper.h"

namespace forte2 {

namespace {

// A helper function to run a loop in parallel over a given number of indices. The work function is
// called with the index as argument.
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
    if (use_claude_algorithms_) {
        H2aa_claude(b_span, s_span);
        H2bb_claude(b_span, s_span);
        H2ab_claude(b_span, s_span);
    } else {
        H2aa(b_span, s_span);
        H2bb(b_span, s_span);
        H2ab(b_span, s_span);
    }
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

    // Here we choose to loop over the smaller range and look up the determinants in the larger
    // range by using the hash map
    if (iend - istart >= jend - jstart) {
        const auto& i_map = list.second_string_to_det_index()[i];
        for (size_t jj{jstart}; jj < jend; ++jj) {
            const auto idx_j = list.sorted_dets_second_string(jj);
            if (const auto it = i_map.find(idx_j); it != i_map.end()) {
                // NOTE: when find_matching_dets is called from different threads,
                // this increment is contention-free, because each thread is working
                // on its own alpha string (i), and the sigma updates are on disjoint sets of
                // determinants because they have different alpha strings Same logic applies to the
                // other branch of this if statement
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

void SelectedCIHelper::H2aa(std::span<double> basis, std::span<double> sigma) const {
    const auto first_string_size = ab_list_.first_string_size();

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

void SelectedCIHelper::H2bb(std::span<double> basis, std::span<double> sigma) const {
    const auto first_string_size = ba_list_.first_string_size();

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
                                // See the comment in find_matching_dets about why this increment is
                                // contention-free when called from different threads
                                sigma[it->second] += v_pqrs * sign * basis[det_permutation[jj]];
                            }
                        }
                    }
                }
            }
        }
    });
}

void SelectedCIHelper::H2ab_claude(std::span<double> basis, std::span<double> sigma) const {
    const size_t first_string_size = ab_list_.first_string_size();
    const size_t nKa = ab_list_.one_hole_first_strings().size();
    const size_t nKb = ab_list_.one_hole_second_strings().size();
    const size_t nKab = nKa * nKb;

    if (nKa == 0 || nKb == 0)
        return;

    const auto& det_permutation = ab_list_.det_permutation();
    // one_hole_alpha[Ia]     -> (orbital q, Ka, sign_q)  where a_q |Ia> = sign_q |Ka>
    const auto& one_hole_alpha = ab_list_.one_hole_first_string_list();
    // one_hole_alpha_inv[Ka] -> (orbital q, Ja, sign_q)  where a_q |Ja> = sign_q |Ka>
    //   equivalently a+_q |Ka> = sign_q |Ja>
    const auto& one_hole_alpha_inv = ab_list_.one_hole_first_string_list_inv();
    // one_hole_beta[Ib]  -> (orbital s, Kb, sign_s)  where a_s |Ib> = sign_s |Kb>
    const auto& one_hole_beta = ab_list_.one_hole_second_string_list();

    // Build V_mat cache: V_mat[pr * norb2_ + qs] = V(p, r, q, s), invalidated by set_Hamiltonian
    if (h2ab_claude_V_mat_.empty()) {
        h2ab_claude_V_mat_.resize(norb2_ * norb2_);
        for (size_t p = 0; p < norb_; ++p)
            for (size_t r = 0; r < norb_; ++r)
                for (size_t q = 0; q < norb_; ++q)
                    for (size_t s = 0; s < norb_; ++s)
                        h2ab_claude_V_mat_[(p * norb_ + r) * norb2_ + (q * norb_ + s)] =
                            V(p, r, q, s);
    }

    // Enumerate active (Ka, Kb) pairs, iterating Ka first so that each Ka owns a
    // contiguous block of column indices — makes the parallel gather contention-free.
    constexpr size_t INACTIVE = std::numeric_limits<size_t>::max();
    if (h2ab_claude_pair_to_col_.size() != nKab)
        h2ab_claude_pair_to_col_.assign(nKab, INACTIVE);
    else
        std::fill(h2ab_claude_pair_to_col_.begin(), h2ab_claude_pair_to_col_.end(), INACTIVE);

    size_t Kpairs = 0;
    for (size_t Ka = 0; Ka < nKa; ++Ka) {
        for (const auto& [q, Ja, sign_q] : one_hole_alpha_inv[Ka]) {
            const auto& [jstart, jend] = ab_list_.range(Ja);
            for (size_t jj = jstart; jj < jend; ++jj) {
                const size_t Ib = ab_list_.sorted_dets_second_string(jj);
                for (const auto& [s, Kb, sign_s] : one_hole_beta[Ib]) {
                    size_t& col = h2ab_claude_pair_to_col_[Ka * nKb + Kb];
                    if (col == INACTIVE)
                        col = Kpairs++;
                }
            }
        }
    }
    h2ab_claude_Kpairs_ = Kpairs;

    if (Kpairs == 0)
        return;

    // Sparse D: D_cols_[col] stores (qs, value) pairs for each active (Ka,Kb) column.
    // Each column has at most na_*nb_ entries (one per pair of removed alpha/beta electrons).
    // Reserve na_*nb_ upfront to avoid allocator calls during the parallel gather.
    h2ab_claude_D_cols_.resize(Kpairs);
    const size_t reserve_per_col = na_ * nb_;
    for (auto& v : h2ab_claude_D_cols_) {
        v.clear();
        if (v.capacity() < reserve_per_col)
            v.reserve(reserve_per_col);
    }

    // Gather: parallel over Ka (contention-free — Ka=i owns col indices col(Ka=i,*))
    // D_cols_[col(Ka,Kb)].push_back({q*norb+s, sign_q*sign_s*C[Ja,Ib]})
    run_parallel_indices(nKa, num_threads_, [&](size_t Ka) {
        for (const auto& [q, Ja, sign_q] : one_hole_alpha_inv[Ka]) {
            const auto& [jstart, jend] = ab_list_.range(Ja);
            for (size_t jj = jstart; jj < jend; ++jj) {
                const size_t Ib = ab_list_.sorted_dets_second_string(jj);
                const double coeff = basis[det_permutation[jj]];
                for (const auto& [s, Kb, sign_s] : one_hole_beta[Ib]) {
                    const size_t col = h2ab_claude_pair_to_col_[Ka * nKb + Kb];
                    h2ab_claude_D_cols_[col].push_back(
                        {static_cast<uint32_t>(q * norb_ + s), sign_q * sign_s * coeff});
                }
            }
        }
    });

    // Combined multiply-scatter (no intermediate E matrix):
    // sigma[det] += sum_{(p,Ka),(r,Kb)} sign_p*sign_r
    //               * sum_{(qs,d) in D_cols_[col(Ka,Kb)]} V(p, s_ann, q_ann, r_create) * d
    //
    // Variable conventions (matching H2ab reference):
    //   p         = alpha creation orbital (in OUTPUT alpha Ia, p = Ia\Ka)
    //   r_create  = beta  creation orbital (in OUTPUT beta  Ib, r_create = Ib\Kb)
    //   q_ann     = alpha annihilation orbital (qs/norb_), stored in D_cols_ from input alpha
    //   s_ann     = beta  annihilation orbital (qs%norb_), stored in D_cols_ from input beta
    //   V(p, s_ann, q_ann, r_create) = V_mat[(p*norb+s_ann)*norb2 + q_ann*norb + r_create]
    //
    // Parallel over Ia: contention-free since each det_idx belongs to exactly one Ia group.
    run_parallel_indices(first_string_size, num_threads_, [&](size_t Ia) {
        const auto& [start, end] = ab_list_.range(Ia);
        for (size_t ii = start; ii < end; ++ii) {
            const size_t Ib = ab_list_.sorted_dets_second_string(ii);
            const size_t det_idx = det_permutation[ii];
            double contrib = 0.0;
            for (const auto& [p, Ka, sign_p] : one_hole_alpha[Ia]) {
                for (const auto& [r, Kb, sign_r] : one_hole_beta[Ib]) {
                    const size_t col = h2ab_claude_pair_to_col_[Ka * nKb + Kb];
                    const double sign_pr = sign_p * sign_r;
                    for (const auto& [qs, d] : h2ab_claude_D_cols_[col]) {
                        // qs = q_ann * norb_ + s_ann (from gather)
                        // want V(p, s_ann, q_ann, r_create) =
                        // V_mat[(p*norb+s_ann)*norb2+q_ann*norb+r]
                        const size_t s_ann = qs % norb_;
                        const size_t q_ann = qs / norb_;
                        contrib +=
                            sign_pr *
                            h2ab_claude_V_mat_[(p * norb_ + s_ann) * norb2_ + q_ann * norb_ + r] *
                            d;
                    }
                }
            }
            sigma[det_idx] += contrib;
        }
    });

    // Diagonal correction: the scatter above includes the p=q, r=s term (same det contributing
    // to itself), which equals sum_{p,r} V(p,r,p,r)*C[Ia,Ib] already in det_energies_ — subtract.
    run_parallel_indices(first_string_size, num_threads_, [&](size_t Ia) {
        std::vector<size_t> occ_alpha(norb_), occ_beta(norb_);
        size_t n_alpha = 0;
        ab_list_.sorted_first_string(Ia).find_set_bits(occ_alpha, n_alpha);
        const auto& [istart, iend] = ab_list_.range(Ia);
        for (size_t ii = istart; ii < iend; ++ii) {
            const size_t Ib = ab_list_.sorted_dets_second_string(ii);
            const size_t det_idx = det_permutation[ii];
            size_t n_beta = 0;
            ab_list_.sorted_second_string(Ib).find_set_bits(occ_beta, n_beta);
            double diag = 0.0;
            for (size_t pa = 0; pa < n_alpha; ++pa)
                for (size_t pb = 0; pb < n_beta; ++pb)
                    diag += V(occ_alpha[pa], occ_beta[pb], occ_alpha[pa], occ_beta[pb]);
            sigma[det_idx] -= diag * basis[det_idx];
        }
    });
}

void SelectedCIHelper::H2aa_claude(std::span<double> basis, std::span<double> sigma) const {
    const size_t nIa = ab_list_.first_string_size();       // unique alpha strings
    const size_t nIb = ab_list_.second_string_size();      // unique beta strings
    const size_t nKa = ab_list_.two_hole_strings().size(); // unique two-hole alpha strings

    if (nIa == 0 || nIb == 0 || nKa == 0)
        return;

    const auto& det_permutation = ab_list_.det_permutation();
    const auto& two_hole_list = ab_list_.two_hole_string_list();    // [Ia] -> (p,q,Ka,sign)
    const auto& two_hole_inv = ab_list_.two_hole_string_list_inv(); // [Ka] -> (p,q,Ia,sign)

    // ---- Build active (Ka, Ib) -> column mapping ----
    // pair_to_col[Ka * nIb + Ib] = column index in D_cols, or INACTIVE
    constexpr size_t INACTIVE = std::numeric_limits<size_t>::max();
    if (h2aa_claude_pair_to_col_.size() != nKa * nIb)
        h2aa_claude_pair_to_col_.assign(nKa * nIb, INACTIVE);
    else
        std::fill(h2aa_claude_pair_to_col_.begin(), h2aa_claude_pair_to_col_.end(), INACTIVE);

    size_t Kpairs = 0;
    for (size_t Ka = 0; Ka < nKa; ++Ka) {
        for (const auto& [p, q, Ja, sign] : two_hole_inv[Ka]) {
            const auto& [jstart, jend] = ab_list_.range(Ja);
            for (size_t jj = jstart; jj < jend; ++jj) {
                const size_t Ib = ab_list_.sorted_dets_second_string(jj);
                size_t& col = h2aa_claude_pair_to_col_[Ka * nIb + Ib];
                if (col == INACTIVE)
                    col = Kpairs++;
            }
        }
    }

    if (Kpairs == 0)
        return;

    // ---- Allocate and clear D_cols ----
    h2aa_claude_D_cols_.resize(Kpairs);
    const size_t reserve_per_col = na_ * (na_ - 1) / 2;
    for (auto& v : h2aa_claude_D_cols_) {
        v.clear();
        if (v.capacity() < reserve_per_col)
            v.reserve(reserve_per_col);
    }

    // ---- Gather: parallel over Ka (contention-free — each Ka owns disjoint col indices) ----
    // D_cols[col(Ka,Ib)] += {rs_label, sign_rs * basis[det(Ja,Ib)]}
    // where (r,s,Ja,sign_rs) iterates over two_hole_inv[Ka]
    run_parallel_indices(nKa, num_threads_, [&](size_t Ka) {
        for (const auto& [r, s, Ja, sign_rs] : two_hole_inv[Ka]) {
            const auto& [jstart, jend] = ab_list_.range(Ja);
            for (size_t jj = jstart; jj < jend; ++jj) {
                const size_t Ib = ab_list_.sorted_dets_second_string(jj);
                const size_t col = h2aa_claude_pair_to_col_[Ka * nIb + Ib];
                h2aa_claude_D_cols_[col].push_back(
                    {static_cast<uint32_t>(r * norb_ + s), sign_rs * basis[det_permutation[jj]]});
            }
        }
    });

    // ---- Scatter: parallel over Ia (contention-free — each det_idx belongs to one Ia range) ----
    // sigma[det(Ia,Ib)] += sum_{(p,q,Ka)} sign_pq * sum_{(rs,d) in D[Ka,Ib]} Va(p,q,r,s) * d
    run_parallel_indices(nIa, num_threads_, [&](size_t Ia) {
        const auto& [istart, iend] = ab_list_.range(Ia);
        for (size_t ii = istart; ii < iend; ++ii) {
            const size_t Ib = ab_list_.sorted_dets_second_string(ii);
            const size_t det_idx = det_permutation[ii];
            double contrib = 0.0;
            for (const auto& [p, q, Ka, sign_pq] : two_hole_list[Ia]) {
                const size_t col = h2aa_claude_pair_to_col_[Ka * nIb + Ib];
                if (col == INACTIVE)
                    continue;
                for (const auto& [rs, d] : h2aa_claude_D_cols_[col]) {
                    const size_t r = rs / norb_;
                    const size_t s = rs % norb_;
                    contrib += sign_pq * Va(p, q, r, s) * d;
                }
            }
            sigma[det_idx] += contrib;
        }
    });

    // ---- Diagonal correction: D_cols includes Ja==Ia self-contributions; subtract them ----
    // Self-contribution for (Ia,Ib): sum_{p<q in occ_alpha(Ia)} Va(p,q,p,q) * basis[det(Ia,Ib)]
    run_parallel_indices(nIa, num_threads_, [&](size_t Ia) {
        std::vector<size_t> occ(norb_);
        size_t n_occ = 0;
        ab_list_.sorted_first_string(Ia).find_set_bits(occ, n_occ);
        double diag = 0.0;
        for (size_t i = 0; i < n_occ; ++i)
            for (size_t j = i + 1; j < n_occ; ++j)
                diag += Va(occ[i], occ[j], occ[i], occ[j]);
        const auto& [istart, iend] = ab_list_.range(Ia);
        for (size_t ii = istart; ii < iend; ++ii)
            sigma[det_permutation[ii]] -= diag * basis[det_permutation[ii]];
    });
}

void SelectedCIHelper::H2bb_claude(std::span<double> basis, std::span<double> sigma) const {
    // ba_list_ sorts beta first (first string = beta), alpha second — spectator here is alpha.
    const size_t nBb = ba_list_.first_string_size();       // unique beta strings
    const size_t nBa = ba_list_.second_string_size();      // unique alpha strings (spectator)
    const size_t nKb = ba_list_.two_hole_strings().size(); // unique two-hole beta strings

    if (nBb == 0 || nBa == 0 || nKb == 0)
        return;

    const auto& det_permutation = ba_list_.det_permutation();
    const auto& two_hole_list = ba_list_.two_hole_string_list();
    const auto& two_hole_inv = ba_list_.two_hole_string_list_inv();

    // ---- Build active (Kb, Ia) -> column mapping ----
    constexpr size_t INACTIVE = std::numeric_limits<size_t>::max();
    if (h2bb_claude_pair_to_col_.size() != nKb * nBa)
        h2bb_claude_pair_to_col_.assign(nKb * nBa, INACTIVE);
    else
        std::fill(h2bb_claude_pair_to_col_.begin(), h2bb_claude_pair_to_col_.end(), INACTIVE);

    size_t Kpairs = 0;
    for (size_t Kb = 0; Kb < nKb; ++Kb) {
        for (const auto& [p, q, Jb, sign] : two_hole_inv[Kb]) {
            const auto& [jstart, jend] = ba_list_.range(Jb);
            for (size_t jj = jstart; jj < jend; ++jj) {
                const size_t Ia = ba_list_.sorted_dets_second_string(jj);
                size_t& col = h2bb_claude_pair_to_col_[Kb * nBa + Ia];
                if (col == INACTIVE)
                    col = Kpairs++;
            }
        }
    }

    if (Kpairs == 0)
        return;

    // ---- Allocate and clear D_cols ----
    h2bb_claude_D_cols_.resize(Kpairs);
    const size_t reserve_per_col = nb_ * (nb_ - 1) / 2;
    for (auto& v : h2bb_claude_D_cols_) {
        v.clear();
        if (v.capacity() < reserve_per_col)
            v.reserve(reserve_per_col);
    }

    // ---- Gather: parallel over Kb (contention-free) ----
    run_parallel_indices(nKb, num_threads_, [&](size_t Kb) {
        for (const auto& [r, s, Jb, sign_rs] : two_hole_inv[Kb]) {
            const auto& [jstart, jend] = ba_list_.range(Jb);
            for (size_t jj = jstart; jj < jend; ++jj) {
                const size_t Ia = ba_list_.sorted_dets_second_string(jj);
                const size_t col = h2bb_claude_pair_to_col_[Kb * nBa + Ia];
                h2bb_claude_D_cols_[col].push_back(
                    {static_cast<uint32_t>(r * norb_ + s), sign_rs * basis[det_permutation[jj]]});
            }
        }
    });

    // ---- Scatter: parallel over Ib (contention-free) ----
    run_parallel_indices(nBb, num_threads_, [&](size_t Ib) {
        const auto& [istart, iend] = ba_list_.range(Ib);
        for (size_t ii = istart; ii < iend; ++ii) {
            const size_t Ia = ba_list_.sorted_dets_second_string(ii);
            const size_t det_idx = det_permutation[ii];
            double contrib = 0.0;
            for (const auto& [p, q, Kb, sign_pq] : two_hole_list[Ib]) {
                const size_t col = h2bb_claude_pair_to_col_[Kb * nBa + Ia];
                if (col == INACTIVE)
                    continue;
                for (const auto& [rs, d] : h2bb_claude_D_cols_[col]) {
                    const size_t r = rs / norb_;
                    const size_t s = rs % norb_;
                    contrib += sign_pq * Va(p, q, r, s) * d;
                }
            }
            sigma[det_idx] += contrib;
        }
    });

    // ---- Diagonal correction ----
    run_parallel_indices(nBb, num_threads_, [&](size_t Ib) {
        std::vector<size_t> occ(norb_);
        size_t n_occ = 0;
        ba_list_.sorted_first_string(Ib).find_set_bits(occ, n_occ);
        double diag = 0.0;
        for (size_t i = 0; i < n_occ; ++i)
            for (size_t j = i + 1; j < n_occ; ++j)
                diag += Va(occ[i], occ[j], occ[i], occ[j]);
        const auto& [istart, iend] = ba_list_.range(Ib);
        for (size_t ii = istart; ii < iend; ++ii)
            sigma[det_permutation[ii]] -= diag * basis[det_permutation[ii]];
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

    // Here we choose to loop over the smaller range and look up the determinants in the larger
    // range by using the hash map
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

np_matrix SelectedCIHelper::compute_aa_2rdm(size_t left_root, size_t right_root) const {
    // calculate the number of pairs of orbitals p > q
    const size_t npairs = (norb_ * (norb_ - 1)) / 2;
    auto rdm = make_zeros<nb::numpy, double, 2>({npairs, npairs});
    double* rdm_data = rdm.data();

    const auto first_string_size = ab_list_.first_string_size();
    const auto& one_hole_first_strings = ab_list_.one_hole_first_strings();
    // Loop over all unique alpha strings
    for (size_t i{0}; i < first_string_size; ++i) {
        const auto& sublist = ab_list_.two_hole_string_list()[i];
        for (const auto& [p, q, hole_idx, sign_pq] : sublist) { // (p < q)
            const size_t pq = pair_index_gt(p, q);
            const auto& inv_sublist = ab_list_.two_hole_string_list_inv()[hole_idx];
            for (const auto& [r, s, j, sign_rs] : inv_sublist) { // (r < s)
                const size_t rs = pair_index_gt(r, s);
                const double sign = sign_pq * sign_rs;
                rdm_data[pq * npairs + rs] +=
                    find_matching_dets_1rdm(left_root, right_root, ab_list_, i, j, sign);
            }
        }
    }
    return rdm;
}

np_matrix SelectedCIHelper::compute_bb_2rdm(size_t left_root, size_t right_root) const {
    // calculate the number of pairs of orbitals p > q
    const size_t npairs = (norb_ * (norb_ - 1)) / 2;
    auto rdm = make_zeros<nb::numpy, double, 2>({npairs, npairs});
    double* rdm_data = rdm.data();

    const auto first_string_size = ba_list_.first_string_size();
    const auto& one_hole_first_strings = ba_list_.one_hole_first_strings();
    // Loop over all unique alpha strings
    for (size_t i{0}; i < first_string_size; ++i) {
        const auto& sublist = ba_list_.two_hole_string_list()[i];
        for (const auto& [p, q, hole_idx, sign_pq] : sublist) { // (p < q)
            const size_t pq = pair_index_gt(p, q);
            const auto& inv_sublist = ba_list_.two_hole_string_list_inv()[hole_idx];
            for (const auto& [r, s, j, sign_rs] : inv_sublist) { // (r < s)
                const size_t rs = pair_index_gt(r, s);
                const double sign = sign_pq * sign_rs;
                rdm_data[pq * npairs + rs] +=
                    find_matching_dets_1rdm(left_root, right_root, ba_list_, i, j, sign);
            }
        }
    }
    return rdm;
}

np_tensor4 SelectedCIHelper::compute_ab_2rdm(size_t left_root, size_t right_root) const {
    auto rdm = make_zeros<nb::numpy, double, 4>({norb_, norb_, norb_, norb_});
    double* rdm_data = rdm.data();

    const auto first_string_size = ab_list_.first_string_size();
    const auto& det_permutation = ab_list_.det_permutation();
    // Loop over all unique alpha strings
    for (size_t i{0}; i < first_string_size; ++i) {
        const auto& i_map = ab_list_.second_string_to_det_index()[i];
        // Loop over all single excitations in the alpha string.
        // a+_q a_p |i_a> -> +/-|j_a>
        const auto& sublist_a = ab_list_.one_hole_first_string_list()[i];
        for (const auto& [p, hole_idx, sign_p] : sublist_a) {
            const auto& inv_sublist_a = ab_list_.one_hole_first_string_list_inv()[hole_idx];
            for (const auto& [q, j, sign_q] : inv_sublist_a) {
                const auto& [jstart, jend] = ab_list_.range(j);
                // Loop over all the beta strings with the same alpha string
                for (size_t jj{jstart}; jj < jend; ++jj) {
                    const auto idx_j = ab_list_.sorted_dets_second_string(jj);
                    // Loop over single excitations in the beta string.
                    // a+_s a_r |j_b> -> +/-|k_b>
                    const auto& sublist_b = ab_list_.one_hole_second_string_list()[idx_j];
                    for (const auto& [r, hole_idx_b, sign_r] : sublist_b) {
                        const auto& inv_sublist_b =
                            ab_list_.one_hole_second_string_list_inv()[hole_idx_b];
                        for (const auto& [s, k, sign_s] : inv_sublist_b) {
                            const double sign = sign_p * sign_q * sign_r * sign_s;
                            // Here we get (leaving note since this is a bit subtle):
                            //     <j_a k_b|a+_r a_s a+_q a_p|i_a i_b>
                            //   = <j_a k_b|a+_q a+_r a_s a_p|i_a i_b> = gamma2(q_a,r_b,p_a,s_b)
                            // Check if the determinant with the new beta string exists
                            if (const auto it = i_map.find(k); it != i_map.end()) {
                                rdm_data[q * norb3_ + r * norb2_ + p * norb_ + s] +=
                                    sign * c_[nroots_ * it->second + left_root] *
                                    c_[nroots_ * det_permutation[jj] + right_root];
                            }
                        }
                    }
                }
            }
        }
    }
    return rdm;
}

np_tensor4 SelectedCIHelper::compute_sf_2rdm(size_t left_root, size_t right_root) const {
    auto rdm_sf = make_zeros<nb::numpy, double, 4>({norb_, norb_, norb_, norb_});

    if (norb_ < 1) {
        return rdm_sf; // No 2-RDM for less than 1 orbitals
    }

    auto rdm_sf_v = rdm_sf.view();
    // Mixed-spin contribution (1 orbital or more)
    {
        auto rdm_ab = compute_ab_2rdm(left_root, right_root);
        auto rdm_ab_v = rdm_ab.view();
        for (size_t p{0}; p < norb_; ++p) {
            for (size_t q{0}; q < norb_; ++q) {
                for (size_t r{0}; r < norb_; ++r) {
                    for (size_t s{0}; s < norb_; ++s) {
                        rdm_sf_v(p, q, r, s) += rdm_ab_v(p, q, r, s) + rdm_ab_v(q, p, s, r);
                    }
                }
            }
        }
    }

    if (norb_ < 2) {
        return rdm_sf; // No same-spin contributions to the 2-RDM for less than 2 orbitals
    }

    // To reduce the  memory footprint, we compute the aa and bb contributions in a packed
    // format and one at a time.
    for (auto spin : {Spin::Alpha, Spin::Beta}) {
        auto rdm_ss = spin == Spin::Alpha ? compute_aa_2rdm(left_root, right_root)
                                          : compute_bb_2rdm(left_root, right_root);
        auto rdm_ss_v = rdm_ss.view();
        for (size_t p{1}, pq{0}; p < norb_; ++p) {
            for (size_t q{0}; q < p; ++q, ++pq) { // p > q
                for (size_t r{1}, rs{0}; r < norb_; ++r) {
                    for (size_t s{0}; s < r; ++s, ++rs) { // r > s
                        auto element = rdm_ss_v(pq, rs);
                        rdm_sf_v(p, q, r, s) += element;
                        rdm_sf_v(q, p, r, s) -= element;
                        rdm_sf_v(p, q, s, r) -= element;
                        rdm_sf_v(q, p, s, r) += element;
                    }
                }
            }
        }
    }

    return rdm_sf;
}

double SelectedCIHelper::find_matching_dets_1trdm(size_t left_root, size_t right_root,
                                                  const SelectedCIStrings& left_list,
                                                  const SelectedCIStrings& right_list,
                                                  const std::vector<double>& left_c,
                                                  const std::vector<double>& right_c, size_t i,
                                                  size_t j, double sign) const {
    double result = 0.0;

    // Find the range of determinants with the current alpha string
    const auto& [istart, iend] = left_list.range(i);
    const auto& [jstart, jend] = right_list.range(j);
    const auto& right_det_permutation = right_list.det_permutation();
    const auto& left_det_permutation = left_list.det_permutation();

    // Here we choose to loop over the smaller range and look up the determinants in the larger
    // range by using the hash map
    if (iend - istart >= jend - jstart) {
        const auto& i_map = left_list.second_string_to_det_index()[i];
        for (size_t jj{jstart}; jj < jend; ++jj) {
            const auto idx_j = right_list.sorted_dets_second_string(jj);
            if (const auto it = i_map.find(idx_j); it != i_map.end()) {
                result += sign * left_c[nroots_ * it->second + left_root] *
                          right_c[nroots_ * right_det_permutation[jj] + right_root];
            }
        }
    } else {
        const auto& j_map = right_list.second_string_to_det_index()[j];
        for (size_t ii{istart}; ii < iend; ++ii) {
            const auto idx_i = left_list.sorted_dets_second_string(ii);
            if (const auto it = j_map.find(idx_i); it != j_map.end()) {
                result += sign * left_c[nroots_ * left_det_permutation[ii] + left_root] *
                          right_c[nroots_ * it->second + right_root];
            }
        }
    }

    return result;
}

np_matrix SelectedCIHelper::compute_s_1trdm(const SelectedCIHelper& right_helper, size_t left_root,
                                            size_t right_root, Spin spin) const {
    const auto& left_helper = *this;

    const auto& left_c = left_helper.c_;
    const auto& right_c = right_helper.c_;

    auto rdm = make_zeros<nb::numpy, double, 2>({norb_, norb_});
    double* rdm_data = rdm.data();

    // pick the appropriate string lists based on the spin
    const auto& left_list = is_alpha(spin) ? left_helper.ab_list_ : left_helper.ba_list_;
    const auto& right_list = is_alpha(spin) ? right_helper.ab_list_ : right_helper.ba_list_;

    const auto right_first_string_size = right_list.first_string_size();
    const auto& right_one_hole_first_strings = right_list.one_hole_first_strings();
    const auto& left_one_hole_first_strings_index = left_list.one_hole_first_strings_index();

    // Loop over all unique strings of the right state
    for (size_t j{0}; j < right_first_string_size; ++j) {
        const auto& sublist_right = right_list.one_hole_first_string_list()[j];
        // loop over all single excitations in the right string. a_p |j> -> +/-|k>
        for (const auto& [p, right_hole_idx, sign_p] : sublist_right) {
            // get the one-hole alpha string from the right solver
            const auto& K = right_one_hole_first_strings[right_hole_idx];
            // find the index of the one-hole alpha string K in the left solver
            if (const auto it = left_one_hole_first_strings_index.find(K);
                it != left_one_hole_first_strings_index.end()) {
                // if found, get the corresponding inv_sublist on the left
                const auto& inv_sublist = left_list.one_hole_first_string_list_inv()[it->second];
                for (const auto& [q, i, sign_q] : inv_sublist) {
                    const double sign = sign_p * sign_q;
                    rdm_data[p * norb_ + q] += find_matching_dets_1trdm(
                        left_root, right_root, left_list, right_list, left_c, right_c, i, j, sign);
                }
            }
        }
    }

    return rdm;
}

np_matrix SelectedCIHelper::compute_a_1trdm(const SelectedCIHelper& right_helper, size_t left_root,
                                            size_t right_root) const {
    return compute_s_1trdm(right_helper, left_root, right_root, Spin::Alpha);
}

np_matrix SelectedCIHelper::compute_b_1trdm(const SelectedCIHelper& right_helper, size_t left_root,
                                            size_t right_root) const {
    return compute_s_1trdm(right_helper, left_root, right_root, Spin::Beta);
}

} // namespace forte2
