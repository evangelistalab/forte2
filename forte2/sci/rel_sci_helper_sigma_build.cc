#include <complex>

#include "helpers/timer.hpp"
#include "helpers/ndarray.h"
#include "helpers/np_vector_functions.h"
#include "helpers/parallel.h"

#include "rel_sci_helper.h"

namespace forte2 {

void RelSelectedCIHelper::compute_det_energies() {
    // compute the energy of all the determinants (RelSlaterRules::energy returns a real value,
    // the diagonal of a Hermitian matrix)
    const auto istart = det_energies_.size();
    det_energies_.resize(dets_.size());
    for (size_t i{istart}, n{dets_.size()}; i < n; ++i) {
        det_energies_[i] = slater_rules_.energy(dets_[i]);
    }
}

void RelSelectedCIHelper::prepare_strings() {
    // Only the alpha-beta string list is needed: with nb == 0 the beta string is always empty, so
    // there are no beta / beta-alpha excitation classes and the spin-flipped list of the real
    // helper would be redundant.
    std::vector<Determinant> sorted_dets = dets_;
    ab_list_ = SelectedCIStrings(norb_, sorted_dets);
}

void RelSelectedCIHelper::Hamiltonian(np_vector_complex basis, np_vector_complex sigma) const {
    local_timer t;
    vector::zero<std::complex<double>>(sigma);
    auto b_span = vector::as_span<std::complex<double>>(basis);
    auto s_span = vector::as_span<std::complex<double>>(sigma);

    // With an empty beta string only the diagonal, single-alpha and double-alpha-alpha blocks
    // contribute (H1b / H2b / H2ab all vanish).
    H0(b_span, s_span);
    H1a(b_span, s_span);
    H2a(b_span, s_span);
}

void RelSelectedCIHelper::H0(std::span<std::complex<double>> basis,
                             std::span<std::complex<double>> sigma) const {
    // H0 is diagonal in the determinant basis (det_energies_ is real)
    for (size_t i{0}, i_max{dets_.size()}; i < i_max; ++i) {
        sigma[i] = det_energies_[i] * basis[i];
    }
}

// == The nb == 0 scatter (why there is no find_matching_dets) ==
//
// With an empty beta string every determinant is uniquely identified by its alpha string, so the
// sorted first-string index i corresponds to exactly one determinant, det_permutation()[i], and
// there is a single (empty) beta string. The general helper's find_matching_dets -- a size-1
// hash-map lookup plus range / second-string indirection per matrix element -- therefore collapses
// to a direct scatter:
//     sigma[perm[i]] += int_sign * basis[perm[j]].
// No conjugation is applied: Hermiticity is produced by the caller visiting both index orderings
// ((p,q) and (q,p)), which contribute h(p,q) and h(q,p) = conj(h(p,q)) on separate iterations.

void RelSelectedCIHelper::H1a(std::span<std::complex<double>> basis,
                              std::span<std::complex<double>> sigma) const {
    const auto first_string_size = ab_list_.first_string_size();
    const auto& perm = ab_list_.det_permutation();

    parallel_for(first_string_size, [&](size_t i) {
        // Each thread owns a disjoint set of source strings i, hence disjoint destinations perm[i]
        // (perm is a bijection), so the accumulation into sigma below is contention-free.
        const size_t dest = perm[i];
        const auto& sublist = ab_list_.one_hole_first_string_list()[i];
        for (const auto& [p, hole_idx, sign_p] : sublist) {
            const auto& inv_sublist = ab_list_.one_hole_first_string_list_inv()[hole_idx];
            for (const auto& [q, j, sign_q] : inv_sublist) {
                if (p == q)
                    continue; // skip diagonal contribution
                const std::complex<double> h_pq = h(p, q);
                if (std::abs(h_pq) < integral_threshold)
                    continue;
                const double sign = sign_p * sign_q;
                sigma[dest] += (h_pq * sign) * basis[perm[j]];
            }
        }
    });
}

void RelSelectedCIHelper::H2a(std::span<std::complex<double>> basis,
                              std::span<std::complex<double>> sigma) const {
    const auto first_string_size = ab_list_.first_string_size();
    const auto& perm = ab_list_.det_permutation();

    parallel_for(first_string_size, [&](size_t i) {
        const size_t dest = perm[i];
        const auto& sublist = ab_list_.two_hole_string_list()[i];
        for (const auto& [p, q, hole_idx, sign_pq] : sublist) { // (p < q)
            const auto& inv_sublist = ab_list_.two_hole_string_list_inv()[hole_idx];
            for (const auto& [r, s, j, sign_rs] : inv_sublist) { // (r < s)
                if ((p == r) and (q == s))
                    continue; // skip diagonal contribution
                const std::complex<double> v_pqrs = Va(p, q, r, s);
                if (std::abs(v_pqrs) < integral_threshold)
                    continue;
                const double sign = sign_pq * sign_rs;
                sigma[dest] += (v_pqrs * sign) * basis[perm[j]];
            }
        }
    });
}

} // namespace forte2
