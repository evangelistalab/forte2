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

std::vector<double> SelectedCIHelper::compute_spin2() const {
    const size_t num_total_batches = num_batches_per_thread_;
    const double sz = 0.5 * (na_ - nb_);
    const auto sz_contribution_to_s2 = sz * (sz + 1.0);
    std::vector<double> s2_expectation_values(nroots_, sz_contribution_to_s2);

    // Single thread batch processing to limit memory usage
    for (size_t b{0}; b < num_total_batches; ++b) {
        const auto s2_expectation_values_batch = spin2_batch(num_total_batches, b);
        for (size_t r{0}; r < nroots_; ++r) {
            s2_expectation_values[r] += s2_expectation_values_batch[r];
        }
    }

    return s2_expectation_values;
}

std::vector<double> SelectedCIHelper::spin2_batch(size_t num_batches, size_t batch_id) const {
    // We assume all determinants have the same number of electrons
    std::vector<size_t> aocc(na_);
    std::vector<size_t> bocc(nb_);
    std::vector<size_t> avir(norb_ - na_);
    std::vector<size_t> bvir(norb_ - nb_);

    size_t noa, nob;

    std::vector<DetMap> s_plus_map(nroots_);

    const auto a_string_size = ab_list_.first_string_size();
    // Phase factor to account for the sign to permute a(pβ) through all alpha electrons
    const double a_str_phase = (na_ % 2 == 0) ? 1.0 : -1.0;

    Determinant new_det;
    // Loop over all unique alpha strings
    for (size_t i{0}; i < a_string_size; ++i) {
        const String& a_str = ab_list_.sorted_first_string(i);
        const auto& second_string_to_det_index = ab_list_.second_string_to_det_index()[i];

        // find the occupied and empty orbitals for the current alpha string
        a_str.find_set_bits(aocc, noa);
        compute_fast_virtual(aocc, avir, norb_);

        // single alpha creation of an electron (S+ = a^+(pα) a(pβ))
        for (const auto& a : avir) {
            String new_a_str{a_str};
            const double a_sign = new_a_str.create(a);

            // only process this string if it belongs to the current batch
            if (String::Hash()(new_a_str) % num_batches != batch_id) {
                continue;
            }
            new_det.set_a_string(new_a_str);

            // find strings where we can annihilate the electron in orbital a
            for (const auto& [b_str_idx, det_index] : second_string_to_det_index) {
                const String& b_str = ab_list_.sorted_second_string(b_str_idx);
                // only proceed if we can annihilate the beta electron in orbital a
                if (b_str.get_bit(a)) {
                    String new_b_str = b_str;
                    const double b_sign = new_b_str.destroy(a);
                    new_det.set_b_string(new_b_str);
                    const double sign = a_str_phase * a_sign * b_sign;
                    for (size_t r{0}; r < nroots_; ++r) {
                        s_plus_map[r][new_det] += sign * c_[det_index * nroots_ + r];
                    }
                }
            }
        }
    }

    std::vector<double> s2_eigenvalues(nroots_, 0.0);
    for (size_t r{0}; r < nroots_; ++r) {
        const auto& vec = s_plus_map[r];
        double s2_val = 0.0;
        for (const auto& [_, val] : vec) {
            s2_val += val * val;
        }
        s2_eigenvalues[r] = s2_val;
    }

    return s2_eigenvalues;
}

} // namespace forte2
