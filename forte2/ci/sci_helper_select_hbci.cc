
#include "helpers/logger.h"
#include "helpers/timer.hpp"
#include "helpers/sorting.hpp"
#include "helpers/np_matrix_functions.h"

#include "determinant_helpers.h"
#include "sci_helper.h"

namespace forte2 {

static inline void merge_and_keep_unique(std::vector<Determinant>& dets,
                                         std::vector<Determinant>& new_dets) {
    // Sort a copy of the existing determinants
    std::vector<Determinant> dets_sorted = dets;
    std::sort(dets_sorted.begin(), dets_sorted.end());

    // Sort the new determinants
    std::sort(new_dets.begin(), new_dets.end());

    // Keep only unique new determinants
    new_dets.erase(std::unique(new_dets.begin(), new_dets.end()), new_dets.end());

    append_unique_from_sorted_inplace(dets, dets_sorted, new_dets);
}

void SelectedCIHelper::select_hbci2(double threshold) {

    compute_det_energies();
    prepare_strings();

    auto rdm = compute_sf_1rdm(0, 0);
    for (size_t r{1}; r < nroots_; ++r) {
        auto rdm_r = compute_sf_1rdm(r, r);
        matrix::daxpy(1.0, rdm_r, rdm);
    }
    matrix::scale(rdm, 1.0 / nroots_);

    for (size_t i = 0; i < norb_; ++i) {
        epsilon_[i] = h_[i * norb_ + i];
        for (size_t j = 0; j < norb_; ++j) {
            for (size_t k = 0; k < norb_; ++k) {
                epsilon_[i] += rdm(j, k) * (V(i, j, i, k) - 0.5 * V(i, j, k, i));
            }
        }
    }

    update_hbci_ints();

    local_timer selection_timer;

    std::vector<size_t> aocc(na_);
    std::vector<size_t> bocc(nb_);
    std::vector<size_t> avir(norb_ - na_);
    std::vector<size_t> bvir(norb_ - nb_);

    std::vector<Determinant> new_dets;
    size_t checks_count = 0;
    double e_pt2 = 0.0;

    size_t noa, nob;
    for (size_t idx{0}, idx_max{dets_.size()}; idx < idx_max; ++idx) {
        const auto& det = dets_[idx];
        double c2 = 0.0;
        for (size_t r{0}; r < nroots_; ++r) {
            c2 += std::pow(c_[idx * nroots_ + r], 2);
        }

        // Perform selection based on threshold
        det.get_fast_a_occ(aocc, noa);
        det.get_fast_b_occ(bocc, nob);
        compute_fast_virtual(aocc, avir, norb_);
        compute_fast_virtual(bocc, bvir, norb_);
        size_t nva = norb_ - noa;
        size_t nvb = norb_ - nob;

        for (const auto& i : aocc) {
            for (const auto& a : avir) {
                double val = c2 * std::pow(h_[i * norb_ + a], 2.0) /
                             (std::abs(epsilon_[a] - epsilon_[i]) + 1e-3);
                checks_count++;
                if (val > threshold) {
                    new_dets.emplace_back(create_single_a_excitation(det, i, a));
                }
            }
        }

        for (const auto& i : bocc) {
            for (const auto& a : bvir) {
                double val = c2 * std::pow(h_[i * norb_ + a], 2.0) /
                             (std::abs(epsilon_[a] - epsilon_[i]) + 1e-3);
                checks_count++;
                if (val > threshold) {
                    new_dets.emplace_back(create_single_b_excitation(det, i, a));
                }
            }
        }

        for (const auto& i : aocc) {
            for (const auto& j : aocc) {
                if (i >= j)
                    continue;
                const auto& v_list = va_sorted_[i * norb_ + j];
                for (const auto& [val, a, b] : v_list) {
                    checks_count++;
                    if (std::abs(val * c2) < threshold)
                        break;
                    if (det.na(a) or det.na(b))
                        continue;
                    new_dets.emplace_back(create_double_aa_excitation(det, i, j, a, b));
                }
            }
        }

        for (const auto& i : bocc) {
            for (const auto& j : bocc) {
                if (i >= j)
                    continue;
                const auto& v_list = va_sorted_[i * norb_ + j];
                for (const auto& [val, a, b] : v_list) {
                    checks_count++;
                    if (std::abs(val * c2) < threshold)
                        break;
                    if (det.nb(a) or det.nb(b))
                        continue;
                    new_dets.emplace_back(create_double_bb_excitation(det, i, j, a, b));
                }
            }
        }

        for (const auto& i : aocc) {
            for (const auto& j : bocc) {
                const auto& v_list = v_sorted_[i * norb_ + j];
                for (const auto& [val, a, b] : v_list) {
                    checks_count++;
                    if (std::abs(val * c2) < threshold)
                        break;
                    if (det.na(a) or det.nb(b))
                        continue;
                    new_dets.emplace_back(create_double_ab_excitation(det, i, j, a, b));
                }
            }
        }
    }

    merge_and_keep_unique(dets_, new_dets);
    c_.resize(dets_.size(), 0.0);

    LOG(log_level_) << "Tested " << checks_count << " excitations for selection.";
    LOG(log_level_) << "Number of variational determinants after selection: " << dets_.size();
    LOG(log_level_) << "Selection completed in " << selection_timer.elapsed_seconds()
                    << " seconds.";

    compute_det_energies();
    prepare_strings();
}

} // namespace forte2
