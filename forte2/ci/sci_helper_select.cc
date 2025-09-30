
#include "helpers/logger.h"
#include "helpers/timer.hpp"
#include "helpers/sorting.hpp"

#include "determinant_helpers.h"
#include "sci_helper.h"

namespace forte2 {

static inline void
merge_unique_keep_coeff(std::vector<Determinant>& dets,
                        std::vector<double>& c_flat, // row-major [row * nroots + r]
                        const std::vector<Determinant>& new_dets, std::size_t nroots) {
    struct Rec {
        Determinant d;
        std::size_t old_idx; // SIZE_MAX == new (zero-initialized row)
    };

    const std::size_t oldN = dets.size();
    std::vector<Rec> recs;
    recs.reserve(oldN + new_dets.size());

    // Originals first so stable_sort keeps them over identical "new" ones
    for (std::size_t k = 0; k < oldN; ++k)
        recs.push_back(Rec{dets[k], k});
    for (const auto& d : new_dets)
        recs.push_back(Rec{d, std::size_t(-1)});

    // Sort by determinant; stability preserves original-before-new for ties
    std::stable_sort(recs.begin(), recs.end(), [](const Rec& a, const Rec& b) {
        return Determinant::reverse_less_than(a.d, b.d);
    });

    // Unique by determinant (keep first occurrence)
    recs.erase(std::unique(recs.begin(), recs.end(),
                           [](const Rec& a, const Rec& b) { return a.d == b.d; }),
               recs.end());

    // Rebuild dets and c (copy old rows; zeros for new rows)
    const std::size_t newN = recs.size();
    std::vector<Determinant> dets_new;
    dets_new.reserve(newN);
    std::vector<double> c_new(newN * nroots, 0.0);

    for (std::size_t r = 0; r < newN; ++r) {
        dets_new.push_back(recs[r].d);
        const std::size_t oi = recs[r].old_idx;
        if (oi != std::size_t(-1)) {
            std::copy_n(&c_flat[oi * nroots], nroots, &c_new[r * nroots]);
        }
    }

    dets.swap(dets_new);
    c_flat.swap(c_new);
}

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

void SelectedCIHelper::select_cipsi(double threshold) {

    local_timer selection_timer;

    // Placeholder for CIPSI selection algorithm
    // std::cout << "CIPSI selection with threshold: " << threshold << std::endl;

    std::vector<size_t> aocc(norb_);
    std::vector<size_t> bocc(norb_);
    std::vector<size_t> avir(norb_);
    std::vector<size_t> bvir(norb_);

    std::vector<Determinant> new_dets;
    size_t checks_count = 0;

    size_t noa, nob;
    for (size_t idx{0}, idx_max{dets_.size()}; idx < idx_max; ++idx) {
        const auto& det = dets_[idx];
        const double c = c_[idx];

        // std::cout << "Determinant in variational space: " << str(det, norb_) << std::endl;

        // Perform selection based on threshold
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

        // std::cout << "  Alpha occ: ";
        // for (size_t i = 0; i < noa; ++i) {
        //     std::cout << aocc[i] << " ";
        // }
        // std::cout << std::endl;

        // std::cout << "  Beta occ: ";
        // for (size_t i = 0; i < nob; ++i) {
        //     std::cout << bocc[i] << " ";
        // }
        // std::cout << std::endl;

        // std::cout << "  Alpha vir: ";
        // for (size_t i = 0; i < nva; ++i) {
        //     std::cout << avir[i] << " ";
        // }
        // std::cout << std::endl;

        // std::cout << "  Beta vir: ";
        // for (size_t i = 0; i < nvb; ++i) {
        //     std::cout << bvir[i] << " ";
        // }
        // std::cout << std::endl;

        // select_singles(det, aocc, na, avir, nva, threshold,[]{})

        for (const auto& i : aocc_span) {
            for (const auto& a : avir_span) {
                double h_ia = c * h_[i * norb_ + a] / (epsilon_[a] - epsilon_[i]);
                checks_count++;
                if (std::abs(h_ia) > threshold) {
                    // std::cout << "  Selected single excitation (alpha): " << i << " -> " << a
                    //           << " with matrix element " << h_ia << std::endl;
                    new_dets.emplace_back(create_single_a_excitation(det, i, a));
                }
            }
        }

        for (const auto& i : bocc_span) {
            for (const auto& a : bvir_span) {
                double h_ia = c * h_[i * norb_ + a] / (epsilon_[a] - epsilon_[i]);
                checks_count++;
                if (std::abs(h_ia) > threshold) {
                    // std::cout << "  Selected single excitation (beta): " << i << " -> " << a
                    //           << " with matrix element " << h_ia << std::endl;
                    new_dets.emplace_back(create_single_b_excitation(det, i, a));
                }
            }
        }

        for (const auto& i : aocc_span) {
            for (const auto& j : aocc_span) {
                if (i >= j)
                    continue;
                for (const auto& a : avir_span) {
                    for (const auto& b : avir_span) {
                        if (a >= b)
                            continue;
                        checks_count++;
                        double h_ijab = c * Va(i, j, a, b) /
                                        (epsilon_[a] + epsilon_[b] - epsilon_[i] - epsilon_[j]);
                        if (std::abs(h_ijab) > threshold) {
                            // std::cout << "  Selected double excitation (alpha-alpha): " << i << j
                            //           << " -> " << a << b << " with matrix element " << h_ijab
                            //           << std::endl;
                            new_dets.emplace_back(create_double_aa_excitation(det, i, j, a, b));
                        }
                    }
                }
            }
        }

        for (const auto& i : bocc_span) {
            for (const auto& j : bocc_span) {
                if (i >= j)
                    continue;
                for (const auto& a : bvir_span) {
                    for (const auto& b : bvir_span) {
                        if (a >= b)
                            continue;
                        checks_count++;
                        double h_ijab = c * Va(i, j, a, b) /
                                        (epsilon_[a] + epsilon_[b] - epsilon_[i] - epsilon_[j]);
                        if (std::abs(h_ijab) > threshold) {
                            // std::cout << "  Selected double excitation (beta-beta): " << i << j
                            //           << " -> " << a << b << " with matrix element " << h_ijab
                            //           << std::endl;
                            new_dets.emplace_back(create_double_bb_excitation(det, i, j, a, b));
                        }
                    }
                }
            }
        }

        for (const auto& i : aocc_span) {
            for (const auto& j : bocc_span) {
                for (const auto& a : avir_span) {
                    for (const auto& b : bvir_span) {
                        double h_ijab = c * V(i, j, a, b) /
                                        (epsilon_[a] + epsilon_[b] - epsilon_[i] - epsilon_[j]);
                        checks_count++;
                        if (std::abs(h_ijab) > threshold) {
                            // std::cout << "  Selected double excitation (alpha-beta): " << i << "
                            // "
                            //          << j << " -> " << a << " " << b << " with matrix element "
                            //          << h_ijab << std::endl;
                            new_dets.emplace_back(create_double_ab_excitation(det, i, j, a, b));
                        }
                    }
                }
            }
        }
    }
    // add new determinants to the variational space
    dets_.insert(dets_.end(), new_dets.begin(), new_dets.end());
    // remove duplicates
    std::sort(dets_.begin(), dets_.end());
    // keep only unique determinants
    auto last = std::unique(dets_.begin(), dets_.end());
    // resize the vector to the new size
    dets_.erase(last, dets_.end());

    // resize the coefficient vector and set new coefficients to zero
    c_.resize(dets_.size(), 0.0);

    LOG(log_level_) << "Checked " << checks_count << " excitations for selection";
    LOG(log_level_) << "CIPSI selection completed in " << selection_timer.elapsed();
}

void SelectedCIHelper::select_hbci(double threshold) {

    local_timer selection_timer;

    std::vector<size_t> aocc(norb_);
    std::vector<size_t> bocc(norb_);
    std::vector<size_t> avir(norb_);
    std::vector<size_t> bvir(norb_);

    std::vector<Determinant> new_dets;
    size_t checks_count = 0;

    size_t noa, nob;
    for (size_t idx{0}, idx_max{dets_.size()}; idx < idx_max; ++idx) {
        const auto& det = dets_[idx];
        const double c = c_[idx];

        // std::cout << "Determinant in variational space: " << str(det, norb_) << std::endl;

        // Perform selection based on threshold
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
            for (const auto& a : avir_span) {
                double h_ia = c * h_[i * norb_ + a] / (epsilon_[a] - epsilon_[i]);
                checks_count++;
                if (std::abs(h_ia) > threshold) {
                    new_dets.emplace_back(create_single_a_excitation(det, i, a));
                }
            }
        }

        for (const auto& i : bocc_span) {
            for (const auto& a : bvir_span) {
                double h_ia = c * h_[i * norb_ + a] / (epsilon_[a] - epsilon_[i]);
                checks_count++;
                if (std::abs(h_ia) > threshold) {
                    new_dets.emplace_back(create_single_b_excitation(det, i, a));
                }
            }
        }

        for (const auto& i : aocc_span) {
            for (const auto& j : aocc_span) {
                if (i >= j)
                    continue;
                const auto& v_list = va_sorted_[i * norb_ + j];
                for (const auto& [v_ijab, a, b] : v_list) {
                    checks_count++;
                    if (std::abs(v_ijab * c) < threshold)
                        break;
                    if (det.na(a) or det.na(b))
                        continue;
                    new_dets.emplace_back(create_double_aa_excitation(det, i, j, a, b));
                }
            }
        }

        for (const auto& i : bocc_span) {
            for (const auto& j : bocc_span) {
                if (i >= j)
                    continue;
                const auto& v_list = va_sorted_[i * norb_ + j];
                for (const auto& [v_ijab, a, b] : v_list) {
                    checks_count++;
                    if (std::abs(v_ijab * c) < threshold)
                        break;
                    if (det.nb(a) or det.nb(b))
                        continue;
                    new_dets.emplace_back(create_double_bb_excitation(det, i, j, a, b));
                }
            }
        }

        for (const auto& i : aocc_span) {
            for (const auto& j : bocc_span) {
                const auto& v_list = v_sorted_[i * norb_ + j];
                for (const auto& [v_ijab, a, b] : v_list) {
                    checks_count++;
                    if (std::abs(v_ijab * c) < threshold)
                        break;
                    if (det.na(a) or det.nb(b))
                        continue;
                    new_dets.emplace_back(create_double_ab_excitation(det, i, j, a, b));
                }
            }
        }
    }
    LOG(log_level_) << "HBCI selection completed in " << selection_timer.elapsed();
    LOG(log_level_) << "Checked " << checks_count << " excitations for selection";

    merge_and_keep_unique(dets_, new_dets);
    c_.resize(dets_.size(), 0.0);
    prepare_sigma_build();
    for (size_t i = 0; i < dets_.size(); ++i) {
        LOG(log_level_) << std::to_string(i) << " " << str(dets_[i], norb_) << "  c = " << c_[i]
                        << " E = " << det_energies_[i];
    }

    LOG(log_level_) << "After HBCI selection, number of determinants: " << dets_.size();
}

} // namespace forte2
