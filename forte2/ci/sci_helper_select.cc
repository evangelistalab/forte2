
#include "determinant_helpers.h"
#include "sci_helper.h"

namespace forte2 {

void SelectedCIHelper::select_cipsi(double threshold) {
    // Placeholder for CIPSI selection algorithm
    // std::cout << "CIPSI selection with threshold: " << threshold << std::endl;

    std::vector<size_t> aocc(norb_);
    std::vector<size_t> bocc(norb_);
    std::vector<size_t> avir(norb_);
    std::vector<size_t> bvir(norb_);

    std::vector<Determinant> new_dets;

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
                        double h_ijab =
                            c *
                            v_a_[i * norb_ * norb_ * norb_ + j * norb_ * norb_ + a * norb_ + b] /
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
                        double h_ijab =
                            c *
                            v_a_[i * norb_ * norb_ * norb_ + j * norb_ * norb_ + a * norb_ + b] /
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
                        double h_ijab =
                            c * v_[i * norb_ * norb_ * norb_ + j * norb_ * norb_ + a * norb_ + b] /
                            (epsilon_[a] + epsilon_[b] - epsilon_[i] - epsilon_[j]);
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
}

} // namespace forte2
