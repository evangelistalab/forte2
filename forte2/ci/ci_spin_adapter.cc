#include <set>
#include <cmath>
#include <cassert>
#include <iostream>
#include <iomanip>

#include "helpers/timer.hpp"
#include "helpers/np_vector_functions.h"

#include "ci_spin_adapter.h"

namespace forte2 {

// Utility functions

/// @brief A function to compute the Clebsch-Gordan coefficient
/// @param twoS Twice the value of S
/// @param twoM Twice the value of M
/// @param dtwoS Twice the change in S
/// @param dtwoM Twice the change in M
double ClebschGordan(double twoS, double twoM, int dtwoS, double dtwoM) {
    if (dtwoS == 1)
        return std::sqrt(0.5 * (twoS + dtwoM * twoM) / twoS);
    if (dtwoS == -1)
        return -dtwoM * std::sqrt(0.5 * (twoS + 2. - dtwoM * twoM) / (twoS + 2.));
    return 0.0;
}

/// @brief A function to compute the overlap between a determinant and a CSF
/// @param N The number of unpaired electrons
/// @param spin_coupling The spin coupling of the CSF (up = 0, down = 1)
/// @param det_occ The spin occupation of the determinant (up = alpha = 0, down = beta = 1)
double overlap(int N, const String& spin_coupling, const String& det_occ) {
    double overlap = 1.0;
    int pi = 0;
    int qi = 0;
    for (int i = 0; i < N; i++) {
        const int dpi = spin_coupling[i];
        const int dqi = det_occ[i];
        int dtwoS = 1 - 2 * dpi;
        int dtwoM = 1 - 2 * dqi;
        pi += dtwoS;
        qi += dtwoM;
        if (std::abs(qi) > pi)
            return 0.0;
        overlap *= ClebschGordan(pi, qi, dtwoS, dtwoM);
    }
    return overlap;
}

// CISpinAdapter class

CISpinAdapter::CISpinAdapter(int twoS, int twoMs, int norb)
    : twoS_(twoS), twoMs_(twoMs), norb_(norb), N_ncsf_(norb + 1, 0),
      N_to_det_occupations_(norb + 1), N_to_overlaps_(norb + 1), N_to_noverlaps_(norb + 1) {}

size_t CISpinAdapter::ncsf() const { return ncsf_; }

size_t CISpinAdapter::ndet() const { return ndet_; }

void CISpinAdapter::det_C_to_csf_C(np_vector det_C, np_vector csf_C) {
    vector::zero(csf_C);
    // loop over all the elements of csf_to_det_coeff_ and add the contribution to csf_C
    for (size_t i{0}; i < ncsf_; i++) {
        const auto& start = csf_to_det_bounds_[i];
        const auto& end = csf_to_det_bounds_[i + 1];
        for (size_t j{start}; j < end; j++) {
            const auto& [det_idx, coeff] = csf_to_det_coeff_[j];
            csf_C(i) += coeff * det_C(det_idx);
        }
    }
}

void CISpinAdapter::csf_C_to_det_C(np_vector csf_C, np_vector det_C) {
    vector::zero(det_C);

    // loop over all the elements of csf_to_det_coeff_ and add the contribution to det_C
    for (size_t i = 0; i < ncsf_; i++) {
        const auto& start = csf_to_det_bounds_[i];
        const auto& end = csf_to_det_bounds_[i + 1];
        for (size_t j = start; j < end; j++) {
            const auto& [det_idx, coeff] = csf_to_det_coeff_[j];
            det_C(det_idx) += coeff * csf_C(i);
        }
    }
}

auto CISpinAdapter::compute_unique_couplings() {
    // compute the number of couplings and CSFs for each allowed value of N
    size_t ncoupling = 0;
    size_t ncsf = 0;
    for (size_t N = 0; N < N_ncsf_.size(); N++) {
        if (N_ncsf_[N] > 0) {
            const auto spin_couplings = make_spin_couplings(N, twoS_);
            const auto determinant_occ = make_determinant_occupations(N, twoMs_);

            std::vector<std::tuple<size_t, size_t, double>> overlaps;
            std::vector<size_t> noverlaps_;

            size_t ncoupling_N = 0;
            size_t ncsf_N = 0;
            for (const auto& spin_coupling : spin_couplings) {
                size_t ndet_N = 0;
                size_t nonzero_overlap = 0;
                for (const auto& det_occ : determinant_occ) {
                    auto o = overlap(N, spin_coupling, det_occ);
                    if (std::fabs(o) > 0.0) {
                        overlaps.push_back(std::make_tuple(ncsf_N, ndet_N, o));
                        nonzero_overlap++;
                    }
                    ndet_N++;
                }
                ncoupling_N += nonzero_overlap;
                noverlaps_.push_back(nonzero_overlap);
                ncsf_N++;
            }
            // save the spin couplings and the determinant occupations
            N_to_det_occupations_[N] = determinant_occ;
            N_to_overlaps_[N] = overlaps;
            N_to_noverlaps_[N] = noverlaps_;
            ncoupling += ncoupling_N * N_ncsf_[N];
            ncsf += ncsf_N * N_ncsf_[N];
        }
    }
    return std::pair(ncoupling, ncsf);
}

void CISpinAdapter::prepare_couplings(const std::vector<Determinant>& dets) {
    ndet_ = dets.size();
    // build the address of each determinant
    det_hash<size_t> det_hash;
    for (size_t i = 0; i < ndet_; i++) {
        det_hash[dets[i]] = i;
    }

    // find all the configurations
    local_timer t1;
    std::set<Configuration> confs;
    for (const auto& d : dets) {
        confs.insert(Configuration(d));
    }

    // count the configurations with the same number of unpaired electrons (N)
    for (const auto& conf : confs) {
        // exclude configurations with more unpaired electrons than twoS
        if (const auto N = conf.count_socc(); N >= twoS_)
            N_ncsf_[N]++;
    }

    // compute the number of couplings and CSFs for each allowed value of N
    const auto [ncoupling, ncsf] = compute_unique_couplings();

    // allocate memory for the couplings and the starting index of each CSF
    csf_to_det_coeff_.resize(ncoupling);
    csf_to_det_bounds_.resize(ncsf + 1);

    confs_ = std::vector<Configuration>(confs.begin(), confs.end());
    // std::cout << "Number of configuration state functions: " << ncsf << std::endl;
    // std::cout << "Number of couplings: " << ncoupling << std::endl;
    // std::cout << "Timing for identifying configurations: " << t1.elapsed_seconds() << std::endl;

    // loop over all the configurations and find the CSFs
    local_timer t2;
    ncsf_ = 0;
    ncoupling_ = 0;
    for (const auto& conf : confs_) {
        if (conf.count_socc() >= twoS_) {
            conf_to_csfs(conf, det_hash);
        }
    }
    // std::cout << "Timing for finding the CSFs: " << t2.elapsed_seconds() << std::endl;

    // check that the number of couplings and CSFs is correct
    assert(ncsf_ == ncsf);
    assert(ncoupling_ == ncoupling);
}

void CISpinAdapter::conf_to_csfs(const Configuration& conf, det_hash<size_t>& det_hash) {
    // number of unpaired electrons
    const auto N = conf.count_socc();
    String docc = conf.get_docc_str();
    std::vector<int> socc_vec(norb_);
    conf.get_socc_vec(norb_, socc_vec);

    const auto& determinant_occ = N_to_det_occupations_[N];
    const auto& noverlaps = N_to_noverlaps_[N];

    csf_to_det_bounds_[0] = 0;
    Determinant det;

    size_t temp = ncoupling_;
    for (const auto& [i, j, o] : N_to_overlaps_[N]) {
        const auto& det_occ = determinant_occ[j];
        det.set_str(docc, docc);
        // keep track of the sign of the singly occupied orbitals
        double sign = 1.0;
        for (int k = N - 1; k >= 0; k--) {
            if (det_occ.get_bit(k)) {
                sign *= det.create_beta_bit(socc_vec[k]);
            } else {
                sign *= det.create_alfa_bit(socc_vec[k]);
            }
        }
        csf_to_det_coeff_[ncoupling_].first = det_hash[det];
        csf_to_det_coeff_[ncoupling_].second = sign * o;
        ncoupling_ += 1;
    }
    for (const auto& n : noverlaps) {
        temp += n;
        ncsf_ += 1;
        csf_to_det_bounds_[ncsf_] = temp;
    }
}

auto CISpinAdapter::make_spin_couplings(int N, int twoS) -> std::vector<String> {
    if (N == 0)
        return std::vector<String>(1, String());
    std::vector<String> couplings;
    auto nup = (N + twoS) / 2;
    String coupling;
    // up = false = 0, down = true = 1
    // The coupling should always start with up
    for (int i = 0; i < nup; i++)
        coupling[i] = false;
    for (int i = nup; i < N; i++)
        coupling[i] = true;
    /// Generate all permutations of the path
    do {
        // check if the path is valid (no negative spin)
        bool valid = true;
        for (int i = 0, p = 0; i < N; i++) {
            p += 1 - 2 * coupling[i];
            if (p < 0)
                valid = false;
        }
        if (valid)
            couplings.push_back(coupling);
        // to keep the first coupling as up we only permute starting from the second element
    } while (std::next_permutation(coupling.begin() + 1, coupling.begin() + N));

    return couplings;
}

auto CISpinAdapter::make_determinant_occupations(int N, int twoMs) -> std::vector<String> {
    std::vector<String> det_occs;
    if (N == 0)
        return std::vector<String>(1, String());
    auto nup = (N + twoMs) / 2;
    String det_occ;
    // true = 1 = up, false = 0 = down
    // The det_occ should always start with up
    for (int i = 0; i < nup; i++)
        det_occ[i] = false;
    for (int i = nup; i < N; i++)
        det_occ[i] = true;
    /// Generate all permutations of the path
    do {
        det_occs.push_back(det_occ);
    } while (std::next_permutation(det_occ.begin(), det_occ.begin() + N));
    return det_occs;
}

} // namespace forte2
