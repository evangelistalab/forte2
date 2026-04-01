#include <algorithm>
#include <iostream>
#include <iomanip>

#include "helpers/indexing.hpp"

#include "sci_helper.h"

namespace forte2 {

SelectedCIHelper::SelectedCIHelper(size_t norb, const std::vector<Determinant>& dets, np_matrix& c,
                                   double E, np_matrix& H, np_tensor4& V, int log_level)
    : norb_(norb), norb2_(norb * norb), norb3_(norb * norb * norb), dets_(dets), c_guess_(c),
      log_level_(log_level), slater_rules_(norb, E, H, V) {
    if (dets.empty()) {
        throw std::runtime_error("The list of determinants cannot be empty.");
    }

    set_Hamiltonian(E, H, V);
    set_c(c);
    root_energies_.resize(nroots_, 0.0);
    ept2_var_.resize(nroots_, 0.0);
    ept2_pt_.resize(nroots_, 0.0);

    na_ = dets_[0].count_a();
    nb_ = dets_[0].count_b();

    for (const auto& det : dets_) {
        if (det.count_a() != na_ || det.count_b() != nb_) {
            throw std::runtime_error("All determinants must have the same number of electrons.");
        }
    }

    compute_det_energies();
    prepare_strings();
}

void SelectedCIHelper::set_Hamiltonian(double E, np_matrix H, np_tensor4 V) {
    E_ = E;

    if (H.ndim() != 2) {
        throw std::runtime_error("H must be a 2D matrix.");
    }
    if (H.shape(0) != norb_ || H.shape(1) != norb_) {
        throw std::runtime_error("H shape does not match the number of orbitals.");
    }
    H_ = H;

    // Initialize the one-electron integrals epsilon and h
    epsilon_.resize(norb_);
    h_.resize(norb_ * norb_);
    auto h = H.view();
    for (size_t p{0}; p < norb_; ++p) {
        epsilon_[p] = h(p, p);
        for (size_t q{0}; q < norb_; ++q) {
            h_[p * norb_ + q] = h(p, q);
        }
    }

    // Initialize the two-electron integrals v_pr_qs and v_pr_qs_a
    if (V.ndim() != 4) {
        throw std::runtime_error("V must be a 4D tensor.");
    }
    if (V.shape(0) != norb_ || V.shape(1) != norb_ || V.shape(2) != norb_ || V.shape(3) != norb_) {
        throw std::runtime_error("V shape does not match the number of orbitals.");
    }
    V_ = V;

    v_.resize(norb_ * norb_ * norb_ * norb_);   // V[p][q][r][s] = <pq|rs>
    v_a_.resize(norb_ * norb_ * norb_ * norb_); // V_a[p][q][r][s] = <pq||rs> = <pq|rs> - <pq|sr>

    auto v = V.view();
    // Loop over all pairs (p, r) and (q, s) to fill v_
    for (size_t p{0}, pqrs{0}; p < norb_; ++p) {
        for (size_t q{0}; q < norb_; ++q) {
            for (size_t r{0}; r < norb_; ++r) {
                for (size_t s{0}; s < norb_; ++s, ++pqrs) {
                    v_[pqrs] = v(p, q, r, s);
                    v_a_[pqrs] = v(p, q, r, s) - v(p, q, s, r);
                }
            }
        }
    }

    update_hbci_ints();
}

void SelectedCIHelper::set_frozen_creation(const std::vector<size_t>& frozen_creation) {
    frozen_creation_mask_.clear();
    for (auto i : frozen_creation) {
        if (i >= norb_) {
            throw std::runtime_error("Frozen creation orbital index is out of range.");
        }
        frozen_creation_mask_.set_bit(i, true);
    }
}

void SelectedCIHelper::set_frozen_annihilation(const std::vector<size_t>& frozen_annihilation) {
    frozen_annihilation_mask_.clear();
    for (auto i : frozen_annihilation) {
        if (i >= norb_) {
            throw std::runtime_error("Frozen annihilation orbital index is out of range.");
        }
        frozen_annihilation_mask_.set_bit(i, true);
    }
}

double evaluate_criterion(double delta, double v, ScreeningCriterion criterion) {
    if (criterion == ScreeningCriterion::eHBCI) {
        return v * v / (std::fabs(delta) + 1e-3);
    }
    return std::fabs(v);
}

void SelectedCIHelper::update_hbci_ints() {
    // Precompute sorted lists of two-electron integrals for each (p, q) pair
    // (p,q) -> [|<pq|rs>^2/(ep+eq-er-es)|, r, s), ...] sorted in descending order
    v_sorted_.resize(norb_ * norb_);
    for (size_t p{0}; p < norb_; ++p) {
        for (size_t q{0}; q < norb_; ++q) {
            std::vector<std::tuple<double, double, u_int32_t, u_int32_t>> v_list;
            v_list.reserve(norb_ * norb_);
            for (size_t r{0}; r < norb_; ++r) {
                for (size_t s{0}; s < norb_; ++s) {
                    const double delta = epsilon_[p] + epsilon_[q] - epsilon_[r] - epsilon_[s];
                    const double v = V(p, q, r, s);
                    const double val = evaluate_criterion(delta, v, screening_criterion_);
                    if (std::fabs(val) > integral_threshold)
                        v_list.emplace_back(val, v, r, s);
                }
            }
            // sort in descending order by absolute value of the integral
            std::sort(v_list.rbegin(), v_list.rend());
            v_sorted_[p * norb_ + q] = std::move(v_list);
        }
    }

    // Precompute sorted lists of two-electron integrals for each (p, q) pair
    // (p,q) -> [|<pq||rs>^2/(ep+eq-er-es)|, r, s), ...] sorted in descending order
    va_sorted_.resize(norb_ * norb_);
    for (size_t p{0}; p < norb_; ++p) {
        for (size_t q{0}; q < norb_; ++q) {
            std::vector<std::tuple<double, double, u_int32_t, u_int32_t>> v_list;
            v_list.reserve(norb_ * norb_);
            for (size_t r{0}; r < norb_; ++r) {
                if (!creation_allowed(r))
                    continue;
                for (size_t s{0}; s < norb_; ++s) {
                    if (!creation_allowed(s))
                        continue;
                    const double delta = epsilon_[p] + epsilon_[q] - epsilon_[r] - epsilon_[s];
                    const double v = Va(p, q, r, s);
                    const double val = evaluate_criterion(delta, v, screening_criterion_);
                    if (std::fabs(val) > integral_threshold)
                        v_list.emplace_back(val, v, r, s);
                }
            }
            // sort in descending order by absolute value of the integral
            std::sort(v_list.rbegin(), v_list.rend());
            va_sorted_[p * norb_ + q] = std::move(v_list);
        }
    }

    // Precompute sorted lists of two-electron integrals for each (p, q) pair
    // (p,q) -> [|<pq|rs>^2/(ep+eq-er-es)|, r, s), ...] sorted in descending order
    vab_sorted_.resize(norb_ * norb_);
    for (size_t p{0}; p < norb_; ++p) {
        for (size_t r{0}; r < norb_; ++r) {
            std::vector<std::tuple<double, double, u_int32_t, u_int32_t>> v_list;
            v_list.reserve(norb_ * norb_);
            for (size_t q{0}; q < norb_; ++q) {
                if (!annihilation_allowed(q))
                    continue;
                for (size_t s{0}; s < norb_; ++s) {
                    if (!creation_allowed(s))
                        continue;
                    const double delta = epsilon_[p] + epsilon_[q] - epsilon_[r] - epsilon_[s];
                    const double v = V(p, q, r, s);
                    const double val = evaluate_criterion(delta, v, screening_criterion_);
                    if (std::fabs(val) > integral_threshold)
                        v_list.emplace_back(val, v, q, s);
                }
            }
            // sort in descending order by absolute value of the integral
            std::sort(v_list.rbegin(), v_list.rend());
            vab_sorted_[p * norb_ + r] = std::move(v_list);
        }
    }
}

void SelectedCIHelper::set_c(np_matrix& c) {
    nroots_ = c.shape(1);
    if (c.shape(0) != dets_.size()) {
        throw std::runtime_error("The number of rows in c must match the number of determinants.");
    }
    auto c_view = c.view();
    c_.resize(dets_.size() * nroots_);
    for (size_t i{0}; i < dets_.size(); ++i) {
        for (size_t r{0}; r < nroots_; ++r) {
            c_[i * nroots_ + r] = c_view(i, r);
        }
    }
}

void SelectedCIHelper::set_energies(np_vector e) {
    if (e.shape(0) != nroots_) {
        throw std::runtime_error("The length of e must match the number of roots.");
    }
    root_energies_.resize(nroots_);
    for (size_t r{0}; r < nroots_; ++r) {
        root_energies_[r] = e(r);
    }
}

void SelectedCIHelper::set_screening_criterion(const std::string& criterion) {
    if (criterion == "ehbci") {
        screening_criterion_ = ScreeningCriterion::eHBCI;
    } else if (criterion == "hbci") {
        screening_criterion_ = ScreeningCriterion::HBCI;
    } else {
        throw std::runtime_error("Unknown screening criterion: " + criterion +
                                 ". Supported criteria are 'hbci' and 'ehbci'.");
    }
}

void SelectedCIHelper::set_energy_correction(const std::string& correction) {
    if (correction == "variational") {
        energy_correction_ = EnergyCorrection::Variational;
    } else if (correction == "pt2") {
        energy_correction_ = EnergyCorrection::PT2;
    } else {
        throw std::runtime_error("Unknown energy correction method: " + correction +
                                 ". Supported methods are 'variational' and 'pt2'.");
    }
}

void SelectedCIHelper::set_pt2_regularizer(const std::string& regularizer, double strength) {
    if (regularizer == "none") {
        pt2_regularizer_ = PT2Regularizer::None;
        pt2_regularizer_strength_ = 0.0;
    } else if (regularizer == "shift") {
        pt2_regularizer_ = PT2Regularizer::Shift;
        pt2_regularizer_strength_ = strength;
    } else if (regularizer == "dsrg") {
        pt2_regularizer_ = PT2Regularizer::DSRG;
        pt2_regularizer_strength_ = strength;
    } else {
        throw std::runtime_error("Unknown PT2 regularization method: " + regularizer +
                                 ". Supported methods are 'none', 'shift', and 'dsrg'.");
    }
}

np_vector SelectedCIHelper::Hdiag() const {
    auto Hdiag = make_zeros<nb::numpy, double, 1>({dets_.size()});
    auto Hdiag_view = Hdiag.view();
    for (size_t i{0}; i < dets_.size(); ++i) {
        Hdiag_view(i) = det_energies_[i];
    }
    return Hdiag;
}
} // namespace forte2
