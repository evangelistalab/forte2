#include <algorithm>
#include <iostream>
#include <iomanip>

#include "helpers/indexing.hpp"

#include "sci_helper.h"

namespace forte2 {

SelectedCIHelper::SelectedCIHelper(size_t norb, const std::vector<Determinant>& dets, np_matrix& c,
                                   double E, np_matrix& H, np_tensor4& V, int log_level,
                                   const std::string& screening_criterion,
                                   const std::vector<size_t>& gas_size,
                                   const std::vector<int>& gas_min, const std::vector<int>& gas_max)
    : norb_(norb), norb2_(norb * norb), norb3_(norb * norb * norb), dets_(dets), c_guess_(c),
      log_level_(log_level), slater_rules_(norb, E, H, V) {
    if (dets.empty()) {
        throw std::runtime_error("The list of determinants cannot be empty.");
    }

    set_screening_criterion(screening_criterion);
    set_gas_limits(gas_size, gas_min, gas_max);
    set_Hamiltonian(E, H, V);
    set_c(c);
    root_energies_.resize(nroots_, 0.0);
    ept2_var_.resize(nroots_, 0.0);
    ept2_pt_.resize(nroots_, 0.0);
    ept2_gas_.resize(nroots_, 0.0);

    na_ = dets_[0].count_alpha();
    nb_ = dets_[0].count_beta();

    for (const auto& det : dets_) {
        if (det.count_alpha() != na_ || det.count_beta() != nb_) {
            throw std::runtime_error("All determinants must have the same number of electrons.");
        }
    }

    compute_det_energies();
    prepare_strings();
}

void SelectedCIHelper::set_Hamiltonian(std::optional<double> E, std::optional<np_matrix> H,
                                       std::optional<np_tensor4> V) {
    if (E) {
        E_ = *E;
    }

    if (H) {
        if (H->ndim() != 2) {
            throw std::runtime_error("H must be a 2D matrix.");
        }
        if (H->shape(0) != norb_ || H->shape(1) != norb_) {
            throw std::runtime_error("H shape does not match the number of orbitals.");
        }
        // Initialize the one-electron integrals epsilon and h
        epsilon_.resize(norb_);
        h_.resize(norb_ * norb_);
        auto h = H->view();
        for (size_t p{0}; p < norb_; ++p) {
            epsilon_[p] = h(p, p);
            for (size_t q{0}; q < norb_; ++q) {
                h_[p * norb_ + q] = h(p, q);
            }
        }
    }

    if (V) {
        if (V->ndim() != 4) {
            throw std::runtime_error("V must be a 4D tensor.");
        }
        if (V->shape(0) != norb_ || V->shape(1) != norb_ || V->shape(2) != norb_ ||
            V->shape(3) != norb_) {
            throw std::runtime_error("V shape does not match the number of orbitals.");
        }
        v_.resize(norb_ * norb_ * norb_ * norb_); // V[p][q][r][s] = <pq|rs>
        v_a_.resize(norb_ * norb_ * norb_ *
                    norb_); // V_a[p][q][r][s] = <pq||rs> = <pq|rs> - <pq|sr>

        auto v = V->view();
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
    }

    // va_sorted_/vab_sorted_ mix epsilon_ (H-derived) and v_/v_a_ (V-derived), so they must be
    // rebuilt whenever either H or V changes.
    if (H || V) {
        update_hbci_ints();
    }

    // slater_rules_ is a separate view of the same integrals (used by compute_det_energies())
    // and is not touched by the code above.
    if (E || H || V) {
        slater_rules_.update_integrals(static_cast<int>(norb_), E, H, V);
        // det_energies_ is an incremental cache: compute_det_energies() only fills entries past
        // its current size, so every determinant already in the variational space would keep an
        // energy computed under the previous Hamiltonian unless the cache is cleared and eagerly
        // rebuilt here (eagerly, so this object is never left with a stale/empty cache regardless
        // of what the caller does next).
        det_energies_.clear();
        compute_det_energies();
    }
}

void SelectedCIHelper::set_gas_limits(const std::vector<size_t>& gas_size,
                                      const std::vector<int>& gas_min,
                                      const std::vector<int>& gas_max) {
    size_t first = 0;
    for (size_t g{0}; g < gas_size.size(); ++g) {
        const size_t capacity = 2 * gas_size[g];
        const size_t nmin = g < gas_min.size() ? std::max(gas_min[g], 0) : 0;
        const size_t nmax = g < gas_max.size() ? std::max(gas_max[g], 0) : capacity;
        // keep only the spaces whose limits exclude some occupation
        if (nmin > 0 || nmax < capacity) {
            Determinant mask = Determinant::zero();
            for (size_t i{first}; i < first + gas_size[g]; ++i) {
                mask.set_na(i, true);
                mask.set_nb(i, true);
            }
            gas_masks_.push_back(mask);
            gas_min_.push_back(nmin);
            gas_max_.push_back(nmax);
        }
        first += gas_size[g];
    }
    if (first != 0 && first != norb_) {
        throw std::runtime_error("The GAS sizes must add up to the number of orbitals.");
    }
}

double evaluate_criterion(double delta, double v, ScreeningCriterion criterion) {
    if (criterion == ScreeningCriterion::eHBCI) {
        return v * v / (std::fabs(delta) + 1e-3);
    }
    return std::fabs(v);
}

void SelectedCIHelper::update_hbci_ints() {
    // Precompute sorted lists of two-electron integrals for each (p, q) pair with p < q
    // (p,q) -> [|<pq||rs>^2/(ep+eq-er-es)|, r, s), ...] with r < s, sorted in descending order.
    // <pq||rs> is antisymmetric in r and s, and the same-spin doubles use only p < q and r < s.
    va_sorted_.resize(norb_ * norb_);
    for (size_t p{0}; p < norb_; ++p) {
        for (size_t q{p + 1}; q < norb_; ++q) {
            std::vector<std::tuple<double, double, u_int32_t, u_int32_t>> v_list;
            v_list.reserve(norb_ * (norb_ - 1) / 2);
            for (size_t r{0}; r < norb_; ++r) {
                for (size_t s{r + 1}; s < norb_; ++s) {
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
                for (size_t s{0}; s < norb_; ++s) {
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
