#include <algorithm>

#include "helpers/indexing.hpp"

#include "rel_sci_helper.h"

namespace forte2 {

RelSelectedCIHelper::RelSelectedCIHelper(size_t norb, const std::vector<Determinant>& dets,
                                         np_matrix_complex& c, double E, np_matrix_complex& H,
                                         np_tensor4_complex& V, int log_level,
                                         const std::string& screening_criterion,
                                         const std::vector<size_t>& frozen_creation,
                                         const std::vector<size_t>& frozen_annihilation)
    : norb_(norb), norb2_(norb * norb), norb3_(norb * norb * norb), E_(E),
      slater_rules_(static_cast<int>(norb), E, H, V), c_guess_(c), dets_(dets) {
    log_level_ = log_level;
    if (dets.empty()) {
        throw std::runtime_error("The list of determinants cannot be empty.");
    }

    set_screening_criterion(screening_criterion);
    set_frozen_creation(frozen_creation);
    set_frozen_annihilation(frozen_annihilation);
    set_Hamiltonian(E, H, V);
    set_c(c);
    root_energies_.resize(nroots_, 0.0);
    ept2_var_.resize(nroots_, 0.0);
    ept2_pt_.resize(nroots_, 0.0);

    na_ = dets_[0].count_alpha();
    nb_ = dets_[0].count_beta();

    if (nb_ != 0) {
        throw std::runtime_error(
            "RelSelectedCIHelper requires all electrons in the alpha string (nb == 0).");
    }

    for (const auto& det : dets_) {
        if (det.count_alpha() != na_ || det.count_beta() != nb_) {
            throw std::runtime_error("All determinants must have the same number of electrons.");
        }
    }

    compute_det_energies();
    prepare_strings();
}

void RelSelectedCIHelper::set_Hamiltonian(double E, np_matrix_complex H, np_tensor4_complex V) {
    E_ = E;

    if (H.ndim() != 2) {
        throw std::runtime_error("H must be a 2D matrix.");
    }
    if (H.shape(0) != norb_ || H.shape(1) != norb_) {
        throw std::runtime_error("H shape does not match the number of orbitals.");
    }

    // Initialize the one-electron integrals epsilon (real) and h (complex).
    epsilon_.resize(norb_);
    h_.resize(norb_ * norb_);
    auto h = H.view();
    for (size_t p{0}; p < norb_; ++p) {
        // The diagonal of a Hermitian matrix is real; the PT2 denominator uses only epsilon.
        epsilon_[p] = h(p, p).real();
        for (size_t q{0}; q < norb_; ++q) {
            h_[p * norb_ + q] = h(p, q);
        }
    }

    // Initialize the two-electron integrals v_ = <pq|rs> and v_a_ = <pq||rs>.
    if (V.ndim() != 4) {
        throw std::runtime_error("V must be a 4D tensor.");
    }
    if (V.shape(0) != norb_ || V.shape(1) != norb_ || V.shape(2) != norb_ || V.shape(3) != norb_) {
        throw std::runtime_error("V shape does not match the number of orbitals.");
    }

    v_.resize(norb_ * norb_ * norb_ * norb_);
    v_a_.resize(norb_ * norb_ * norb_ * norb_);

    auto v = V.view();
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

void RelSelectedCIHelper::set_frozen_creation(const std::vector<size_t>& frozen_creation) {
    frozen_creation_mask_.clear();
    for (auto i : frozen_creation) {
        if (i >= norb_) {
            throw std::runtime_error("Frozen creation orbital index is out of range.");
        }
        frozen_creation_mask_.set_bit(i, true);
    }
}

void RelSelectedCIHelper::set_frozen_annihilation(const std::vector<size_t>& frozen_annihilation) {
    frozen_annihilation_mask_.clear();
    for (auto i : frozen_annihilation) {
        if (i >= norb_) {
            throw std::runtime_error("Frozen annihilation orbital index is out of range.");
        }
        frozen_annihilation_mask_.set_bit(i, true);
    }
}

void RelSelectedCIHelper::update_hbci_ints() {
    // Precompute, for each occupied pair (p, q), a list of (criterion_key, <pq||rs>, r, s) sorted
    // in descending order by the (real) key. Only the double alpha-alpha excitation class uses
    // these lists (the beta / alpha-beta lists of the real helper are not needed when nb == 0).
    va_sorted_.resize(norb_ * norb_);
    for (size_t p{0}; p < norb_; ++p) {
        for (size_t q{0}; q < norb_; ++q) {
            std::vector<std::tuple<double, std::complex<double>, u_int32_t, u_int32_t>> v_list;
            v_list.reserve(norb_ * norb_);
            for (size_t r{0}; r < norb_; ++r) {
                if (!creation_allowed(r))
                    continue;
                for (size_t s{0}; s < norb_; ++s) {
                    if (!creation_allowed(s))
                        continue;
                    const std::complex<double> v = Va(p, q, r, s);
                    // Plain HBCI uses |V| as the screening key. (eHBCI, which would fold in the
                    // energy denominator, is not supported in the two-component helper.)
                    const double key = std::abs(v);
                    if (key > integral_threshold)
                        v_list.emplace_back(key, v, r, s);
                }
            }
            // sort in descending order by the (real) criterion key
            std::sort(v_list.rbegin(), v_list.rend(), [](const auto& lhs, const auto& rhs) {
                return std::get<0>(lhs) < std::get<0>(rhs);
            });
            va_sorted_[p * norb_ + q] = std::move(v_list);
        }
    }
}

void RelSelectedCIHelper::set_c(np_matrix_complex& c) {
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

void RelSelectedCIHelper::set_energies(np_vector e) {
    if (e.shape(0) != nroots_) {
        throw std::runtime_error("The length of e must match the number of roots.");
    }
    root_energies_.resize(nroots_);
    for (size_t r{0}; r < nroots_; ++r) {
        root_energies_[r] = e(r);
    }
}

void RelSelectedCIHelper::set_screening_criterion(const std::string& criterion) {
    if (criterion == "hbci") {
        screening_criterion_ = ScreeningCriterion::HBCI;
    } else if (criterion == "ehbci") {
        throw std::runtime_error(
            "The eHBCI screening criterion is not supported for two-component selected CI.");
    } else {
        throw std::runtime_error("Unknown screening criterion: " + criterion +
                                 ". Only 'hbci' is supported for two-component selected CI.");
    }
}

void RelSelectedCIHelper::set_energy_correction(const std::string& correction) {
    if (correction == "variational") {
        energy_correction_ = EnergyCorrection::Variational;
    } else if (correction == "pt2") {
        energy_correction_ = EnergyCorrection::PT2;
    } else {
        throw std::runtime_error("Unknown energy correction method: " + correction +
                                 ". Supported methods are 'variational' and 'pt2'.");
    }
}

void RelSelectedCIHelper::set_pt2_regularizer(const std::string& regularizer, double strength) {
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

double RelSelectedCIHelper::compute_delta_ept2(double delta, double abs_v) const {
    if (energy_correction_ == EnergyCorrection::Variational) {
        return -0.5 * (delta + std::sqrt(delta * delta + 4.0 * abs_v * abs_v));
    } else if (energy_correction_ == EnergyCorrection::PT2) {
        if (pt2_regularizer_ == PT2Regularizer::Shift) {
            return abs_v * abs_v / (delta + pt2_regularizer_strength_);
        } else if (pt2_regularizer_ == PT2Regularizer::DSRG) {
            return abs_v * abs_v * regularized_denominator(delta, pt2_regularizer_strength_);
        } else {
            return abs_v * abs_v / delta;
        }
    }
    throw std::runtime_error("Unknown energy correction method");
}

std::complex<double> RelSelectedCIHelper::singles_coupling_a(size_t i, size_t a,
                                                             const Determinant& d) const {
    // <J|H|new_det> for the single excitation i -> a, matching RelSlaterRules::slater_rules for a
    // single connection: h(i,a) + sum_{j occ} <ij||aj>. The beta loop of the non-relativistic
    // SlaterRules::singles_coupling_a is empty here (nb == 0). The j == a term contributes
    // <ia||aa> = 0 by antisymmetry, so no exclusion of j is needed.
    std::complex<double> coupling = h(i, a);
    d.for_each_a_occ([&](size_t j) { coupling += Va(i, j, a, j); });
    return coupling;
}

np_vector RelSelectedCIHelper::Hdiag() const {
    auto Hdiag = make_zeros<nb::numpy, double, 1>({dets_.size()});
    auto Hdiag_view = Hdiag.view();
    for (size_t i{0}; i < dets_.size(); ++i) {
        Hdiag_view(i) = det_energies_[i];
    }
    return Hdiag;
}

} // namespace forte2
