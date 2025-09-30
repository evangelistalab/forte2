#include <iostream>
#include <iomanip>

// #include "helpers/timer.hpp"
// #include "helpers/np_vector_functions.h"
// #include "helpers/np_matrix_functions.h"
#include "helpers/indexing.hpp"
// #include "helpers/blas.h"
// #include "helpers/logger.h"

#include "sci_helper.h"

namespace forte2 {

SelectedCIHelper::SelectedCIHelper(size_t norb, const std::vector<Determinant>& dets, np_matrix& c,
                                   double E, np_matrix& H, np_tensor4& V, int log_level)
    : norb_(norb), norb2_(norb * norb), norb3_(norb * norb * norb), dets_(dets), c_guess_(c),
      log_level_(log_level), slater_rules_(norb, E, H, V) {
    set_Hamiltonian(E, H, V);
    set_c(c);

    // LOG(log_level_) << "\nAllocating CI temporary buffers of size 2 x " << max_size << " ("
    //                 << 2 * max_size * sizeof(double) / (1024 * 1024) << " MB).\n";
}

void SelectedCIHelper::set_Hamiltonian(double E, np_matrix H, np_tensor4 V) {
    const auto norb = norb_;

    E_ = E;

    if (H.ndim() != 2) {
        throw std::runtime_error("H must be a 2D matrix.");
    }
    if (H.shape(0) != norb || H.shape(1) != norb) {
        throw std::runtime_error("H shape does not match the number of orbitals.");
    }
    H_ = H;

    // Initialize the one-electron integrals epsilon and h
    epsilon_.resize(norb);
    h_.resize(norb * norb);
    auto h = H.view();
    for (size_t p{0}; p < norb; ++p) {
        epsilon_[p] = h(p, p);
        for (size_t q{0}; q < norb; ++q) {
            h_[p * norb + q] = h(p, q);
        }
    }

    // Initialize the two-electron integrals v_pr_qs and v_pr_qs_a
    if (V.ndim() != 4) {
        throw std::runtime_error("V must be a 4D tensor.");
    }
    if (V.shape(0) != norb || V.shape(1) != norb || V.shape(2) != norb || V.shape(3) != norb) {
        throw std::runtime_error("V shape does not match the number of orbitals.");
    }
    V_ = V;

    v_.resize(norb * norb * norb * norb);   // V[p][q][r][s] = <pq|rs>
    v_a_.resize(norb * norb * norb * norb); // V_a[p][q][r][s] = <pq||rs> = <pq|rs> - <pq|sr>

    auto v = V.view();
    // Loop over all pairs (p, r) and (q, s) to fill v_
    for (size_t p{0}, pqrs{0}; p < norb; ++p) {
        for (size_t q{0}; q < norb; ++q) {
            for (size_t r{0}; r < norb; ++r) {
                for (size_t s{0}; s < norb; ++s, ++pqrs) {
                    v_[pqrs] = v(p, q, r, s);
                    v_a_[pqrs] = v(p, q, r, s) - v(p, q, s, r);
                }
            }
        }
    }

    // Precompute sorted lists of two-electron integrals for each (p, q) pair
    // (p,q) -> [|<pq|rs>|^2/(ep+eq-er-es)|, r, s), ...] sorted in descending order
    v_sorted_.resize(norb * norb);
    for (size_t p{0}; p < norb; ++p) {
        for (size_t q{0}; q < norb; ++q) {
            std::vector<std::tuple<double, size_t, size_t>> v_list;
            v_list.reserve(norb * norb);
            for (size_t r{0}; r < norb; ++r) {
                for (size_t s{0}; s < norb; ++s) {
                    const double den = epsilon_[p] + epsilon_[q] - epsilon_[r] - epsilon_[s];
                    const double val = std::fabs(std::pow(V(p, q, r, s), 2.0) / den);
                    v_list.emplace_back(val, r, s);
                }
            }
            // Sort in descending order by absolute value of the integral
            std::sort(v_list.rbegin(), v_list.rend());

            v_sorted_[p * norb_ + q] = std::move(v_list);
        }
    }

    // Precompute sorted lists of two-electron integrals for each (p, q) pair
    // (p,q) -> [|<pq||rs>|^2/(ep+eq-er-es)|, r, s), ...] sorted in descending order
    va_sorted_.resize(norb * norb);
    for (size_t p{0}; p < norb; ++p) {
        for (size_t q{0}; q < norb; ++q) {
            std::vector<std::tuple<double, size_t, size_t>> v_list;
            v_list.reserve(norb * norb);
            for (size_t r{0}; r < norb; ++r) {
                for (size_t s{0}; s < norb; ++s) {
                    const double den = epsilon_[p] + epsilon_[q] - epsilon_[r] - epsilon_[s];
                    const double val = std::fabs(std::pow(Va(p, q, r, s), 2.0) / den);
                    v_list.emplace_back(val, r, s);
                }
            }
            // Sort in descending order by absolute value of the integral
            std::sort(v_list.rbegin(), v_list.rend());

            va_sorted_[p * norb_ + q] = std::move(v_list);
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

np_vector SelectedCIHelper::Hdiag() const {
    auto Hdiag = make_zeros<nb::numpy, double, 1>({dets_.size()});
    auto Hdiag_view = Hdiag.view();
    for (size_t i{0}; i < dets_.size(); ++i) {
        Hdiag_view(i) = det_energies_[i];
    }
    return Hdiag;
}
} // namespace forte2
