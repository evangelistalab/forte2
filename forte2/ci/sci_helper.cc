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
      log_level_(log_level) {
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

    const size_t npairs = (norb * (norb - 1)) / 2; // Number of pairs (p, r) with p > r
    v_.resize(norb * norb * norb * norb);          // Full storage for V[p][q][r][s]
    v_a_.resize(npairs * npairs); // Antisymmetrized storage for V[p][q][r][s] with p > r and q > s

    auto v = V.view();
    // Loop over all pairs (p, r) and (q, s) to fill v_
    for (size_t p{0}, pqrs{0}; p < norb; ++p) {
        for (size_t q{0}; q < norb; ++q) {
            for (size_t r{0}; r < norb; ++r) {
                for (size_t s{0}; s < norb; ++s, ++pqrs) {
                    v_[pqrs] = v(p, r, q, s);
                }
            }
        }
    }
    // Loop over all pairs (p, r) and (q, s) to fill v_a_ with p > r and q > s
    for (int p = 1; p < norb; ++p) {
        for (int r = 0; r < p; ++r) {
            const auto pr_index = (p * (p - 1)) / 2 + r;
            for (int q = 1; q < norb; ++q) {
                for (int s = 0; s < q; ++s) {
                    const auto qs_index = pair_index_gt(q, s);
                    v_a_[pr_index * npairs + qs_index] = v(p, r, q, s) - v(p, r, s, q);
                }
            }
        }
    }
}

void SelectedCIHelper::set_c(np_matrix& c) {
    nroots_ = c.shape(1);
    if (c.shape(0) != dets_.size()) {
        throw std::runtime_error("The number of rows in c must match the number of determinants.");
    }
    c_.resize(dets_.size() * nroots_);
    auto c_view = c.view();
    for (size_t i{0}; i < dets_.size(); ++i) {
        for (size_t r{0}; r < nroots_; ++r) {
            c_[i * nroots_ + r] = c_view(i, r);
        }
    }
}

} // namespace forte2
