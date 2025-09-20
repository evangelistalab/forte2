#include <iostream>

#include "helpers/timer.hpp"
#include "helpers/np_matrix_functions.h"
#include "helpers/np_vector_functions.h"
#include "helpers/indexing.hpp"
#include "helpers/blas.h"

#include "rel_ci_sigma_builder.h"
#include "ci_sigma_builder.h"
#include "sparse/sparse_state.h"

namespace forte2 {

SparseState get_sparse_state(np_vector_complex c, const CIStrings& lists) {
    auto c_span = vector::as_span<std::complex<double>>(c);
    SparseState state;
    auto dets = lists.make_determinants();
    for (size_t i{0}, maxi{dets.size()}; i < maxi; ++i) {
        state[dets[i]] = c_span[i];
    }
    return state;
}

np_matrix_complex RelCISigmaBuilder::compute_1rdm_debug(np_vector_complex C_left,
                                                        np_vector_complex C_right) const {
    const auto norb = lists_.norb();
    auto rdm = make_zeros<nb::numpy, std::complex<double>, 2>({norb, norb});
    auto rdm_v = rdm.view();

    auto state_vector_l = get_sparse_state(C_left, lists_);
    auto state_vector_r = get_sparse_state(C_right, lists_);

    Determinant J;

    for (size_t p{0}; p < norb; p++) {
        for (size_t q{0}; q < norb; ++q) {
            std::complex<double> element = 0.0;
            for (const auto& [I, c_I] : state_vector_r) {
                J = I;
                double sign = 1.0;
                sign *= J.destroy_a(q);
                sign *= J.create_a(p);
                if (sign != 0) {
                    if (state_vector_l.count(J) != 0) {
                        element += static_cast<std::complex<double>>(sign) *
                                   std::conj(state_vector_l[J]) * c_I;
                    }
                }
            }
            rdm_v(p, q) = element;
        }
    }
    return rdm;
}

np_tensor4_complex RelCISigmaBuilder::compute_2rdm_debug(np_vector_complex C_left,
                                                         np_vector_complex C_right) const {
    const size_t norb = lists_.norb();
    const size_t npairs = (norb * (norb - 1)) / 2;
    auto rdm = make_zeros<nb::numpy, std::complex<double>, 2>({npairs, npairs});
    auto rdm_v = rdm.view();

    auto state_vector_l = get_sparse_state(C_left, lists_);
    auto state_vector_r = get_sparse_state(C_right, lists_);

    Determinant J;

    for (size_t p{1}, pq{0}; p < norb; ++p) {
        for (size_t q{0}; q < p; ++q, ++pq) {
            for (size_t r{1}, rs{0}; r < norb; ++r) {
                for (size_t s{0}; s < r; ++s, ++rs) {
                    std::complex<double> element = 0.0;
                    for (const auto& [I, c_I] : state_vector_r) {
                        J = I;
                        double sign = 1.0;
                        sign *= J.destroy_a(r);
                        sign *= J.destroy_a(s);
                        sign *= J.create_a(q);
                        sign *= J.create_a(p);
                        if (sign != 0) {
                            if (state_vector_l.count(J) != 0) {
                                element += static_cast<std::complex<double>>(sign) *
                                           std::conj(state_vector_l[J]) * c_I;
                            }
                        }
                    }
                    rdm_v(pq, rs) = element;
                }
            }
        }
    }
    return matrix::packed_tensor4_to_tensor4<std::complex<double>>(rdm);
}

np_tensor6_complex RelCISigmaBuilder::compute_3rdm_debug(np_vector_complex C_left,
                                                         np_vector_complex C_right) const {
    const size_t norb = lists_.norb();
    const size_t ntriplets = (norb * (norb - 1) * (norb - 2)) / 6;

    auto rdm = make_zeros<nb::numpy, std::complex<double>, 2>({ntriplets, ntriplets});
    auto rdm_v = rdm.view();

    auto state_vector_l = get_sparse_state(C_left, lists_);
    auto state_vector_r = get_sparse_state(C_right, lists_);

    Determinant J;

    for (size_t p{2}, pqr{0}; p < norb; ++p) {
        for (size_t q{1}; q < p; ++q) {
            for (size_t r{0}; r < q; ++r, ++pqr) {
                for (size_t s{2}, stu{0}; s < norb; ++s) {
                    for (size_t t{1}; t < s; ++t) {
                        for (size_t u{0}; u < t; ++u, ++stu) {
                            std::complex<double> element = 0.0;
                            for (const auto& [I, c_I] : state_vector_r) {
                                J = I;
                                double sign = 1.0;
                                sign *= J.destroy_a(s);
                                sign *= J.destroy_a(t);
                                sign *= J.destroy_a(u);
                                sign *= J.create_a(r);
                                sign *= J.create_a(q);
                                sign *= J.create_a(p);
                                if (sign != 0) {
                                    if (state_vector_l.count(J) != 0) {
                                        element += static_cast<std::complex<double>>(sign) *
                                                   std::conj(state_vector_l[J]) * c_I;
                                    }
                                }
                            }
                            rdm_v(pqr, stu) = element;
                        }
                    }
                }
            }
        }
    }

    auto rdm_full =
        make_zeros<nb::numpy, std::complex<double>, 6>({norb, norb, norb, norb, norb, norb});
    auto rdm_full_v = rdm_full.view();
    for (size_t p{2}, pqr{0}; p < norb; ++p) {
        for (size_t q{1}; q < p; ++q) {
            for (size_t r{0}; r < q; ++r, ++pqr) {
                for (size_t s{2}, stu{0}; s < norb; ++s) {
                    for (size_t t{1}; t < s; ++t) {
                        for (size_t u{0}; u < t; ++u, ++stu) {
                            // grab the unique element of the 3-RDM
                            const auto el = rdm_v(pqr, stu);

                            // Place the element in all valid 36 antisymmetric index
                            // permutations
                            rdm_full_v(p, q, r, s, t, u) = +el;
                            rdm_full_v(p, q, r, s, u, t) = -el;
                            rdm_full_v(p, q, r, u, s, t) = +el;
                            rdm_full_v(p, q, r, u, t, s) = -el;
                            rdm_full_v(p, q, r, t, u, s) = +el;
                            rdm_full_v(p, q, r, t, s, u) = -el;

                            rdm_full_v(p, r, q, s, t, u) = -el;
                            rdm_full_v(p, r, q, s, u, t) = +el;
                            rdm_full_v(p, r, q, u, s, t) = -el;
                            rdm_full_v(p, r, q, u, t, s) = +el;
                            rdm_full_v(p, r, q, t, u, s) = -el;
                            rdm_full_v(p, r, q, t, s, u) = +el;

                            rdm_full_v(r, p, q, s, t, u) = +el;
                            rdm_full_v(r, p, q, s, u, t) = -el;
                            rdm_full_v(r, p, q, u, s, t) = +el;
                            rdm_full_v(r, p, q, u, t, s) = -el;
                            rdm_full_v(r, p, q, t, u, s) = +el;
                            rdm_full_v(r, p, q, t, s, u) = -el;

                            rdm_full_v(r, q, p, s, t, u) = -el;
                            rdm_full_v(r, q, p, s, u, t) = +el;
                            rdm_full_v(r, q, p, u, s, t) = -el;
                            rdm_full_v(r, q, p, u, t, s) = +el;
                            rdm_full_v(r, q, p, t, u, s) = -el;
                            rdm_full_v(r, q, p, t, s, u) = +el;

                            rdm_full_v(q, r, p, s, t, u) = +el;
                            rdm_full_v(q, r, p, s, u, t) = -el;
                            rdm_full_v(q, r, p, u, s, t) = +el;
                            rdm_full_v(q, r, p, u, t, s) = -el;
                            rdm_full_v(q, r, p, t, u, s) = +el;
                            rdm_full_v(q, r, p, t, s, u) = -el;

                            rdm_full_v(q, p, r, s, t, u) = -el;
                            rdm_full_v(q, p, r, s, u, t) = +el;
                            rdm_full_v(q, p, r, u, s, t) = -el;
                            rdm_full_v(q, p, r, u, t, s) = +el;
                            rdm_full_v(q, p, r, t, u, s) = -el;
                            rdm_full_v(q, p, r, t, s, u) = +el;
                        }
                    }
                }
            }
        }
    }

    return rdm_full;
}

} // namespace forte2
