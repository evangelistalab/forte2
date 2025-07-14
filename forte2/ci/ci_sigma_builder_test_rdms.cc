#include "helpers/timer.hpp"
#include "helpers/np_matrix_functions.h"
#include "helpers/np_vector_functions.h"
#include "helpers/indexing.hpp"
#include "helpers/blas.h"

#include "ci_sigma_builder.h"
#include "sparse_state.h"

namespace forte2 {

SparseState get_sparse_state(np_vector c, const CIStrings& lists) {
    auto c_span = vector::as_span(c);
    SparseState state;
    auto dets = lists.make_determinants();
    for (size_t i{0}, maxi{dets.size()}; i < maxi; ++i) {
        state[dets[i]] = c_span[i];
    }
    return state;
}

np_matrix CISigmaBuilder::compute_1rdm_a_debug(np_vector C_left, np_vector C_right, bool alfa) {
    const size_t norb = lists_.norb();
    auto g1_ref = make_zeros<nb::numpy, double, 2>({norb, norb});

    auto state_vector_l = get_sparse_state(C_left, lists_);
    auto state_vector_r = get_sparse_state(C_right, lists_);

    Determinant J;

    for (size_t p{0}; p < norb; p++) {
        for (size_t q{0}; q < norb; ++q) {
            double rdm = 0.0;
            for (const auto& [I, c_I] : state_vector_r) {
                J = I;
                double sign = 1.0;
                if (alfa) {
                    sign *= J.destroy_alfa_bit(q);
                    sign *= J.create_alfa_bit(p);
                } else {
                    sign *= J.destroy_beta_bit(q);
                    sign *= J.create_beta_bit(p);
                }
                if (sign != 0) {
                    if (state_vector_l.count(J) != 0) {
                        rdm += sign * to_double(state_vector_l[J] * c_I);
                    }
                }
            }
            g1_ref(p, q) = rdm;
        }
    }
    return g1_ref;
}

np_matrix CISigmaBuilder::compute_2rdm_aa_debug(np_vector C_left, np_vector C_right, bool alfa) {
    const size_t norb = lists_.norb();
    const size_t npairs = (norb * (norb - 1)) / 2;
    auto g2_ref = make_zeros<nb::numpy, double, 2>({npairs, npairs});

    auto state_vector_l = get_sparse_state(C_left, lists_);
    auto state_vector_r = get_sparse_state(C_right, lists_);

    Determinant J;

    for (size_t p{1}, pq{0}; p < norb; ++p) {
        for (size_t q{0}; q < p; ++q, ++pq) {
            for (size_t r{1}, rs{0}; r < norb; ++r) {
                for (size_t s{0}; s < r; ++s, ++rs) {
                    double rdm = 0.0;
                    for (const auto& [I, c_I] : state_vector_r) {
                        J = I;
                        double sign = 1.0;
                        if (alfa) {
                            sign *= J.destroy_alfa_bit(r);
                            sign *= J.destroy_alfa_bit(s);
                            sign *= J.create_alfa_bit(q);
                            sign *= J.create_alfa_bit(p);
                        } else {
                            sign *= J.destroy_beta_bit(r);
                            sign *= J.destroy_beta_bit(s);
                            sign *= J.create_beta_bit(q);
                            sign *= J.create_beta_bit(p);
                        }
                        if (sign != 0) {
                            if (state_vector_l.count(J) != 0) {
                                rdm += sign * to_double(state_vector_l[J] * c_I);
                            }
                        }
                    }
                    g2_ref(pq, rs) = rdm;
                }
            }
        }
    }
    return g2_ref;
}

np_tensor4 CISigmaBuilder::compute_2rdm_ab_debug(np_vector C_left, np_vector C_right) {
    const size_t norb = lists_.norb();
    auto g2_ref = make_zeros<nb::numpy, double, 4>({norb, norb, norb, norb});

    auto state_vector_l = get_sparse_state(C_left, lists_);
    auto state_vector_r = get_sparse_state(C_right, lists_);

    Determinant J;

    for (size_t p{0}; p < norb; ++p) {
        for (size_t q{0}; q < norb; ++q) {
            for (size_t r{0}; r < norb; ++r) {
                for (size_t s{0}; s < norb; ++s) {
                    double rdm = 0.0;
                    for (const auto& [I, c_I] : state_vector_r) {
                        J = I;
                        double sign = 1.0;
                        sign *= J.destroy_alfa_bit(r);
                        sign *= J.destroy_beta_bit(s);
                        sign *= J.create_beta_bit(q);
                        sign *= J.create_alfa_bit(p);
                        if (sign != 0) {
                            if (state_vector_l.count(J) != 0) {
                                rdm += sign * to_double(state_vector_l[J] * c_I);
                            }
                        }
                    }
                    g2_ref(p, q, r, s) = rdm;
                }
            }
        }
    }
    return g2_ref;
}

np_matrix CISigmaBuilder::compute_3rdm_aaa_debug(np_vector C_left, np_vector C_right, bool alfa) {
    const size_t norb = lists_.norb();
    const size_t ntriplets = (norb * (norb - 1) * (norb - 2)) / 6;
    auto g3_ref = make_zeros<nb::numpy, double, 2>({ntriplets, ntriplets});

    auto state_vector_l = get_sparse_state(C_left, lists_);
    auto state_vector_r = get_sparse_state(C_right, lists_);

    Determinant J;

    for (size_t p{2}, pqr{0}; p < norb; ++p) {
        for (size_t q{1}; q < p; ++q) {
            for (size_t r{0}; r < q; ++r, ++pqr) {
                for (size_t s{2}, stu{0}; s < norb; ++s) {
                    for (size_t t{1}; t < s; ++t) {
                        for (size_t u{0}; u < t; ++u, ++stu) {
                            double rdm = 0.0;
                            for (const auto& [I, c_I] : state_vector_r) {
                                J = I;
                                double sign = 1.0;
                                if (alfa) {
                                    sign *= J.destroy_alfa_bit(s);
                                    sign *= J.destroy_alfa_bit(t);
                                    sign *= J.destroy_alfa_bit(u);
                                    sign *= J.create_alfa_bit(r);
                                    sign *= J.create_alfa_bit(q);
                                    sign *= J.create_alfa_bit(p);
                                } else {
                                    sign *= J.destroy_beta_bit(s);
                                    sign *= J.destroy_beta_bit(t);
                                    sign *= J.destroy_beta_bit(u);
                                    sign *= J.create_beta_bit(r);
                                    sign *= J.create_beta_bit(q);
                                    sign *= J.create_beta_bit(p);
                                }
                                if (sign != 0) {
                                    if (state_vector_l.count(J) != 0) {
                                        rdm += sign * to_double(state_vector_l[J] * c_I);
                                    }
                                }
                            }
                            g3_ref(pqr, stu) = rdm;
                        }
                    }
                }
            }
        }
    }
    return g3_ref;
}

np_tensor4 CISigmaBuilder::compute_3rdm_aab_debug(np_vector C_left, np_vector C_right) {
    const size_t norb = lists_.norb();
    // the number of orbital pairs i > j of the same spin
    const size_t npair = (norb * (norb - 1)) / 2;

    auto g3_ref = make_zeros<nb::numpy, double, 4>({npair, norb, npair, norb});

    auto state_vector_l = get_sparse_state(C_left, lists_);
    auto state_vector_r = get_sparse_state(C_right, lists_);

    Determinant J;

    for (size_t p{1}, pq{0}; p < norb; ++p) {
        for (size_t q{0}; q < p; ++q, ++pq) {
            for (size_t r{0}; r < norb; ++r) {
                for (size_t s{1}, st{0}; s < norb; ++s) {
                    for (size_t t{0}; t < s; ++t, ++st) {
                        for (size_t u{0}; u < norb; ++u) {
                            double rdm = 0.0;
                            for (const auto& [I, c_I] : state_vector_r) {
                                J = I;
                                double sign = 1.0;
                                sign *= J.destroy_alfa_bit(s);
                                sign *= J.destroy_alfa_bit(t);
                                sign *= J.destroy_beta_bit(u);
                                sign *= J.create_beta_bit(r);
                                sign *= J.create_alfa_bit(q);
                                sign *= J.create_alfa_bit(p);
                                if (sign != 0) {
                                    if (state_vector_l.count(J) != 0) {
                                        rdm += sign * to_double(state_vector_l[J] * c_I);
                                    }
                                }
                            }
                            g3_ref(pq, r, st, u) = rdm;
                        }
                    }
                }
            }
        }
    }
    return g3_ref;
}

np_tensor4 CISigmaBuilder::compute_3rdm_abb_debug(np_vector C_left, np_vector C_right) {
    const size_t norb = lists_.norb();
    // the number of orbital pairs i > j of the same spin
    const size_t npair = (norb * (norb - 1)) / 2;

    auto g3_ref = make_zeros<nb::numpy, double, 4>({norb, npair, norb, npair});

    auto state_vector_l = get_sparse_state(C_left, lists_);
    auto state_vector_r = get_sparse_state(C_right, lists_);

    Determinant J;

    for (size_t p{0}; p < norb; ++p) {
        for (size_t q{1}, qr{0}; q < norb; ++q) {
            for (size_t r{0}; r < q; ++r, ++qr) {
                for (size_t s{0}; s < norb; ++s) {
                    for (size_t t{1}, tu{0}; t < norb; ++t) {
                        for (size_t u{0}; u < t; ++u, ++tu) {
                            double rdm = 0.0;
                            for (const auto& [I, c_I] : state_vector_r) {
                                J = I;
                                double sign = 1.0;
                                sign *= J.destroy_alfa_bit(s);
                                sign *= J.destroy_beta_bit(t);
                                sign *= J.destroy_beta_bit(u);
                                sign *= J.create_beta_bit(r);
                                sign *= J.create_beta_bit(q);
                                sign *= J.create_alfa_bit(p);
                                if (sign != 0) {
                                    if (state_vector_l.count(J) != 0) {
                                        rdm += sign * to_double(state_vector_l[J] * c_I);
                                    }
                                }
                            }
                            g3_ref(p, qr, s, tu) = rdm;
                        }
                    }
                }
            }
        }
    }
    return g3_ref;
}

np_matrix CISigmaBuilder::compute_4rdm_aaaa_debug(np_vector C_left, np_vector C_right, bool alfa) {
    const size_t norb = lists_.norb();
    const size_t quadruplets = (norb * (norb - 1) * (norb - 2) * (norb - 3)) / 24;
    auto g4_ref = make_zeros<nb::numpy, double, 2>({quadruplets, quadruplets});

    auto state_vector_l = get_sparse_state(C_left, lists_);
    auto state_vector_r = get_sparse_state(C_right, lists_);

    Determinant J;

    for (size_t p{3}, pqrs{0}; p < norb; ++p) {
        for (size_t q{2}; q < p; ++q) {
            for (size_t r{1}; r < q; ++r) {
                for (size_t s{0}; s < r; ++s, ++pqrs) {
                    for (size_t t{3}, tuvw{0}; t < norb; ++t) {
                        for (size_t u{2}; u < t; ++u) {
                            for (size_t v{1}; v < u; ++v) {
                                for (size_t w{0}; w < v; ++w, ++tuvw) {
                                    double rdm = 0.0;
                                    for (const auto& [I, c_I] : state_vector_r) {
                                        J = I;
                                        double sign = 1.0;
                                        if (alfa) {
                                            sign *= J.destroy_alfa_bit(t);
                                            sign *= J.destroy_alfa_bit(u);
                                            sign *= J.destroy_alfa_bit(v);
                                            sign *= J.destroy_alfa_bit(w);
                                            sign *= J.create_alfa_bit(s);
                                            sign *= J.create_alfa_bit(r);
                                            sign *= J.create_alfa_bit(q);
                                            sign *= J.create_alfa_bit(p);
                                        } else {
                                            sign *= J.destroy_beta_bit(t);
                                            sign *= J.destroy_beta_bit(u);
                                            sign *= J.destroy_beta_bit(v);
                                            sign *= J.destroy_beta_bit(w);
                                            sign *= J.create_beta_bit(s);
                                            sign *= J.create_beta_bit(r);
                                            sign *= J.create_beta_bit(q);
                                            sign *= J.create_beta_bit(p);
                                        }
                                        if (sign != 0) {
                                            if (state_vector_l.count(J) != 0) {
                                                rdm += sign * to_double(state_vector_l[J] * c_I);
                                            }
                                        }
                                    }
                                    g4_ref(pqrs, tuvw) = rdm;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return g4_ref;
}

np_tensor4 CISigmaBuilder::compute_4rdm_aaab_debug(np_vector C_left, np_vector C_right) {
    const size_t norb = lists_.norb();
    const size_t ntriplets = (norb * (norb - 1) * (norb - 2)) / 6;

    auto g4_ref = make_zeros<nb::numpy, double, 4>({ntriplets, norb, ntriplets, norb});

    auto state_vector_l = get_sparse_state(C_left, lists_);
    auto state_vector_r = get_sparse_state(C_right, lists_);

    Determinant J;

    for (size_t p{2}, pqr{0}; p < norb; ++p) {
        for (size_t q{1}; q < p; ++q) {
            for (size_t r{0}; r < q; ++r, ++pqr) {
                for (size_t s{0}; s < norb; ++s) {
                    for (size_t t{2}, tuv{0}; t < norb; ++t) {
                        for (size_t u{1}; u < t; ++u) {
                            for (size_t v{0}; v < u; ++v, ++tuv) {
                                for (size_t w{0}; w < norb; ++w) {
                                    double rdm = 0.0;
                                    for (const auto& [I, c_I] : state_vector_r) {
                                        J = I;
                                        double sign = 1.0;
                                        sign *= J.destroy_alfa_bit(t);
                                        sign *= J.destroy_alfa_bit(u);
                                        sign *= J.destroy_alfa_bit(v);
                                        sign *= J.destroy_beta_bit(w);
                                        sign *= J.create_beta_bit(s);
                                        sign *= J.create_alfa_bit(r);
                                        sign *= J.create_alfa_bit(q);
                                        sign *= J.create_alfa_bit(p);
                                        if (sign != 0) {
                                            if (state_vector_l.count(J) != 0) {
                                                rdm += sign * to_double(state_vector_l[J] * c_I);
                                            }
                                        }
                                    }
                                    g4_ref(pqr, s, tuv, w) = rdm;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return g4_ref;
}

np_tensor4 CISigmaBuilder::compute_4rdm_aabb_debug(np_vector C_left, np_vector C_right) {
    const size_t norb = lists_.norb();
    const size_t npair = (norb * (norb - 1)) / 2;

    auto g4_ref = make_zeros<nb::numpy, double, 4>({npair, npair, npair, npair});

    auto state_vector_l = get_sparse_state(C_left, lists_);
    auto state_vector_r = get_sparse_state(C_right, lists_);

    Determinant J;

    for (size_t p{1}, pq{0}; p < norb; ++p) {
        for (size_t q{0}; q < p; ++q, ++pq) {
            for (size_t r{1}, rs{0}; r < norb; ++r) {
                for (size_t s{0}; s < r; ++s, ++rs) {
                    for (size_t t{1}, tu{0}; t < norb; ++t) {
                        for (size_t u{0}; u < t; ++u, ++tu) {
                            for (size_t v{1}, vw{0}; v < norb; ++v) {
                                for (size_t w{0}; w < v; ++w, ++vw) {
                                    double rdm = 0.0;
                                    for (const auto& [I, c_I] : state_vector_r) {
                                        J = I;
                                        double sign = 1.0;
                                        sign *= J.destroy_alfa_bit(t);
                                        sign *= J.destroy_alfa_bit(u);
                                        sign *= J.destroy_beta_bit(v);
                                        sign *= J.destroy_beta_bit(w);
                                        sign *= J.create_beta_bit(s);
                                        sign *= J.create_beta_bit(r);
                                        sign *= J.create_alfa_bit(q);
                                        sign *= J.create_alfa_bit(p);
                                        if (sign != 0) {
                                            if (state_vector_l.count(J) != 0) {
                                                rdm += sign * to_double(state_vector_l[J] * c_I);
                                            }
                                        }
                                    }
                                    g4_ref(pq, rs, tu, vw) = rdm;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return g4_ref;
}

np_tensor4 CISigmaBuilder::compute_4rdm_abbb_debug(np_vector C_left, np_vector C_right) {
    const size_t norb = lists_.norb();
    const size_t ntriplet = (norb * (norb - 1) * (norb - 2)) / 6;

    auto g4_ref = make_zeros<nb::numpy, double, 4>({norb, ntriplet, norb, ntriplet});

    auto state_vector_l = get_sparse_state(C_left, lists_);
    auto state_vector_r = get_sparse_state(C_right, lists_);

    Determinant J;

    for (size_t p{0}; p < norb; ++p) {
        for (size_t q{2}, qrs{0}; q < norb; ++q) {
            for (size_t r{1}, rs{0}; r < q; ++r) {
                for (size_t s{0}; s < r; ++s, ++qrs) {
                    for (size_t t{0}; t < norb; ++t) {
                        for (size_t u{2}, uvw{0}; u < norb; ++u) {
                            for (size_t v{1}; v < u; ++v) {
                                for (size_t w{0}; w < v; ++w, ++uvw) {
                                    double rdm = 0.0;
                                    for (const auto& [I, c_I] : state_vector_r) {
                                        J = I;
                                        double sign = 1.0;
                                        sign *= J.destroy_alfa_bit(t);
                                        sign *= J.destroy_beta_bit(u);
                                        sign *= J.destroy_beta_bit(v);
                                        sign *= J.destroy_beta_bit(w);
                                        sign *= J.create_beta_bit(s);
                                        sign *= J.create_beta_bit(r);
                                        sign *= J.create_beta_bit(q);
                                        sign *= J.create_alfa_bit(p);
                                        if (sign != 0) {
                                            if (state_vector_l.count(J) != 0) {
                                                rdm += sign * to_double(state_vector_l[J] * c_I);
                                            }
                                        }
                                    }
                                    g4_ref(p, qrs, t, uvw) = rdm;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return g4_ref;
}

np_matrix CISigmaBuilder::compute_sf_1rdm_debug(np_vector C_left, np_vector C_right) {
    auto rdm_sf = compute_1rdm_a_debug(C_left, C_right, true);
    auto rdm_b = compute_1rdm_a_debug(C_left, C_right, false);

    auto norb = lists_.norb();
    auto rdm_sf_v = rdm_sf.view();
    auto rdm_b_v = rdm_b.view();

    for (size_t p{0}; p < norb; ++p) {
        for (size_t q{0}; q < norb; ++q) {
            rdm_sf_v(p, q) += rdm_b_v(p, q);
        }
    }
    return rdm_sf;
}

np_tensor4 CISigmaBuilder::compute_sf_2rdm_debug(np_vector C_left, np_vector C_right) {
    auto norb = lists_.norb();
    auto rdm_sf = make_zeros<nb::numpy, double, 4>({norb, norb, norb, norb});

    auto rdm_aa = compute_2rdm_aa_debug(C_left, C_right, true);
    auto rdm_bb = compute_2rdm_aa_debug(C_left, C_right, false);
    auto rdm_ab = compute_2rdm_ab_debug(C_left, C_right);

    auto rdm_sf_v = rdm_sf.view();
    auto rdm_aa_v = rdm_aa.view();
    auto rdm_ab_v = rdm_ab.view();
    auto rdm_bb_v = rdm_bb.view();

    for (size_t p{1}, pq{0}; p < norb; ++p) {
        for (size_t q{0}; q < p; ++q, ++pq) {
            for (size_t r{1}, rs{0}; r < norb; ++r) {
                for (size_t s{0}; s < r; ++s, ++rs) {
                    const auto el = rdm_aa_v(pq, rs) + rdm_bb_v(pq, rs);
                    rdm_sf_v(p, q, r, s) += el;
                    rdm_sf_v(p, q, s, r) -= el;
                    rdm_sf_v(q, p, r, s) -= el;
                    rdm_sf_v(q, p, s, r) += el;
                }
            }
        }
    }
    for (size_t p{0}; p < norb; ++p) {
        for (size_t q{0}; q < norb; ++q) {
            for (size_t r{0}; r < norb; ++r) {
                for (size_t s{0}; s < norb; ++s) {
                    rdm_sf_v(p, q, r, s) += rdm_ab_v(p, q, r, s) + rdm_ab_v(q, p, s, r);
                }
            }
        }
    }

    return rdm_sf;
}
} // namespace forte2
