#pragma once

#include "sparse/sparse_state.h"

namespace forte2 {

enum class Spin1 { a, b };
enum class Spin2 { aa, ab, bb };
enum class Spin3 { aaa, aab, abb, bbb };
enum class Spin4 { aaaa, aaab, aabb, abbb, bbbb };

template <Spin1 S>
np_matrix compute_1rdm(const SparseState& state_left, const SparseState& state_right,
                       std::size_t norb) {
    auto g1_ref = make_zeros<nb::numpy, double, 2>({norb, norb});

    Determinant J;

    for (std::size_t p{0}; p < norb; p++) {
        for (std::size_t q{0}; q < norb; ++q) {
            double rdm = 0.0;
            for (const auto& [I, c_I] : state_right) {
                J = I;
                double sign = 1.0;
                if constexpr (S == Spin1::a) {
                    sign *= J.destroy_a(q);
                    sign *= J.create_a(p);
                } else {
                    sign *= J.destroy_b(q);
                    sign *= J.create_b(p);
                }
                if (sign != 0) {
                    auto it = state_left.find(J);
                    if (it != state_left.end()) {
                        rdm += sign * to_double(it->second * c_I);
                    }
                }
            }
            g1_ref(p, q) = rdm;
        }
    }
    return g1_ref;
}

template <Spin2 S>
np_tensor4 compute_2rdm(const SparseState& state_left, const SparseState& state_right,
                        std::size_t norb) {
    auto g2_ref = make_zeros<nb::numpy, double, 4>({norb, norb, norb, norb});
    auto g2_view = g2_ref.view();

    Determinant J;

    for (std::size_t p{0}; p < norb; ++p) {
        for (std::size_t q{0}; q < norb; ++q) {
            for (std::size_t r{0}; r < norb; ++r) {
                for (std::size_t s{0}; s < norb; ++s) {
                    double rdm = 0.0;
                    for (const auto& [I, c_I] : state_right) {
                        J = I;
                        double sign = 1.0;
                        if constexpr (S == Spin2::aa) {
                            sign *= J.destroy_a(r);
                            sign *= J.destroy_a(s);
                            sign *= J.create_a(q);
                            sign *= J.create_a(p);
                        } else if constexpr (S == Spin2::bb) {
                            sign *= J.destroy_b(r);
                            sign *= J.destroy_b(s);
                            sign *= J.create_b(q);
                            sign *= J.create_b(p);
                        } else {
                            sign *= J.destroy_a(r);
                            sign *= J.destroy_b(s);
                            sign *= J.create_b(q);
                            sign *= J.create_a(p);
                        }
                        if (sign != 0) {
                            auto it = state_left.find(J);
                            if (it != state_left.end()) {
                                rdm += sign * to_double(it->second * c_I);
                            }
                        }
                    }
                    g2_view(p, q, r, s) = rdm;
                }
            }
        }
    }
    return g2_ref;
}

template <Spin3 S>
np_tensor6 compute_3rdm(const SparseState& state_left, const SparseState& state_right,
                        std::size_t norb) {
    auto g3_ref = make_zeros<nb::numpy, double, 6>({norb, norb, norb, norb, norb, norb});
    auto g3_view = g3_ref.view();

    Determinant J;

    for (std::size_t p{0}; p < norb; ++p) {
        for (std::size_t q{0}; q < norb; ++q) {
            for (std::size_t r{0}; r < norb; ++r) {
                for (std::size_t s{0}; s < norb; ++s) {
                    for (std::size_t t{0}; t < norb; ++t) {
                        for (std::size_t u{0}; u < norb; ++u) {
                            double rdm = 0.0;
                            for (const auto& [I, c_I] : state_right) {
                                J = I;
                                double sign = 1.0;
                                if constexpr (S == Spin3::aaa) {
                                    sign *= J.destroy_a(s);
                                    sign *= J.destroy_a(t);
                                    sign *= J.destroy_a(u);
                                    sign *= J.create_a(r);
                                    sign *= J.create_a(q);
                                    sign *= J.create_a(p);
                                } else if constexpr (S == Spin3::bbb) {
                                    sign *= J.destroy_b(s);
                                    sign *= J.destroy_b(t);
                                    sign *= J.destroy_b(u);
                                    sign *= J.create_b(r);
                                    sign *= J.create_b(q);
                                    sign *= J.create_b(p);
                                } else if constexpr (S == Spin3::aab) {
                                    sign *= J.destroy_a(s);
                                    sign *= J.destroy_a(t);
                                    sign *= J.destroy_b(u);
                                    sign *= J.create_b(r);
                                    sign *= J.create_a(q);
                                    sign *= J.create_a(p);
                                } else {
                                    sign *= J.destroy_a(s);
                                    sign *= J.destroy_b(t);
                                    sign *= J.destroy_b(u);
                                    sign *= J.create_b(r);
                                    sign *= J.create_b(q);
                                    sign *= J.create_a(p);
                                }
                                if (sign != 0) {
                                    auto it = state_left.find(J);
                                    if (it != state_left.end()) {
                                        rdm += sign * to_double(it->second * c_I);
                                    }
                                }
                            }
                            g3_view(p, q, r, s, t, u) = rdm;
                        }
                    }
                }
            }
        }
    }
    return g3_ref;
}

template <Spin4 S>
np_tensor8 compute_4rdm(const SparseState& state_left, const SparseState& state_right,
                        std::size_t norb) {
    auto g4_ref =
        make_zeros<nb::numpy, double, 8>({norb, norb, norb, norb, norb, norb, norb, norb});
    auto g4_view = g4_ref.view();

    Determinant J;

    for (std::size_t p{0}; p < norb; ++p) {
        for (std::size_t q{0}; q < norb; ++q) {
            for (std::size_t r{0}; r < norb; ++r) {
                for (std::size_t s{0}; s < norb; ++s) {
                    for (std::size_t t{0}; t < norb; ++t) {
                        for (std::size_t u{0}; u < norb; ++u) {
                            for (std::size_t v{0}; v < norb; ++v) {
                                for (std::size_t w{0}; w < norb; ++w) {
                                    double rdm = 0.0;
                                    for (const auto& [I, c_I] : state_right) {
                                        J = I;
                                        double sign = 1.0;
                                        if constexpr (S == Spin4::aaaa) {
                                            sign *= J.destroy_a(t);
                                            sign *= J.destroy_a(u);
                                            sign *= J.destroy_a(v);
                                            sign *= J.destroy_a(w);
                                            sign *= J.create_a(s);
                                            sign *= J.create_a(r);
                                            sign *= J.create_a(q);
                                            sign *= J.create_a(p);
                                        } else if constexpr (S == Spin4::bbbb) {
                                            sign *= J.destroy_b(t);
                                            sign *= J.destroy_b(u);
                                            sign *= J.destroy_b(v);
                                            sign *= J.destroy_b(w);
                                            sign *= J.create_b(s);
                                            sign *= J.create_b(r);
                                            sign *= J.create_b(q);
                                            sign *= J.create_b(p);
                                        } else if constexpr (S == Spin4::aaab) {
                                            sign *= J.destroy_a(t);
                                            sign *= J.destroy_a(u);
                                            sign *= J.destroy_a(v);
                                            sign *= J.destroy_b(w);
                                            sign *= J.create_b(s);
                                            sign *= J.create_a(r);
                                            sign *= J.create_a(q);
                                            sign *= J.create_a(p);
                                        } else if constexpr (S == Spin4::aabb) {
                                            sign *= J.destroy_a(t);
                                            sign *= J.destroy_a(u);
                                            sign *= J.destroy_b(v);
                                            sign *= J.destroy_b(w);
                                            sign *= J.create_b(s);
                                            sign *= J.create_b(r);
                                            sign *= J.create_a(q);
                                            sign *= J.create_a(p);
                                        } else {
                                            sign *= J.destroy_a(t);
                                            sign *= J.destroy_b(u);
                                            sign *= J.destroy_b(v);
                                            sign *= J.destroy_b(w);
                                            sign *= J.create_b(s);
                                            sign *= J.create_b(r);
                                            sign *= J.create_b(q);
                                            sign *= J.create_a(p);
                                        }
                                        if (sign != 0) {
                                            auto it = state_left.find(J);
                                            if (it != state_left.end()) {
                                                rdm += sign * to_double(it->second * c_I);
                                            }
                                        }
                                    }
                                    g4_view(p, q, r, s, t, u, v, w) = rdm;
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

auto compute_a_1rdm(const SparseState& state_left, const SparseState& state_right,
                    std::size_t norb) {
    return compute_1rdm<Spin1::a>(state_left, state_right, norb);
}

auto compute_b_1rdm(const SparseState& state_left, const SparseState& state_right,
                    std::size_t norb) {
    return compute_1rdm<Spin1::b>(state_left, state_right, norb);
}

auto compute_aa_2rdm(const SparseState& state_left, const SparseState& state_right,
                     std::size_t norb) {
    return compute_2rdm<Spin2::aa>(state_left, state_right, norb);
}

auto compute_ab_2rdm(const SparseState& state_left, const SparseState& state_right,
                     std::size_t norb) {
    return compute_2rdm<Spin2::ab>(state_left, state_right, norb);
}

auto compute_bb_2rdm(const SparseState& state_left, const SparseState& state_right,
                     std::size_t norb) {
    return compute_2rdm<Spin2::bb>(state_left, state_right, norb);
}

auto compute_aaa_3rdm(const SparseState& state_left, const SparseState& state_right,
                      std::size_t norb) {
    return compute_3rdm<Spin3::aaa>(state_left, state_right, norb);
}

auto compute_aab_3rdm(const SparseState& state_left, const SparseState& state_right,
                      std::size_t norb) {
    return compute_3rdm<Spin3::aab>(state_left, state_right, norb);
}

auto compute_abb_3rdm(const SparseState& state_left, const SparseState& state_right,
                      std::size_t norb) {
    return compute_3rdm<Spin3::abb>(state_left, state_right, norb);
}

auto compute_bbb_3rdm(const SparseState& state_left, const SparseState& state_right,
                      std::size_t norb) {
    return compute_3rdm<Spin3::bbb>(state_left, state_right, norb);
}

auto compute_aaaa_4rdm(const SparseState& state_left, const SparseState& state_right,
                       std::size_t norb) {
    return compute_4rdm<Spin4::aaaa>(state_left, state_right, norb);
}

auto compute_aaab_4rdm(const SparseState& state_left, const SparseState& state_right,
                       std::size_t norb) {
    return compute_4rdm<Spin4::aaab>(state_left, state_right, norb);
}

auto compute_aabb_4rdm(const SparseState& state_left, const SparseState& state_right,
                       std::size_t norb) {
    return compute_4rdm<Spin4::aabb>(state_left, state_right, norb);
}

auto compute_abbb_4rdm(const SparseState& state_left, const SparseState& state_right,
                       std::size_t norb) {
    return compute_4rdm<Spin4::abbb>(state_left, state_right, norb);
}

auto compute_bbbb_4rdm(const SparseState& state_left, const SparseState& state_right,
                       std::size_t norb) {
    return compute_4rdm<Spin4::bbbb>(state_left, state_right, norb);
}

} // namespace forte2