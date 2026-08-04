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
                    sign *= J.destroy_alpha(q);
                    sign *= J.create_alpha(p);
                } else {
                    sign *= J.destroy_beta(q);
                    sign *= J.create_beta(p);
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
                            sign *= J.destroy_alpha(r);
                            sign *= J.destroy_alpha(s);
                            sign *= J.create_alpha(q);
                            sign *= J.create_alpha(p);
                        } else if constexpr (S == Spin2::bb) {
                            sign *= J.destroy_beta(r);
                            sign *= J.destroy_beta(s);
                            sign *= J.create_beta(q);
                            sign *= J.create_beta(p);
                        } else {
                            sign *= J.destroy_alpha(r);
                            sign *= J.destroy_beta(s);
                            sign *= J.create_beta(q);
                            sign *= J.create_alpha(p);
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
                                    sign *= J.destroy_alpha(s);
                                    sign *= J.destroy_alpha(t);
                                    sign *= J.destroy_alpha(u);
                                    sign *= J.create_alpha(r);
                                    sign *= J.create_alpha(q);
                                    sign *= J.create_alpha(p);
                                } else if constexpr (S == Spin3::bbb) {
                                    sign *= J.destroy_beta(s);
                                    sign *= J.destroy_beta(t);
                                    sign *= J.destroy_beta(u);
                                    sign *= J.create_beta(r);
                                    sign *= J.create_beta(q);
                                    sign *= J.create_beta(p);
                                } else if constexpr (S == Spin3::aab) {
                                    sign *= J.destroy_alpha(s);
                                    sign *= J.destroy_alpha(t);
                                    sign *= J.destroy_beta(u);
                                    sign *= J.create_beta(r);
                                    sign *= J.create_alpha(q);
                                    sign *= J.create_alpha(p);
                                } else {
                                    sign *= J.destroy_alpha(s);
                                    sign *= J.destroy_beta(t);
                                    sign *= J.destroy_beta(u);
                                    sign *= J.create_beta(r);
                                    sign *= J.create_beta(q);
                                    sign *= J.create_alpha(p);
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
                                            sign *= J.destroy_alpha(t);
                                            sign *= J.destroy_alpha(u);
                                            sign *= J.destroy_alpha(v);
                                            sign *= J.destroy_alpha(w);
                                            sign *= J.create_alpha(s);
                                            sign *= J.create_alpha(r);
                                            sign *= J.create_alpha(q);
                                            sign *= J.create_alpha(p);
                                        } else if constexpr (S == Spin4::bbbb) {
                                            sign *= J.destroy_beta(t);
                                            sign *= J.destroy_beta(u);
                                            sign *= J.destroy_beta(v);
                                            sign *= J.destroy_beta(w);
                                            sign *= J.create_beta(s);
                                            sign *= J.create_beta(r);
                                            sign *= J.create_beta(q);
                                            sign *= J.create_beta(p);
                                        } else if constexpr (S == Spin4::aaab) {
                                            sign *= J.destroy_alpha(t);
                                            sign *= J.destroy_alpha(u);
                                            sign *= J.destroy_alpha(v);
                                            sign *= J.destroy_beta(w);
                                            sign *= J.create_beta(s);
                                            sign *= J.create_alpha(r);
                                            sign *= J.create_alpha(q);
                                            sign *= J.create_alpha(p);
                                        } else if constexpr (S == Spin4::aabb) {
                                            sign *= J.destroy_alpha(t);
                                            sign *= J.destroy_alpha(u);
                                            sign *= J.destroy_beta(v);
                                            sign *= J.destroy_beta(w);
                                            sign *= J.create_beta(s);
                                            sign *= J.create_beta(r);
                                            sign *= J.create_alpha(q);
                                            sign *= J.create_alpha(p);
                                        } else {
                                            sign *= J.destroy_alpha(t);
                                            sign *= J.destroy_beta(u);
                                            sign *= J.destroy_beta(v);
                                            sign *= J.destroy_beta(w);
                                            sign *= J.create_beta(s);
                                            sign *= J.create_beta(r);
                                            sign *= J.create_beta(q);
                                            sign *= J.create_alpha(p);
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

// == Complex (two-component) RDMs ==
//
// The relativistic (two-component) selected CI stores every active electron in the alpha
// string (the beta string is empty), so only the alpha 1-RDM and the alpha-alpha 2-RDM are
// needed. Unlike the real versions above (which drop the imaginary part via to_double and do
// NOT conjugate the bra), these overloads accumulate the full complex value and conjugate the
// left (bra) coefficient so the resulting RDMs are Hermitian. This matches the convention of
// RelCISigmaBuilder::compute_1rdm / compute_2rdm (see rel_ci_sigma_builder_rdm.cc), so that
// gamma1[p,q] = <L| a^+_p a_q |R> and gamma2[p,q,r,s] = <L| a^+_p a^+_q a_s a_r |R>.

/// @brief Compute the complex alpha 1-RDM between two SparseStates
/// @return gamma1[p][q] = <L| a^+_p a_q |R> as a complex (norb, norb) matrix
inline np_matrix_complex compute_a_1rdm_complex(const SparseState& state_left,
                                                const SparseState& state_right,
                                                std::size_t norb) {
    auto g1 = make_zeros<nb::numpy, sparse_scalar_t, 2>({norb, norb});
    auto g1_v = g1.view();

    Determinant J;
    for (std::size_t p{0}; p < norb; ++p) {
        for (std::size_t q{0}; q < norb; ++q) {
            sparse_scalar_t rdm = 0.0;
            for (const auto& [I, c_I] : state_right) {
                J = I;
                double sign = 1.0;
                sign *= J.destroy_alpha(q);
                sign *= J.create_alpha(p);
                if (sign != 0) {
                    auto it = state_left.find(J);
                    if (it != state_left.end()) {
                        rdm += sign * std::conj(it->second) * c_I;
                    }
                }
            }
            g1_v(p, q) = rdm;
        }
    }
    return g1;
}

/// @brief Compute the complex alpha-alpha 2-RDM between two SparseStates
/// @return gamma2[p][q][r][s] = <L| a^+_p a^+_q a_s a_r |R> as a complex (norb, norb, norb, norb)
///         tensor (full, antisymmetric in p<->q and r<->s)
inline np_tensor4_complex compute_aa_2rdm_complex(const SparseState& state_left,
                                                  const SparseState& state_right,
                                                  std::size_t norb) {
    auto g2 = make_zeros<nb::numpy, sparse_scalar_t, 4>({norb, norb, norb, norb});
    auto g2_v = g2.view();

    Determinant J;
    for (std::size_t p{0}; p < norb; ++p) {
        for (std::size_t q{0}; q < norb; ++q) {
            for (std::size_t r{0}; r < norb; ++r) {
                for (std::size_t s{0}; s < norb; ++s) {
                    sparse_scalar_t rdm = 0.0;
                    for (const auto& [I, c_I] : state_right) {
                        J = I;
                        double sign = 1.0;
                        sign *= J.destroy_alpha(r);
                        sign *= J.destroy_alpha(s);
                        sign *= J.create_alpha(q);
                        sign *= J.create_alpha(p);
                        if (sign != 0) {
                            auto it = state_left.find(J);
                            if (it != state_left.end()) {
                                rdm += sign * std::conj(it->second) * c_I;
                            }
                        }
                    }
                    g2_v(p, q, r, s) = rdm;
                }
            }
        }
    }
    return g2;
}

} // namespace forte2