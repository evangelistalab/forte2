#pragma once

#include <string>
#include <complex>
#include <iostream>

#include "helpers/indexing.hpp"
#include "ndarray.h"

namespace forte2::matrix {

/// @brief Zero out a np_matrix
/// @param mat The matrix to zero out
template <typename T> void zero(nb::ndarray<nb::numpy, T, nb::ndim<2>> mat) {
    auto mat_view = mat.view();
    for (size_t i{0}, maxi{mat.shape(0)}; i < maxi; ++i) {
        for (size_t j{0}, maxj{mat.shape(1)}; j < maxj; ++j) {
            mat_view(i, j) = 0.0;
        }
    }
}

/// @brief Copy the contents of one np_matrix to another
/// @param src The source matrix to copy from
/// @param dest The destination matrix to copy to
template <typename T>
void copy(nb::ndarray<nb::numpy, T, nb::ndim<2>> src, nb::ndarray<nb::numpy, T, nb::ndim<2>> dest) {
    if (src.shape(0) != dest.shape(0) || src.shape(1) != dest.shape(1)) {
        throw std::runtime_error("Source and destination matrices must have the same shape.");
    }
    auto src_view = src.view();
    auto dest_view = dest.view();
    for (size_t i{0}, maxi{src.shape(0)}; i < maxi; ++i) {
        for (size_t j{0}, maxj{src.shape(1)}; j < maxj; ++j) {
            dest_view(i, j) = src_view(i, j);
        }
    }
}

/// @brief Scale a np_matrix by a factor
/// @param mat The matrix to scale
/// @param factor The scaling factor
template <typename T> void scale(nb::ndarray<nb::numpy, T, nb::ndim<2>> mat, double factor) {
    auto mat_view = mat.view();
    for (size_t i{0}, maxi{mat.shape(0)}; i < maxi; ++i) {
        for (size_t j{0}, maxj{mat.shape(1)}; j < maxj; ++j) {
            mat_view(i, j) *= static_cast<T>(factor);
        }
    }
}

/// @brief Perform the operation y[:,:] = a * x[:,:] + y[:,:]
template <typename T>
void daxpy(double a, nb::ndarray<nb::numpy, T, nb::ndim<2>> x,
           nb::ndarray<nb::numpy, T, nb::ndim<2>> y) {
    if (x.shape(0) != y.shape(0) || x.shape(1) != y.shape(1)) {
        throw std::runtime_error("Source and destination matrices must have the same shape.");
    }
    auto x_view = x.view();
    auto y_view = y.view();
    for (size_t i{0}, maxi{x.shape(0)}; i < maxi; ++i) {
        for (size_t j{0}, maxj{x.shape(1)}; j < maxj; ++j) {
            y_view(i, j) += static_cast<T>(a) * x_view(i, j);
        }
    }
}

/// @brief Print the contents of a np_matrix to standard output
template <typename T> void print(nb::ndarray<nb::numpy, T, nb::ndim<2>> mat, std::string label) {
    std::cout << label << ":" << std::endl;
    auto mat_view = mat.view();
    for (size_t i{0}, maxi{mat.shape(0)}; i < maxi; ++i) {
        for (size_t j{0}, maxj{mat.shape(1)}; j < maxj; ++j) {
            std::cout << mat_view(i, j) << " ";
        }
        std::cout << std::endl;
    }
}

/// @brief Expand a packed 4D tensor T([p>q],[r>s]) stored as a 2D matrix into a full 4D tensor with
/// elements:   T[p,q,r,s] = +T([p>q],[r>s])
///             T[p,q,s,r] = -T([p>q],[r>s])
///             T[q,p,r,s] = -T([p>q],[r>s])
///             T[q,p,s,r] = +T([p>q],[r>s])
///
/// @param M The input 2D matrix.
/// @return The expanded 4D tensor.
template <typename T>
nb::ndarray<nb::numpy, T, nb::ndim<4>>
packed_tensor4_to_tensor4(nb::ndarray<nb::numpy, T, nb::ndim<2>> m) {
    const auto nrows = m.shape(0);
    const auto ncols = m.shape(1);
    auto nr = inv_pair_index_gt(nrows - 1).first + 1;
    auto nl = inv_pair_index_gt(ncols - 1).first + 1;

    auto m_v = m.view();
    auto t = make_zeros<nb::numpy, T, 4>({nr, nr, nl, nl});
    auto t_v = t.view();

    for (size_t p{1}, pq{0}; p < nr; ++p) {
        for (size_t q{0}; q < p; ++q, ++pq) { // p > q
            for (size_t r{1}, rs{0}; r < nl; ++r) {
                for (size_t s{0}; s < r; ++s, ++rs) { // r > s
                    auto element = m_v(pq, rs);
                    t_v(p, q, r, s) = element;
                    t_v(q, p, s, r) = element;
                    t_v(p, q, s, r) = -element;
                    t_v(q, p, r, s) = -element;
                }
            }
        }
    }
    return t;
}

/// @brief Pack a 4D tensor that is antisymmetric in its first and in its last pair of indices
/// into a 2D matrix with elements M([p>q],[r>s]) = T[p,q,r,s]. This is the inverse of
/// packed_tensor4_to_tensor4.
///
/// @param t The input 4D tensor.
/// @return The packed 2D matrix.
template <typename T>
nb::ndarray<nb::numpy, T, nb::ndim<2>>
tensor4_to_packed_tensor4(nb::ndarray<nb::numpy, T, nb::ndim<4>> t) {
    const size_t nr = t.shape(0);
    const size_t nl = t.shape(2);
    auto t_v = t.view();
    auto m = make_zeros<nb::numpy, T, 2>({nr * (nr - 1) / 2, nl * (nl - 1) / 2});
    auto m_v = m.view();
    for (size_t p{1}; p < nr; ++p) {
        for (size_t q{0}; q < p; ++q) {
            const auto pq = pair_index_gt(p, q);
            for (size_t r{1}; r < nl; ++r) {
                for (size_t s{0}; s < r; ++s) {
                    m_v(pq, pair_index_gt(r, s)) = t_v(p, q, r, s);
                }
            }
        }
    }
    return m;
}

namespace detail {
/// @brief Call f(a, b, c, sign) for each permutation (a, b, c) of (i, j, k), where sign is the
/// sign of the permutation.
template <typename F> void for_each_permutation3(size_t i, size_t j, size_t k, F&& f) {
    f(i, j, k, 1.0);
    f(j, k, i, 1.0);
    f(k, i, j, 1.0);
    f(j, i, k, -1.0);
    f(i, k, j, -1.0);
    f(k, j, i, -1.0);
}

inline size_t ntriplets_gt(size_t n) { return n < 3 ? 0 : n * (n - 1) * (n - 2) / 6; }
} // namespace detail

/// @brief Expand a packed 6D tensor M([p>q>r],[s>t>u]) stored as a 2D matrix into a full 6D tensor
/// that is antisymmetric under any permutation of (p, q, r) and of (s, t, u).
///
/// @param m The input 2D matrix.
/// @param n The dimension of each index of the full tensor.
/// @return The expanded 6D tensor.
template <typename T>
nb::ndarray<nb::numpy, T, nb::ndim<6>>
packed_tensor6_to_tensor6(nb::ndarray<nb::numpy, T, nb::ndim<2>> m, size_t n) {
    const auto ntriplets = detail::ntriplets_gt(n);
    if (m.shape(0) != ntriplets || m.shape(1) != ntriplets) {
        throw std::runtime_error("packed_tensor6_to_tensor6: the matrix must have one row and "
                                 "one column per triplet p > q > r of n indices.");
    }
    auto m_v = m.view();
    auto out = make_zeros<nb::numpy, T, 6>({n, n, n, n, n, n});
    auto out_v = out.view();
    for (size_t p{2}; p < n; ++p) {
        for (size_t q{1}; q < p; ++q) {
            for (size_t r{0}; r < q; ++r) {
                const auto pqr = triplet_index_gt(p, q, r);
                for (size_t s{2}; s < n; ++s) {
                    for (size_t t{1}; t < s; ++t) {
                        for (size_t u{0}; u < t; ++u) {
                            const auto element = m_v(pqr, triplet_index_gt(s, t, u));
                            detail::for_each_permutation3(
                                p, q, r, [&](size_t a, size_t b, size_t c, double sign_abc) {
                                    detail::for_each_permutation3(
                                        s, t, u,
                                        [&](size_t d, size_t e, size_t f, double sign_def) {
                                            out_v(a, b, c, d, e, f) =
                                                static_cast<T>(sign_abc * sign_def) * element;
                                        });
                                });
                        }
                    }
                }
            }
        }
    }
    return out;
}

/// @brief Pack a 6D tensor that is antisymmetric under any permutation of its first and of its
/// last three indices into a 2D matrix with elements M([p>q>r],[s>t>u]) = T[p,q,r,s,t,u]. This is
/// the inverse of packed_tensor6_to_tensor6.
///
/// @param tensor The input 6D tensor.
/// @return The packed 2D matrix.
template <typename T>
nb::ndarray<nb::numpy, T, nb::ndim<2>>
tensor6_to_packed_tensor6(nb::ndarray<nb::numpy, T, nb::ndim<6>> tensor) {
    const size_t n = tensor.shape(0);
    const auto ntriplets = detail::ntriplets_gt(n);
    auto tensor_v = tensor.view();
    auto m = make_zeros<nb::numpy, T, 2>({ntriplets, ntriplets});
    auto m_v = m.view();
    for (size_t p{2}; p < n; ++p) {
        for (size_t q{1}; q < p; ++q) {
            for (size_t r{0}; r < q; ++r) {
                const auto pqr = triplet_index_gt(p, q, r);
                for (size_t s{2}; s < n; ++s) {
                    for (size_t t{1}; t < s; ++t) {
                        for (size_t u{0}; u < t; ++u) {
                            m_v(pqr, triplet_index_gt(s, t, u)) = tensor_v(p, q, r, s, t, u);
                        }
                    }
                }
            }
        }
    }
    return m;
}

/// @brief Expand a packed alpha-alpha-beta 6D tensor M([p>q],r,[s>t],u), stored as a 4D tensor,
/// into a full 6D tensor that is antisymmetric under p <-> q and under s <-> t.
///
/// @param m The input 4D tensor, with shape (npair, n, npair, n).
/// @return The expanded 6D tensor.
template <typename T>
nb::ndarray<nb::numpy, T, nb::ndim<6>>
packed_aab_tensor6_to_tensor6(nb::ndarray<nb::numpy, T, nb::ndim<4>> m) {
    const size_t n = m.shape(1);
    auto m_v = m.view();
    auto out = make_zeros<nb::numpy, T, 6>({n, n, n, n, n, n});
    auto out_v = out.view();
    for (size_t p{1}; p < n; ++p) {
        for (size_t q{0}; q < p; ++q) {
            const auto pq = pair_index_gt(p, q);
            for (size_t s{1}; s < n; ++s) {
                for (size_t t{0}; t < s; ++t) {
                    const auto st = pair_index_gt(s, t);
                    for (size_t r{0}; r < n; ++r) {
                        for (size_t u{0}; u < n; ++u) {
                            const auto element = m_v(pq, r, st, u);
                            out_v(p, q, r, s, t, u) = element;
                            out_v(q, p, r, t, s, u) = element;
                            out_v(q, p, r, s, t, u) = -element;
                            out_v(p, q, r, t, s, u) = -element;
                        }
                    }
                }
            }
        }
    }
    return out;
}

/// @brief Pack a 6D tensor that is antisymmetric under p <-> q and under s <-> t into a 4D
/// tensor with elements M([p>q],r,[s>t],u) = T[p,q,r,s,t,u]. This is the inverse of
/// packed_aab_tensor6_to_tensor6.
///
/// @param tensor The input 6D tensor.
/// @return The packed 4D tensor, with shape (npair, n, npair, n).
template <typename T>
nb::ndarray<nb::numpy, T, nb::ndim<4>>
tensor6_to_packed_aab_tensor6(nb::ndarray<nb::numpy, T, nb::ndim<6>> tensor) {
    const size_t n = tensor.shape(0);
    const size_t npair = n * (n - 1) / 2;
    auto tensor_v = tensor.view();
    auto m = make_zeros<nb::numpy, T, 4>({npair, n, npair, n});
    auto m_v = m.view();
    for (size_t p{1}; p < n; ++p) {
        for (size_t q{0}; q < p; ++q) {
            const auto pq = pair_index_gt(p, q);
            for (size_t s{1}; s < n; ++s) {
                for (size_t t{0}; t < s; ++t) {
                    const auto st = pair_index_gt(s, t);
                    for (size_t r{0}; r < n; ++r) {
                        for (size_t u{0}; u < n; ++u) {
                            m_v(pq, r, st, u) = tensor_v(p, q, r, s, t, u);
                        }
                    }
                }
            }
        }
    }
    return m;
}

/// @brief Expand a packed alpha-beta-beta 6D tensor M(p,[q>r],s,[t>u]), stored as a 4D tensor,
/// into a full 6D tensor that is antisymmetric under q <-> r and under t <-> u.
///
/// @param m The input 4D tensor, with shape (n, npair, n, npair).
/// @return The expanded 6D tensor.
template <typename T>
nb::ndarray<nb::numpy, T, nb::ndim<6>>
packed_abb_tensor6_to_tensor6(nb::ndarray<nb::numpy, T, nb::ndim<4>> m) {
    const size_t n = m.shape(0);
    auto m_v = m.view();
    auto out = make_zeros<nb::numpy, T, 6>({n, n, n, n, n, n});
    auto out_v = out.view();
    for (size_t q{1}; q < n; ++q) {
        for (size_t r{0}; r < q; ++r) {
            const auto qr = pair_index_gt(q, r);
            for (size_t t{1}; t < n; ++t) {
                for (size_t u{0}; u < t; ++u) {
                    const auto tu = pair_index_gt(t, u);
                    for (size_t p{0}; p < n; ++p) {
                        for (size_t s{0}; s < n; ++s) {
                            const auto element = m_v(p, qr, s, tu);
                            out_v(p, q, r, s, t, u) = element;
                            out_v(p, r, q, s, u, t) = element;
                            out_v(p, r, q, s, t, u) = -element;
                            out_v(p, q, r, s, u, t) = -element;
                        }
                    }
                }
            }
        }
    }
    return out;
}

/// @brief Pack a 6D tensor that is antisymmetric under q <-> r and under t <-> u into a 4D
/// tensor with elements M(p,[q>r],s,[t>u]) = T[p,q,r,s,t,u]. This is the inverse of
/// packed_abb_tensor6_to_tensor6.
///
/// @param tensor The input 6D tensor.
/// @return The packed 4D tensor, with shape (n, npair, n, npair).
template <typename T>
nb::ndarray<nb::numpy, T, nb::ndim<4>>
tensor6_to_packed_abb_tensor6(nb::ndarray<nb::numpy, T, nb::ndim<6>> tensor) {
    const size_t n = tensor.shape(0);
    const size_t npair = n * (n - 1) / 2;
    auto tensor_v = tensor.view();
    auto m = make_zeros<nb::numpy, T, 4>({n, npair, n, npair});
    auto m_v = m.view();
    for (size_t q{1}; q < n; ++q) {
        for (size_t r{0}; r < q; ++r) {
            const auto qr = pair_index_gt(q, r);
            for (size_t t{1}; t < n; ++t) {
                for (size_t u{0}; u < t; ++u) {
                    const auto tu = pair_index_gt(t, u);
                    for (size_t p{0}; p < n; ++p) {
                        for (size_t s{0}; s < n; ++s) {
                            m_v(p, qr, s, tu) = tensor_v(p, q, r, s, t, u);
                        }
                    }
                }
            }
        }
    }
    return m;
}
} // namespace forte2::matrix
