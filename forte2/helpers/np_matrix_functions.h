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

/// @brief Expand a packed 6D tensor T([p>q>r],[s>t>u]) stored as a 2D matrix into a full 6D tensor
/// with elements:   T[p,q,r,s,t,u] = +T([p>q>r],[s>t>u])
///             T[p,q,r,s,u,t] = -T([p>q>r],[s>t>u])
///             T[p,r,q,s,t,u] = -T([p>q>r],[s>t>u])
///             T[p,r,q,s,u,t] = +T([p>q>r],[s>t>u])
///             T[q,p,r,s,t,u] = -T([p>q>r],[s>t>u])
///             T[q,p,r,s,u,t] = +T([p>q>r],[s>t>u])
/// @param m The input 2D matrix.
/// @return The expanded 6D tensor.
// np_tensor6 packed_tensor6_to_tensor6(np_matrix m);
} // namespace forte2::matrix
