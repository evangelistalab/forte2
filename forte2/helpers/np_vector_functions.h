#pragma once

#include <string>
#include <span>
#include <complex>
#include <iostream>

#include "ndarray.h"

namespace forte2::vector {

/// @brief Zero out a ndarray<T, 1>
/// @param vec The vector to zero out
template <typename T, typename... Extra>
void zero(nb::ndarray<nb::numpy, T, nb::ndim<1>, Extra...> vec) {
    auto v = vec.view();
    for (size_t i = 0, n = vec.shape(0); i < n; ++i) {
        v(i) = T(0);
    }
}

/// @brief Copy the contents of one ndarray<double, 1> to another
/// @param src The source vector to copy from
/// @param dest The destination vector to copy to
template <typename T> void copy(ndarray<T, 1> src, ndarray<T, 1> dest) {
    if (src.shape(0) != dest.shape(0)) {
        throw std::runtime_error("Source and destination vectors must have the same shape.");
    }
    auto src_view = src.view();
    auto dest_view = dest.view();
    for (size_t i{0}, maxi{src.shape(0)}; i < maxi; ++i) {
        dest_view(i) = src_view(i);
    }
}
/// @brief Scale a ndarray<double, 1> by a factor
/// @param vec The vector to scale
/// @param factor The scaling factor
template <typename T> void scale(ndarray<T, 1> vec, double factor) {
    auto vec_view = vec.view();
    for (size_t i{0}, maxi{vec.shape(0)}; i < maxi; ++i) {
        vec_view(i) *= static_cast<T>(factor);
    }
}

/// @brief Perform the operation y = a * x + y for specific rows in two vectors
/// @param a The scalar multiplier
/// @param x The source vector to multiply
/// @param y The destination vector to add to
template <typename T> void daxpy(double a, ndarray<T, 1> x, ndarray<T, 1> y) {
    if (x.shape(0) != y.shape(0)) {
        throw std::runtime_error("Source and destination vectors must have the same shape.");
    }
    auto x_view = x.view();
    auto y_view = y.view();
    for (size_t i{0}, maxi{x.shape(0)}; i < maxi; ++i) {
        y_view(i) += static_cast<T>(a) * x_view(i);
    }
}

/// @brief Check ndarray<double, 1> for contiguity and make a std::span from it
template <typename T> std::span<T> as_span(ndarray<T, 1> vec) {
    if (vec.ndim() != 1) {
        throw std::runtime_error(
            "ndarray<double, 1> must be 1-dimensional to convert to std::span.");
    }
    if (vec.stride(0) != 1) {
        throw std::runtime_error("ndarray<double, 1> must have stride 1 to convert to std::span.");
    }
    return std::span<T>(vec.data(), vec.shape(0));
}

template <typename T> std::span<T> as_span(ndarray<T, 1, nb::c_contig> vec) {
    return std::span<T>(vec.data(), vec.shape(0));
}

/// @brief Print the contents of a ndarray<double, 1> to standard output
template <typename T> void print(ndarray<T, 1> vec, std::string label) {
    std::cout << label << ":" << std::endl;
    auto vec_view = vec.view();
    for (size_t i{0}, maxi{vec.shape(0)}; i < maxi; ++i) {
        std::cout << vec_view(i) << std::endl;
    }
    std::cout << std::endl;
}

} // namespace forte2::vector
