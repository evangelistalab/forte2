#pragma once

#include <string>
#include <span>
#include <complex>
#include <iostream>

#include "ndarray.h"

namespace forte2::vector {

/// @brief Zero out a np_vector
/// @param vec The vector to zero out
template <typename T> void zero(nb::ndarray<nb::numpy, T, nb::ndim<1>> vec) {
    auto vec_view = vec.view();
    for (size_t i{0}, maxi{vec.shape(0)}; i < maxi; ++i) {
        vec_view(i) = 0.0;
    }
}

/// @brief Copy the contents of one np_vector to another
/// @param src The source vector to copy from
/// @param dest The destination vector to copy to
template <typename T>
void copy(nb::ndarray<nb::numpy, T, nb::ndim<1>> src, nb::ndarray<nb::numpy, T, nb::ndim<1>> dest) {
    if (src.shape(0) != dest.shape(0)) {
        throw std::runtime_error("Source and destination vectors must have the same shape.");
    }
    auto src_view = src.view();
    auto dest_view = dest.view();
    for (size_t i{0}, maxi{src.shape(0)}; i < maxi; ++i) {
        dest_view(i) = src_view(i);
    }
}
/// @brief Scale a np_vector by a factor
/// @param vec The vector to scale
/// @param factor The scaling factor
template <typename T> void scale(nb::ndarray<nb::numpy, T, nb::ndim<1>> vec, double factor) {
    auto vec_view = vec.view();
    for (size_t i{0}, maxi{vec.shape(0)}; i < maxi; ++i) {
        vec_view(i) *= static_cast<T>(factor);
    }
}

/// @brief Perform the operation y = a * x + y for specific rows in two vectors
/// @param a The scalar multiplier
/// @param x The source vector to multiply
/// @param y The destination vector to add to
template <typename T>
void daxpy(double a, nb::ndarray<nb::numpy, T, nb::ndim<1>> x,
           nb::ndarray<nb::numpy, T, nb::ndim<1>> y) {
    if (x.shape(0) != y.shape(0)) {
        throw std::runtime_error("Source and destination vectors must have the same shape.");
    }
    auto x_view = x.view();
    auto y_view = y.view();
    for (size_t i{0}, maxi{x.shape(0)}; i < maxi; ++i) {
        y_view(i) += static_cast<T>(a) * x_view(i);
    }
}

/// @brief Check np_vector for contiguity and make a std::span from it
template <typename T> std::span<T> as_span(nb::ndarray<nb::numpy, T, nb::ndim<1>> vec) {
    if (vec.ndim() != 1) {
        throw std::runtime_error("np_vector must be 1-dimensional to convert to std::span.");
    }
    if (vec.stride(0) != 1) {
        throw std::runtime_error("np_vector must have stride 1 to convert to std::span.");
    }
    return std::span<T>(vec.data(), vec.shape(0));
}

/// @brief Print the contents of a np_vector to standard output
template <typename T> void print(nb::ndarray<nb::numpy, T, nb::ndim<1>> vec, std::string label) {
    std::cout << label << ":" << std::endl;
    auto vec_view = vec.view();
    for (size_t i{0}, maxi{vec.shape(0)}; i < maxi; ++i) {
        std::cout << vec_view(i) << std::endl;
    }
    std::cout << std::endl;
}

} // namespace forte2::vector
