#include <iostream>

#include "helpers/np_vector_functions.h"

namespace forte2::vector {
void copy(np_vector src, np_vector dest) {
    if (src.shape(0) != dest.shape(0)) {
        throw std::runtime_error("Source and destination vectors must have the same shape.");
    }
    auto src_view = src.view();
    auto dest_view = dest.view();
    for (size_t i{0}, maxi{src.shape(0)}; i < maxi; ++i) {
        dest_view(i) = src_view(i);
    }
}

void zero(np_vector vec) {
    auto vec_view = vec.view();
    for (size_t i{0}, maxi{vec.shape(0)}; i < maxi; ++i) {
        vec_view(i) = 0.0;
    }
}

void scale(np_vector vec, double factor) {
    auto vec_view = vec.view();
    for (size_t i{0}, maxi{vec.shape(0)}; i < maxi; ++i) {
        vec_view(i) *= factor;
    }
}

void daxpy(double a, np_vector x, np_vector y) {
    if (x.shape(0) != y.shape(0)) {
        throw std::runtime_error("Source and destination vectors must have the same shape.");
    }
    auto x_view = x.view();
    auto y_view = y.view();
    for (size_t i{0}, maxi{x.shape(0)}; i < maxi; ++i) {
        y_view(i) += a * x_view(i);
    }
}

std::span<double> as_span(np_vector vec) {
    if (vec.ndim() != 1) {
        throw std::runtime_error("np_vector must be 1-dimensional to convert to std::span.");
    }
    if (vec.stride(0) != 1) {
        throw std::runtime_error("np_vector must have stride 1 to convert to std::span.");
    }
    return std::span<double>(vec.data(), vec.shape(0));
}

void print(np_vector vec, std::string label) {
    std::cout << label << ":" << std::endl;
    auto vec_view = vec.view();
    for (size_t i{0}, maxi{vec.shape(0)}; i < maxi; ++i) {
        std::cout << vec_view(i) << std::endl;
    }
    std::cout << std::endl;
}
} // namespace forte2::vector

namespace forte2::vector_complex {
void copy(np_vector_complex src, np_vector_complex dest) {
    if (src.shape(0) != dest.shape(0)) {
        throw std::runtime_error("Source and destination vectors must have the same shape.");
    }
    auto src_view = src.view();
    auto dest_view = dest.view();
    for (size_t i{0}, maxi{src.shape(0)}; i < maxi; ++i) {
        dest_view(i) = src_view(i);
    }
}

void zero(np_vector_complex vec) {
    auto vec_view = vec.view();
    for (size_t i{0}, maxi{vec.shape(0)}; i < maxi; ++i) {
        vec_view(i) = 0.0;
    }
}

void scale(np_vector_complex vec, double factor) {
    auto vec_view = vec.view();
    for (size_t i{0}, maxi{vec.shape(0)}; i < maxi; ++i) {
        vec_view(i) *= factor;
    }
}

void daxpy(double a, np_vector_complex x, np_vector_complex y) {
    if (x.shape(0) != y.shape(0)) {
        throw std::runtime_error("Source and destination vectors must have the same shape.");
    }
    auto x_view = x.view();
    auto y_view = y.view();
    for (size_t i{0}, maxi{x.shape(0)}; i < maxi; ++i) {
        y_view(i) += a * x_view(i);
    }
}

std::span<std::complex<double>> as_span(np_vector_complex vec) {
    if (vec.ndim() != 1) {
        throw std::runtime_error("np_vector_complex must be 1-dimensional to convert to std::span.");
    }
    if (vec.stride(0) != 1) {
        throw std::runtime_error("np_vector_complex must have stride 1 to convert to std::span.");
    }
    return std::span<std::complex<double>>(vec.data(), vec.shape(0));
}

void print(np_vector_complex vec, std::string label) {
    std::cout << label << ":" << std::endl;
    auto vec_view = vec.view();
    for (size_t i{0}, maxi{vec.shape(0)}; i < maxi; ++i) {
        std::cout << vec_view(i) << std::endl;
    }
    std::cout << std::endl;
}
} // namespace forte2::vector_complex
