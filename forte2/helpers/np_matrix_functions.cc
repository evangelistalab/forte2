#include <iostream>

#include "helpers/np_matrix_functions.h"

namespace forte2::matrix {
void copy(np_matrix src, np_matrix dest) {
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

void zero(np_matrix mat) {
    auto mat_view = mat.view();
    for (size_t i{0}, maxi{mat.shape(0)}; i < maxi; ++i) {
        for (size_t j{0}, maxj{mat.shape(1)}; j < maxj; ++j) {
            mat_view(i, j) = 0.0;
        }
    }
}

void scale(np_matrix mat, double factor) {
    auto mat_view = mat.view();
    for (size_t i{0}, maxi{mat.shape(0)}; i < maxi; ++i) {
        for (size_t j{0}, maxj{mat.shape(1)}; j < maxj; ++j) {
            mat_view(i, j) *= factor;
        }
    }
}

void daxpy(double a, np_matrix x, np_matrix y) {
    if (x.shape(0) != y.shape(0) || x.shape(1) != y.shape(1)) {
        throw std::runtime_error("Source and destination matrices must have the same shape.");
    }
    auto x_view = x.view();
    auto y_view = y.view();
    for (size_t i{0}, maxi{x.shape(0)}; i < maxi; ++i) {
        for (size_t j{0}, maxj{x.shape(1)}; j < maxj; ++j) {
            y_view(i, j) += a * x_view(i, j);
        }
    }
}

void daxpy_rows(double a, np_matrix x, int row_x, np_matrix y, int row_y) {
    if (x.shape(1) != y.shape(1)) {
        throw std::runtime_error("Source and destination matrices must have the same shape.");
    }
    auto x_view = x.view();
    auto y_view = y.view();
    for (size_t i{0}, maxi{x.shape(1)}; i < maxi; ++i) {
        y_view(row_y, i) += a * x_view(row_x, i);
    }
}

double dot_rows(np_matrix x, int row_x, np_matrix y, int row_y, size_t max_col) {
    if (x.shape(1) != y.shape(1)) {
        throw std::runtime_error(
            "Source and destination matrices must have the same number of columns.");
    }
    auto x_view = x.view();
    auto y_view = y.view();
    double result = 0.0;
    max_col = (max_col == 0 ? x.shape(1) : max_col);
    for (size_t i{0}; i < max_col; ++i) {
        result += x_view(row_x, i) * y_view(row_y, i);
    }
    return result;
}

void print(np_matrix mat, std::string label) {
    std::cout << label << ":" << std::endl;
    auto mat_view = mat.view();
    for (size_t i{0}, maxi{mat.shape(0)}; i < maxi; ++i) {
        for (size_t j{0}, maxj{mat.shape(1)}; j < maxj; ++j) {
            std::cout << mat_view(i, j) << " ";
        }
        std::cout << std::endl;
    }
}
} // namespace forte2::matrix