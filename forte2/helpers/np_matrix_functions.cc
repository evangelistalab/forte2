#include <iostream>

#include "helpers/indexing.hpp"
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

np_tensor4 packed_tensor4_to_tensor4(np_matrix m) {
    const auto nrows = m.shape(0);
    const auto ncols = m.shape(1);
    auto nr = inv_pair_index_gt(nrows - 1).first + 1;
    auto nl = inv_pair_index_gt(ncols - 1).first + 1;

    auto m_v = m.view();
    auto t = make_zeros<nb::numpy, double, 4>({nr, nr, nl, nl});
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
} // namespace forte2::matrix

namespace forte2::matrix_complex {
void copy(np_matrix_complex src, np_matrix_complex dest) {
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

void zero(np_matrix_complex mat) {
    auto mat_view = mat.view();
    for (size_t i{0}, maxi{mat.shape(0)}; i < maxi; ++i) {
        for (size_t j{0}, maxj{mat.shape(1)}; j < maxj; ++j) {
            mat_view(i, j) = 0.0;
        }
    }
}

void scale(np_matrix_complex mat, double factor) {
    auto mat_view = mat.view();
    for (size_t i{0}, maxi{mat.shape(0)}; i < maxi; ++i) {
        for (size_t j{0}, maxj{mat.shape(1)}; j < maxj; ++j) {
            mat_view(i, j) *= factor;
        }
    }
}

void daxpy(double a, np_matrix_complex x, np_matrix_complex y) {
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

void daxpy_rows(double a, np_matrix_complex x, int row_x, np_matrix_complex y, int row_y) {
    if (x.shape(1) != y.shape(1)) {
        throw std::runtime_error("Source and destination matrices must have the same shape.");
    }
    auto x_view = x.view();
    auto y_view = y.view();
    for (size_t i{0}, maxi{x.shape(1)}; i < maxi; ++i) {
        y_view(row_y, i) += a * x_view(row_x, i);
    }
}

std::complex<double> dot_rows(np_matrix_complex x, int row_x, np_matrix_complex y, int row_y,
                              size_t max_col) {
    if (x.shape(1) != y.shape(1)) {
        throw std::runtime_error(
            "Source and destination matrices must have the same number of columns.");
    }
    auto x_view = x.view();
    auto y_view = y.view();
    std::complex<double> result = 0.0;
    max_col = (max_col == 0 ? x.shape(1) : max_col);
    for (size_t i{0}; i < max_col; ++i) {
        result += x_view(row_x, i) * y_view(row_y, i);
    }
    return result;
}

void print(np_matrix_complex mat, std::string label) {
    std::cout << label << ":" << std::endl;
    auto mat_view = mat.view();
    for (size_t i{0}, maxi{mat.shape(0)}; i < maxi; ++i) {
        for (size_t j{0}, maxj{mat.shape(1)}; j < maxj; ++j) {
            std::cout << mat_view(i, j) << " ";
        }
        std::cout << std::endl;
    }
}

np_tensor4_complex packed_tensor4_to_tensor4(np_matrix_complex m) {
    const auto nrows = m.shape(0);
    const auto ncols = m.shape(1);
    auto nr = inv_pair_index_gt(nrows - 1).first + 1;
    auto nl = inv_pair_index_gt(ncols - 1).first + 1;

    auto m_v = m.view();
    auto t = make_zeros<nb::numpy, std::complex<double>, 4>({nr, nr, nl, nl});
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

} // namespace forte2::matrix_complex