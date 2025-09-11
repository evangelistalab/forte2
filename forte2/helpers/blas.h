#pragma once
#include <complex>
namespace forte2 {

extern "C" {
// Note: all Fortran BLAS routines expect single‐letter char arguments by address,
// and integer args are usually Fortran‐INTEGER (often 32‐bit). Adjust if your BLAS uses 64‐bit
// ints.
void daxpy_(int* length, double* a, const double* x, int* inc_x, double* y, int* inc_y);

void zaxpy_(int* length, std::complex<double>* a, const std::complex<double>* x, int* inc_x,
            std::complex<double>* y, int* inc_y);

void dgemm_(const char* transA, const char* transB, const int* M, const int* N, const int* K,
            const double* alpha, const double* A, const int* lda, const double* B, const int* ldb,
            const double* beta, double* C, const int* ldc);

void zgemm_(const char* transA, const char* transB, const int* M, const int* N, const int* K,
            const std::complex<double>* alpha, const std::complex<double>* A, const int* lda,
            const std::complex<double>* B, const int* ldb, const std::complex<double>* beta,
            std::complex<double>* C, const int* ldc);

double ddot_(int* n, const double* x, int* incx, const double* y, int* incy);
} // extern "C"

inline void matrix_product(char transa, char transb, int m, int n, int k, double alpha, double* a,
                           int lda, double* b, int ldb, double beta, double* c, int ldc) {
    if (m == 0 || n == 0 || k == 0)
        return;
    dgemm_(&transb, &transa, &n, &m, &k, &alpha, b, &ldb, a, &lda, &beta, c, &ldc);
}

inline void matrix_product(char transa, char transb, int m, int n, int k,
                           std::complex<double> alpha, std::complex<double>* a, int lda,
                           std::complex<double>* b, int ldb, std::complex<double> beta,
                           std::complex<double>* c, int ldc) {
    if (m == 0 || n == 0 || k == 0)
        return;
    zgemm_(&transb, &transa, &n, &m, &k, &alpha, b, &ldb, a, &lda, &beta, c, &ldc);
}

/// @brief // Add a scaled vector to another vector using BLAS daxpy routine Y = a * X + Y.
/// @param length
/// @param a
/// @param x
/// @param inc_x
/// @param y
/// @param inc_y
inline void add(size_t length, double a, const double* x, int inc_x, double* y, int inc_y) {
    int big_blocks = (int)(length / INT_MAX);
    int small_size = (int)(length % INT_MAX);
    for (int block = 0; block <= big_blocks; block++) {
        const double* x_s = &x[static_cast<size_t>(block) * inc_x * INT_MAX];
        double* y_s = &y[static_cast<size_t>(block) * inc_y * INT_MAX];
        signed int length_s = (block == big_blocks) ? small_size : INT_MAX;
        daxpy_(&length_s, &a, x_s, &inc_x, y_s, &inc_y);
    }
}

inline void add(size_t length, std::complex<double> a, const std::complex<double>* x, int inc_x,
                std::complex<double>* y, int inc_y) {
    int big_blocks = (int)(length / INT_MAX);
    int small_size = (int)(length % INT_MAX);
    for (int block = 0; block <= big_blocks; block++) {
        const std::complex<double>* x_s = &x[static_cast<size_t>(block) * inc_x * INT_MAX];
        std::complex<double>* y_s = &y[static_cast<size_t>(block) * inc_y * INT_MAX];
        signed int length_s = (block == big_blocks) ? small_size : INT_MAX;
        zaxpy_(&length_s, &a, x_s, &inc_x, y_s, &inc_y);
    }
}

inline double dot(size_t length, const double* const x, int inc_x, const double* const y,
                  int inc_y) {
    if (length == 0)
        return 0.0;

    double reg = 0.0;

    int big_blocks = (int)(length / INT_MAX);
    int small_size = (int)(length % INT_MAX);
    for (int block = 0; block <= big_blocks; block++) {
        const double* x_s = &x[static_cast<size_t>(block) * inc_x * INT_MAX];
        const double* y_s = &y[static_cast<size_t>(block) * inc_y * INT_MAX];
        signed int length_s = (block == big_blocks) ? small_size : INT_MAX;
        reg += ddot_(&length_s, x_s, &inc_x, y_s, &inc_y);
    }

    return reg;
}

} // namespace forte2