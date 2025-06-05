
namespace forte2 {

extern "C" {
// Note: all Fortran BLAS routines expect single‐letter char arguments by address,
// and integer args are usually Fortran‐INTEGER (often 32‐bit). Adjust if your BLAS uses 64‐bit
// ints.
void dgemm_(const char* transA, const char* transB, const int* M, const int* N, const int* K,
            const double* alpha, const double* A, const int* lda, const double* B, const int* ldb,
            const double* beta, double* C, const int* ldc);
}

inline void matrix_product(char transa, char transb, int m, int n, int k, double alpha, double* a,
                           int lda, double* b, int ldb, double beta, double* c, int ldc) {
    if (m == 0 || n == 0 || k == 0)
        return;
    dgemm_(&transb, &transa, &n, &m, &k, &alpha, b, &ldb, a, &lda, &beta, c, &ldc);
}

} // namespace forte2