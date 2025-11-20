#pragma once

#include <complex.h>
#include <variant>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "helpers/ndarray.h"

#if FORTE2_USE_LIBCINT
extern "C" {
#include <cint.h>

int CINTcgto_spheric(const int i, const int* bas);
int CINTcgto_spinor(const int i, const int* bas);
}
#endif

// Standard Libcint function pointer type
using CIntorFunc = int (*)(double* buf, int* dims, int* shls, int* atm, int natm, int* bas,
                           int nbas, double* env, CINTOpt* opt, double* cache);
using CIntorFuncSpinor = int (*)(double _Complex* buf, int* dims, int* shls, int* atm, int natm,
                                 int* bas, int nbas, double* env, CINTOpt* opt, double* cache);

namespace forte2 {

// templated function to compute two-center integrals with M components
// (i.e., 1 for scalar integrals, 3 for dipoles integrals, etc.)
template <std::size_t M>
np_tensor3_f cint_int2c(CIntorFunc intor, const std::vector<int>& shell_slice, np_matrix_int atm,
                        np_matrix_int bas, np_vector env) {

    const int ish_0 = static_cast<int>(shell_slice[0]);
    const int ish_1 = static_cast<int>(shell_slice[1]);
    const int jsh_0 = static_cast<int>(shell_slice[2]);
    const int jsh_1 = static_cast<int>(shell_slice[3]);
    const int nish = ish_1 - ish_0;
    const int njsh = jsh_1 - jsh_0;

    int natm = atm.shape(0);
    int nbas = bas.shape(0);

    auto* atm_data = atm.data();
    auto* bas_data = bas.data();
    auto* env_data = env.data();

    std::vector<int> ao_offset(nbas + 1, 0);
    for (int i = 0; i < nbas; ++i) {
        ao_offset[i + 1] = ao_offset[i] + CINTcgto_spheric(i, bas_data);
    }

    const int nao_i = ao_offset[ish_1] - ao_offset[ish_0];
    const int nao_j = ao_offset[jsh_1] - ao_offset[jsh_0];

    auto ints = make_zeros<nb::numpy, double, 3, nb::f_contig>(
        std::array<size_t, 3>{static_cast<size_t>(nao_i), static_cast<size_t>(nao_j), M});

    double* buf = ints.data();

    int shells[2];
    int dims[2] = {nao_i, nao_j};

    int shell_offset_i, shell_offset_j;

    for (int i = ish_0; i < ish_1; ++i) {
        shells[0] = i;
        shell_offset_i = ao_offset[i] - ao_offset[ish_0];
        for (int j = jsh_0; j < jsh_1; ++j) {
            shells[1] = j;
            shell_offset_j = ao_offset[j] - ao_offset[jsh_0];
            // Fortran ordering: i changes fastest
            auto buf_ij = buf + shell_offset_i + shell_offset_j * nao_i;
            intor(buf_ij, dims, shells, atm_data, natm, bas_data, nbas, env_data, NULL, NULL);
        }
    }

    return ints;
}

template <std::size_t M>
np_tensor3_complex_f cint_int2c_spinor(CIntorFuncSpinor intor, const std::vector<int>& shell_slice,
                                       np_matrix_int atm, np_matrix_int bas, np_vector env) {

    const int ish_0 = static_cast<int>(shell_slice[0]);
    const int ish_1 = static_cast<int>(shell_slice[1]);
    const int jsh_0 = static_cast<int>(shell_slice[2]);
    const int jsh_1 = static_cast<int>(shell_slice[3]);
    const int nish = ish_1 - ish_0;
    const int njsh = jsh_1 - jsh_0;

    int natm = atm.shape(0);
    int nbas = bas.shape(0);

    auto* atm_data = atm.data();
    auto* bas_data = bas.data();
    auto* env_data = env.data();

    std::vector<int> ao_offset(nbas + 1, 0);
    for (int i = 0; i < nbas; ++i) {
        ao_offset[i + 1] = ao_offset[i] + CINTcgto_spinor(i, bas_data);
    }

    const int nao_i = ao_offset[ish_1] - ao_offset[ish_0];
    const int nao_j = ao_offset[jsh_1] - ao_offset[jsh_0];

    auto ints = make_zeros<nb::numpy, std::complex<double>, 3, nb::f_contig>(
        std::array<size_t, 3>{static_cast<size_t>(nao_i), static_cast<size_t>(nao_j), M});

    auto buf = reinterpret_cast<double _Complex*>(ints.data());

    int shells[2];
    int dims[2] = {nao_i, nao_j};

    int shell_offset_i, shell_offset_j;

    for (int i = ish_0; i < ish_1; ++i) {
        shells[0] = i;
        shell_offset_i = ao_offset[i] - ao_offset[ish_0];
        for (int j = jsh_0; j < jsh_1; ++j) {
            shells[1] = j;
            shell_offset_j = ao_offset[j] - ao_offset[jsh_0];
            // Fortran ordering: i changes fastest
            auto buf_ij = buf + shell_offset_i + shell_offset_j * nao_i;
            intor(buf_ij, dims, shells, atm_data, natm, bas_data, nbas, env_data, NULL, NULL);
        }
    }

    return ints;
}

// function to compute three-center integrals
np_tensor3_f cint_int3c(CIntorFunc intor, const std::vector<int>& shell_slice, np_matrix_int atm,
                        np_matrix_int bas, np_vector env) {
    const int ish_0 = static_cast<int>(shell_slice[0]);
    const int ish_1 = static_cast<int>(shell_slice[1]);
    const int jsh_0 = static_cast<int>(shell_slice[2]);
    const int jsh_1 = static_cast<int>(shell_slice[3]);
    const int ksh_0 = static_cast<int>(shell_slice[4]);
    const int ksh_1 = static_cast<int>(shell_slice[5]);

    const int nish = ish_1 - ish_0;
    const int njsh = jsh_1 - jsh_0;
    const int nksh = ksh_1 - ksh_0;

    int natm = atm.shape(0);
    int nbas = bas.shape(0);

    auto* atm_data = atm.data();
    auto* bas_data = bas.data();
    auto* env_data = env.data();

    std::vector<int> ao_offset(nbas + 1, 0);
    for (int i = 0; i < nbas; ++i) {
        ao_offset[i + 1] = ao_offset[i] + CINTcgto_spheric(i, bas_data);
    }

    const int nao_i = ao_offset[ish_1] - ao_offset[ish_0];
    const int nao_j = ao_offset[jsh_1] - ao_offset[jsh_0];
    const int nao_k = ao_offset[ksh_1] - ao_offset[ksh_0];

    auto ints = make_zeros<nb::numpy, double, 3, nb::f_contig>(std::array<size_t, 3>{
        static_cast<size_t>(nao_i), static_cast<size_t>(nao_j), static_cast<size_t>(nao_k)});

    double* buf = ints.data();

    int shells[3];
    int dims[3] = {nao_i, nao_j, nao_k};

    int shell_offset_i, shell_offset_j, shell_offset_k;

    for (int i = ish_0; i < ish_1; ++i) {
        shells[0] = i;
        shell_offset_i = ao_offset[i] - ao_offset[ish_0];
        for (int j = jsh_0; j < jsh_1; ++j) {
            shells[1] = j;
            shell_offset_j = ao_offset[j] - ao_offset[jsh_0];
            for (int k = ksh_0; k < ksh_1; ++k) {
                shells[2] = k;
                shell_offset_k = ao_offset[k] - ao_offset[ksh_0];
                // Fortran ordering: i changes fastest
                auto buf_ijk =
                    buf + shell_offset_i + shell_offset_j * nao_i + shell_offset_k * nao_i * nao_j;
                intor(buf_ijk, dims, shells, atm_data, natm, bas_data, nbas, env_data, NULL, NULL);
            }
        }
    }

    return ints;
}

} // namespace forte2
