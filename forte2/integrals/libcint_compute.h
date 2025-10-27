#pragma once

#include <complex.h>
#include <variant>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "helpers/ndarray.h"

extern "C" {
#include <cint.h>

int CINTcgto_spheric(const int i, const int* bas);
int CINTcgto_spinor(const int i, const int* bas);
}

// Standard Libcint function pointer type
using CIntorFunc = int (*)(double* buf, int* dims, int* shls, int* atm, int natm, int* bas,
                           int nbas, double* env, CINTOpt* opt, double* cache);
using CIntorFuncSpinor = int (*)(double _Complex* buf, int* dims, int* shls, int* atm, int natm,
                                 int* bas, int nbas, double* env, CINTOpt* opt, double* cache);

namespace forte2 {

// templated function to compute two-center integrals with M components 
// (i.e., 1 for scalar integrals, 3 for dipoles integrals, etc.)
template <std::size_t M>
np_tensor3_f cint_int2c(CIntorFunc intor, const std::vector<int>& shell_slice,
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
        ao_offset[i + 1] = ao_offset[i] + CINTcgto_spheric(i, bas_data);
    }

    const int nao_i = ao_offset[ish_1] - ao_offset[ish_0];
    const int nao_j = ao_offset[jsh_1] - ao_offset[jsh_0];

    auto ints = make_zeros<nb::numpy, double, 3, nb::f_contig>(
        std::array<size_t, 3>{static_cast<size_t>(nao_i), static_cast<size_t>(nao_j), M});

    double* buf = ints.data();

    int shells[2];
    int dims[2] = {nao_i, nao_j};

    for (int i = 0; i < nish; ++i) {
        for (int j = 0; j < njsh; ++j) {
            shells[0] = i;
            shells[1] = j;
            // Fortran ordering: i changes fastest
            auto buf_ij = buf + ao_offset[i] + ao_offset[j] * nao_i;
            intor(buf_ij, dims, shells, atm_data, natm, bas_data, nbas, env_data, NULL, NULL);
        }
    }

    return ints;
}

np_matrix_complex_f cint_int2c_1comp_spinor(CIntorFuncSpinor intor,
                                            const std::vector<int>& shell_slice, np_matrix_int atm,
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
        ao_offset[i + 1] = ao_offset[i] + CINTcgto_spinor(i, bas_data);
    }

    const int nao_i = ao_offset[ish_1] - ao_offset[ish_0];
    const int nao_j = ao_offset[jsh_1] - ao_offset[jsh_0];

    auto ints = make_zeros<nb::numpy, std::complex<double>, 2, nb::f_contig>(
        std::array<size_t, 2>{static_cast<size_t>(nao_i), static_cast<size_t>(nao_j)});

    auto buf = reinterpret_cast<double _Complex*>(ints.data());

    int shells[2];
    int dims[2] = {nao_i, nao_j};

    for (int i = 0; i < nish; ++i) {
        for (int j = 0; j < njsh; ++j) {
            shells[0] = i;
            shells[1] = j;
            // Fortran ordering: i changes fastest
            auto buf_ij = buf + ao_offset[i] + ao_offset[j] * nao_i;
            intor(buf_ij, dims, shells, atm_data, natm, bas_data, nbas, env_data, NULL, NULL);
        }
    }

    return ints;
}
} // namespace forte2