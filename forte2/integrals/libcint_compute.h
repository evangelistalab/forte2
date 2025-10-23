#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "helpers/ndarray.h"

extern "C" {
#include <cint.h>

int CINTcgto_spheric(const int i, const int* bas);
}

using CIntorFunc = int (*)(double* buf, int* dims, int* shls, int* atm, int natm, int* bas,
                           int nbas, double* env, CINTOpt* opt, double* cache);

namespace forte2 {

np_matrix_f cint_int1e_1comp(CIntorFunc intor, const size_t nao, np_matrix_int atm,
                             np_matrix_int bas, np_vector env) {
    int natm = atm.shape(0);
    int nbas = bas.shape(0);

    auto* atm_data = atm.data();
    auto* bas_data = bas.data();
    auto* env_data = env.data();

    std::vector<int> ao_offset(nbas + 1, 0);
    for (int i = 0; i < nbas; ++i) {
        ao_offset[i + 1] = ao_offset[i] + CINTcgto_spheric(i, bas_data);
    }

    auto ints = make_zeros<nb::numpy, double, 2, nb::f_contig>(std::array<size_t, 2>{nao, nao});
    double* buf = ints.data();

    int shells[2];
    int dims[2] = {static_cast<int>(nao), static_cast<int>(nao)};

    for (int i = 0; i < nbas; ++i) {
        // dims[0] = ao_offset[i + 1] - ao_offset[i];
        for (int j = 0; j < nbas; ++j) {
            // dims[1] = ao_offset[j + 1] - ao_offset[j];
            shells[0] = i;
            shells[1] = j;
            // Fortran ordering: i changes fastest
            double* buf_ij = buf + ao_offset[i] + ao_offset[j] * nao;
            intor(buf_ij, dims, shells, atm_data, natm, bas_data, nbas, env_data, NULL, NULL);
        }
    }

    return ints;
}
} // namespace forte2