#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "helpers/ndarray.h"

extern "C" {
#include <cint.h>

int int1e_ovlp_sph(double* buf, int* dims, int* shls, int* atm, int natm, int* bas, int nbas,
                   double* env, CINTOpt* opt, double* cache);
int int1e_kin_sph(double* buf, int* dims, int* shls, int* atm, int natm, int* bas, int nbas,
                  double* env, CINTOpt* opt, double* cache);
int int1e_nuc_sph(double* buf, int* dims, int* shls, int* atm, int natm, int* bas, int nbas,
                  double* env, CINTOpt* opt, double* cache);
}

namespace forte2 {
np_matrix_f cint_int1e_ovlp_sph(const size_t nao, np_matrix_int atm, np_matrix_int bas,
                                np_vector env);
np_matrix_f cint_int1e_kin_sph(const size_t nao, np_matrix_int atm, np_matrix_int bas,
                               np_vector env);
np_matrix_f cint_int1e_nuc_sph(const size_t nao, np_matrix_int atm, np_matrix_int bas,
                               np_vector env);
} // namespace forte2