#pragma once

#include <complex.h>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "helpers/ndarray.h"

extern "C" {
#include <cint.h>

int int1e_ovlp_sph(double* buf, int* dims, int* shls, int* atm, int natm, int* bas, int nbas,
                   double* env, CINTOpt* opt, double* cache);
int int1e_ovlp_spinor(double _Complex* buf, int* dims, int* shls, int* atm, int natm, int* bas,
                      int nbas, double* env, CINTOpt* opt, double* cache);
int int1e_kin_sph(double* buf, int* dims, int* shls, int* atm, int natm, int* bas, int nbas,
                  double* env, CINTOpt* opt, double* cache);
int int1e_kin_spinor(double _Complex* buf, int* dims, int* shls, int* atm, int natm, int* bas,
                     int nbas, double* env, CINTOpt* opt, double* cache);
int int1e_nuc_sph(double* buf, int* dims, int* shls, int* atm, int natm, int* bas, int nbas,
                  double* env, CINTOpt* opt, double* cache);
int int1e_nuc_spinor(double _Complex* buf, int* dims, int* shls, int* atm, int natm, int* bas,
                     int nbas, double* env, CINTOpt* opt, double* cache);
int int1e_spnucsp_spinor(double _Complex* buf, int* dims, int* shls, int* atm, int natm, int* bas,
                         int nbas, double* env, CINTOpt* opt, double* cache);
int int1e_r_sph(double* buf, int* dims, int* shls, int* atm, int natm, int* bas, int nbas,
                double* env, CINTOpt* opt, double* cache);
int int2c2e_sph(double* buf, int* dims, int* shls, int* atm, int natm, int* bas, int nbas,
                double* env, CINTOpt* opt, double* cache);
}

namespace forte2 {
np_tensor3_f cint_int1e_ovlp_sph(const std::vector<int>& shell_slice, np_matrix_int atm,
                                 np_matrix_int bas, np_vector env);
np_matrix_complex_f cint_int1e_ovlp_spinor(const std::vector<int>& shell_slice, np_matrix_int atm,
                                           np_matrix_int bas, np_vector env);
np_tensor3_f cint_int1e_kin_sph(const std::vector<int>& shell_slice, np_matrix_int atm,
                                np_matrix_int bas, np_vector env);
np_matrix_complex_f cint_int1e_kin_spinor(const std::vector<int>& shell_slice, np_matrix_int atm,
                                          np_matrix_int bas, np_vector env);
np_tensor3_f cint_int1e_nuc_sph(const std::vector<int>& shell_slice, np_matrix_int atm,
                                np_matrix_int bas, np_vector env);
np_matrix_complex_f cint_int1e_nuc_spinor(const std::vector<int>& shell_slice, np_matrix_int atm,
                                          np_matrix_int bas, np_vector env);
np_matrix_complex_f cint_int1e_spnucsp_spinor(const std::vector<int>& shell_slice,
                                              np_matrix_int atm, np_matrix_int bas, np_vector env);
np_tensor3_f cint_int1e_r_sph(const std::vector<int>& shell_slice, np_matrix_int atm,
                              np_matrix_int bas, np_vector env);
np_tensor3_f cint_int2c2e_sph(const std::vector<int>& shell_slice, np_matrix_int atm,
                              np_matrix_int bas, np_vector env);
} // namespace forte2