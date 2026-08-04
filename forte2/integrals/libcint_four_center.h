// This header defines libcint-backed four-center integral wrappers.
// It is only compiled when FORTE2_USE_LIBCINT is enabled.
#pragma once

#if FORTE2_USE_LIBCINT

#include <complex.h>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "integrals/libcint_compute.h"
#include "helpers/ndarray.h"

extern "C" {
#include <cint.h>

// Declare the libcint four-center two-electron integral in spherical harmonics.
// To add a new integral, consult the available Libcint functions,
// at https://github.com/sunqm/libcint/blob/master/include/cint_funcs.h
#define DECL_CINT_FUNC_SPH(name)                                                                   \
    int name##_sph(double* buf, int* dims, int* shls, int* atm, int natm, int* bas, int nbas,      \
                   double* env, CINTOpt* opt, double* cache);
DECL_CINT_FUNC_SPH(int2e)
#undef DECL_CINT_FUNC_SPH
}

namespace forte2 {
// Forte2 wrappers for the libcint four-center integrals. These build on the generic helpers in
// libcint_compute.h and match the layout of the libint2-backed coulomb_4c_{diagonal,pair_block}.

// Diagonal (mn|mn), row-major over the AO-pair index, length nbf * nbf.
inline np_vector cint_int2e_diagonal_sph(np_matrix_int atm, np_matrix_int bas, np_vector env) {
    return cint_int2e_diagonal(int2e_sph, atm, bas, env);
}

// Dense block (AB|CD) for the given bra and ket shell-pair lists.
inline np_matrix cint_int2e_pair_block_sph(np_matrix_int atm, np_matrix_int bas, np_vector env,
                                           np_matrix_int bra_pairs, np_matrix_int ket_pairs) {
    return cint_int2e_pair_block(int2e_sph, atm, bas, env, bra_pairs, ket_pairs);
}

} // namespace forte2

#endif // FORTE2_USE_LIBCINT
