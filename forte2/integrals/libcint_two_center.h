// This header defines libcint-backed two-center integral wrappers.
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

// Declare APIs to Libcint functions: they have identical signatures,
// except for the buffer type (double* for spherical, double _Complex* for spinor)

// These generate declarations of the form:
// int int1e_ovlp_sph(double* buf, int* dims, int* shls, int* atm, int natm, int* bas, int nbas,
//                   double* env, CINTOpt* opt, double* cache);

// To add a new integral, consult the available Libcint functions,
// at https://github.com/sunqm/libcint/blob/master/include/cint_funcs.h
// and see the number of components it will require,
// at https://github.com/pyscf/pyscf/blob/master/pyscf/gto/moleintor.py
#define DECL_CINT_FUNC_SPH(name)                                                                   \
    int name##_sph(double* buf, int* dims, int* shls, int* atm, int natm, int* bas, int nbas,      \
                   double* env, CINTOpt* opt, double* cache);
DECL_CINT_FUNC_SPH(int1e_ovlp)
DECL_CINT_FUNC_SPH(int1e_kin)
DECL_CINT_FUNC_SPH(int1e_nuc)
DECL_CINT_FUNC_SPH(int1e_spnucsp)
DECL_CINT_FUNC_SPH(int1e_r)
DECL_CINT_FUNC_SPH(int2c2e)
#undef DECL_CINT_FUNC_SPH

#define DECL_CINT_FUNC_SPINOR(name)                                                                \
    int name##_spinor(double _Complex* buf, int* dims, int* shls, int* atm, int natm, int* bas,    \
                      int nbas, double* env, CINTOpt* opt, double* cache);
DECL_CINT_FUNC_SPINOR(int1e_ovlp)
DECL_CINT_FUNC_SPINOR(int1e_kin)
DECL_CINT_FUNC_SPINOR(int1e_nuc)
DECL_CINT_FUNC_SPINOR(int1e_spnucsp)
#undef DECL_CINT_FUNC_SPINOR
}

namespace forte2 {
// define the Forte2 wrappers for the Libcint integrals
// These generate definitions of the form:
// ndarray<double, 3, nb::f_contig> cint_int1e_ovlp_sph(const std::vector<int>& shell_slice,
// ndarray<int, 2> atm, ndarray<int, 2> bas,
//                                 ndarray<double, 1> env) {
//     return cint_int2c<1>(int1e_ovlp_sph, shell_slice, atm, bas, env);
// }
#define DECL_CINT_FORTE2_FUNC_SPH(name, comp)                                                      \
    ndarray<double, 3, nb::f_contig> cint_##name##_sph(const std::vector<int>& shell_slice,        \
                                                       ndarray<int, 2> atm, ndarray<int, 2> bas,   \
                                                       ndarray<double, 1> env) {                   \
        return cint_int2c<comp>(name##_sph, shell_slice, atm, bas, env);                           \
    }

DECL_CINT_FORTE2_FUNC_SPH(int1e_ovlp, 1)
DECL_CINT_FORTE2_FUNC_SPH(int1e_kin, 1)
DECL_CINT_FORTE2_FUNC_SPH(int1e_nuc, 1)
DECL_CINT_FORTE2_FUNC_SPH(int1e_spnucsp, 4)
DECL_CINT_FORTE2_FUNC_SPH(int1e_r, 3)
DECL_CINT_FORTE2_FUNC_SPH(int2c2e, 1)
#undef DECL_CINT_FORTE2_FUNC_SPH

#define DECL_CINT_FORTE2_FUNC_SPINOR(name, comp)                                                   \
    ndarray<std::complex<double>, 3, nb::f_contig> cint_##name##_spinor(                           \
        const std::vector<int>& shell_slice, ndarray<int, 2> atm, ndarray<int, 2> bas,             \
        ndarray<double, 1> env) {                                                                  \
        return cint_int2c_spinor<comp>(name##_spinor, shell_slice, atm, bas, env);                 \
    }
DECL_CINT_FORTE2_FUNC_SPINOR(int1e_ovlp, 1)
DECL_CINT_FORTE2_FUNC_SPINOR(int1e_kin, 1)
DECL_CINT_FORTE2_FUNC_SPINOR(int1e_nuc, 1)
DECL_CINT_FORTE2_FUNC_SPINOR(int1e_spnucsp, 1)
#undef DECL_CINT_FORTE2_FUNC_SPINOR
} // namespace forte2

#endif // FORTE2_USE_LIBCINT
