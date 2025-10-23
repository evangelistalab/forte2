#include "integrals/libcint_compute.h"
#include "integrals/libcint_one_electron.h"

namespace forte2 {
np_matrix_f cint_int1e_ovlp_sph(const size_t nao, np_matrix_int atm, np_matrix_int bas,
                                np_vector env) {
    return cint_int1e_1comp(int1e_ovlp_sph, nao, atm, bas, env);
}

np_matrix_f cint_int1e_kin_sph(const size_t nao, np_matrix_int atm, np_matrix_int bas,
                               np_vector env) {
    return cint_int1e_1comp(int1e_kin_sph, nao, atm, bas, env);
}

np_matrix_f cint_int1e_nuc_sph(const size_t nao, np_matrix_int atm, np_matrix_int bas,
                               np_vector env) {
    return cint_int1e_1comp(int1e_nuc_sph, nao, atm, bas, env);
}
} // namespace forte2