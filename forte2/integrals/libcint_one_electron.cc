#include "integrals/libcint_compute.h"
#include "integrals/libcint_one_electron.h"

namespace forte2 {
np_matrix_f cint_int1e_ovlp_sph(const std::vector<int>& shell_slice, np_matrix_int atm,
                                np_matrix_int bas, np_vector env) {
    return cint_int1e_1comp(int1e_ovlp_sph, shell_slice, atm, bas, env);
}

np_matrix_f cint_int1e_kin_sph(const std::vector<int>& shell_slice, np_matrix_int atm,
                               np_matrix_int bas, np_vector env) {
    return cint_int1e_1comp(int1e_kin_sph, shell_slice, atm, bas, env);
}

np_matrix_f cint_int1e_nuc_sph(const std::vector<int>& shell_slice, np_matrix_int atm,
                               np_matrix_int bas, np_vector env) {
    return cint_int1e_1comp(int1e_nuc_sph, shell_slice, atm, bas, env);
}
} // namespace forte2