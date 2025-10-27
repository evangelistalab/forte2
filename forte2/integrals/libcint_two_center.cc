#include "integrals/libcint_compute.h"
#include "integrals/libcint_two_center.h"

namespace forte2 {
np_tensor3_f cint_int1e_ovlp_sph(const std::vector<int>& shell_slice, np_matrix_int atm,
                                 np_matrix_int bas, np_vector env) {
    return cint_int2c<1>(int1e_ovlp_sph, shell_slice, atm, bas, env);
}

np_matrix_complex_f cint_int1e_ovlp_spinor(const std::vector<int>& shell_slice, np_matrix_int atm,
                                           np_matrix_int bas, np_vector env) {
    return cint_int2c_1comp_spinor(int1e_ovlp_spinor, shell_slice, atm, bas, env);
}

np_tensor3_f cint_int1e_kin_sph(const std::vector<int>& shell_slice, np_matrix_int atm,
                                np_matrix_int bas, np_vector env) {
    return cint_int2c<1>(int1e_kin_sph, shell_slice, atm, bas, env);
}

np_matrix_complex_f cint_int1e_kin_spinor(const std::vector<int>& shell_slice, np_matrix_int atm,
                                          np_matrix_int bas, np_vector env) {
    return cint_int2c_1comp_spinor(int1e_kin_spinor, shell_slice, atm, bas, env);
}

np_tensor3_f cint_int1e_nuc_sph(const std::vector<int>& shell_slice, np_matrix_int atm,
                                np_matrix_int bas, np_vector env) {
    return cint_int2c<1>(int1e_nuc_sph, shell_slice, atm, bas, env);
}

np_matrix_complex_f cint_int1e_nuc_spinor(const std::vector<int>& shell_slice, np_matrix_int atm,
                                          np_matrix_int bas, np_vector env) {
    return cint_int2c_1comp_spinor(int1e_nuc_spinor, shell_slice, atm, bas, env);
}

np_matrix_complex_f cint_int1e_spnucsp_spinor(const std::vector<int>& shell_slice,
                                              np_matrix_int atm, np_matrix_int bas, np_vector env) {
    return cint_int2c_1comp_spinor(int1e_spnucsp_spinor, shell_slice, atm, bas, env);
}

np_tensor3_f cint_int1e_r_sph(const std::vector<int>& shell_slice, np_matrix_int atm,
                              np_matrix_int bas, np_vector env) {
    return cint_int2c<3>(int1e_r_sph, shell_slice, atm, bas, env);
}

np_tensor3_f cint_int2c2e_sph(const std::vector<int>& shell_slice, np_matrix_int atm,
                              np_matrix_int bas, np_vector env) {
    return cint_int2c<1>(int2c2e_sph, shell_slice, atm, bas, env);
}
} // namespace forte2