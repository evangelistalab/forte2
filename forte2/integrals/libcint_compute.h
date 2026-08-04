#pragma once
#include <complex.h>
#include <variant>
#include <future>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "helpers/parallel.h"
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
np_tensor3_c cint_int2c(CIntorFunc intor, const std::vector<int>& shell_slice, np_matrix_int atm,
                        np_matrix_int bas, np_vector env) {

    const int jsh_0 = static_cast<int>(shell_slice[0]);
    const int jsh_1 = static_cast<int>(shell_slice[1]);
    const int ish_0 = static_cast<int>(shell_slice[2]);
    const int ish_1 = static_cast<int>(shell_slice[3]);
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

    return fortran_to_c<nb::numpy, double, 3>(ints);
}

template <std::size_t M>
np_tensor3_complex_c cint_int2c_spinor(CIntorFuncSpinor intor, const std::vector<int>& shell_slice,
                                       np_matrix_int atm, np_matrix_int bas, np_vector env) {

    const int jsh_0 = static_cast<int>(shell_slice[0]);
    const int jsh_1 = static_cast<int>(shell_slice[1]);
    const int ish_0 = static_cast<int>(shell_slice[2]);
    const int ish_1 = static_cast<int>(shell_slice[3]);
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

    return fortran_to_c<nb::numpy, std::complex<double>, 3>(ints);
}

void fill_3c_sym(np_tensor3_f& ints) {
    auto data = ints.data();
    const auto nsym = ints.shape(0);
    const auto nk = ints.shape(2);

    // The symmetry is in the i and j indices
    // Fortran looping: k changes slowest, then j, then i
    // This will be slightly wasteful since we will overwrite values in
    // the upper triangulars of the diagonal shell pairs
    parallel_for(nk, [&](std::size_t k) {
        for (std::size_t j = 0; j < nsym; ++j) {
            for (std::size_t i = j + 1; i < nsym; ++i) {
                data[i + j * nsym + k * nsym * nsym] = data[j + i * nsym + k * nsym * nsym];
            }
        }
    });
}

// function to compute three-center integrals
np_tensor3_c cint_int3c(CIntorFunc intor, const std::vector<int>& shell_slice, np_matrix_int atm,
                        np_matrix_int bas, np_vector env, np_tensor3_c& ints) {
    const int ksh_0 = static_cast<int>(shell_slice[0]);
    const int ksh_1 = static_cast<int>(shell_slice[1]);
    const int jsh_0 = static_cast<int>(shell_slice[2]);
    const int jsh_1 = static_cast<int>(shell_slice[3]);
    const int ish_0 = static_cast<int>(shell_slice[4]);
    const int ish_1 = static_cast<int>(shell_slice[5]);

    // Whether i and j shells are identical, if they are we can exploit symmetry
    bool ij_sym = (ish_0 == jsh_0) && (ish_1 == jsh_1);

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

    if (ints.shape(0) != static_cast<size_t>(nao_k) ||
        ints.shape(1) != static_cast<size_t>(nao_j) ||
        ints.shape(2) != static_cast<size_t>(nao_i)) {
        std::string msg = "Buffer shape must be (";
        msg += std::to_string(nao_k) + ", " + std::to_string(nao_j) + ", " + std::to_string(nao_i) +
               ")";
        throw std::invalid_argument(msg);
    }

    // ints.shape = {k, j, i}
    // ints_f.shape = {i, j, k}
    auto ints_f = c_to_fortran<nb::numpy, double, 3>(ints);

    double* buf = ints_f.data();
    int dims[3] = {nao_i, nao_j, nao_k};

    // we collapse the k and j loops to hopefully get better load balancing
    auto kernel_ijsym = [&](std::size_t kj) {
        int shells[3];
        int shell_offset_i, shell_offset_j, shell_offset_k;
        // kj = j + k * njsh
        std::size_t j = jsh_0 + kj % njsh;
        std::size_t k = ksh_0 + kj / njsh;
        // Fortran looping: k changes slowest so it is the outermost loop
        shells[2] = static_cast<int>(k);
        shell_offset_k = ao_offset[k] - ao_offset[ksh_0];
        shells[1] = static_cast<int>(j);
        shell_offset_j = ao_offset[j] - ao_offset[jsh_0];
        for (int i = ish_0; i <= j; ++i) {
            shells[0] = i;
            shell_offset_i = ao_offset[i] - ao_offset[ish_0];
            // Fortran ordering: i changes fastest
            auto buf_ijk =
                buf + shell_offset_i + shell_offset_j * nao_i + shell_offset_k * nao_i * nao_j;
            intor(buf_ijk, dims, shells, atm_data, natm, bas_data, nbas, env_data, NULL, NULL);
        }
    };

    auto kernel_nosym = [&](std::size_t kj) {
        int shells[3];
        int shell_offset_i, shell_offset_j, shell_offset_k;
        // kj = j + k * njsh
        std::size_t j = jsh_0 + kj % njsh;
        std::size_t k = ksh_0 + kj / njsh;
        // Fortran looping: k changes slowest so it is the outermost loop
        shells[2] = static_cast<int>(k);
        shell_offset_k = ao_offset[k] - ao_offset[ksh_0];
        shells[1] = static_cast<int>(j);
        shell_offset_j = ao_offset[j] - ao_offset[jsh_0];
        for (int i = ish_0; i < ish_1; ++i) {
            shells[0] = i;
            shell_offset_i = ao_offset[i] - ao_offset[ish_0];
            // Fortran ordering: i changes fastest
            auto buf_ijk =
                buf + shell_offset_i + shell_offset_j * nao_i + shell_offset_k * nao_i * nao_j;
            intor(buf_ijk, dims, shells, atm_data, natm, bas_data, nbas, env_data, NULL, NULL);
        }
    };

    std::size_t kj_0 = 0;
    std::size_t kj_1 = static_cast<std::size_t>(njsh) * static_cast<std::size_t>(nksh);

    if (ij_sym) {
        parallel_for(kj_0, kj_1, kernel_ijsym);
    } else {
        parallel_for(kj_0, kj_1, kernel_nosym);
    }

    if (ij_sym) {
        fill_3c_sym(ints_f);
    }

    // no copy of underlying storage, but python object changed
    return fortran_to_c<nb::numpy, double, 3>(ints_f);
}

// function to compute three-center integrals
np_tensor3_c cint_int3c(CIntorFunc intor, const std::vector<int>& shell_slice, np_matrix_int atm,
                        np_matrix_int bas, np_vector env) {
    const int ksh_0 = static_cast<int>(shell_slice[0]);
    const int ksh_1 = static_cast<int>(shell_slice[1]);
    const int jsh_0 = static_cast<int>(shell_slice[2]);
    const int jsh_1 = static_cast<int>(shell_slice[3]);
    const int ish_0 = static_cast<int>(shell_slice[4]);
    const int ish_1 = static_cast<int>(shell_slice[5]);

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

    auto ints = make_zeros<nb::numpy, double, 3, nb::c_contig>(std::array<size_t, 3>{
        static_cast<size_t>(nao_k), static_cast<size_t>(nao_j), static_cast<size_t>(nao_i)});

    return cint_int3c(intor, shell_slice, atm, bas, env, ints);
}

/// @brief Compute a multi-component three-center integral tensor.
/// @details Libcint stores components after the three AO dimensions. The returned C-contiguous
/// tensor reverses that layout to `(component, k, j, i)`, consistent with the component-first
/// convention used by the other Python-facing Libcint wrappers.
template <std::size_t M>
np_tensor4_c cint_int3c_components(CIntorFunc intor, const std::vector<int>& shell_slice,
                                   np_matrix_int atm, np_matrix_int bas, np_vector env) {
    const int ksh_0 = static_cast<int>(shell_slice[0]);
    const int ksh_1 = static_cast<int>(shell_slice[1]);
    const int jsh_0 = static_cast<int>(shell_slice[2]);
    const int jsh_1 = static_cast<int>(shell_slice[3]);
    const int ish_0 = static_cast<int>(shell_slice[4]);
    const int ish_1 = static_cast<int>(shell_slice[5]);

    const int njsh = jsh_1 - jsh_0;
    const int nksh = ksh_1 - ksh_0;
    const int natm = static_cast<int>(atm.shape(0));
    const int nbas = static_cast<int>(bas.shape(0));

    auto* atm_data = atm.data();
    auto* bas_data = bas.data();
    auto* env_data = env.data();

    std::vector<int> ao_offset(nbas + 1, 0);
    for (int shell = 0; shell < nbas; ++shell) {
        ao_offset[shell + 1] = ao_offset[shell] + CINTcgto_spheric(shell, bas_data);
    }

    const int nao_i = ao_offset[ish_1] - ao_offset[ish_0];
    const int nao_j = ao_offset[jsh_1] - ao_offset[jsh_0];
    const int nao_k = ao_offset[ksh_1] - ao_offset[ksh_0];

    auto ints = make_zeros<nb::numpy, double, 4, nb::c_contig>(std::array<size_t, 4>{
        M, static_cast<size_t>(nao_k), static_cast<size_t>(nao_j), static_cast<size_t>(nao_i)});
    auto ints_f = c_to_fortran<nb::numpy, double, 4>(ints);

    double* buf = ints_f.data();
    int dims[3] = {nao_i, nao_j, nao_k};

    auto kernel = [&](std::size_t kj) {
        int shells[3];
        const int j = jsh_0 + static_cast<int>(kj % static_cast<std::size_t>(njsh));
        const int k = ksh_0 + static_cast<int>(kj / static_cast<std::size_t>(njsh));
        shells[2] = k;
        shells[1] = j;

        const int shell_offset_j = ao_offset[j] - ao_offset[jsh_0];
        const int shell_offset_k = ao_offset[k] - ao_offset[ksh_0];
        for (int i = ish_0; i < ish_1; ++i) {
            shells[0] = i;
            const int shell_offset_i = ao_offset[i] - ao_offset[ish_0];
            auto* buf_ijk =
                buf + shell_offset_i + shell_offset_j * nao_i + shell_offset_k * nao_i * nao_j;
            intor(buf_ijk, dims, shells, atm_data, natm, bas_data, nbas, env_data, NULL, NULL);
        }
    };

    parallel_for(static_cast<std::size_t>(njsh) * static_cast<std::size_t>(nksh), kernel);
    return fortran_to_c<nb::numpy, double, 4>(ints_f);
}

// Compute the diagonal (mn|mn) of the four-center two-electron integral matrix over all shells in
// `bas`, laid out row-major over the AO-pair index p = m * nbf + n (length nbf * nbf). This is the
// libcint analogue of compute_two_electron_4c_diagonal; the result matches it element-wise because
// forte2 hands both libraries the same spherical-AO ordering.
inline np_vector cint_int2e_diagonal(CIntorFunc intor, np_matrix_int atm, np_matrix_int bas,
                                     np_vector env) {
    const int natm = atm.shape(0);
    const int nbas = bas.shape(0);
    auto* atm_data = atm.data();
    auto* bas_data = bas.data();
    auto* env_data = env.data();

    // Per-shell AO counts and offsets (spherical harmonics).
    std::vector<int> ao_size(nbas), ao_offset(nbas + 1, 0);
    int max_size = 0;
    for (int s = 0; s < nbas; ++s) {
        ao_size[s] = CINTcgto_spheric(s, bas_data);
        ao_offset[s + 1] = ao_offset[s] + ao_size[s];
        max_size = std::max(max_size, ao_size[s]);
    }
    const std::size_t nbf = static_cast<std::size_t>(ao_offset[nbas]);

    auto diag = make_zeros<nb::numpy, double, 1>({nbf * nbf});
    double* data = diag.data();

    // Parallelize over the first shell; each chunk owns a scratch quartet buffer and writes
    // disjoint rows of the diagonal.
    auto kernel = [&](std::size_t s1_begin, std::size_t s1_end) {
        std::vector<double> buf(static_cast<std::size_t>(max_size) * max_size * max_size * max_size);
        for (std::size_t s1 = s1_begin; s1 < s1_end; ++s1) {
            const int n1 = ao_size[s1];
            const int f1 = ao_offset[s1];
            for (int s2 = 0; s2 < nbas; ++s2) {
                const int n2 = ao_size[s2];
                const int f2 = ao_offset[s2];
                int shells[4] = {static_cast<int>(s1), s2, static_cast<int>(s1), s2};
                int dims[4] = {n1, n2, n1, n2};
                intor(buf.data(), dims, shells, atm_data, natm, bas_data, nbas, env_data, NULL,
                      NULL);
                // libcint buffer is Fortran-order: idx(a,b,c,d) = a + b*n1 + c*n1*n2 + d*n1*n2*n1.
                // Diagonal (a,b,a,b) reduces to (1 + n1*n2) * (a + b*n1).
                const std::size_t stride = static_cast<std::size_t>(1) + n1 * n2;
                for (int a = 0; a < n1; ++a) {
                    for (int b = 0; b < n2; ++b) {
                        data[(f1 + a) * nbf + (f2 + b)] = buf[stride * (a + b * n1)];
                    }
                }
            }
        }
    };

    parallel_for_chunked(static_cast<std::size_t>(nbas), kernel);

    return diag;
}

// Compute a dense block (AB|CD) of the four-center two-electron integral matrix for a list of bra
// shell-pairs (rows) and ket shell-pairs (columns). Output shape is (n_bra_ao_pairs,
// n_ket_ao_pairs), row-major, matching compute_two_electron_4c_pair_block. Within a shell-pair the
// AO-pair order is iA * nB + iB; blocks are concatenated in the given shell-pair order.
inline np_matrix cint_int2e_pair_block(CIntorFunc intor, np_matrix_int atm, np_matrix_int bas,
                                       np_vector env, np_matrix_int bra_pairs,
                                       np_matrix_int ket_pairs) {
    if (bra_pairs.shape(1) != 2 || ket_pairs.shape(1) != 2) {
        throw std::invalid_argument("bra_pairs and ket_pairs must have shape (n, 2)");
    }

    const int natm = atm.shape(0);
    const int nbas = bas.shape(0);
    auto* atm_data = atm.data();
    auto* bas_data = bas.data();
    auto* env_data = env.data();

    std::vector<int> ao_size(nbas);
    int max_size = 0;
    for (int s = 0; s < nbas; ++s) {
        ao_size[s] = CINTcgto_spheric(s, bas_data);
        max_size = std::max(max_size, ao_size[s]);
    }

    const std::size_t n_bra = bra_pairs.shape(0);
    const std::size_t n_ket = ket_pairs.shape(0);
    auto bra_view = bra_pairs.view();
    auto ket_view = ket_pairs.view();

    // Shell indices per pair and row/column offset of each block.
    std::vector<std::array<int, 2>> bra_sh(n_bra), ket_sh(n_ket);
    std::vector<std::size_t> row_off(n_bra + 1, 0), col_off(n_ket + 1, 0);
    for (std::size_t pb = 0; pb < n_bra; ++pb) {
        const int A = bra_view(pb, 0), B = bra_view(pb, 1);
        if (A < 0 || B < 0 || A >= nbas || B >= nbas)
            throw std::invalid_argument("bra_pairs shell index out of range");
        bra_sh[pb] = {A, B};
        row_off[pb + 1] = row_off[pb] + static_cast<std::size_t>(ao_size[A]) * ao_size[B];
    }
    for (std::size_t pk = 0; pk < n_ket; ++pk) {
        const int C = ket_view(pk, 0), D = ket_view(pk, 1);
        if (C < 0 || D < 0 || C >= nbas || D >= nbas)
            throw std::invalid_argument("ket_pairs shell index out of range");
        ket_sh[pk] = {C, D};
        col_off[pk + 1] = col_off[pk] + static_cast<std::size_t>(ao_size[C]) * ao_size[D];
    }

    const std::size_t nrow = row_off[n_bra];
    const std::size_t ncol = col_off[n_ket];

    auto out = make_zeros<nb::numpy, double, 2>({nrow, ncol});
    double* data = out.data();

    // Parallelize over bra pairs; each chunk owns a scratch quartet buffer and writes disjoint rows.
    auto kernel = [&](std::size_t pb_begin, std::size_t pb_end) {
        std::vector<double> buf(static_cast<std::size_t>(max_size) * max_size * max_size * max_size);
        for (std::size_t pb = pb_begin; pb < pb_end; ++pb) {
            const int A = bra_sh[pb][0], B = bra_sh[pb][1];
            const int nA = ao_size[A], nB = ao_size[B];
            const std::size_t r0 = row_off[pb];
            for (std::size_t pk = 0; pk < n_ket; ++pk) {
                const int C = ket_sh[pk][0], D = ket_sh[pk][1];
                const int nC = ao_size[C], nD = ao_size[D];
                const std::size_t c0 = col_off[pk];
                int shells[4] = {A, B, C, D};
                int dims[4] = {nA, nB, nC, nD};
                intor(buf.data(), dims, shells, atm_data, natm, bas_data, nbas, env_data, NULL,
                      NULL);
                // libcint Fortran-order: idx(iA,iB,iC,iD) = iA + iB*nA + iC*nA*nB + iD*nA*nB*nC.
                for (int iA = 0; iA < nA; ++iA) {
                    for (int iB = 0; iB < nB; ++iB) {
                        const std::size_t r = r0 + static_cast<std::size_t>(iA) * nB + iB;
                        for (int iC = 0; iC < nC; ++iC) {
                            for (int iD = 0; iD < nD; ++iD) {
                                const std::size_t idx =
                                    iA + static_cast<std::size_t>(iB) * nA +
                                    static_cast<std::size_t>(iC) * nA * nB +
                                    static_cast<std::size_t>(iD) * nA * nB * nC;
                                data[r * ncol + c0 + static_cast<std::size_t>(iC) * nD + iD] =
                                    buf[idx];
                            }
                        }
                    }
                }
            }
        }
    };

    parallel_for_chunked(n_bra, kernel);

    return out;
}

} // namespace forte2
