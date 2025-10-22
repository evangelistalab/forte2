#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "integrals/basis.h"
#include "helpers/ndarray.h"

extern "C" {
# include <cint.h>

int CINTcgto_spheric(const int i, const int* bas);
int cint1e_ovlp_sph(double* buf, int* shls, int* atm, int natm, int* bas, int nbas, double* env);
}

namespace forte2 {

nb::ndarray<nb::numpy, double, nb::ndim<2>, nb::f_contig>
cint_int1e_ovlp_sph(const size_t nao, nb::ndarray<nb::numpy, int, nb::ndim<2>> atm,
                    nb::ndarray<nb::numpy, int, nb::ndim<2>> bas,
                    nb::ndarray<nb::numpy, double, nb::ndim<1>> env) {
    int natm = atm.shape(0);
    int nbas = bas.shape(0);

    std::vector<int> ao_offset(nbas + 1, 0);
    for (int i = 0; i < nbas; ++i) {
        ao_offset[i + 1] = ao_offset[i] + CINTcgto_spheric(i, bas.data());
    }
    auto ints = make_zeros<nb::numpy, double, 2, nb::f_contig>(std::array<size_t, 2>{nao, nao});
    double* buf = ints.data();

    auto* atm_data = atm.data();
    auto* bas_data = bas.data();
    auto* env_data = env.data();

    int di, dj;

    for (int i = 0; i < nbas; ++i) {
        di = CINTcgto_spheric(i, bas_data);
        for (int j = 0; j <= i; ++j) {
            dj = CINTcgto_spheric(j, bas_data);
            int shells[2] = {i, j};
            double* buf_ij = buf + ao_offset[i] * nao + ao_offset[j];
            cint1e_ovlp_sph(buf_ij, shells, atm_data, natm, bas_data, nbas, env_data);
        }
    }

    return ints;
}
} // namespace forte2