#pragma once

#include <cmath>
#include <cassert>
#include "helpers/ndarray.h"

#define MACHEPS 1e-9
#define TAYLOR_THRES 1e-3

namespace forte2 {

double taylor_exp(double z) {
    int n = static_cast<int>(0.5 * (15.0 / TAYLOR_THRES + 1)) + 1;
    if (n > 0) {
        double value = z;
        double tmp = z;
        for (int x = 0; x < n - 1; x++) {
            tmp *= -1.0 * z * z / (x + 2);
            value += tmp;
        }
        return value;
    } else {
        return 0.0;
    }
}

double regularized_denominator(double x, double s) {
    double z = std::sqrt(s) * x;
    if (fabs(z) <= MACHEPS) {
        return taylor_exp(z) * std::sqrt(s);
    } else {
        return (1. - std::exp(-s * x * x)) / x;
    }
}

void compute_T2_block(np_tensor4_complex& t2, np_vector& ei, np_vector& ej, np_vector& ea,
                      np_vector& eb, double flow_param) {
    auto t2_v = t2.view();
    auto ei_v = ei.view();
    auto ej_v = ej.view();
    auto ea_v = ea.view();
    auto eb_v = eb.view();

    size_t ni = ei.shape(0);
    size_t nj = ej.shape(0);
    size_t na = ea.shape(0);
    size_t nb = eb.shape(0);

    assert(ni == t2.shape(0));
    assert(nj == t2.shape(1));
    assert(na == t2.shape(2));
    assert(nb == t2.shape(3));
    double denom;
    for (size_t i = 0; i < ni; i++) {
        for (size_t j = 0; j < nj; j++) {
            for (size_t a = 0; a < na; a++) {
                for (size_t b = 0; b < nb; b++) {
                    denom = ei_v(i) + ej_v(j) - ea_v(a) - eb_v(b);
                    t2_v(i, j, a, b) *= static_cast<std::complex<double>>(
                        regularized_denominator(denom, flow_param));
                }
            }
        }
    }
}

void compute_T1_block(np_matrix_complex& t1, np_vector& ei, np_vector& ea, double flow_param) {
    auto t1_v = t1.view();
    auto ei_v = ei.view();
    auto ea_v = ea.view();

    size_t ni = ei.shape(0);
    size_t na = ea.shape(0);

    assert(ni == t1.shape(0));
    assert(na == t1.shape(1));
    double denom;
    for (size_t i = 0; i < ni; i++) {
        for (size_t a = 0; a < na; a++) {
            denom = ei_v(i) - ea_v(a);
            t1_v(i, a) *=
                static_cast<std::complex<double>>(regularized_denominator(denom, flow_param));
        }
    }
}

void renormalize_V_block(np_tensor4_complex& v, np_vector& ei, np_vector& ej, np_vector& ea,
                         np_vector& eb, double flow_param) {
    auto v_v = v.view();
    auto ei_v = ei.view();
    auto ej_v = ej.view();
    auto ea_v = ea.view();
    auto eb_v = eb.view();

    size_t ni = ei.shape(0);
    size_t nj = ej.shape(0);
    size_t na = ea.shape(0);
    size_t nb = eb.shape(0);

    assert(ni == v.shape(0));
    assert(nj == v.shape(1));
    assert(na == v.shape(2));
    assert(nb == v.shape(3));
    double denom;
    for (size_t i = 0; i < ni; i++) {
        for (size_t j = 0; j < nj; j++) {
            for (size_t a = 0; a < na; a++) {
                for (size_t b = 0; b < nb; b++) {
                    denom = ei_v(i) + ej_v(j) - ea_v(a) - eb_v(b);
                    v_v(i, j, a, b) += v_v(i, j, a, b) * static_cast<std::complex<double>>(
                                                             std::exp(-flow_param * denom * denom));
                }
            }
        }
    }
}

void renormalize_3index(np_tensor3_complex& v, double& ep, np_vector& eq, np_vector& er,
                        np_vector& es, double flow_param) {
    auto v_v = v.view();
    auto eq_v = eq.view();
    auto er_v = er.view();
    auto es_v = es.view();

    size_t nq = eq.shape(0);
    size_t nr = er.shape(0);
    size_t ns = es.shape(0);

    assert(nq == v.shape(0));
    assert(nr == v.shape(1));
    assert(ns == v.shape(2));
    double denom;
    for (size_t q = 0; q < nq; q++) {
        for (size_t r = 0; r < nr; r++) {
            for (size_t s = 0; s < ns; s++) {
                denom = ep + eq_v(q) - er_v(r) - es_v(s);
                v_v(q, r, s) *=
                    static_cast<std::complex<double>>((1 + std::exp(-flow_param * denom * denom)) *
                                                      regularized_denominator(denom, flow_param));
            }
        }
    }
}

} // namespace forte2